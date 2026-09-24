"""
do-MNIST experiment: a CopSens latent-factor ball around a prefit CNN.

Three things differ from the linear experiments and shape everything here.

1. **The estimand is analytic.** h_*(x) = E[Y | do(x)] is a function of the digit
   label alone (`sem.ate_of`, `sem.h_star`), never of the pixels, so `sem.f` does
   not exist. The query path scores the exemplars through `estimand`, the sweep
   path fills `SweepData.estimand` from the population draw's labels.

2. **The outcome nets are PREFIT.** One ERM on the observed draw and one DA+ERM on
   the augmented draw are trained per replicate and handed to every PI variant as
   its centre; a third, the ERM+INV net (the ERM subject to invariance on the DA
   pairs), is trained when PI+INV is centred on it (`inv_recenter: inv`) or
   ERM+INV is listed. That inverts the usual build order: methods can only be
   built AFTER the data exists, so the runners rebuild through the method factory
   once the nets are there.

3. **The split and mix-in protocol.** The 60k training images are partitioned
   once (split A trains the nets, B is the only thing the PI machinery sees, C is
   what the gamma selection bisects on); a `mix_in` share of the DA measure's rows
   is swapped back to its observed row on the DA+ family only (Asm. 1b), so the
   DA+ methods fit on a mixed `GX` while the PI+INV pairs stay unmixed (`GX_inv`).
   `draw_replicate` performs the whole protocol in the reference order, so the
   torch stream is consumed identically on the query and the sweep path.

The query path is the exemplar figure plus the population `run.json`, and, under
`experiment.query.tint`, one tint sweep per digit (`run_tint_sweep`): a single image
rendered from blue to red and scored by every fitted method, with the target
constant along it. Of the sweeps only `gamma` is wired (a ratio grid around the DECLARED gamma); see
`DoMNISTOrchestrator.get_sweep_runner_cls` for what the others still need.
"""

import dataclasses
import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from loguru import logger

from src.data_augmentors.do_mnist import DoMNISTDA as DA
from src.experiments.base import ExperimentDataContext, ExperimentOrchestrator, SweepData
from src.experiments.configs import ANNOTATE_SWEEP_PLOT, DOMNIST_CONFIG, MethodRegistry, TintSpec
from src.experiments.generic_runner import STRATEGIES, GenericQuerySweep
from src.experiments.utils import save
from src.experiments.utils.constants import SUBDIR_QUERY
from src.experiments.utils.diagnostics import erm_report, net_noise, prescreen, probe
from src.experiments.utils.metrics import STATUS_CATEGORIES, evaluate_queries
from src.experiments.utils.plotting import create_digit_sweep_plot, create_query_sweep_plot
from src.methods.regression import GradientDescentERM, InvariantGradientDescentERM
from src.oracle import OracleParameters
from src.sem.do_mnist import SPLIT_DEFAULT, DoMNISTSEM, split_key, tint_of

EXPERIMENT_NAME = "do_mnist"

#: the point estimators of the backend; everything else predicts an interval
POINT_METHODS: frozenset = frozenset({"ERM", "DA+ERM", "ERM+INV"})


class Flatten:
    """Duck-types PolynomialFeatures, so images reach the methods through the
    existing `poly_transform` hook and no base class has to learn about pixels.
    A 2-D input is already flat and comes back as is."""

    def fit_transform(self, X):
        X = np.asarray(X)
        # reshape on a C-contiguous array is a VIEW, which is what keeps the 1.2M
        # draw at ~2.8 GB rather than 5.6 GB. Enforce it, so a future slicing
        # change cannot silently double the footprint.
        if not X.flags["C_CONTIGUOUS"]:
            raise ValueError("Flatten needs C-contiguous input to stay a view")
        return X.reshape(len(X), -1)


def load_1min() -> float:
    """The 1-minute load average (NaN where the OS has none), read outside every
    timer so the results table can state what the timings ran under."""
    return float(os.getloadavg()[0]) if hasattr(os, "getloadavg") else float("nan")


def flat(X) -> np.ndarray:
    """(N, ...) images to (N, d) float32 rows, the layout every method sees."""
    return np.ascontiguousarray(np.asarray(X).reshape(len(X), -1), dtype=np.float32)


def train_pair(X, GX, y, init_seed, net="domnist-fast", **train_kw):
    """The matched ERM / DA+ERM pair: same architecture, same init_seed, same batch
    order, differing only in their inputs."""
    logger.info(f"do-mnist: training the ERM pair on {len(X):,} draws")
    return {
        "X": GradientDescentERM(net).fit(X, y, init_seed=init_seed, **train_kw),
        "GX": GradientDescentERM(net).fit(GX, y, init_seed=init_seed, **train_kw),
    }


def erm_inv_fit_kwargs(train_kw: dict[str, Any], tau: float, epochs: int | None = None) -> dict[str, Any]:
    """The ERM+INV net's fit kwargs: the pair's training kwargs, the augmented
    Lagrangian's constants from `DOMNIST_CONFIG`, the invariance target `tau`, and
    the epoch count (`epochs`, else `erm_inv_epochs`, else the pair's)."""
    kw = dict(train_kw)
    epochs = DOMNIST_CONFIG.erm_inv_epochs if epochs is None else epochs
    if epochs is not None:
        kw["epochs"] = int(epochs)
    kw.update(
        al_tau=float(tau),
        al_mu0=DOMNIST_CONFIG.erm_inv_mu0,
        al_growth=DOMNIST_CONFIG.erm_inv_growth,
        al_updates_per_epoch=DOMNIST_CONFIG.erm_inv_updates_per_epoch,
        al_mu_max=DOMNIST_CONFIG.erm_inv_mu_max,
    )
    return kw


def train_erm_inv(X, GX, y, init_seed, tau, train_kw, net="domnist-fast", epochs=None):
    """The ERM+INV net on the UNMIXED pairs, matched to the pair by `init_seed`.
    `train_kw` is the pair's training dict, taken whole: its `epochs` yields to
    `epochs`, then to `erm_inv_epochs` (`erm_inv_fit_kwargs`)."""
    kw = erm_inv_fit_kwargs(train_kw, tau, epochs)
    logger.info(f"do-mnist: training ERM+INV on {len(X):,} pairs, tau {tau:g}, {kw.get('epochs', 1)} epoch(s)")
    return InvariantGradientDescentERM(net).fit(X, y, GX=GX, init_seed=init_seed, **kw)


def e_inv(net, X, GX) -> float:
    """E[(mu(X) - mu(GX))^2] with mu clipped as the PI+INV constraint reads it."""
    lo_hi = DOMNIST_CONFIG.attainable if DOMNIST_CONFIG.mu_clip else None
    mu_x = np.asarray(net.predict_mean(X), dtype=float).ravel()
    mu_gx = np.asarray(net.predict_mean(GX), dtype=float).ravel()
    if lo_hi is not None:
        mu_x, mu_gx = np.clip(mu_x, *lo_hi), np.clip(mu_gx, *lo_hi)
    return float(np.mean((mu_x - mu_gx) ** 2))


def _net_rho(nets, X, GX, y) -> float:
    """sigma~^2/sigma^2 on the prefit pair (`diagnostics.net_noise`): the GX net's
    MSE on GX over the X net's MSE on X, both against y on the SAME rows. NaN-free by
    construction unless the X net fits y exactly, in which case rho is left at 1.
    The prescreen's `rho` is the same number on the replicate's B rows."""
    if nets is None:
        return 1.0
    noise = net_noise(nets, X, GX, y)
    if not noise["rho_ok"]:
        logger.warning("do-mnist: rho on the prefit nets not computable; falling back to 1.")
    return noise["rho"]


def log_vacuous(bounds: dict[str, np.ndarray]):
    """An all-INFEASIBLE method draws nothing, so the figure shows only a legend
    entry. Say so, or it reads as a plotting bug."""
    for name, prediction in bounds.items():
        prediction = np.asarray(prediction)
        if prediction.ndim == 3 and not np.isfinite(prediction).any():
            logger.warning(
                f"{name}: INFEASIBLE at every query -- it contributes a "
                "legend entry and no band. Check the floor/budget pair "
                "logged above."
            )


#: (constrained method, the UNCONSTRAINED model on the same ball), by the PI+INV
#: centre: under `off` PI+INV is the X-centred ball with the pairs bolted on, so
#: PI is its parent; under `on` (`RecentredInvCopSens`) it is the post-DA ball;
#: under `inv` its parent would be a plain ball around the ERM+INV net, which is
#: not built, so PI+INV has no entry
def nested_in(inv_recenter: str) -> dict[str, str]:
    nesting = {
        "PI+INV": "DA+PI" if inv_recenter == "on" else "PI",
        "DA+PI+IV": "DA+PI",
        "PI&DA+PI": "DA+PI",
        "PI&DA+PI+IV": "DA+PI",
    }
    if inv_recenter == "inv":
        del nesting["PI+INV"]
    return nesting


def log_nesting(bounds: dict[str, np.ndarray], inv_recenter: str = "off"):
    """The SLSQP multi-start is non-convex, so `constrained subset parent` is not
    guaranteed. Logged, never raised: a violation is information about the
    optimiser, not a reason to discard the run.

    Adding a constraint can only shrink the feasible set, so a constrained method
    must never come out WIDER than its parent. Coming out much narrower while the
    constraint is inactive is the other tell, and it is the one that under-covers.
    """
    for name, parent in nested_in(inv_recenter).items():
        if not {name, parent} <= set(bounds):
            continue
        child = np.asarray(bounds[name])
        base = np.asarray(bounds[parent])
        if child.ndim != 3 or base.ndim != 3:
            continue
        width_child = child[..., 1] - child[..., 0]
        width_parent = base[..., 1] - base[..., 0]
        violations = int(np.nansum(width_child > width_parent + 1e-9))
        if violations:
            logger.warning(
                f"{name} wider than its parent {parent} at {violations}/"
                f"{width_parent.size} queries: the non-convex multi-start missed an "
                "optimum."
            )


def domnist_oracle(sem) -> OracleParameters:
    """The oracle record of a do-MNIST pair. The augmentation is exactly invariant
    by construction (a small translation, rotation or colour shift never moves
    E[Y | do(x)]), so eps* and the T piece are 0, and gamma* is Lemma 2's
    bias^2/sigma^2, which the CopSens LATENT budget does not measure (the sweeps
    read the declared gamma instead, `DoMNISTMixin.fit_gamma`). Nothing here
    calls the pixel-level oracle, which would need a `sem.f`."""
    bias_sq, sigma_sq = float(sem.bias_sq), float(sem.sigma_sq)
    logger.info(
        f"do-mnist oracle: bias^2 {bias_sq:.4g} sigma^2 {sigma_sq:.4g} (Lemma-2 gamma* {bias_sq / sigma_sq:.4g}, "
        "not the CopSens budget); eps* 0 by exact invariance."
    )
    return OracleParameters(
        gamma_star=bias_sq / sigma_sq,
        epsilon_star=0.0,
        gamma_z_star=None,
        bias_sq=bias_sq,
        sigma_sq=sigma_sq,
        rho=None,
        eps_iv_star=0.0,
        eps_iv_z_star=0.0,
        eps_rms=0.0,
        eta=0.0,
        epsilon_star_pointwise=0.0,
    )


def population(sem_test, n: int, seed: int):
    """The evaluation population: `n` observational draws from MNIST TEST at `seed`
    (never in A/B/C) with the analytic target. Coverage and width are reported HERE,
    not on the exemplars. Returns (P, h_star, h_erm, f) with P flat float32."""
    X_img, _ = sem_test.sample(n, seed=seed)
    last = dict(sem_test.last_)
    return flat(X_img), sem_test.h_star(last["f"]), sem_test.h_erm(last["f"], last["C"]), last["f"]


# =============================================================================
# THE REPLICATE PROTOCOL
# =============================================================================


@dataclass
class DoMNISTData:
    """One replicate's fit arrays, flat float32 rows from split B.

    `GX` and `G` are the MIXED augmentation (the DA+ family's measure, identity
    rows where the mix-in swapped the observed row back in); `GX_inv` is the same
    rows' UNMIXED augmentation, the PI+INV pairs. `nets` is the prefit pair, plus
    the ERM+INV net under "INV" when the replicate trained it.
    """

    X: np.ndarray
    GX: np.ndarray
    GX_inv: np.ndarray
    y: np.ndarray
    G: np.ndarray
    nets: dict[str, Any]
    mask_a: np.ndarray
    mask_b: np.ndarray
    split_key: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


def draw_replicate(
    sem_train,
    sem_test,
    da,
    seed: int,
    n_samples: int,
    n_pi: int,
    mix_in: float,
    hyperparameters: dict[str, Any] | None,
    net: str,
    split: dict[str, int] | None,
    split_seed: int,
    train_inv: bool = False,
    erm_inv_tau: float | None = None,
    inv_fits=None,
) -> DoMNISTData:
    """The reference protocol, in the reference order, so the torch stream is
    consumed identically by every caller:

    1. the image-level split of the training set at `split_seed`;
    2. the nets' draw from A at `seed`, augmented on the global torch stream;
    3. the ERM on the observed draw, then (`train_inv`) the ERM+INV net on the
       unmixed pairs, the mix-in IN PLACE on the augmented draw (seed `[seed, 1]`),
       the DA+ERM on the mixture;
    4. the ERM report on the held-out probe;
    5. the PI rows from B at `seed + 1`, augmented, and a mixed COPY at `[seed, 2]`
       for the DA+ family; the pairs stay unmixed;
    6. the prescreen (rho off the ERM and DA+ERM nets, the linear rho_linear and
       the shift-operator spectrum) on the mixed B rows.

    The DA+ERM fit reseeds torch (CPU and CUDA) at its start, so everything from
    it on is bit-identical with or without the ERM+INV fit. `inv_fits`, when
    given, replaces that fit: it is called at the same point as
    `inv_fits(X=X, GX=GX, y=y, init_seed=seed, train_kw=train_kw)` and returns the
    net to keep (the diagnostic script fits several there).
    """
    from src.experiments.utils import set_seed

    set_seed(seed)
    parts = sem_train.split(split or SPLIT_DEFAULT, split_seed)
    key = split_key(parts)
    logger.info(f"do-mnist: split { ({k: len(v) for k, v in parts.items()}) } seed {split_seed} key {key}")

    train_kw = dict(hyperparameters or {})
    sem_a = parts["A"]
    X_img, y = sem_a.sample(n_samples, seed=seed)
    start = time.perf_counter()
    GX_img, _ = da(X_img)
    augment_seconds_a = time.perf_counter() - start
    X = flat(X_img)
    del X_img
    GX = flat(GX_img)
    del GX_img

    # every net in a replicate shares init_seed: a matched pair, differing only in inputs
    seconds, loads = {}, {}
    loads["X"] = [load_1min()]
    start = time.perf_counter()
    erm = GradientDescentERM(net).fit(X, y, init_seed=seed, **train_kw)
    seconds["X"] = time.perf_counter() - start
    loads["X"].append(load_1min())
    inv_net = None
    # the ERM+INV net needs the UNMIXED pairs, so it trains before the mix-in
    if train_inv or inv_fits is not None:
        tau = DOMNIST_CONFIG.erm_inv_tau if erm_inv_tau is None else float(erm_inv_tau)
        loads["INV"] = [load_1min()]
        start = time.perf_counter()
        if inv_fits is not None:
            inv_net = inv_fits(X=X, GX=GX, y=y, init_seed=seed, train_kw=train_kw)
        else:
            inv_net = train_erm_inv(X, GX, y, init_seed=seed, tau=tau, train_kw=train_kw, net=net)
        seconds["INV"] = time.perf_counter() - start
        loads["INV"].append(load_1min())
    # mix-in IN PLACE on the A-draw GX: the ERM (and ERM+INV) are done with the
    # unmixed pairs, and the DA+ERM is the only consumer left. X is never written
    GX, _, mask_a = da.mix_in(X, GX, None, mix_in, seed=[seed, 1], inplace=True)
    logger.info(f"do-mnist: mix_in A-draw {int(mask_a.sum()):,} / {len(GX):,} rows observed (DA+ERM)")
    loads["GX"] = [load_1min()]
    start = time.perf_counter()
    da_erm = GradientDescentERM(net).fit(GX, y, init_seed=seed, **train_kw)
    seconds["GX"] = time.perf_counter() - start
    loads["GX"].append(load_1min())
    del X, GX
    nets = {"X": erm, "GX": da_erm} if inv_net is None else {"X": erm, "GX": da_erm, "INV": inv_net}

    diagnostics = erm_report(erm, probe(sem_test, DOMNIST_CONFIG.probe_samples, DOMNIST_CONFIG.probe_seed))
    diagnostics.update({f"train_seconds_{key}": value for key, value in seconds.items()})
    diagnostics.update({f"train_load_{key}": value for key, value in loads.items()})

    # the PI rows: a draw from B at seed + 1, never the nets' draw
    sem_b = parts["B"]
    Xb_img, yb = sem_b.sample(n_pi, seed=seed + 1)
    idx = sem_b.last_["idx"]
    if not (np.isin(idx, sem_b.subset_).all() and not np.isin(idx, sem_a.subset_).any()):
        raise RuntimeError("do-mnist: the PI rows left split B")
    start = time.perf_counter()
    GXb_img, Gb = da(Xb_img)
    augment_seconds_b = time.perf_counter() - start
    Xb = flat(Xb_img)
    del Xb_img
    GXb = flat(GXb_img)
    del GXb_img
    # a mixed COPY for the DA+ family (DA+PI, DA+PI+IV and the prescreen whose rho
    # feeds DA+PI+IV); GXb / Gb stay unmixed for the PI+INV pairs in every mode
    GXb_da, Gb_da, mask_b = da.mix_in(Xb, GXb, Gb, mix_in, seed=[seed, 2])
    logger.info(f"do-mnist: mix_in B-rows {int(mask_b.sum()):,} / {len(GXb):,} (DA+PI, DA+PI+IV, prescreen)")
    diagnostics.update(mix_in_n_A=float(mask_a.sum()), mix_in_n_B=float(mask_b.sum()))
    # the two DA passes, charged by the table to the methods that consume them
    diagnostics.update(augment_seconds_A=augment_seconds_a, augment_seconds_B=augment_seconds_b)
    # each net's invariance error on the UNMIXED B pairs, clipped as PI+INV reads it
    for name, model in nets.items():
        diagnostics[f"E_inv_B_{name}"] = e_inv(model, Xb, GXb)
    if "INV" in nets:
        diagnostics.update(_erm_inv_record(nets["INV"]))
        logger.info(
            f"do-mnist: E_inv on the B pairs ERM {diagnostics['E_inv_B_X']:.4g} DA+ERM {diagnostics['E_inv_B_GX']:.4g} "
            f"ERM+INV {diagnostics['E_inv_B_INV']:.4g} (tau {diagnostics.get('erm_inv_al_tau', float('nan')):g}); "
            f"ERM+INV sha1 {diagnostics['erm_inv_state_sha1']}"
        )
    diagnostics.update(
        prescreen(Xb, yb, GXb_da, link=DOMNIST_CONFIG.link, keep=DOMNIST_CONFIG.spectrum_keep, nets=nets)
    )
    logger.info(
        f"do-mnist seed {seed}: rho={diagnostics['rho']:.4f} (nets; linear {diagnostics['rho_linear']:.4f}) "
        f"tr(S)/k={diagnostics['tr_S_over_k']:.4f} contracts={diagnostics['contracts']}"
    )
    return DoMNISTData(
        X=Xb,
        GX=GXb_da,
        GX_inv=GXb,
        y=yb,
        G=np.asarray(Gb_da, dtype=float),
        nets=nets,
        mask_a=mask_a,
        mask_b=mask_b,
        split_key=key,
        diagnostics=diagnostics,
    )


def _erm_inv_record(model) -> dict[str, Any]:
    """The ERM+INV net's provenance for `run.json`: its hash, its target and the
    augmented Lagrangian's trace `(step, c_bar, lam, mu)` with its last window."""
    record: dict[str, Any] = {"erm_inv_state_sha1": model.state_sha1()}
    trace = getattr(model, "al_trace_", None)
    if trace is not None:
        record["erm_inv_al_tau"] = float(model.al_tau_)
        record["erm_inv_al_trace"] = [list(row) for row in trace]
        if trace:
            _, c_bar, lam, mu = trace[-1]
            record.update(erm_inv_al_updates=len(trace), erm_inv_al_c_bar=c_bar, erm_inv_al_lam=lam, erm_inv_al_mu=mu)
    return record


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    return value


# =============================================================================
# QUERY SWEEP
# =============================================================================


class DoMNISTQuerySweep(GenericQuerySweep):
    """GenericQuerySweep builds methods BEFORE data exists; prefit nets need the
    reverse order, so the early build is suppressed and redone once nets exist."""

    def __init__(
        self,
        method_factory=None,
        sem_test_factory=None,
        n_pi: int = 60_000,
        n_queries: int = 2_000,
        mix_in: float = 0.05,
        split: dict[str, int] | None = None,
        split_seed: int = 420,
        pop_seed: int = 44,
        exemplar_seed: int = 420,
        net: str = "domnist-fast",
        inv_recenter: str = "off",
        train_inv: bool = False,
        erm_inv_tau: float | None = None,
        default_gamma: float = 0.1,
        default_epsilon: float = 0.05,
        **kwargs,
    ):
        self.sem_test_factory = sem_test_factory
        self.n_pi, self.n_queries, self.mix_in = n_pi, n_queries, mix_in
        self.split, self.split_seed, self.pop_seed = split, split_seed, pop_seed
        self.exemplar_seed, self.net, self.inv_recenter = exemplar_seed, net, inv_recenter
        self.train_inv, self.erm_inv_tau = train_inv, erm_inv_tau
        self.nets = None
        self.data_ = None
        # FORWARD the budgets: GenericQuerySweep stores them. Setting them here
        # instead would be silently overwritten by super's own defaults.
        super().__init__(method_factory=None, default_gamma=default_gamma, default_epsilon=default_epsilon, **kwargs)

        # rebuilt here, not in the base: handing the factory up would make
        # GenericQuerySweep read `epsilon_iv`, which runs the linear constraint
        # floor on the pixel design and logs a meaningless number. The T-as-IV
        # budget is the block epsilon (the reference's `from_eps`)
        if method_factory is not None:
            self.methods = method_factory(
                gamma=self.default_gamma,
                epsilon=self.default_epsilon,
                epsilon_iv=self.default_epsilon,
                rho=self.fit_rho(),
                outcome_models=self.nets,
            )

    def prepare_pair(self, sem, da, features=None):
        return domnist_oracle(sem)

    def fit_rho(self) -> float:
        """rho on the prefit nets' MSE over the B rows, the same ratio the
        intersections read off their branches (`IntersectedCopSens.rho`)."""
        return _net_rho(self.nets, self.X, self.GX, self.y)

    def _load_data(self):
        """The replicate protocol at the runner's seed; (X, GX, y, G) flat, the
        unmixed pairs kept on `data_` for `setup_data`."""
        self.sem_test = self.sem_test_factory()
        self.data_ = draw_replicate(
            self.sem,
            self.sem_test,
            self.da,
            seed=self.seed,
            n_samples=self.n_samples,
            n_pi=self.n_pi,
            mix_in=self.mix_in,
            hyperparameters=self.hyperparameters,
            net=self.net,
            split=self.split,
            split_seed=self.split_seed,
            train_inv=self.train_inv,
            erm_inv_tau=self.erm_inv_tau,
        )
        self.nets = self.data_.nets
        return self.data_.X, self.data_.GX, self.data_.y, self.data_.G

    def setup_data(self) -> ExperimentDataContext:
        context = super().setup_data()
        context.GX_inv = self.data_.GX_inv
        return context

    def _load(self, name: str, phase: str) -> None:
        """The 1-minute load average at a phase boundary, outside every timer, so
        the table can say what the latency was timed under. Same for every method."""
        if not hasattr(self, "load_"):
            self.load_ = {}
        self.load_.setdefault(name, {})[phase] = load_1min()

    def before_fit(self, name: str) -> None:
        self._load(name, "fit_before")

    def after_fit(self, name: str, model) -> None:
        """Solve each constrained model's floor once, timed, before its first
        predict: PI+INV and DA+PI+IV directly, an intersection through its DA
        branch. The floor is cached on (radius, n_starts), so the floor gate inside
        the first predict reuses it and the predict time is the solves alone. The
        fit phase's closing load is read after the floor, the last part it charges."""
        if not hasattr(self, "floor_seconds_"):
            self.floor_seconds_ = {}
        branch = getattr(model, "augmented", None)
        target = model if branch is None else branch
        budget = getattr(target, "_budget", None)
        if budget is not None and budget() is not None:
            start = time.perf_counter()
            target.constraint_floor(target._radius(target.gamma))
            self.floor_seconds_[name] = time.perf_counter() - start
        self._load(name, "fit_after")

    def get_sweep_values(self) -> np.ndarray:
        """The frozen digit exemplars from the UNRESTRICTED training set at
        `exemplar_seed`, a visualisation set that is never scored. GenericQuerySweep
        would run a PCA over the images here, which is a crash rather than a bad plot."""
        colors = DOMNIST_CONFIG.exemplar_colors
        exemplars, self.digits_ = self.sem.exemplars(self.exemplar_seed, colors=colors)
        # same digits and tints at full resolution: `subsample` is a modelling
        # choice, and the figure has no reason to inherit it
        self.exemplar_images_, _ = self.sem.exemplars(self.exemplar_seed, colors=colors, subsample=1)
        logger.info(f"do-mnist: exemplar digits {np.asarray(self.digits_).tolist()}")
        return flat(exemplars)

    def estimand(self, queries) -> np.ndarray:
        """h_* at the exemplars, off their digit labels."""
        return self.sem.ate_of(self.digits_)

    # ------------------------------------------------------------- population

    def evaluate_population(self) -> dict[str, Any]:
        """Every fitted model scored on the evaluation population: coverage, width,
        approximation and worst error, wall clock (predict seconds per query) and
        the status split for the intervals, RMSE for the point estimators, each
        method's `fit_model` and floor seconds, the NaN-row share, and the
        INV floor / budget pair where PI+INV was fitted, and under `inv` the
        ERM+INV centre's own constraint value `erm_inv_con0`. A method whose every
        interval is NaN reads coverage NaN (`evaluate_queries`), not 0."""
        P, h_star, _, _ = population(self.sem_test, self.n_queries, self.pop_seed)
        target = np.asarray(h_star).reshape(-1, 1)  # (n, 1), the shape `sem.f` gives elsewhere
        out: dict[str, Any] = {}
        outcomes: dict[str, np.ndarray] = {}
        for name, model in self.models_.items():
            self._load(name, "predict_before")
            start = time.perf_counter()
            prediction = np.asarray(model.predict(P))
            elapsed = time.perf_counter() - start
            self._load(name, "predict_after")
            outcomes[name] = prediction
            out[f"load_{name}"] = dict(self.load_[name])
            out[f"fit_seconds_{name}"] = float(self.fit_seconds_.get(name, float("nan")))
            if name in getattr(self, "floor_seconds_", {}):
                out[f"floor_seconds_{name}"] = float(self.floor_seconds_[name])
            if name in POINT_METHODS or prediction.ndim == 1 or prediction.shape[1] != 2:
                out[f"rmse_{name}"] = float(np.sqrt(np.nanmean((prediction.ravel() - target.ravel()) ** 2)))
                continue
            record = evaluate_queries(
                target, prediction, statuses=getattr(model, "query_status", None), elapsed=elapsed
            )
            out[f"coverage_{name}"] = float(record.coverage)
            out[f"width_{name}"] = float(record.interval_width)
            out[f"approx_error_{name}"] = float(record.approximation_error)
            out[f"worst_error_{name}"] = float(record.worst_error)
            out[f"wall_clock_{name}"] = float(record.wall_clock)
            out[f"nan_{name}"] = float(np.isnan(prediction).any(axis=1).mean())
            out[f"status_{name}"] = dict(zip(STATUS_CATEGORIES, [int(c) for c in record.status_counts], strict=True))
            if name == "PI+INV":
                out["inv_floor"] = float(model.constraint_floor(model._radius(model.gamma)))
                out["inv_budget"] = float(model._budget())
                logger.info(f"PI+INV: floor {out['inv_floor']:.4g} against budget eps^2 {out['inv_budget']:.4g}")
                if self.inv_recenter == "inv":
                    out["erm_inv_con0"] = float(model._con_at_zero())
                    self._check_inv_centre(out["erm_inv_con0"], out["inv_floor"], out["inv_budget"])
        # the ERM+INV net scored as a point estimate even when it is not plotted
        if self.nets is not None and "INV" in self.nets and "ERM+INV" not in self.models_:
            prediction = np.asarray(self.nets["INV"].predict(P))
            out["rmse_ERM+INV"] = float(np.sqrt(np.nanmean((prediction.ravel() - target.ravel()) ** 2)))
        # per query, so the table can bootstrap its bands instead of reading means
        save(target, "population_values", EXPERIMENT_NAME, "pkl", subdir=SUBDIR_QUERY)
        save(outcomes, "population_outcomes", EXPERIMENT_NAME, "pkl", subdir=SUBDIR_QUERY)
        return out

    @staticmethod
    def _check_inv_centre(con0: float, floor: float, budget: float):
        """The one feasibility check of the ERM+INV centre: its own invariance error
        on the PI+INV constraint rows, and the floor, inside the eps^2 ball. Warns,
        never stops, never moves tau."""
        logger.info(f"PI+INV (inv): the ERM+INV centre's constraint value {con0:.4g} against eps^2 {budget:.4g}")
        if con0 > budget or floor > budget:
            logger.warning(
                f"PI+INV (inv): centre value {con0:.4g} / floor {floor:.4g} against eps^2 {budget:.4g}: PI+INV is "
                "likely infeasible at this epsilon."
            )


# =============================================================================
# TINT SWEEP
# =============================================================================


#: the per-digit tint figure's legend column, a figure fraction (the digit sweep's)
TINT_LEGEND_WIDTH: float = 0.24


def _tints_of_rows(X) -> np.ndarray:
    """`tint_of` on flat (n, 3 s^2) rows; `s` from the row width, not hard-coded."""
    X = np.asarray(X)
    side = int(round(np.sqrt(X.shape[1] / 3)))
    if 3 * side * side != X.shape[1]:
        raise ValueError(f"rows of width {X.shape[1]} are not (3, s, s) images")
    return tint_of(X.reshape(len(X), 3, side, side))


def run_tint_sweep(runner, spec: TintSpec, experiment: str = EXPERIMENT_NAME) -> dict[str, Any]:
    """One image per digit rendered at every tint of the grid and scored by every
    fitted model of the query run (no refit). Saves per digit `tint_{d}_values`,
    `tint_{d}_outcomes` (the query results layout, ATE first) and `tint_{d}_images`
    (the full-resolution blue and red endpoints), the figure `tint_{d}_sweep`, and
    once `tint_density` (the B rows' tints before and after DA). Returns the
    `run.json` entry."""
    tints = np.linspace(spec.range[0], spec.range[1], spec.sweep_samples)
    digits = sorted(spec.digits)
    source = DOMNIST_CONFIG.tint_image_source
    sem = runner.sem_test if source == "test" else runner.sem
    n = len(tints)
    queries = np.concatenate([flat(sem.tinted(runner.exemplar_seed, d, tints)) for d in digits])
    target = np.asarray(runner.sem.ate_of(np.repeat(digits, n)), dtype=float).reshape(-1, 1)

    # one predict per method over every digit at once: the per-predict set-up is paid once
    predictions, status = {}, {}
    for name, model in runner.models_.items():
        prediction = np.asarray(model.predict(queries))
        predictions[name] = prediction
        if name not in POINT_METHODS and prediction.ndim == 2 and prediction.shape[1] == 2:
            record = evaluate_queries(target, prediction, statuses=getattr(model, "query_status", None))
            status[name] = dict(zip(STATUS_CATEGORIES, [int(c) for c in record.status_counts], strict=True))

    for i, digit in enumerate(digits):
        rows = slice(i * n, (i + 1) * n)
        outcomes = {"ATE": target[rows]} if "ATE" in runner.methods else {}
        for name in runner.methods:
            if name not in predictions:
                continue
            p = predictions[name][rows]
            outcomes[name] = p.reshape(n, 1) if name in POINT_METHODS else p[:, np.newaxis, :]
        endpoints = sem.tinted(runner.exemplar_seed, digit, tints[[0, -1]], subsample=1)
        save(tints, f"tint_{digit}_values", experiment, "pkl", subdir=SUBDIR_QUERY)
        save(outcomes, f"tint_{digit}_outcomes", experiment, "pkl", subdir=SUBDIR_QUERY)
        save(endpoints, f"tint_{digit}_images", experiment, "pkl", subdir=SUBDIR_QUERY)
        log_vacuous(outcomes)
        create_query_sweep_plot(
            tints,
            outcomes,
            fname=f"tint_{digit}",
            experiment=experiment,
            legend_width=TINT_LEGEND_WIDTH,
            mark_missing=True,
            **ANNOTATE_SWEEP_PLOT["tint"],
        )

    # the DA measure the DA+ methods fit on (the mixed GX), against the observed rows
    density = {"before": _tints_of_rows(runner.data_.X), "after": _tints_of_rows(runner.data_.GX)}
    save(density, "tint_density", experiment, "pkl", subdir=SUBDIR_QUERY)
    logger.info(f"do-mnist tint sweep: digits {digits}, {n} tints on {spec.range}, image from {source}")
    return dict(
        digits=digits,
        grid=tints.tolist(),
        range=list(spec.range),
        sweep_samples=int(spec.sweep_samples),
        image_source=source,
        image_seed=int(runner.exemplar_seed),
        status=status,
        # provenance, so the aggregate can tell sweeps of different runs apart
        split_key=runner.data_.split_key,
        inv_recenter=runner.inv_recenter,
        gamma=float(runner.default_gamma),
        epsilon=float(runner.default_epsilon),
    )


# =============================================================================
# PARAMETER SWEEP MIXIN
# =============================================================================


class DoMNISTMixin:
    """Prefit nets and the B rows, composed OVER a sweep strategy.

    A mixin and not a base class: it has to sit ahead of the strategy in the MRO
    rather than replace it. The sweep is data-constant: the B draw is the step's
    data, so no re-augmentation happens per step.
    """

    def __init__(
        self,
        sem_test_factory=None,
        n_pi: int = 60_000,
        n_queries: int = 2_000,
        mix_in: float = 0.05,
        split: dict[str, int] | None = None,
        split_seed: int = 420,
        pop_seed: int = 44,
        net: str = "domnist-fast",
        train_inv: bool = False,
        erm_inv_tau: float | None = None,
        **kwargs,
    ):
        self.sem_test_factory = sem_test_factory
        self.n_pi, self.n_queries, self.mix_in = n_pi, n_queries, mix_in
        self.split, self.split_seed, self.pop_seed, self.net = split, split_seed, pop_seed, net
        self.train_inv, self.erm_inv_tau = train_inv, erm_inv_tau
        self._nets = {}
        self._data = {}
        self._population = None
        super().__init__(**kwargs)

    def setup_sems_and_das(self):
        super().setup_sems_and_das()
        # the test SEM holds the held-out MNIST images; the population is drawn from it
        self.sems_test = [self.sem_test_factory() for _ in range(self.n_experiments)]

    def prepare_pair(self, sem, da, features=None):
        return domnist_oracle(sem)

    # -------------------------------------------------------- per-experiment policy

    def fit_gamma(self, experiment_index: int) -> float:
        """The DECLARED gamma, not the oracle gamma*: the CopSens budget is a
        latent-space quantity (a radius on the factor scores of the outcome net's
        input) that Lemma 2's bias^2/sigma^2 does not measure. The gamma sweep
        therefore runs its ratio grid around the selected value."""
        return float(self.default_gamma)

    def fit_epsilon(self, experiment_index: int, step_index: int = 0, data=None) -> float:
        """A MODELLING ASSUMPTION, not estimable: the true invariance defect of h_*
        is 0 for these ops (none of them can change E[Y | do(x)]), and the
        configured value is finite-sample slack. No floor report: the linear
        `constraint_floor` does not apply to the latent ball."""
        return float(self.default_epsilon)

    def fit_epsilon_iv(self, experiment_index: int, step_index: int = 0, data=None, ratio: float = 1.0) -> float:
        """Same reasoning as fit_epsilon: the T-as-IV budget is the block epsilon
        (the reference's `from_eps`), and `ratio` (the epsilon sweep's per-step
        call) does not move it."""
        return float(self.default_epsilon)

    def fit_rho(self, experiment_index: int, data=None) -> float:
        """rho on the prefit nets' MSE over the step's rows (see `_net_rho`); the
        linear `rho_hat` would refit OLS on pixels, which is not the class here."""
        if data is None or getattr(data, "GX", None) is None:
            return 1.0
        return _net_rho(self._nets.get(experiment_index), data.X, data.GX, data.y)

    def method_kwargs(self, experiment_index: int) -> dict[str, object]:
        return {"outcome_models": self._nets[experiment_index]}

    # ------------------------------------------------------------- base sample

    def _shared_population(self):
        """One population for every experiment, at `pop_seed` from MNIST test."""
        if self._population is None:
            P, h_star, _, _ = population(self.sems_test[0], self.n_queries, self.pop_seed)
            self._population = (P, np.asarray(h_star).reshape(-1, 1))
        return self._population

    def _draw_base(self, experiment_index: int, n_samples=None):
        """The replicate protocol at `seed + j`; the population as the test set and
        its analytic h_* as the estimand."""
        n_total = self.n_samples if n_samples is None else int(n_samples)
        data = draw_replicate(
            self.sems[experiment_index],
            self.sems_test[experiment_index],
            self.das[experiment_index],
            seed=self.seed + experiment_index,
            n_samples=n_total,
            n_pi=self.n_pi,
            mix_in=self.mix_in,
            hyperparameters=self.hyperparameters,
            net=self.net,
            split=self.split,
            split_seed=self.split_seed,
            train_inv=self.train_inv,
            erm_inv_tau=self.erm_inv_tau,
        )
        self._nets[experiment_index] = data.nets
        self._data[(experiment_index, n_samples)] = data
        P, h_star = self._shared_population()
        return (data.X, data.X, data.y, P, h_star)

    def _sweep_data(
        self, experiment_index: int, n_samples: int | None = None, common_random: bool = False, **augment_kwargs
    ) -> SweepData:
        """The B draw as the step's data: the mixed GX and G for the DA+ family, the
        unmixed pairs for PI+INV. No second DA pass, no common-random-number knob."""
        if augment_kwargs:
            raise NotImplementedError(f"do-mnist: the DA is not re-drawn per step ({sorted(augment_kwargs)}).")
        X_raw, X, y, X_test, estimand, Z = self._base_data(experiment_index, n_samples)
        data = self._data[(experiment_index, n_samples)]
        return SweepData(X=X, y=y, GX=data.GX, G=data.G, X_test=X_test, estimand=estimand, Z=Z, GX_inv=data.GX_inv)


# =============================================================================
# ORCHESTRATOR
# =============================================================================


class DoMNISTOrchestrator(ExperimentOrchestrator):
    """Orchestrator for do-MNIST experiments."""

    build_panel = False  # queries are digit exemplars, not a PC sweep

    def __init__(
        self,
        augmentation: str,
        gamma: float,
        epsilon: float,
        n_pi: int = 60_000,
        n_queries: int = 2_000,
        net: str = "domnist-fast",
        n_components: int = 32,
        mix_in: float = 0.05,
        target_coverage: float = 0.99,
        inv_recenter: str = "off",
        erm_inv_tau: float | None = None,
        gamma_z_star: float = 0.0,
        calibrate_sigma: bool = True,
        split: dict[str, int] | None = None,
        split_seed: int = 420,
        pop_seed: int = 44,
        exemplar_seed: int = 420,
        augmentation_amounts: dict[str, float] | None = None,
        **kwargs,
    ):
        self.augmentation = augmentation
        self.augmentation_amounts = augmentation_amounts
        self.gamma, self.epsilon = gamma, epsilon
        self.n_pi, self.n_queries, self.net = n_pi, n_queries, net
        self.n_components, self.mix_in = n_components, mix_in
        self.target_coverage = target_coverage
        self.inv_recenter = {True: "on", False: "off"}.get(inv_recenter, str(inv_recenter).strip().lower())
        self.erm_inv_tau = float(DOMNIST_CONFIG.erm_inv_tau if erm_inv_tau is None else erm_inv_tau)
        self.gamma_z_star, self.calibrate_sigma = gamma_z_star, calibrate_sigma
        self.split = {k: int(v) for k, v in (split or SPLIT_DEFAULT).items()}
        self.split_seed, self.pop_seed, self.exemplar_seed = split_seed, pop_seed, exemplar_seed
        self.tint_ = None  # the query tint spec, set by `run`
        self.toggles = dict(
            recalibrate=kwargs.get("recalibrate", True),
            pad=kwargs.get("pad", False),
            clipy=kwargs.get("clipy", True),
            n_jobs=kwargs.get("n_jobs", 1),
            mean_match=kwargs.get("mean_match", True),
        )
        outer = self

        class DoMNISTRegistry(MethodRegistry):
            @staticmethod
            def build_methods(names):
                return outer._build(names, gamma=gamma, epsilon=epsilon, epsilon_iv=epsilon, **outer.toggles)

        super().__init__(EXPERIMENT_NAME, DoMNISTRegistry(), **kwargs)

    @property
    def train_inv(self) -> bool:
        """The ERM+INV net is trained only when something reads it: PI+INV centred
        on it, or ERM+INV listed. A PI / DA+PI selection run never pays for it."""
        methods = set(self.kwargs.get("methods") or ())
        return (self.inv_recenter == "inv" and "PI+INV" in methods) or "ERM+INV" in methods

    # ---------------------------------------------------------------- factories

    def _sem_factory(self, train: bool = True):
        return DoMNISTSEM(
            seed=self.kwargs["seed"],
            train=train,
            alpha=DOMNIST_CONFIG.alpha,
            beta=DOMNIST_CONFIG.beta,
            eta=DOMNIST_CONFIG.eta,
            subsample=DOMNIST_CONFIG.subsample,
        )

    def _sem_test_factory(self):
        return self._sem_factory(train=False)

    def _da_factory(self, sem=None, append=None):
        if append is not None:
            raise NotImplementedError("do-mnist: the DA chain is fixed by the block; no robustness chain is appended.")
        return DA(self.augmentation, amounts=self.augmentation_amounts)

    def _poly_factory(self):
        return Flatten()

    def _build(self, names, gamma, epsilon, epsilon_iv=None, rho=1.0, outcome_models=None, **toggles):
        return MethodRegistry.build_methods(
            names,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_iv=epsilon_iv,
            rho=rho,
            backend="copsens",
            outcome_models=outcome_models,
            n_components=self.n_components,
            inv_recenter=self.inv_recenter,
            calibrate_sigma=self.calibrate_sigma,
            gamma_z_star=self.gamma_z_star,
            **toggles,
        )

    def build_methods(
        self, gamma: float, epsilon: float, epsilon_iv=None, n_jobs=None, rho=1.0, outcome_models=None, epsilon_iv_z=0.0
    ):
        """Methods at explicit budgets. `n_jobs` overrides the toggle. `epsilon_iv_z`
        is accepted because the runner hands it to every factory; the backend has
        no real-Z instrument and never reads it."""
        toggles = self.toggles if n_jobs is None else {**self.toggles, "n_jobs": n_jobs}
        return self._build(
            self.kwargs["methods"],
            gamma=gamma,
            epsilon=epsilon,
            epsilon_iv=self.epsilon if epsilon_iv is None else epsilon_iv,
            rho=rho,
            outcome_models=outcome_models,
            **toggles,
        )

    # ------------------------------------------------------------------ runners

    def _runner_kwargs(self) -> dict[str, Any]:
        return dict(
            sem_factory=self._sem_factory,
            sem_test_factory=self._sem_test_factory,
            da_factory=self._da_factory,
            poly_transform=self._poly_factory(),
            default_gamma=self.gamma,
            default_epsilon=self.epsilon,
            n_pi=self.n_pi,
            n_queries=self.n_queries,
            mix_in=self.mix_in,
            split=self.split,
            split_seed=self.split_seed,
            pop_seed=self.pop_seed,
            net=self.net,
            train_inv=self.train_inv,
            erm_inv_tau=self.erm_inv_tau,
        )

    def get_query_runner_cls(self) -> type[GenericQuerySweep]:
        outer = self

        class ConfiguredQuerySweep(DoMNISTQuerySweep):
            def __init__(inner, **kwargs):
                super().__init__(
                    method_factory=outer.build_methods,
                    exemplar_seed=outer.exemplar_seed,
                    inv_recenter=outer.inv_recenter,
                    **outer._runner_kwargs(),
                    **kwargs,
                )

        return ConfiguredQuerySweep

    def get_sweep_runner_cls(self, param: str) -> type:
        if param != "gamma":
            raise NotImplementedError(
                f"do_mnist {param} sweep is not wired. What each needs: epsilon, an "
                "absolute grid (eps* is 0 here, so a ratio of it is empty); omega, an "
                "augment_kwargs_fn on DA.strength with common random numbers through "
                "the torch generator; n, the nets retrained per step; m, a tiled fit "
                "of the prefit nets' ball; recalibrate, a rho the latent ball reads."
            )

        outer, Strategy = self, STRATEGIES[param]

        class ConfiguredSweep(DoMNISTMixin, Strategy):  # MRO: mixin first
            def __init__(inner, **kwargs):
                super().__init__(
                    test_fraction=DOMNIST_CONFIG.test_fraction,
                    experiment_name=EXPERIMENT_NAME,
                    **outer._runner_kwargs(),
                    **kwargs,
                )

        return ConfiguredSweep

    def run(self, plan):
        """Keep the tint spec for `_plot_query_sweep`, and hand the base a plan
        without it (the base refuses one)."""
        self.tint_ = plan.tint
        super().run(dataclasses.replace(plan, tint=None))

    def _run_perf(self, perf_spec):
        """The perf sweeps run on the epsilon grid, which is not wired here
        (`get_sweep_runner_cls`)."""
        logger.warning("do-mnist: perf skipped, the epsilon sweep it runs on is not wired here.")

    # --------------------------------------------------------------------- plot

    def _plot_query_sweep(self, runner, results):
        """x-axis is the digit exemplars, so thumbnails replace a numeric axis. The
        population metrics, the ERM report, the prescreen and the provenance go to
        `run.json` beside the pkls."""
        log_nesting(results, self.inv_recenter)
        log_vacuous(results)
        digits = [int(d) for d in runner.digits_]

        save(np.asarray(digits), "treatment_values", self.name, "pkl", subdir=SUBDIR_QUERY)
        save(results, "outcome_values", self.name, "pkl", subdir=SUBDIR_QUERY)

        ate = np.asarray(results["ATE"]).reshape(len(digits), -1) if "ATE" in results else runner.estimand(None)
        record = dict(runner.data_.diagnostics)
        record.update(runner.evaluate_population())
        record.update(self._exemplar_summary(results, ate))
        if self.tint_ is not None:
            # after the population: its status split is read right after its own predict
            record["tint"] = run_tint_sweep(runner, self.tint_, self.name)
        record.update(
            gamma=float(self.gamma),
            epsilon=float(self.epsilon),
            target_coverage=float(self.target_coverage),
            gamma_z_star=float(self.gamma_z_star),
            inv_recenter=self.inv_recenter,
            erm_inv_tau=float(self.erm_inv_tau),
            train_inv=bool(self.train_inv),
            iv_rho=float(runner.fit_rho()),
            n_components=int(self.n_components),
            calibrate_sigma=bool(self.calibrate_sigma),
            link=DOMNIST_CONFIG.link,
            mix_in=float(self.mix_in),
            augmentation=self.augmentation,
            augmentation_amounts=self.augmentation_amounts,
            n=int(self.kwargs["n_samples"]),
            n_pi=int(self.n_pi),
            n_queries=int(self.n_queries),
            seed=int(self.kwargs["seed"]),
            alpha=DOMNIST_CONFIG.alpha,
            beta=DOMNIST_CONFIG.beta,
            eta=DOMNIST_CONFIG.eta,
            subsample=DOMNIST_CONFIG.subsample,
            split=self.split,
            split_seed=int(self.split_seed),
            split_key=runner.data_.split_key,
            net=self.net,
            pop_seed=int(self.pop_seed),
            exemplar_seed=int(self.exemplar_seed),
            digits=digits,
            ate=np.asarray(ate).ravel().tolist(),
            methods=list(self.kwargs["methods"]),
            **{f"toggle_{k}": v for k, v in self.toggles.items()},
            # the compute the timings were read on, for the results table
            blas_threads=DOMNIST_CONFIG.blas_threads,
            cpu_count=len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count(),
        )
        save(_jsonable(record), "run", self.name, "json", subdir=SUBDIR_QUERY)
        headline = {k: v for k, v in record.items() if k.startswith(("coverage_", "width_", "rmse_"))}
        logger.info(f"do-mnist population: {headline}")

        create_digit_sweep_plot(runner.exemplar_images_, results, labels=digits, experiment=self.name)

    @staticmethod
    def _exemplar_summary(results, ate) -> dict[str, float]:
        """EXEMPLAR-only width and coverage (the visualisation queries). The
        headline numbers are the population ones. `ate` is (n_queries, 1)."""
        out = {}
        for name, p in results.items():
            if name == "ATE":
                continue
            p = np.asarray(p)
            if p.ndim == 3:
                lo, hi = p[:, :, 0], p[:, :, 1]
                out[f"width_exemplar_{name}"] = float(np.nanmean(hi - lo))
                out[f"coverage_exemplar_{name}"] = float(np.nanmean((lo <= ate) & (ate <= hi)))
            else:
                out[f"rmse_exemplar_{name}"] = float(np.sqrt(np.nanmean((p - ate) ** 2)))
        return out
