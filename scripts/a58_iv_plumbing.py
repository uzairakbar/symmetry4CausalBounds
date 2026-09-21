"""A58: the instrument reaches the runner: split at the draw, carried beside X, handed to every solver.

refactor3 touches the carrier and its consumers only (SS8.1): `SEM.iv_width` and
`split_instruments`, the two split sites in generic_runner.py, `SweepData.Z` and
`ExperimentDataContext.Z` (None spelled (n, 0) on construction), `fit_model`'s
`Z`, `eps_iv_z_star` in the oracle, and the floor guard on the joint instrument,
skippable for a declared budget. No shipped SEM emits an instrument yet (the sim
and cigarette SEMs learn to in batch C), so the legs that need `iv_width > 0` run
on two SEM stubs defined here: a generator (the linear SEM with a valid Z drawn
beside X) and a recorded pool with a row-aligned `iv_pool`. Legs:

  (D)   the digest leg (scripts/digest_leg.py), as a56 and a57: one query panel and
        one gamma sweep step per dataset at the shipped configuration against the
        reference recorded on 14509db, no tolerance. Catches: a reordered draw, a
        Z-tilde that is not G elementwise, a budget that moved, a (n, 0) that is not
        exactly today's None. Misses: every path with a real Z, which is what (i),
        (iv) and (v) are for.
  (i)   every consumer in SS8.1's table, once, on the stubs: both split sites
        (`_draw_base` on a pool and on a generator, `_load_data`), `f` and `extent`
        on the stripped test set, the oracle on the stripped draw and on `iv_pool`
        (`eps_iv_z_star` against an independent computation of the same norm),
        `pool` staying treatment only, the n-sweep slicing Z, the m-sweep tiling it
        with the untiled copy beside `X_base`, the recalibrate and omega sweeps
        carrying it, `fit_model` handing the real Z to the non-DA +IV methods and
        the intersection, the joint (G, Z) to the DA+ ones and nothing to the rest,
        the m-sweep's baselines fitting the untiled Z, a 5-tuple `_draw_base` and a
        4-tuple `_load_data` (the do-MNIST overrides' shapes) padded with (n, 0),
        do_mnist.py untouched since the base commit, both dataclasses spelling None
        as (n, 0), and every +IV method solving OK end to end through `runner.run`.
        Catches: a consumer seeing the wide X (the stub's `f` and `extent` raise on
        it), Z dropped or mis-sliced anywhere between the draw and the fit, a
        method handed the wrong instrument. Misses: the shipped SEMs' own emitters.
  (ii)  with an empty set, on sim, optical and cigarettes through the production
        orchestrators: `Z` is (n, 0), `column_stack([G, Z])` is G elementwise with
        an identical QR, `fit_model` hands DA+PI+IV exactly G and the intersection
        an (n, 0) Z, and the DA+PI+IV and intersection DA-branch fitted arrays are
        bit-equal to today's direct `Z=G` fit; optical still applies the poly
        features. Catches: a None reaching `column_stack`, a reshaped or copied G
        that no longer QRs the same, the intersection pre-stacked to [G, G, Z].
        Misses: the numbers, (D).
  (iii) `eps_iv_z_star` is exactly 0.0 for an empty Z (None and (n, 0), touching
        no RNG), and the runner's `epsilon_iv` (param sweeps and the query runner,
        all three datasets) is bit-identical to the old formula recomputed here from
        the oracle, EPS_TOL and the floor on (GX, G). Catches: a tolerance or a
        jitter inside the Z piece, a floor that moved off G alone. Misses: a wrong Z
        piece under a real Z, which (i)'s independent norm covers.
  (iv)  row alignment survives a bootstrap: on a pool whose Z column is 2 X_0 + 1
        and whose `sample` resamples rows with replacement, the rows come off
        `train_test_split` before the columns do, so Z_train is 2 X_train_0 + 1 to
        the bit, through `_draw_base`, `_sweep_data`, the n-slice and the m-tiling.
        Catches: columns split before rows (the plan's break-it), Z sliced or tiled
        differently from X. Misses: the cigarette SEM's own bootstrap (batch C).
  (v)   the floor guard is skipped exactly when the budget is declared, and the
        floor is logged: on a pool with an INVALID instrument the T constraint's own
        floor sits above r_T^2; `declared_iv=True` returns r_T = eps_iv_star + EPS_TOL
        untouched and logs one INFO line naming that floor (the one on G alone, not
        on the stacked instrument) and "never raised"; `declared_iv=False` reads the
        same T piece and would raise a budget under the floor to sqrt(9 floor); the
        query runner does the same. The oracle measures `eps_iv_z_star` on both
        paths, and it is the radius of a separate constraint.
        Catches: the skip dropped, the declared path reading the joint budget, the
        floor computed on G alone. Misses: the declared gamma_z reaching the
        solver, which is the orchestrator's (batch C).
  (vi)  the parallel dispatch: with the instrument in play, PI+IV, DA+PI+IV and the
        intersection at n_jobs 4 return bounds and statuses array_equal to n_jobs 1
        on the same draw, and a gamma sweep through the runner at n_jobs 4 matches
        the serial one. Leg (D) is serial now, so this is the only leg through loky.
        Catches: an unpicklable attribute on the fitted IV classes, a chunk that
        solves a different problem. Misses: timing.
  (vii) the two budgets per SS2.6 row on the oracle path (the batch B rulings): the
        runner hands the factory `epsilon_iv_z = eps_iv_z_star + EPS_TOL` beside the
        T-side `epsilon_iv`; PI+IV, PI+INV+IV and the intersection's baseline branch
        carry it, DA+PI+IV carries the T-side term, and the intersection is feasible
        on every query under a real Z, both branches OK. Under an empty Z the term
        is exactly 0.0 (inert). Catches: `epsilon_iv_z` not forwarded or ignored
        (the baseline bound is then 0 and every query INFEASIBLE), the T-side term
        leaking into the non-DA methods. Misses: a floor on this term, by ruling.
  (viii) the declared path: with `declared_iv=True` and a declared gamma_z handed
        to the registry (as the cigarette orchestrator will), the runner hands
        `epsilon_iv_z = 0.0`, so standalone PI+IV solves at exactly r_Z = s sqrt(gamma_z)
        to the bit, as does the intersection's baseline, while DA+PI+IV keeps
        r_T = eps_iv_star + EPS_TOL and every one of them solves OK. Catches: the
        T-side budget handed to the non-DA methods (their bound would read the joint
        0.0699 rather than 0.0625 on the cigarette panel), the declared skip lost.
        Misses: gamma_z's own route from the yaml (batch C).

    MPLBACKEND=Agg python scripts/a58_iv_plumbing.py [--seed 0] [--reference JSON] [--skip-digest]

`--skip-digest` drops leg (D) and exists for the break-it runs of the other legs;
the committed state is always gated with it on.
"""

import argparse
import inspect
import os
import subprocess
import sys

import numpy as np
from loguru import logger

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import digest_leg  # noqa: E402
from munch import munchify  # noqa: E402

import src.experiments.generic_runner as runner_module  # noqa: E402
from src.data_augmentors.simulation import NullSpaceTranslation  # noqa: E402
from src.experiments.base import ExperimentDataContext, SweepData  # noqa: E402
from src.experiments.configs import EPS_TOL, FLOOR_GUARD_R, MethodRegistry, resolve_dataset_block  # noqa: E402
from src.experiments.do_mnist import DoMNISTMixin, DoMNISTQuerySweep  # noqa: E402
from src.experiments.generic_runner import STRATEGIES, GenericQuerySweep  # noqa: E402
from src.experiments.utils import set_seed  # noqa: E402
from src.experiments.utils.model_fitting import fit_model, instrument_columns, joint_instrument  # noqa: E402
from src.main import ORCHESTRATORS  # noqa: E402
from src.methods.sensitivity_models import (  # noqa: E402
    InstrumentalVariablePartialR2 as IVPartialR2,
)
from src.methods.sensitivity_models import SolveStatus, constraint_floor  # noqa: E402
from src.oracle import eps_iv_z_star  # noqa: E402
from src.sem.abstract import StructuralEquationModel as SEM  # noqa: E402
from src.sem.simulation import OUTCOME_NOISE_STD, LinearSimulationSEM  # noqa: E402

BASE_COMMIT = "2f07683"
# the oracle estimates the real-Z piece on CALIBRATION_SAMPLES rows, so the stub's
# draw has the same n, or its moment noise outruns the oracle's plus the tolerance
N, K, M = 2048, 6, 2
TEST_FRACTION = 0.02
TOGGLES = dict(recalibrate=True, pad=False, clipy=False, mean_match=True, n_jobs=1)
NAMES = [
    "PI",
    "PI+INV",
    "PI+IV",
    "PI+INV+IV",
    "DA+PI",
    "DA+PI+IV",
    "PI&DA+PI",
    "PI&DA+PI+IV",
    "ERM",
    "ERM+IV",
    "DA+ERM+IV",
]
IV_NAMES = ("PI+IV", "PI+INV+IV", "DA+PI+IV", "PI&DA+PI+IV")
NON_DA_IV = ("PI+IV", "PI+INV+IV")
INTERSECTION = "PI&DA+PI+IV"
DECLARED_GAMMA_Z = 2**-8
METRICS = ("interval_width", "coverage", "worst_error", "approximation_error")
GAMMA, EPSILON = 0.5, EPS_TOL
# the gamma sweep's ratios: at and above gamma*, where the oracle budgets admit h*
# by construction; below it the IV ball can legitimately miss the moment line (SS3.3)
GAMMA_GRID = [1.0, 2.0]
ORACLE_POOL_DRAWS, ORACLE_POOL_SEED = runner_module.ORACLE_POOL_DRAWS, runner_module.ORACLE_POOL_SEED
FAIL = []
SKIPPED = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAIL.append(name)


def skip(tag, reason):
    """A leg that cannot run here: printed and counted in the summary, never a silent PASS."""
    print(f"  [SKIP] {tag} {reason}")
    SKIPPED.append(f"{tag} {reason}")


# ------------------------------------------------------------------ SEM stubs


class InstrumentedSEM(LinearSimulationSEM):
    """The linear SEM emitting m instrument columns after X: Z = X U + e, U
    orthonormal and orthogonal to W_XXi, so Z is valid and correlated with X.
    A stand-in for the SEM batch C builds; `f` and `extent` refuse a wide X."""

    def __init__(self, iv_dim: int = M, **kwargs):
        super().__init__(**kwargs)
        self._iv_dim = int(iv_dim)
        basis = np.random.randn(self.treatment_dimension, self._iv_dim)
        basis -= np.outer(self.W_XXi, self.W_XXi @ basis)
        self.U = np.linalg.qr(basis)[0]

    @property
    def iv_width(self) -> int:
        return self._iv_dim

    def sample(self, N: int = 1, intervention: bool = False, **kwargs):
        X, y = super().sample(N=N, intervention=intervention, **kwargs)
        Z = X @ self.U + 0.5 * np.random.randn(N, self._iv_dim)
        return np.column_stack([X, Z]), y

    def f(self, X):
        if np.shape(X)[1] != self.treatment_dimension:
            raise ValueError(f"f saw {np.shape(X)[1]} columns; the treatment has {self.treatment_dimension}")
        return super().f(X)

    def extent(self, X):
        if np.shape(X)[1] != self.treatment_dimension:
            raise ValueError(f"extent saw {np.shape(X)[1]} columns; the treatment has {self.treatment_dimension}")
        return np.zeros(len(X))


class PooledInstrumentedSEM(SEM):
    """A recorded (X, y) with an instrument pool beside it. `sample` returns
    [X | Z] for the first N rows or, with `bootstrap`, a row resample with
    replacement off the global stream, like the cigarette cluster bootstrap. Z's
    first column is 2 X_0 + 1, so alignment is checkable to the bit; its second is
    X_1 plus noise when `valid`, else xi plus noise, an INVALID instrument that
    puts the joint IV floor above the T-as-IV budget."""

    def __init__(self, n: int = N, k: int = K, bootstrap: bool = False, valid: bool = True, seed: int = 1):
        rng = np.random.RandomState(seed)
        self.bootstrap = bootstrap
        direction = rng.randn(k)
        self.W_XXi = direction / np.linalg.norm(direction)
        self.W_XY = rng.randn(k, 1)
        self._kappa_sq = 0.5
        kappa = np.sqrt(self._kappa_sq)
        X = rng.randn(n, k)
        xi = kappa * (X @ self.W_XXi) + np.sqrt(1.0 - self._kappa_sq) * rng.randn(n)
        self.X = X
        self.y = ((X @ self.W_XY).ravel() + xi + OUTCOME_NOISE_STD * rng.randn(n)).reshape(-1, 1)
        second = (X[:, 1] if valid else xi) + 0.3 * rng.randn(n)
        self.Z = np.column_stack([2.0 * X[:, 0] + 1.0, second])

    @property
    def iv_width(self) -> int:
        return self.Z.shape[1]

    @property
    def pool(self):
        return self.X, self.y

    @property
    def iv_pool(self):
        return self.Z

    @property
    def bias_sq(self) -> float:
        return self._kappa_sq

    @property
    def sigma_sq(self) -> float:
        return 1.0 - self._kappa_sq + OUTCOME_NOISE_STD**2

    def sample(self, N: int = 1, **kwargs):
        index = np.random.randint(0, len(self.X), N) if self.bootstrap else np.arange(min(N, len(self.X)))
        return np.column_stack([self.X[index], self.Z[index]]), self.y[index]


class Recorder:
    """Stands in for a model: keeps what `fit_model` hands it."""

    def __init__(self, name):
        self.name = name
        self.kw = None

    def fit(self, **kw):
        self.kw = kw
        return self

    def predict(self, X, **kw):
        return np.zeros((len(X), 2)) if "PI" in self.name else np.zeros(len(X))


# ------------------------------------------------------------------ builders


def factory(gamma, epsilon, epsilon_iv=None, rho=1.0, n_jobs=None, epsilon_iv_z=0.0, gamma_z=0.0, **kwargs):
    toggles = TOGGLES if n_jobs is None else {**TOGGLES, "n_jobs": n_jobs}
    return MethodRegistry.build_methods(
        NAMES,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_iv=epsilon_iv,
        epsilon_iv_z=epsilon_iv_z,
        gamma_z=gamma_z,
        rho=rho,
        **toggles,
    )


def recorders(names):
    return {name: (lambda n=name: Recorder(n)) for name in names}


def make_generator():
    return InstrumentedSEM(treatment_dimension=K, gamma=GAMMA)


def da_from_sem(sem, strength=0.0):
    """The sim's null-space translation; `strength` > 0 adds an out-of-kernel
    component, so h* is no longer T-invariant and the T-side term is O(strength)."""
    return NullSpaceTranslation(sem.W_XY, kernel_dim=0, strength=strength)


# the DA strength (vii) runs at, so eps_iv_star is far from 0 and the T-side budget
# cannot coincide with the Z term (a T-invariant h* makes the two meet to the bit)
MISSPECIFIED_STRENGTH = 0.5


def sweep_runner(param, sem_factory, grid, seed=0, declared_iv=False, n_jobs=1, gamma_z=0.0, strength=0.0, **extra):
    """One configured strategy on a stub, the way the orchestrators build them; a
    `gamma_z` closes over the factory as the cigarette orchestrator's will."""
    return STRATEGIES[param](
        sem_factory=sem_factory,
        da_factory=lambda sem: da_from_sem(sem, strength),
        poly_transform=None,
        test_fraction=TEST_FRACTION,
        default_gamma=GAMMA,
        default_epsilon=EPSILON,
        experiment_name="simulation",
        param_grid_override=grid,
        methods=factory(GAMMA, EPSILON, n_jobs=n_jobs),
        method_factory=lambda **kw: factory(**{**kw, "n_jobs": n_jobs, "gamma_z": gamma_z}),
        seed=seed,
        n_samples=N,
        n_experiments=1,
        sweep_samples=2,
        declared_iv=declared_iv,
        **{**TOGGLES, "n_jobs": n_jobs},
        **extra,
    )


def query_runner(sem, seed=0, declared_iv=False, cls=GenericQuerySweep, strength=0.0):
    return cls(
        sem_factory=lambda: sem,
        da_factory=lambda: da_from_sem(sem, strength),
        poly_transform=None,
        method_factory=factory,
        default_gamma=GAMMA,
        default_epsilon=EPSILON,
        eps_tol=EPS_TOL,
        raw_gamma=False,
        methods=factory(GAMMA, EPSILON),
        seed=seed,
        n_samples=N,
        n_experiments=1,
        sweep_samples=4,
        declared_iv=declared_iv,
        **TOGGLES,
    )


def production(name, kind="sweep"):
    """The shipped block through its orchestrator, as leg (D) and src/main.py run it."""
    block = {**digest_leg.TOGGLES, **digest_leg.BLOCKS[name], "n_experiments": 1, "sweep_samples": 4}
    block = resolve_dataset_block(name, block)
    set_seed(block["seed"])
    orchestrator = ORCHESTRATORS[name](**block, hyperparameters=munchify(digest_leg.HYPERPARAMETERS))
    if kind == "query":
        return orchestrator.get_query_runner_cls()(
            methods=orchestrator.methods, **{**orchestrator._get_clean_kwargs(), "n_experiments": 1}
        )
    return orchestrator.get_sweep_runner_cls("gamma")(
        methods=orchestrator.methods, method_factory=orchestrator.build_methods, **orchestrator._get_clean_kwargs()
    )


def budgets(runner, data):
    return dict(
        gamma=runner.fit_gamma(0),
        epsilon=runner.fit_epsilon(0, 0, data),
        epsilon_iv=runner.fit_epsilon_iv(0, 0, data),
        epsilon_iv_z=runner.fit_epsilon_iv_z(0, data),
        rho=runner.fit_rho(0, data),
    )


def recorded(runner, data):
    """What `fit_model` hands each of the runner's own methods, through `build_models`."""
    runner.methods, runner.method_factory = recorders(list(runner.methods)), None
    return {name: model.kw for name, model in runner.build_models(0, 0, data).items()}


def expected_instrument(name, Z, Z_solo, G):
    """The instrument BLOCKS each method must be handed (SS2.6): the PI classes take
    `T` and `Z` separately, the point estimate takes the one stacked matrix. `{}`
    means no instrument at all."""
    T = np.reshape(G, (len(G), -1))
    if name in ("PI+IV", "ERM+IV"):
        return {"Z": Z_solo}
    if name in ("PI+INV+IV", "PI&DA+PI+IV"):
        return {"Z": Z}
    if name == "DA+PI+IV":
        return {"Z": Z, "T": T}
    if name == "DA+ERM+IV":
        return {"Z": np.column_stack([T, Z])}
    return {}


def check_dispatch(tag, kws, Z, Z_solo, G):
    for name, kw in kws.items():
        want = expected_instrument(name, Z, Z_solo, G)
        if not want:
            handed = [key for key in ("Z", "T") if kw.get(key) is not None]
            check(f"{tag} {name}: no instrument handed", not handed, f"got {handed}")
            continue
        for key, block in want.items():
            got = kw.get(key)
            same = got is not None and np.shape(got) == np.shape(block) and np.array_equal(got, block)
            check(f"{tag} {name}: {key} block {np.shape(block)} handed exactly", same, f"got {np.shape(got)}")
        for key in {"Z", "T"} - set(want):
            check(f"{tag} {name}: no {key} block", kw.get(key) is None, f"got {np.shape(kw.get(key))}")


def finite_widths(results, expect_empty=()):
    """Every IV method's width finite at every step, except the families named in
    `expect_empty`, which must be EMPTY at every step.

    On this fixture's n-sweep they are, and the cause is the floor guard, which now
    measures the T CONSTRAINT'S OWN floor (SS2.6): on G alone that floor is ~1e-7,
    so the guard does not fire and `epsilon_iv` stays at the oracle 0.03125, where
    it used to be raised to ~0.4 off the STACKED instrument's floor of ~0.019. The
    inflation was never the T constraint's to claim. At the honest budget the
    program is empty here -- and so is the OLD pooled one at the pooled radius, so
    this is a budget fact, not a decoupling one. RECORDED rather than asserted
    away; the shipped `nEfficiencyFig13` n-sweep is unaffected."""
    bad = [
        name
        for name in IV_NAMES
        if name not in expect_empty and not np.all(np.isfinite(results[name]["interval_width"]))
    ]
    still_finite = [name for name in expect_empty if np.any(np.isfinite(results[name]["interval_width"]))]
    if bad:
        print(f"      non-finite widths: {bad}")
    if still_finite:
        print(f"      expected empty but finite: {still_finite}")
    return not bad and not still_finite


def statuses(model):
    return np.bincount(np.asarray(model.query_status), minlength=3).tolist()


def independent_z_piece(sem, da, X, y, Z, draws, seed):
    """SS2.6's Z budget written out with no oracle code: the larger of the Z moment
    of r_* = y - h_*(X) and of r_* - W#, centred, RMS-pooled over the seeded DA
    draws the oracle pools. W# is w minus its OLS fit on the augmented design, the
    same residual the T budget projects."""
    values = []
    state = np.random.get_state()
    Q = np.linalg.qr(Z)[0]

    def moment(r):
        r = r - r.mean()
        return float(np.linalg.norm(Q.T @ r) / np.sqrt(len(r)))

    residual = np.asarray(y).ravel() - (np.asarray(X) @ sem.W_XY).ravel()
    for draw in range(draws):
        np.random.seed(seed + draw)
        GX, _ = da(X)
        w = ((np.asarray(X) - GX) @ sem.W_XY).ravel()
        Phi = GX - GX.mean(axis=0)
        w_centred = w - w.mean()
        sharp = w_centred - Phi @ np.linalg.lstsq(Phi, w_centred, rcond=None)[0]
        values.append(max(moment(residual), moment(residual - sharp)) ** 2)
    np.random.set_state(state)
    return float(np.sqrt(np.mean(values)))


# ------------------------------------------------------------------ legs


def leg_d(reference):
    print("(D) the shipped configuration does not move: query panel and gamma sweep step, three datasets")
    for label, ok, detail in digest_leg.compare(REPO, reference):
        check(f"(D) {label}", ok, detail)


def leg_i(seed):
    print("(i) every consumer of the trailing-Z carrier, on a SEM stub that emits an instrument")
    np.random.seed(seed)
    runner = sweep_runner("gamma", make_generator, GAMMA_GRID, seed=seed)
    sem = runner.sems[0]
    data = runner.generate_data(0, GAMMA_GRID[0])
    n_test = int(TEST_FRACTION * N)
    check("(i) generator: not a finite pool", not runner.finite_pool)
    check("(i) generator: X_train is treatment only", data.X.shape == (N, K), f"{data.X.shape}")
    check("(i) generator: Z_train rides beside it", data.Z.shape == (N, M), f"{data.Z.shape}")
    check("(i) generator: f saw the stripped test set", data.X_test.shape[1] == K and len(data.estimand) == n_test)
    check("(i) generator: extent saw the stripped test set", np.shape(data.extent) == (n_test,))
    oracle = runner.get_oracle(0)
    z_piece = oracle.eps_iv_z_star
    check(
        "(i) oracle: the draw's instrument reached eps_iv_z_star",
        z_piece is not None and np.isfinite(z_piece) and z_piece > 0.0,
        f"{z_piece!r}",
    )
    check(
        "(i) oracle: the two pieces are separate",
        oracle.eps_iv_star > 0 and z_piece > 0 and not hasattr(oracle, "iv_budget"),
        f"T {oracle.eps_iv_star:.6g}, Z {z_piece:.6g}",
    )
    G = np.asarray(data.G)
    kws = recorded(runner, data)
    check_dispatch("(i) gamma", kws, data.Z, data.Z, G)
    check("(i) non-DA +IV methods never see G", kws["PI+IV"]["Z"].shape[1] == M)

    # real solves, then the whole sweep
    runner = sweep_runner("gamma", lambda: sem, GAMMA_GRID, seed=seed)
    models = runner.build_models(0, 0, data)
    for name in IV_NAMES:
        bounds = models[name].predict(data.X_test)
        status = np.asarray(models[name].query_status)
        ok = np.all(status == SolveStatus.OK) and np.all(np.isfinite(bounds))
        check(
            f"(i) {name} solves OK with the instrument",
            ok,
            f"statuses [OK, INFEASIBLE, FAILURE] {statuses(models[name])}",
        )
    check("(i) ERM+IV point estimate is finite", np.all(np.isfinite(models["ERM+IV"].predict(data.X_test))))
    check("(i) the gamma sweep runs end to end with the instrument", finite_widths(runner.run("gamma")[1]))

    # n-sweep: Z sliced with X
    runner_n = sweep_runner("n", lambda: sem, [64, 128], seed=seed)
    base_Z = runner_n._base_data(0)[5]
    data_n = runner_n.generate_data(0, 64)
    check("(i) n-sweep: Z sliced to n_train beside X", data_n.Z.shape == (64, M) and data_n.X.shape[0] == 64)
    check("(i) n-sweep: the slice is the first rows of the base Z", np.array_equal(data_n.Z, base_Z[:64]))
    check_dispatch("(i) n-sweep", recorded(runner_n, data_n), data_n.Z, data_n.Z, np.asarray(data_n.G))
    # RECORDED: on this fixture the pair is empty at both n. The block below shows
    # why: the guard no longer inflates the T budget off the stacked floor
    check(
        "(i) n-sweep runs end to end, the decoupled pair empty throughout",
        finite_widths(
            sweep_runner("n", lambda: sem, [64, 128], seed=seed).run("n")[1],
            expect_empty=("DA+PI+IV", INTERSECTION),
        ),
    )

    # a FRESH runner: `recorded` above replaced the other one's method_factory
    solo_runner = sweep_runner("n", lambda: sem, [64, 128], seed=seed)
    data_solo = solo_runner.generate_data(0, 64)
    solo_builders = MethodRegistry.build_methods(
        ["DA+PI+IV(T)", "DA+PI+IV(Z)"], **budgets(solo_runner, data_solo), **TOGGLES
    )
    for name, solo in (("DA+PI+IV(T)", "T"), ("DA+PI+IV(Z)", "Z")):
        model = solo_builders[name]()
        fit_model(model=model, method_name=name, **data_solo.fit_arrays)
        model.predict(data_solo.X_test)
        solved = np.all(np.asarray(model.query_status) == SolveStatus.OK)
        check(f"(i) n-sweep: the {solo} constraint ALONE is feasible at these budgets", solved)
    # the cause, measured: the T constraint's own floor is ~1e-7 on G alone against
    # ~1e-2 on the stacked instrument, so the guard has nothing to raise
    common_floor = dict(
        mean_match=solo_runner.mean_match,
        rho=solo_runner.fit_rho(0, data_solo),
        recalibrate=solo_runner.recalibrate,
        kind="iv",
    )
    G_solo = np.reshape(np.asarray(data_solo.G), (len(data_solo.X), -1))
    floor_t = constraint_floor(data_solo.GX, data_solo.y, solo_runner.fit_gamma(0), Z=G_solo, **common_floor)
    floor_stacked = constraint_floor(
        data_solo.GX,
        data_solo.y,
        solo_runner.fit_gamma(0),
        Z=np.column_stack([G_solo, data_solo.Z]),
        **common_floor,
    )
    budget = solo_runner.fit_epsilon_iv(0, 0, data_solo)
    print(f"      RECORDED n-sweep: r_T {budget:.6g}, floor on G {floor_t:.3g}, on [G, Z] {floor_stacked:.3g}")
    check(
        "(i) n-sweep: the guard has nothing to raise, the T floor being far under r_T^2",
        budget**2 >= floor_t and floor_stacked > floor_t,
        f"r_T^2 {budget**2:.3g} vs {floor_t:.3g}",
    )
    # and the OLD pooled program is equally empty at the pooled radius, so nothing
    # about the PAIR is what empties it
    pooled = float(np.hypot(budget, solo_runner.fit_epsilon_iv_z(0, data_solo)))
    old = IVPartialR2(
        gamma=solo_runner.fit_gamma(0),
        epsilon=solo_runner.fit_epsilon(0, 0, data_solo),
        epsilon_iv=pooled,
        epsilon_iv_z=0.0,
        gamma_z=0.0,
        rho=solo_runner.fit_rho(0, data_solo),
        pad=TOGGLES.get("pad", False),
        **{k: v for k, v in TOGGLES.items() if k != "pad"},
    ).fit(data_solo.GX, data_solo.y, T=np.column_stack([G_solo, data_solo.Z]))
    old.predict(data_solo.X_test)
    old_empty = np.all(np.asarray(old.query_status) == SolveStatus.INFEASIBLE)
    check("(i) n-sweep: the OLD pooled program is empty at the pooled radius too", old_empty, f"{statuses(old)}")

    # m-sweep: Z tiled with X, the untiled copy beside X_base
    runner_m = sweep_runner("m", lambda: sem, [1, 2], seed=seed)
    base_Z = runner_m._base_data(0, None)[5]
    data_m = runner_m.generate_data(0, 2)
    check("(i) m-sweep: Z tiled m-fold beside X", data_m.Z.shape == (2 * N, M) and data_m.X.shape[0] == 2 * N)
    check("(i) m-sweep: the tiling is np.tile of the base Z", np.array_equal(data_m.Z, np.tile(base_Z, (2, 1))))
    check("(i) m-sweep: Z_base is the untiled copy", np.array_equal(data_m.Z_base, base_Z) and len(data_m.X_base) == N)
    check_dispatch("(i) m-sweep", recorded(runner_m, data_m), data_m.Z, data_m.Z_base, np.asarray(data_m.G))
    check("(i) m-sweep runs end to end", finite_widths(sweep_runner("m", lambda: sem, [1, 2], seed=seed).run("m")[1]))

    for param, grid in (("recalibrate", [0.0, 1.0]), ("omega", [0.5, 1.0])):
        runner_p = sweep_runner(param, lambda: sem, grid, seed=seed)
        data_p = runner_p.generate_data(0, grid[0])
        check(f"(i) {param}-sweep carries Z", data_p.Z.shape == (N, M) and data_p.X.shape == (N, K))
        check(f"(i) {param}-sweep runs end to end", finite_widths(runner_p.run(param)[1]))

    # the query runner's split site
    np.random.seed(seed)
    query = query_runner(make_generator(), seed=seed)
    check("(i) query: X_raw is treatment only", query.X_raw.shape == (N, K), f"{query.X_raw.shape}")
    check("(i) query: Z rides beside it", query.Z.shape == (N, M), f"{query.Z.shape}")
    context = query.setup_data()
    check("(i) query: the context carries the same Z", context.Z.shape == (N, M) and np.array_equal(context.Z, query.Z))
    check("(i) query: the oracle saw the draw's instrument", query.oracle.eps_iv_z_star > 0.0)
    check("(i) query: epsilon_iv is finite", np.isfinite(query.epsilon_iv))
    query.methods = recorders(NAMES)
    query.run("query")
    # `run` builds and drops its recorders; refit through the same call to read them
    kws = {}
    for name in NAMES:
        model = Recorder(name)
        fit_model(model=model, method_name=name, X=context.X, y=context.y, GX=context.GX, G=context.G, Z=context.Z)
        kws[name] = model.kw
    check_dispatch("(i) query", kws, query.Z, query.Z, np.asarray(query.G))

    # the do-MNIST shapes: a 5-tuple `_draw_base`, a 4-tuple `_load_data`
    class LegacyDraw(STRATEGIES["gamma"]):
        def _draw_base(self, experiment_index, n_samples=None):
            return super()._draw_base(experiment_index, n_samples)[:5]

    np.random.seed(seed)
    legacy = sweep_runner("gamma", LinearSimulationSEM, [0.5], seed=seed)
    legacy.__class__ = LegacyDraw
    check(
        "(i) a 5-tuple _draw_base (do-MNIST's shape) is padded with an (n, 0) Z",
        legacy._base_data(0)[5].shape == (N, 0),
    )

    class LegacyLoad(GenericQuerySweep):
        def _load_data(self):
            return super()._load_data()[:4]

    np.random.seed(seed)
    legacy_query = query_runner(LinearSimulationSEM(treatment_dimension=K), seed=seed, cls=LegacyLoad)
    check("(i) a 4-tuple _load_data (do-MNIST's shape) is padded with an (n, 0) Z", legacy_query.Z.shape == (N, 0))
    source = inspect.getsource(DoMNISTMixin._draw_base) + inspect.getsource(DoMNISTQuerySweep._load_data)
    unaware = "split_instruments" not in source and "iv_" not in source
    check("(i) do-MNIST overrides know nothing of the carrier", unaware)
    # the batch B ruling adds `epsilon_iv_z` to every orchestrator's `build_methods`;
    # nothing else in do_mnist.py may move, the two overrides least of all. Round 12
    # adds the `_run_perf` stub that logs and skips (the lines from its `def` to the
    # blank line that ends it) and reworded the docstring's two "Phase" lines; those
    # are set aside before the count, by anchor, not by word. Needs a checkout: on
    # an archive copy (a break-it outside the worktree) it is skipped
    checkout = subprocess.run(["git", "-C", REPO, "rev-parse", "--git-dir"], capture_output=True, text=True)
    if checkout.returncode != 0:
        skip(f"(i) do_mnist.py diff since {BASE_COMMIT}", "not a git checkout")
    else:
        diff = subprocess.run(
            ["git", "-C", REPO, "diff", BASE_COMMIT, "--", "src/experiments/do_mnist.py"],
            capture_output=True,
            text=True,
        ).stdout
        changed = [line for line in diff.split("\n") if line[:1] in "+-" and line[:3] not in ("+++", "---")]
        kept, in_stub = [], False
        for line in changed:
            if line.startswith("+") and "def _run_perf(" in line:
                in_stub = True
            if in_stub:
                in_stub = line.strip() != "+"  # the stub ends at its blank line
                continue
            # the omega rename's text-only edit of the Phase 2 blocker list: its
            # added line is set aside; the removed line it replaces still counts
            if (
                "Phase 1 is the query sweep" in line
                or "Phase 2 -- see" in line
                or "for what each one still needs" in line
                or "(b) omega --" in line
            ):
                continue
            kept.append(line)
        changed = kept
        outside = [
            line for line in changed if any(k in line for k in ("_draw_base", "_load_data", "sample_paired", "X_raw"))
        ]
        # 12 became 20 when `fit_epsilon_iv` gained its `ratio` argument and the
        # docstring that explains why the do-MNIST override ignores it (SS2.6); the
        # data path is still what this pins, through `outside`
        check(
            f"(i) do_mnist.py since {BASE_COMMIT}: only the budget overrides moved (the perf stub set aside)",
            not outside and len(changed) <= 20 and "build_methods" in diff,
            f"{len(changed)} changed lines",
        )

    # the containers spell None as (n, 0)
    X = np.zeros((10, 3))
    legacy_data = SweepData.coerce((X, np.zeros((10, 1)), X, np.zeros((10, 2)), X[:2], np.zeros(2)))
    check("(i) SweepData from the legacy 6-tuple carries Z (n, 0)", legacy_data.Z.shape == (10, 0))
    column = np.arange(10.0)[:, None]
    tiled = SweepData(np.tile(X, (2, 1)), None, None, None, None, None, X_base=X, Z=np.tile(column, (2, 1)))
    check("(i) SweepData derives Z_base as the first block of a tiled Z", np.array_equal(tiled.Z_base, column))
    check("(i) fit_arrays carries Z and Z_base", {"Z", "Z_base"} <= set(tiled.fit_arrays))
    context = ExperimentDataContext(sem=None, da=None, X=X, y=None, GX=X, G=None)
    check("(i) ExperimentDataContext spells None as (n, 0)", context.Z.shape == (10, 0))
    check("(i) instrument_columns: a 1-d Z becomes a column", instrument_columns(np.arange(10.0), 10).shape == (10, 1))
    try:
        instrument_columns(np.zeros((9, 1)), 10)
        misrow = False
    except ValueError:
        misrow = True
    check("(i) instrument_columns rejects a row mismatch", misrow)

    # the pool: `pool` stays treatment only, `iv_pool` reaches the oracle
    np.random.seed(seed)
    pooled = PooledInstrumentedSEM()
    runner_pool = sweep_runner("gamma", lambda: pooled, [0.5, 1.0], seed=seed)
    data_pool = runner_pool.generate_data(0, 0.5)
    n_train = N - int(np.ceil(TEST_FRACTION * N))  # sklearn's test count
    check("(i) pool: a finite pool", runner_pool.finite_pool)
    check("(i) pool: `pool` stays treatment only", pooled.pool[0].shape == (N, K))
    check("(i) pool: Z_train beside X_train", data_pool.Z.shape == (n_train, M) and data_pool.X.shape == (n_train, K))
    oracle = runner_pool.get_oracle(0)
    da = runner_pool.das[0]
    want = independent_z_piece(pooled, da, pooled.X, pooled.y, pooled.Z, ORACLE_POOL_DRAWS, ORACLE_POOL_SEED)
    check(
        "(i) pool: eps_iv_z_star off iv_pool equals the independent norm to 1e-12",
        abs(oracle.eps_iv_z_star - want) < 1e-12,
        f"{oracle.eps_iv_z_star:.9f} vs {want:.9f}",
    )
    check("(i) pool: eps_iv_star is the T piece alone (unchanged routine)", oracle.eps_iv_star > 0.0)


def leg_ii():
    print("(ii) an empty set on sim, optical and cigarettes: Z is (n, 0) and the T block is exactly G")
    for name in digest_leg.DATASETS:
        runner = production(name)
        data = runner.generate_data(0, 1.0)
        G = np.reshape(np.asarray(data.G), (len(data.G), -1))
        tilde = joint_instrument(data.G, data.Z)
        check(f"(ii) {name}: Z is (n, 0)", data.Z.shape == (len(data.X), 0), f"{data.Z.shape}")
        check(f"(ii) {name}: Z-tilde is G elementwise", np.array_equal(tilde, G) and tilde.shape == G.shape)
        Q_t, R_t = np.linalg.qr(tilde)
        Q_g, R_g = np.linalg.qr(G)
        check(f"(ii) {name}: qr(Z-tilde) == qr(G) bit for bit", np.array_equal(Q_t, Q_g) and np.array_equal(R_t, R_g))
        if name == "optical_device":
            check(
                "(ii) optical: the poly features still apply to the stripped X",
                data.X.shape[1] == runner.poly.n_output_features_ and data.X.shape[1] == data.X_test.shape[1],
                f"{data.X.shape[1]} features",
            )
        at = budgets(runner, data)
        builders = runner.method_factory(**at)
        kws = recorded(runner, data)
        check(f"(ii) {name}: DA+PI+IV is handed exactly G in its T block", np.array_equal(kws["DA+PI+IV"]["T"], G))
        raw = kws["PI&DA+PI+IV"]["Z"]
        check(f"(ii) {name}: the intersection is handed the raw (n, 0) Z", raw.shape == (len(data.X), 0))
        check(f"(ii) {name}: PI is handed no instrument", "Z" not in kws["PI"])
        # the fitted arrays against today's direct Z=G fit
        direct = builders["DA+PI+IV"]().fit(data.GX, data.y, T=data.G)
        runner.methods = {key: builders[key] for key in kws}
        models = runner.build_models(0, 0, data)
        through, intersection = models["DA+PI+IV"], models["PI&DA+PI+IV"]
        same = all(
            np.array_equal(getattr(through, field), getattr(direct, field))
            for field in ("T_projector_R", "t_residual_base", "R_constraint", "h_erm")
        )
        check(f"(ii) {name}: DA+PI+IV through fit_model equals a direct T=G fit, bit for bit", same)
        check(
            f"(ii) {name}: and it carries a T constraint and no Z one",
            through._has_t and not through._has_z and through.Z_projector_R is None,
        )
        check(
            f"(ii) {name}: the intersection's DA branch equals that fit and its baseline reads no instrument",
            np.array_equal(intersection.augmented.T_projector_R, direct.T_projector_R)
            and np.array_equal(intersection.augmented.t_residual_base, direct.t_residual_base)
            and not intersection.baseline._has_iv,
        )
        query = production(name, "query")
        check(f"(ii) {name}: the query runner's Z is (n, 0)", query.Z.shape == (len(query.X_raw), 0))


def old_param_budget(runner, data):
    """Today's `fit_epsilon_iv`, written out: eps_iv_star + EPS_TOL, raised to
    sqrt(FLOOR_GUARD_R floor) on the (GX, G) constraint only when infeasible."""
    budget = float(runner.get_oracle(0).eps_iv_star) + EPS_TOL
    floor = constraint_floor(
        data.GX,
        data.y,
        runner.fit_gamma(0),
        kind="iv",
        Z=data.G,
        mean_match=runner.mean_match,
        rho=runner.fit_rho(0, data),
        recalibrate=runner.recalibrate,
    )
    return budget if budget**2 >= floor else float(np.sqrt(FLOOR_GUARD_R * max(floor, 0.0)))


def old_query_budget(query):
    budget = float(query.oracle.eps_iv_star) + query.eps_tol
    floor = constraint_floor(
        query.GX,
        query.y,
        query.default_gamma,
        kind="iv",
        Z=query.G,
        mean_match=query.mean_match,
        rho=query.fit_rho(),
        recalibrate=query.recalibrate,
    )
    return budget if budget**2 >= floor else float(np.sqrt(FLOOR_GUARD_R * max(floor, 0.0)))


def leg_iii(seed):
    print("(iii) eps_iv_z_star is exactly 0.0 under an empty Z, so epsilon_iv is today's to the bit")
    np.random.seed(seed)
    sem = LinearSimulationSEM(treatment_dimension=K)
    da = da_from_sem(sem)
    X, y = sem(N=N)
    before = np.random.get_state()[1].copy()
    for tag, Z in (("None", None), ("zeros((n, 0))", np.zeros((N, 0)))):
        value = eps_iv_z_star(sem, da, X=X, y=y, Z=Z, mean_match=True)
        check(f"(iii) Z={tag}: eps_iv_z_star == 0.0 exactly", value == 0.0 and isinstance(value, float), f"{value!r}")
    check("(iii) an empty Z touches no RNG", np.array_equal(np.random.get_state()[1], before))
    for name in digest_leg.DATASETS:
        runner = production(name)
        data = runner.generate_data(0, 1.0)
        oracle = runner.get_oracle(0)
        check(f"(iii) {name}: oracle eps_iv_z_star == 0.0", oracle.eps_iv_z_star == 0.0, f"{oracle.eps_iv_z_star!r}")
        check(f"(iii) {name}: eps_iv_z_star is 0.0 under an empty set", oracle.eps_iv_z_star == 0.0)
        new, old = runner.fit_epsilon_iv(0, 0, data), old_param_budget(runner, data)
        check(f"(iii) {name}: sweep epsilon_iv bit-identical to today's", new == old, f"{new!r} vs {old!r}")
        query = production(name, "query")
        new, old = query.epsilon_iv, old_query_budget(query)
        check(f"(iii) {name}: query epsilon_iv bit-identical to today's", new == old, f"{new!r} vs {old!r}")


def aligned(Z, X):
    return np.array_equal(Z[:, 0], 2.0 * X[:, 0] + 1.0)


def leg_iv(seed):
    print("(iv) row alignment survives the bootstrap: Z follows X through the split, the slice and the tiling")
    np.random.seed(seed)
    pooled = PooledInstrumentedSEM(bootstrap=True)
    runner = sweep_runner("gamma", lambda: pooled, [0.5, 1.0], seed=seed)
    X_raw, X, y, X_test, _, Z = runner._draw_base(0)
    n_train = N - int(np.ceil(TEST_FRACTION * N))  # sklearn's test count
    unique = len({tuple(row) for row in X_raw})
    check("(iv) the draw is a resample with repeated rows", unique < len(X_raw), f"{unique} unique of {len(X_raw)}")
    check("(iv) _draw_base: Z_train is 2 X_train_0 + 1 to the bit", aligned(Z, X_raw) and len(Z) == n_train)
    data = runner.generate_data(0, 0.5)
    check("(iv) _sweep_data: aligned", aligned(data.Z, data.X))
    data_n = sweep_runner("n", lambda: pooled, [64, 128], seed=seed).generate_data(0, 64)
    check(
        "(iv) n-slice: aligned", aligned(data_n.Z, data_n.X) and len(data_n.Z) == int(round((1.0 - TEST_FRACTION) * 64))
    )
    data_m = sweep_runner("m", lambda: pooled, [1, 3], seed=seed).generate_data(0, 3)
    both = aligned(data_m.Z, data_m.X) and aligned(data_m.Z_base, data_m.X_base)
    check("(iv) m-tiling: aligned, tiled and untiled", both)
    np.random.seed(seed)
    query = query_runner(PooledInstrumentedSEM(bootstrap=True), seed=seed)
    check("(iv) _load_data: aligned", aligned(query.Z, query.X_raw))


def leg_v(seed):
    print("(v) the floor guard is skipped exactly when the budget is declared, and the floor is logged")
    np.random.seed(seed)
    pooled = PooledInstrumentedSEM(valid=False)
    records = []
    sink = logger.add(lambda message: records.append(message.record), level="DEBUG")
    try:
        declared = sweep_runner("gamma", lambda: pooled, [0.5, 1.0], seed=seed, declared_iv=True)
        oracle_path = sweep_runner("gamma", lambda: pooled, [0.5, 1.0], seed=seed, declared_iv=False)
        data = declared.generate_data(0, 0.5)
        records.clear()
        r_t = declared.fit_epsilon_iv(0, 0, data)
        declared_lines = [r for r in records if "declared path" in r["message"]]
        records.clear()
        joint = oracle_path.fit_epsilon_iv(0, 0, oracle_path.generate_data(0, 0.5))
        oracle_lines = [r for r in records if "declared path" in r["message"]]
    finally:
        logger.remove(sink)
    o_declared, o_oracle = declared.get_oracle(0), oracle_path.get_oracle(0)
    measured = o_declared.eps_iv_z_star > 0.0 and o_declared.eps_iv_z_star == o_oracle.eps_iv_z_star
    check("(v) both paths measure eps_iv_z_star", measured, f"{o_declared.eps_iv_z_star:.6g}")
    common = dict(mean_match=True, rho=declared.fit_rho(0, data), recalibrate=True)
    floor = constraint_floor(data.GX, data.y, declared.fit_gamma(0), kind="iv", Z=data.G, **common)
    floor_stacked = constraint_floor(
        data.GX, data.y, declared.fit_gamma(0), kind="iv", Z=joint_instrument(data.G, data.Z), **common
    )
    want = float(o_declared.eps_iv_star) + EPS_TOL
    detail = f"r_T^2 {want**2:.4g} vs T floor {floor:.4g}, stacked floor {floor_stacked:.4g}"
    # the whole point of measuring the floor on G alone: the INVALID instrument
    # lifts the STACKED floor above r_T^2 and leaves the T constraint's own below it
    check(
        "(v) the invalid instrument lifts the stacked floor above r_T^2, not the T constraint's own",
        want**2 < floor_stacked and want**2 >= floor,
        detail,
    )
    check("(v) declared: epsilon_iv is r_T = eps_iv_star + EPS_TOL, not raised", r_t == want, f"{r_t!r} vs {want!r}")
    check("(v) declared: one INFO line", len(declared_lines) == 1 and declared_lines[0]["level"].name == "INFO")
    message = declared_lines[0]["message"] if declared_lines else ""
    printed = float(message.split("floor ")[1].split(";")[0].strip()) if "floor " in message else np.nan
    # the line prints the floor to 4 significant digits, so 5e-4 relative is its precision
    named = abs(printed - floor) <= 6e-4 * max(floor, 1e-12)
    check("(v) declared: the line names the T constraint's own floor", named, message)
    check(
        "(v) declared: that floor is on G alone, not on the stacked instrument",
        abs(floor - floor_stacked) > 1e-9,
        f"vs stacked {floor_stacked:.4g}",
    )
    check("(v) declared: the line says never raised", "never raised" in message)
    check("(v) oracle path: no declared line", not oracle_lines)
    joint_ok = joint == float(o_oracle.eps_iv_star) + EPS_TOL
    check("(v) oracle path: the T budget is eps_iv_star + EPS_TOL", joint_ok, f"{joint:.6g}")
    small = 0.5 * np.sqrt(floor)
    raised = oracle_path._floor_guard(small, data, "iv", 0, "epsilon_iv")
    kept = declared._floor_guard(small, data, "iv", 0, "epsilon_iv", declared=True)
    lifted = abs(raised - np.sqrt(FLOOR_GUARD_R * floor)) < 1e-12
    check("(v) the guard raises an infeasible oracle budget to sqrt(9 floor)", lifted, f"{small:.4g} -> {raised:.4g}")
    check("(v) and leaves the same budget alone when declared", kept == small)
    # the query runner
    records.clear()
    sink = logger.add(lambda message: records.append(message.record), level="DEBUG")
    try:
        np.random.seed(seed)
        q_declared = query_runner(PooledInstrumentedSEM(valid=False), seed=seed, declared_iv=True)
        q_lines = [r for r in records if "declared path" in r["message"]]
        records.clear()
        np.random.seed(seed)
        q_oracle = query_runner(PooledInstrumentedSEM(valid=False), seed=seed, declared_iv=False)
        q_oracle_lines = [r for r in records if "declared path" in r["message"]]
    finally:
        logger.remove(sink)
    want = float(q_declared.oracle.eps_iv_star) + q_declared.eps_tol
    check("(v) query declared: epsilon_iv is r_T, not raised", q_declared.epsilon_iv == want)
    logged_once = len(q_lines) == 1 and "never raised" in q_lines[0]["message"]
    check("(v) query declared: the floor is logged once at build", logged_once)
    # both paths read `eps_iv_star` now, so the budgets AGREE unless the oracle
    # path's guard fires; what still separates them is the declared log line
    check(
        "(v) query oracle path: no declared line, and the same T piece unless the guard fired",
        not q_oracle_lines and q_oracle.epsilon_iv >= want,
        f"{q_oracle.epsilon_iv:.6g} vs {want:.6g}",
    )


def leg_vi(seed):
    print("(vi) the parallel dispatch at n_jobs 4 equals the serial one on the same draw, instrument included")
    np.random.seed(seed)
    sem = make_generator()
    runner = sweep_runner("gamma", lambda: sem, GAMMA_GRID, seed=seed)
    data = runner.generate_data(0, GAMMA_GRID[0])
    at = budgets(runner, data)
    for name in IV_NAMES:
        out = {}
        for n_jobs in (1, 4):
            model = factory(**at, n_jobs=n_jobs)[name]()
            fit_model(model=model, method_name=name, **data.fit_arrays)
            out[n_jobs] = (np.asarray(model.predict(data.X_test)), np.asarray(model.query_status))
        same = np.array_equal(out[1][0], out[4][0]) and np.array_equal(out[1][1], out[4][1])
        solved = np.all(out[1][1] == SolveStatus.OK)
        check(f"(vi) {name}: n_jobs 4 == n_jobs 1, bounds and statuses", same and solved, f"statuses {statuses(model)}")
    serial = sweep_runner("gamma", lambda: sem, GAMMA_GRID, seed=seed).run("gamma")[1]
    parallel = sweep_runner("gamma", lambda: sem, GAMMA_GRID, seed=seed, n_jobs=4).run("gamma")[1]
    same = all(np.array_equal(serial[name][metric], parallel[name][metric]) for name in IV_NAMES for metric in METRICS)
    check("(vi) a gamma sweep through the runner at n_jobs 4 equals the serial one", same)


def leg_vii(seed):
    print("(vii) the oracle path: non-DA +IV methods carry eps_iv_z_star + EPS_TOL, the intersection is feasible")
    # the routing on a MISSPECIFIED symmetry (DA strength > 0), where the T-side
    # budget is far from the Z term, so substituting one for the other is visible
    np.random.seed(seed)
    runner = sweep_runner("gamma", make_generator, GAMMA_GRID, seed=seed, strength=MISSPECIFIED_STRENGTH)
    data = runner.generate_data(0, GAMMA_GRID[0])
    oracle = runner.get_oracle(0)
    want_z, want_t = float(oracle.eps_iv_z_star) + EPS_TOL, runner.fit_epsilon_iv(0, 0, data)
    apart = oracle.eps_iv_star > 10 * EPS_TOL and want_z != want_t
    check("(vii) the Z term and the T-side budget differ on this draw", apart, f"{want_z:.6g} vs {want_t:.6g}")
    check("(vii) fit_epsilon_iv_z is eps_iv_z_star + EPS_TOL, unguarded", runner.fit_epsilon_iv_z(0, data) == want_z)
    models = runner.build_models(0, 0, data)
    for name in NON_DA_IV:
        model = models[name]
        carries = model.epsilon_iv_z == want_z and model.z_bound == want_z and not model._has_t
        check(f"(vii) {name} carries epsilon_iv_z as its Z radius, with no T block", carries, f"{model.z_bound!r}")
    inter = models[INTERSECTION]
    baseline = inter.baseline
    check(
        "(vii) the intersection's baseline carries epsilon_iv_z and no T block",
        baseline.epsilon_iv_z == want_z and baseline.z_bound == want_z and not baseline._has_t,
    )
    t_side = inter.augmented.t_bound == want_t == models["DA+PI+IV"].t_bound
    check("(vii) its DA branch and DA+PI+IV carry the T budget on their T constraint", t_side)
    # feasibility on the WELL-SPECIFIED symmetry, the ruling's case: both budgets
    # admit h* there, so the two branches overlap at every query. (Under the
    # misspecified DA above the recalibrated DA ball can exclude h*, Thm. 1's
    # regime, and the intersection empties at some queries: not plumbing.)
    np.random.seed(seed)
    runner = sweep_runner("gamma", make_generator, GAMMA_GRID, seed=seed)
    data = runner.generate_data(0, GAMMA_GRID[0])
    inter = runner.build_models(0, 0, data)[INTERSECTION]
    inter.predict(data.X_test)
    feasible = all(
        np.all(np.asarray(m.query_status) == SolveStatus.OK) for m in (inter, inter.baseline, inter.augmented)
    )
    check(
        "(vii) the intersection is feasible on every query under a real Z, both branches OK",
        feasible,
        f"intersection {statuses(inter)}, baseline {statuses(inter.baseline)}, DA {statuses(inter.augmented)}",
    )
    empty = production("simulation")
    check(
        "(vii) an empty Z gives epsilon_iv_z exactly 0.0", empty.fit_epsilon_iv_z(0, empty.generate_data(0, 1.0)) == 0.0
    )
    query = query_runner(make_generator(), seed=seed, strength=MISSPECIFIED_STRENGTH)
    same_term = query.epsilon_iv_z == float(query.oracle.eps_iv_z_star) + EPS_TOL
    check("(vii) the query runner's epsilon_iv_z is the same term", same_term)
    check("(vii) and it differs from the query runner's T-side budget", query.epsilon_iv_z != query.epsilon_iv)
    check("(vii) and its PI+IV carries it", query.methods["PI+IV"]().epsilon_iv_z == query.epsilon_iv_z)


def leg_viii(seed):
    print("(viii) the declared path: PI+IV solves at exactly r_Z = s sqrt(gamma_z), DA+PI+IV keeps r_T")
    np.random.seed(seed)
    pooled = PooledInstrumentedSEM()
    runner = sweep_runner("gamma", lambda: pooled, GAMMA_GRID, seed=seed, declared_iv=True, gamma_z=DECLARED_GAMMA_Z)
    data = runner.generate_data(0, GAMMA_GRID[0])
    check("(viii) fit_epsilon_iv_z is 0.0 on the declared path", runner.fit_epsilon_iv_z(0, data) == 0.0)
    r_t = float(runner.get_oracle(0).eps_iv_star) + EPS_TOL
    models = runner.build_models(0, 0, data)
    for name in IV_NAMES:
        model = models[name]
        bounds = model.predict(data.X_test)
        solved = np.all(np.asarray(model.query_status) == SolveStatus.OK) and np.all(np.isfinite(bounds))
        check(f"(viii) {name} solves OK", solved, f"statuses {statuses(model)}")
    baseline = models[INTERSECTION].baseline
    for name, model in (
        ("PI+IV", models["PI+IV"]),
        ("PI+INV+IV", models["PI+INV+IV"]),
        ("the baseline branch", baseline),
    ):
        r_z = np.sqrt(model.sigma_sq / model.rho * DECLARED_GAMMA_Z)
        exact = not model._has_t and model.z_bound == r_z
        check(f"(viii) {name}: no T block and z_bound exactly r_Z", exact, f"{model.z_bound!r}")
    for name, model in (("DA+PI+IV", models["DA+PI+IV"]), ("the DA branch", models[INTERSECTION].augmented)):
        check(f"(viii) {name}: t_bound is r_T and z_bound is its own", model.t_bound == r_t and model.z_bound > 0.0)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    os.chdir(REPO)
    os.environ.setdefault("MPLBACKEND", "Agg")
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reference", default=digest_leg.DEFAULT_REFERENCE)
    parser.add_argument("--skip-digest", action="store_true")
    args = parser.parse_args()
    print(f"tree: {runner_module.__file__}")
    legs = [
        ("(i)", leg_i, (args.seed,)),
        ("(ii)", leg_ii, ()),
        ("(iii)", leg_iii, (args.seed,)),
        ("(iv)", leg_iv, (args.seed,)),
        ("(v)", leg_v, (args.seed,)),
        ("(vi)", leg_vi, (args.seed,)),
        ("(vii)", leg_vii, (args.seed,)),
        ("(viii)", leg_viii, (args.seed,)),
    ]
    if not args.skip_digest:
        legs.append(("(D)", leg_d, (args.reference,)))
    for tag, leg, leg_args in legs:
        try:
            leg(*leg_args)
        except Exception as error:  # a raise is a FAIL line, and the later legs still report
            check(f"{tag} ran without raising", False, f"{type(error).__name__}: {error}")
    if args.skip_digest:
        print("(D) SKIPPED by --skip-digest: a break-it run, not the committed state")
    skipped = f" ({len(SKIPPED)} SKIPPED: {'; '.join(SKIPPED)})" if SKIPPED else ""
    if not FAIL:
        print(f"A58 PASS{skipped}")
    else:
        print(f"A58 FAIL: {FAIL}{skipped}")
        sys.exit(1)
