"""
US state cigarette demand experiment using generic runners.
"""

import csv
import os
from functools import partial

import numpy as np
from loguru import logger

from src.data_augmentors.cigarettes import ScaleTranslation as DA
from src.experiments.base import ExperimentOrchestrator
from src.experiments.configs import (
    ANNOTATE_SWEEP_PLOT,
    CIGARETTE_CONFIG,
    EPS_TOL,
    GAMMA_Z_DEFAULT,
    QUERY_GAMMA,
    MethodRegistry,
)
from src.experiments.generic_runner import STRATEGIES, GenericQuerySweep
from src.experiments.utils import PanelBuilder, create_query_sweep_plot, create_sweep_plot, save
from src.experiments.utils.constants import SUBDIR_QUERY, TEX_MAPPER, iv_mode, parse_method
from src.methods.sensitivity_models import constraint_floor
from src.oracle import epsilon_star, preserve_rng
from src.sem.cigarettes import (
    OUTCOME,
    SPECS,
    TREATMENTS,
    V,
    build_design,
    clustered_vcov,
    clusters,
    controls,
    data_directory,
    first_stage_f,
    homogeneity_f,
    residualise,
    restricted_fit,
    restricted_gmm,
    rho_max,
    two_stage_fit,
)
from src.sem.cigarettes import CigaretteSEM as SEM

EXPERIMENT_NAME = "cigarettes"

# m-sweep holds n fixed here (PLAN 5.5). 10 whole state histories: a multiple of
# the 50-year history length, so the folds get whole states like every other draw.
FOLD_SWEEP_SAMPLES: int = 500

# eps* is a defect of the TARGET under the DA, and under the translation the defect
# is w = -c (v-hat' b) with b homogeneous on both paths, so it is 0 whatever c comes
# out of the draw. One seeded draw is therefore the population value, not a sample
# of it -- unlike the optical device, where eps* is an RMS over image draws and has
# to be pooled. Fixing the seed keeps the budget a function of (SEM, DA) alone.
EPSILON_STAR_SEED: int = 0

# the query figures: one ray per treatment coordinate, then the ray along v-hat.
# The ids are the `ANNOTATE_SWEEP_PLOT` keys and the filenames both.
DIM_IDS: tuple[str, ...] = ("dim_p", "dim_y", "dim_pn", "dim_cpi")
RAY_ID: str = "ray_v"
# each grid spans this many sd of its own coordinate
GRID_SPAN: float = 3.0
# observed rows behind the width-ratio figure, evenly spaced through the panel
RATIO_QUERIES: int = 512
# normal quantile for the 95% intervals of the unrestricted fit, which the
# coefficient table reports as CONTEXT beside the bounds
NORMAL_95: float = 1.959963984540054

# the headline figures under a configured instrument set (SS10), one pair per
# coefficient in HEADLINE_COEFFICIENTS. F1: the coefficient against the
# confounding budget on the benchmarked range, 1x to 3x the tax-differential
# benchmark. F2: the coefficient against the real-Z radius r_Z = s sqrt(gamma_z)
# at the query budget, the declared radius, the cluster-bootstrap moment and the
# gamma_z benchmarks marked, so the leak assumption is seen against what the
# panel itself says. Both are read off the query panel's fitted models, one band
# per method. The budget range reaches the addiction-stock leak (r_Z 0.558).
HEADLINE_METHODS: tuple[str, ...] = ("PI", "PI+IV", "PI+INV+IV", "DA+PI+IV")
HEADLINE_COEFFICIENTS: tuple[str, ...] = ("pn", "p")
COEFFICIENT_LABELS: dict[str, str] = {"pn": r"$\beta_{p_n}$", "p": r"$\beta_{p}$"}
GAMMA_RANGE: tuple[float, float] = (0.150, 0.450)
BUDGET_RANGE: tuple[float, float] = (2**-8, 2**-0.5)
PN: int = TREATMENTS.index("pn")
# the gamma_z benchmarks marked on F2 and read in the IV benchmark table, by
# BENCHMARK_NAMES key: the primary and the channel-specific one, as on F1
IV_BENCHMARKS: tuple[str, ...] = ("lag_q", "tax_diff")
# direct tax elasticities the IV benchmark table converts to gamma_z: a direct
# effect delta of log tax on log sales OUTSIDE the price channel reads
# gamma_z = (delta / sigma)^2 var(z_tax | C) (Conley, Hansen and Rossi's
# plausibly-exogenous support, in the paper's units)
DIRECT_EFFECTS: tuple[float, ...] = (0.01, 0.02, 0.05, 0.10)
# the state-cluster bootstrap of the IV moment at the target, marked on F2: how
# far a resampled panel's moment sits from the pool's (SS2.4)
MOMENT_REPLICATES: int = 400
MOMENT_SEED: int = 0
# the benchmark covariates (SS6.2), in the order the table prints them: the
# addiction stock first (the primary, no tax variation), the tax differential
# (the left edge), then the rest, all far below the feasibility floor
BENCHMARK_NAMES: dict[str, str] = {
    "lag_q": r"$\log q_{s,t-1}$ (addiction stock)",
    "tax_diff": r"$\tau_s - \tau_{sn}$ (own minus lowest-neighbour excise)",
    "log_tax_s": r"$\log \tau_s$",
    "log_tax_ratio": r"$\log(\tau_s / \tau_{sn})$",
    "log_pop": r"$\log \mathrm{pop}$",
    "log_tax_f": r"$\log \tau_f$",
    "log_tax_sn": r"$\log \tau_{sn}$",
    "log_pop_n": r"$\log \mathrm{pop}_n$",
    "log_pop_ratio": r"$\log(\mathrm{pop}_n / \mathrm{pop})$",
}
NEIGHBOURS_FILE: str = "neighbors.csv"


def _cosines(points, precision):
    """|cos(x, v)| per row, in the Sigma^-1 inner product. The width law of the
    ratio figure is exactly sqrt(1 - cos^2) in this metric, and nothing else."""
    direction = V / np.linalg.norm(V)
    numerator = np.abs(np.asarray(points) @ precision @ direction)
    denominator = np.sqrt(
        np.einsum("ij,jk,ik->i", points, precision, points) * float(direction @ precision @ direction)
    )
    return numerator / np.where(denominator > 0.0, denominator, 1.0)


def _two_stage_interval(design, by: str):
    """(lower, upper) 95% interval per coefficient of the UNRESTRICTED 2SLS fit,
    under one clustering. Context for the table, never a target."""
    beta, X_hat = two_stage_fit(design)
    bread = np.linalg.inv(X_hat.T @ X_hat)
    vcov = clustered_vcov(X_hat, design.y - design.X @ beta, bread, design.K, clusters(design, by))
    half = NORMAL_95 * np.sqrt(np.diag(vcov))
    return beta - half, beta + half


# =============================================================================
# BENCHMARKS (SS6)
# =============================================================================


def benchmark_covariates(panel) -> dict[str, np.ndarray]:
    """The candidate omitted variables W, one array per BENCHMARK_NAMES key, NaN
    where a row has none (the first year has no lag; a state with no neighbour in
    the panel has no neighbour population)."""
    state, year = panel["st"], panel["year"]
    position = {(s, t): i for i, (s, t) in enumerate(zip(state, year, strict=True))}
    lag = np.array(
        [
            np.log(panel["q"][position[(s, t - 1)]]) if (s, t - 1) in position else np.nan
            for s, t in zip(state, year, strict=True)
        ]
    )
    neighbour_population = _neighbour_population(panel)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_pop_n = np.where(neighbour_population > 0, np.log(np.maximum(neighbour_population, 1.0)), np.nan)
    return {
        "lag_q": lag,
        "tax_diff": panel["tax_s"] - panel["tax_sn"],
        "log_tax_s": np.log(panel["tax_s"]),
        "log_tax_ratio": np.log(panel["tax_s"] / panel["tax_sn"]),
        "log_pop": np.log(panel["pop"]),
        "log_tax_f": np.log(panel["tax_f"]),
        "log_tax_sn": np.log(panel["tax_sn"]),
        "log_pop_n": log_pop_n,
        "log_pop_ratio": log_pop_n - np.log(panel["pop"]),
    }


def _neighbour_population(panel) -> np.ndarray:
    """Summed population of each state's neighbours in the panel, per row; zeros
    when the adjacency file is not beside the raw sources."""
    path = os.path.join(data_directory(), "raw", NEIGHBOURS_FILE)
    state, year = panel["st"], panel["year"]
    if not os.path.isfile(path):
        logger.warning(f"{path} missing; the neighbour-population benchmarks are skipped.")
        return np.zeros(len(state))
    neighbours: dict[str, list[str]] = {}
    with open(path, newline="") as handle:
        for row in csv.reader(handle):
            if len(row) >= 2 and row[0] != "StateCode":
                neighbours.setdefault(row[0].strip(), []).append(row[1].strip())
    population = {(s, t): value for s, t, value in zip(state, year, panel["pop"], strict=True)}
    return np.array(
        [sum(population.get((q, t), 0.0) for q in neighbours.get(s, [])) for s, t in zip(state, year, strict=True)]
    )


def benchmark_gamma(panel, w: np.ndarray, spec: str) -> tuple[int, float, float, float, float]:
    """(n, r2(Y~W|X,C), r2(W~X|C), gamma_OVB, gamma_CH) for one omitted W (SS6.1).

    Everything is residualised on the controls INSIDE the rows W exists on and
    the outcome is scaled by that sample's OLS residual sd, or the identity does
    not close (draft 1 measured the lag row against a full-sample FWL and the
    two columns differed by 5e-3; re-FWL'd they agree to 4e-16). gamma_OVB is the
    omitted-variable bias of dropping W in the paper's units, c^2 ||P_X W||^2 /
    (n sigma^2); gamma_CH is Cinelli and Hazlett's r2(W~X) r2(Y~W|X) / (1 - r2(W~X)).
    Plain OLS, no clustering: the point estimate of c is all that enters.
    """
    ok, C, X, y, short, W, c = _benchmark_frame(panel, w, spec)
    design = np.column_stack([X, W])
    full = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    projected = X @ np.linalg.lstsq(X, W, rcond=None)[0]
    n = int(ok.sum())
    r2y = 1.0 - float(np.sum(full**2) / np.sum(short**2))
    r2w = float(np.sum(projected**2) / np.sum(W**2))
    gamma_ovb = float(c**2 * np.sum(projected**2) / n / (np.sum(short**2) / n))
    gamma_ch = r2w * r2y / (1.0 - r2w)
    return n, r2y, r2w, gamma_ovb, gamma_ch


def _benchmark_frame(panel, w: np.ndarray, spec: str):
    """The common sample of one benchmark: (rows, controls, X, y, short residual,
    W, c), FWL'd inside W's rows and sigma-scaled, c the coefficient of W in the
    long regression y ~ X + W."""
    ok = np.isfinite(w)
    C = controls(panel, spec)[ok]
    X = residualise(np.log(np.column_stack([panel[name] for name in TREATMENTS]))[ok], C)
    y = residualise(np.log(panel[OUTCOME])[ok], C)
    short = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    sigma = float(np.sqrt(np.mean(short**2)))
    y, short = y / sigma, short / sigma
    W = residualise(w[ok].reshape(-1, 1), C).ravel()
    c = float(np.linalg.lstsq(np.column_stack([X, W]), y, rcond=None)[0][-1])
    return ok, C, X, y, short, W, c


def benchmark_gamma_z(panel, w: np.ndarray, spec: str, iv) -> tuple[int, float, float, float]:
    """(n, r2(W~Z|C), r2(W~tau_s|C), gamma_z) for one omitted W against the
    configured instrument set: the leak of dropping W (SS6.1 with Z for X).

    Same algebra as `benchmark_gamma`, same c, same sample, projection onto the
    instruments instead of the treatments: with y = X b + c W + e and e clean of
    Z, the moment at b is c Z'W / n, so gamma_z = c^2 ||P_Z W||^2 / (n sigma^2)
    (Cinelli and Hazlett's IV framework, the paper's [12]). Nevo and Rosen's
    imperfect-instrument premise is the ratio to `benchmark_gamma`: the set is
    credible where the same W leaks it less than it confounds X. The third
    column isolates the external excise, since a treatment instrumenting itself
    inherits every confounder that touches it.
    """
    ok, C, X, y, short, W, c = _benchmark_frame(panel, w, spec)
    excise = {name: residualise(np.log(panel[name])[ok], C) for name in iv if name not in TREATMENTS}
    Z = np.column_stack([X[:, TREATMENTS.index(name)] if name in TREATMENTS else excise[name] for name in iv])
    projected = Z @ np.linalg.lstsq(Z, W, rcond=None)[0]
    external = np.column_stack(list(excise.values())) if excise else np.zeros((len(W), 0))
    on_excise = external @ np.linalg.lstsq(external, W, rcond=None)[0] if excise else np.zeros_like(W)
    n = int(ok.sum())
    r2z = float(np.sum(projected**2) / np.sum(W**2))
    r2t = float(np.sum(on_excise**2) / np.sum(W**2))
    gamma_z = float(c**2 * np.sum(projected**2) / n / (np.sum(short**2) / n))
    return n, r2z, r2t, gamma_z


def direct_effect_gamma_z(design, delta: float) -> float:
    """gamma_z of a direct effect `delta` of log tax on log sales outside the
    price channel: the moment at b is delta Z'z_tax / n, and z_tax lies in
    span(Z), so gamma_z = (delta / sigma)^2 var(z_tax | C). The panel's y is
    sigma-scaled and its excise column is not, hence the division."""
    z_tax = design.Z[:, 0]
    return float((delta / design.sigma) ** 2 * np.mean(z_tax**2))


def feasibility_floor(design, Z: np.ndarray, bound: float) -> float:
    """The smallest gamma at which the PI+IV set is non-empty at this IV bound:
    the Lem. 2 ball has to reach the moment line (SS3.3). Bisected on the
    closed-form `constraint_floor`."""
    low, high = 1e-6, 4.0
    for _ in range(50):
        mid = 0.5 * (low + high)
        floor = constraint_floor(design.X, design.y, mid, kind="iv", Z=Z, mean_match=True)
        if np.sqrt(floor) <= bound:
            high = mid
        else:
            low = mid
    return high


def moment_quantiles(design, Z: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """(median, p95) of ||Q_Z'(y - X b)|| / sqrt(n) over state-cluster bootstrap
    replicates of the panel, at a fixed stream: what a resampled panel's moment
    reads against the declared radius (SS2.4)."""
    states = np.unique(design.state)
    rows = {state: np.flatnonzero(design.state == state) for state in states}
    rng = np.random.default_rng(MOMENT_SEED)
    values = []
    for _ in range(MOMENT_REPLICATES):
        index = np.concatenate([rows[state] for state in rng.choice(states, len(states), replace=True)])
        Q = np.linalg.qr(Z[index])[0]
        values.append(np.linalg.norm(Q.T @ (design.y[index] - design.X[index] @ b)) / np.sqrt(len(index)))
    return float(np.percentile(values, 50)), float(np.percentile(values, 95))


# =============================================================================
# ORCHESTRATOR
# =============================================================================


class CigaretteOrchestrator(ExperimentOrchestrator):
    """Orchestrator for the cigarette panel experiments."""

    def __init__(
        self,
        target: str = "iv",
        spec: str = "t3",
        anchor: str = "own-tax",
        da_amplitude: float = 1.0,
        sliver: bool = False,
        iv=None,
        gamma_z: float | None = None,
        **kwargs,
    ):
        """
        Args:
            target: `iv` = the restricted-IV point (the tax-moment program solved
                subject to v'b = 0 against the configured instrument set, or the
                anchor set without one); `plasmode` = the real design with a
                synthetic homogeneous h_* and known confounding.
            spec: state FE plus a trend of this order.
            anchor: which excise columns instrument which regressors.
            da_amplitude: DA amplitude along v, in units of sd(X . v-hat).
            sliver: target the leaky-IV set rather than the restricted point.
            iv: the instrument set by panel column name (the yaml's `iv:`); None
                or [] is no instrument and today's run exactly (SS7.1).
            gamma_z: the DECLARED leakiness budget of that set (SS2.6), read only
                when it is non-empty; None is `GAMMA_Z_DEFAULT`.
        """
        self.target, self.spec, self.anchor = target, spec, anchor
        self.sliver = bool(sliver)
        self._benchmarks = None
        self._benchmarks_iv = None
        self.iv_columns = tuple(iv or ())
        # the real-Z radius s sqrt(gamma_z) is DECLARED, never oracle, and it
        # exists only with an instrument to declare it on; 0 keeps the IV classes
        # bit-identical to today's (`iv_bound` is then exactly `epsilon_iv`)
        self.gamma_z = float(GAMMA_Z_DEFAULT if gamma_z is None else gamma_z) if self.iv_columns else 0.0
        if self.iv_columns:
            logger.info(
                f"instrument set {list(self.iv_columns)}: the target is the restricted fit against it, gamma_z "
                f"{self.gamma_z:g} declared (radius {np.sqrt(self.gamma_z):.4f} of the residual sd)."
            )
        self._epsilon_star = None
        self.toggles = dict(
            recalibrate=kwargs.get("recalibrate", True),
            pad=kwargs.get("pad", False),
            clipy=kwargs.get("clipy", True),
            n_jobs=kwargs.get("n_jobs", 1),
            mean_match=kwargs.get("mean_match", True),
        )
        toggles = self.toggles

        # The DA amplitude is a DATASET constant and the query runner calls
        # `da_factory()` with no arguments, so it cannot come off a SEM: the panel
        # is read once here, exactly as the optical orchestrator builds its feature
        # transform from a throwaway SEM.
        design = build_design(SEM.panel(), spec=spec, anchor=anchor)
        self._amplitude = float(da_amplitude) * float(np.std(design.X @ (V / np.linalg.norm(V))))
        logger.info(f"DA amplitude {da_amplitude:g} x sd(X . v-hat) = {self._amplitude:.6f} at {spec}.")

        # DECLARED, per spec, and only the query panel ever sees it: the sweeps
        # solve at gamma*(target) through `ParamSweepRunner.fit_gamma`. See
        # `QUERY_GAMMA` for why the query budget is declared rather than measured.
        self.gamma = QUERY_GAMMA[spec]
        epsilon = self._epsilon_budget(CIGARETTE_CONFIG.epsilon)
        gamma_z_declared = self.gamma_z

        class CigaretteRegistry(MethodRegistry):
            @staticmethod
            def build_methods(names):
                return MethodRegistry.build_methods(
                    names, gamma=QUERY_GAMMA[spec], epsilon=epsilon, gamma_z=gamma_z_declared, **toggles
                )

        super().__init__(EXPERIMENT_NAME, CigaretteRegistry(), **kwargs)

    # ---------------------------------------------------------------- factories

    def _sem_factory(self, bootstrap: bool = False):
        """Factory for creating SEM instances.

        `bootstrap` is the replicate mechanism and is bound PER RUNNER (SS6): the
        query figures and the coefficient table are fit on the full panel, the
        sweeps on replicates of it (`get_sweep_runner_cls` picks which kind).
        `pool` is the whole panel either way, so h_*, gamma* and eps* do not move
        between them. The configured instrument set rides beside every draw.
        """
        return SEM(
            spec=self.spec,
            target=self.target,
            anchor=self.anchor,
            bootstrap=bootstrap,
            sliver=self.sliver,
            iv_columns=self.iv_columns,
            gamma_z=CIGARETTE_CONFIG.gamma_z,
            gamma_true=CIGARETTE_CONFIG.gamma_true,
            confound_direction=CIGARETTE_CONFIG.confound_direction,
            outcome_noise_std=CIGARETTE_CONFIG.outcome_noise_std,
        )

    def _da_factory(self, sem=None, append: str | None = None):
        """Factory for creating DA instances. The direction and the amplitude are
        dataset constants, so neither argument is needed; `append` is accepted
        because the robustness sweep passes it where a chain has components, and
        this DA has none."""
        return DA(V, std=self._amplitude)

    def _oracle_pieces(self):
        """(sem, da, features) for the budget estimators, built once."""
        return self._sem_factory(), self._da_factory(), None

    def measured_epsilon_star(self) -> float:
        """eps* for THIS (SEM, DA): the L2 defect on the panel. Cached."""
        if self._epsilon_star is None:
            sem, da, features = self._oracle_pieces()
            with preserve_rng():
                np.random.seed(EPSILON_STAR_SEED)
                self._epsilon_star = float(epsilon_star(sem, da, X=sem.X, features=features))
            logger.info(f"Cigarette eps*: {self._epsilon_star:.3e}")
        return self._epsilon_star

    def _epsilon_budget(self, configured: float | None, tol: float = EPS_TOL) -> float:
        """PI+INV's ASSUMED invariance bound -- the SS3.1 constraint budget.

        `None` means the measured eps* plus `tol`, which is what the optical
        orchestrator does and for the same reason: a budget under eps* excludes
        h_* from the PI+INV set, which costs validity rather than width. Here eps*
        is 0 by construction on both targets (h_* is exactly homogeneous and the DA
        translates along v), so the budget IS the tolerance and nothing is padded
        to cover a defect that does not exist.
        """
        if configured is not None:
            return float(configured)
        return self.measured_epsilon_star() + tol

    # ------------------------------------------------------------------ runners

    def get_query_runner_cls(self) -> type[GenericQuerySweep]:
        """Return query sweep runner."""
        outer = self

        class CigaretteQuerySweep(GenericQuerySweep):
            def __init__(inner_self, **kwargs):
                super().__init__(
                    sem_factory=outer._sem_factory,  # bootstrap=False: the panel itself
                    da_factory=outer._da_factory,
                    poly_transform=None,
                    method_factory=outer.build_methods,
                    default_gamma=QUERY_GAMMA[outer.spec],
                    default_epsilon=outer._epsilon_budget(CIGARETTE_CONFIG.query_epsilon, tol=CIGARETTE_CONFIG.eps_tol),
                    eps_tol=CIGARETTE_CONFIG.eps_tol,
                    # the panel is sigma-normalised, so gamma is already in the
                    # paper's units and sigma-hat^2 is 1: nothing to rescale
                    raw_gamma=False,
                    # a configured set declares its budget (SS2.6): the runner
                    # hands the solver r_T alone and never raises it
                    declared_iv=bool(outer.iv_columns),
                    **kwargs,
                )

        return CigaretteQuerySweep

    def build_methods(self, gamma: float, epsilon: float, epsilon_iv=None, n_jobs=None, rho=1.0, epsilon_iv_z=0.0):
        """Methods at explicit (per-experiment) budgets. `n_jobs` overrides the
        toggle -- perf needs serial models to time methods, not the harness."""
        toggles = self.toggles if n_jobs is None else {**self.toggles, "n_jobs": n_jobs}
        return MethodRegistry.build_methods(
            self.kwargs["methods"],
            gamma=gamma,
            epsilon=epsilon,
            epsilon_iv=epsilon_iv,
            epsilon_iv_z=epsilon_iv_z,
            gamma_z=self.gamma_z,
            rho=rho,
            **toggles,
        )

    def get_sweep_runner_cls(self, param: str) -> type:
        """Configured strategy for one sweep parameter."""
        outer, Strategy = self, STRATEGIES[param]
        # the sweep replicates: a state-cluster bootstrap under `iv` with no
        # configured instrument (a resampling rate against a fixed target); 90%
        # row splits under a configured set, whose budget is declared against the
        # POOL and whose target is a pool quantity (SS2.4, decision 9); the fresh
        # outcome draw under `plasmode`
        sem_factory = partial(outer._sem_factory, bootstrap=(outer.target == "iv" and not outer.iv_columns))

        class ConfiguredSweep(Strategy):
            def __init__(inner_self, **kwargs):
                extra = {}
                if param == "m":
                    extra["n_samples_override"] = FOLD_SWEEP_SAMPLES
                super().__init__(
                    sem_factory=sem_factory,
                    da_factory=outer._da_factory,
                    poly_transform=None,
                    test_fraction=CIGARETTE_CONFIG.test_fraction,
                    default_gamma=QUERY_GAMMA[outer.spec],
                    default_epsilon=outer._epsilon_budget(CIGARETTE_CONFIG.epsilon),
                    experiment_name=EXPERIMENT_NAME,
                    declared_iv=bool(outer.iv_columns),
                    **extra,
                    **kwargs,
                )

        return ConfiguredSweep

    # ------------------------------------------------------------- query sweep

    def _query_grids(self, X):
        """The six cigarette query grids, as id -> (plotted x, query points).

        Four per-dimension rays over +-`GRID_SPAN` sd of one FWL'd coordinate with
        the other three at the data mean, which after FWL is exactly 0, and the ray
        along v-hat, where every homogeneous h is 0 and the symmetry alone
        identifies the point. The count is forced EVEN: an odd grid over a symmetric
        span puts a query exactly at the origin, where `PartialR2._solve_single`
        short-circuits |x| < 1e-9 to [0, 0] and every method scores a spurious miss.
        """
        points = int(self.kwargs["sweep_samples"])
        points -= points % 2
        if points < 2:
            raise ValueError(f"sweep_samples {self.kwargs['sweep_samples']!r} leaves no even query grid.")

        grids = {}
        for column, name in enumerate(DIM_IDS):
            span = GRID_SPAN * float(np.std(X[:, column]))
            x = np.linspace(-span, span, points)
            grid = np.zeros((points, X.shape[1]))
            grid[:, column] = x
            grids[name] = (x, grid)

        direction = V / np.linalg.norm(V)
        span = GRID_SPAN * float(np.std(X @ direction))
        c = np.linspace(-span, span, points)
        grids[RAY_ID] = (c, c[:, None] * direction[None, :])
        return grids

    def _run_query_sweep(self):
        """The PC panel and the radial sweep, as everywhere else, plus the six
        cigarette query figures, the width-ratio figure and the two tables.

        The panel and every extra figure are fit ONCE, on the full 2450-row panel:
        `_sem_factory` defaults to `bootstrap=False`, so the query design IS the
        pool and the headline ratios are the panel's, not one resample's.
        """
        runner = self.get_query_runner_cls()(methods=self.methods, **{**self._get_clean_kwargs(), "n_experiments": 1})
        panel = PanelBuilder(runner, self.name, "optical" in self.name)
        panel.build(self.kwargs["sweep_samples"])
        self._plot_query_sweep(runner, panel.get_radial_results())

        for name, (x, grid) in self._query_grids(runner.X).items():
            results, _ = panel.predict(grid)
            save(x, f"{name}_values", self.name, "pkl", subdir=SUBDIR_QUERY)
            save(results, f"{name}_outcomes", self.name, "pkl", subdir=SUBDIR_QUERY)
            create_query_sweep_plot(x, results, **ANNOTATE_SWEEP_PLOT[name], fname=name, experiment=self.name)

        self._plot_width_ratio(runner, panel)
        self._write_coefficients(runner, panel)
        if self.iv_columns:
            # the phase-b figures and the benchmark tables exist only under a
            # configured set: the shipped path writes nothing it did not before
            self._plot_headline(runner, panel)
            self._write_benchmarks()
            self._write_benchmarks_iv(runner)
        self._write_ladder()

    # --------------------------------------------------------------- headline

    def benchmarks(self) -> dict[str, tuple[int, float, float, float, float]]:
        """The SS6.2 benchmark rows at this spec, computed once."""
        if self._benchmarks is None:
            panel_data = SEM.panel()
            self._benchmarks = {
                key: benchmark_gamma(panel_data, w, self.spec) for key, w in benchmark_covariates(panel_data).items()
            }
        return self._benchmarks

    def benchmarks_iv(self) -> dict[str, tuple[int, float, float, float]]:
        """The gamma_z rows against the configured set, same W's, computed once."""
        if self._benchmarks_iv is None:
            panel_data = SEM.panel()
            self._benchmarks_iv = {
                key: benchmark_gamma_z(panel_data, w, self.spec, self.iv_columns)
                for key, w in benchmark_covariates(panel_data).items()
            }
        return self._benchmarks_iv

    def _plot_headline(self, runner, panel):
        """F1 and F2 (SS10), off the query panel's fitted models, one pair per
        HEADLINE_COEFFICIENTS entry (`beta_pn_*`, `beta_p_*`).

        F1: the coefficient's interval of each headline method against gamma on
        the benchmarked range, with the PI+IV feasibility floor, the two
        benchmarks (1x tax-differential, 1x addiction stock), gamma*(b) and 3x the
        tax-differential marked, the frame widened to show all five. Reading on
        beta_pn: the lower bound flattens by 0.19 and only the upper end grows
        with the budget (that PI never separates from PI+INV is T1's row, not a
        band here).
        F2: the same methods against the radius r_Z = s sqrt(gamma_z) at the query
        budget. Marks, in order: the cluster-bootstrap median and p95 of the
        moment at the target, the declared radius, then the radius of each
        IV_BENCHMARKS leak, so the declared budget is seen against what a
        resampled panel reads and against what the same omitted variables that
        benchmark gamma would do to the instruments. Every model's `gamma_z` is
        put back after.
        """
        # a spelled default (`DA+PI+IV(T,Z)`) is the headline `DA+PI+IV`; the
        # `(Z)` variant is not a headline and the outcomes stay keyed by HEADLINE_METHODS
        fitted = {parse_method(n)[0] if iv_mode(n) == "T,Z" else n: m for n, m in panel.fitted_models.items()}
        models = {name: fitted[name] for name in HEADLINE_METHODS if name in fitted}
        if not models:
            logger.warning(f"headline figures need one of {HEADLINE_METHODS} in `methods`; skipping.")
            return
        design, scale = runner.sem.design, runner.sem.design.sigma
        points = int(self.kwargs["sweep_samples"])
        Z, b = runner.sem.iv_pool, runner.sem.solution.ravel()
        tax_diff, lag = self.benchmarks()["tax_diff"][3], self.benchmarks()["lag_q"][3]
        # the floor at PI+IV's bound s sqrt(gamma_z): the panel is sigma-normalised
        # (`build_design` divides y by the OLS residual sd), so PI+IV's s is 1 and
        # the bound is sqrt(gamma_z) in outcome units, what SS3.3's 0.1107 was
        # measured at
        f1_marks = (
            feasibility_floor(design, Z, np.sqrt(self.gamma_z)),
            tax_diff,
            lag,
            float(runner.sem.bias_sq / runner.sem.sigma_sq),
            3.0 * tax_diff,
        )
        logger.info(
            f"F1 marks: PI+IV floor {f1_marks[0]:.4f}, 1x tax-diff {f1_marks[1]:.4f}, 1x lag-q {f1_marks[2]:.4f}, "
            f"gamma*(b) {f1_marks[3]:.4f}, 3x tax-diff {f1_marks[4]:.4f}"
        )
        median, p95 = moment_quantiles(design, Z, b)
        leaks = tuple(float(np.sqrt(self.benchmarks_iv()[key][3])) for key in IV_BENCHMARKS)
        f2_marks = (median, p95, float(np.sqrt(self.gamma_z)), *leaks)
        logger.info(
            f"F2 marks: cluster-bootstrap moment at the target, median {median:.4f}, p95 {p95:.4f}; declared r_Z "
            f"{f2_marks[2]:.4f}; leak radii {dict(zip(IV_BENCHMARKS, leaks, strict=True))}"
        )
        gammas = np.linspace(*GAMMA_RANGE, points)
        # F2: r_Z enters the bound as s sqrt(gamma_z), so each radius is one
        # gamma_z per model at that model's own s; PI carries no such term
        radii = np.geomspace(*BUDGET_RANGE, points)

        for coefficient in HEADLINE_COEFFICIENTS:
            query = np.eye(design.k)[TREATMENTS.index(coefficient)][None, :]
            label = COEFFICIENT_LABELS[coefficient]

            def interval(model, gamma, query=query):
                return scale * model.predict(query, gamma=gamma)[0]

            # F1
            results = {name: np.full((points, 1, 2), np.nan) for name in models}
            for i, gamma in enumerate(gammas):
                for name, model in models.items():
                    results[name][i, 0] = interval(model, float(gamma))
            stem = f"beta_{coefficient}_gamma"
            save(gammas, f"{stem}_values", self.name, "pkl", subdir=SUBDIR_QUERY)
            save(results, f"{stem}_outcomes", self.name, "pkl", subdir=SUBDIR_QUERY)
            save(np.array(f1_marks), f"{stem}_vlines", self.name, "pkl", subdir=SUBDIR_QUERY)
            create_query_sweep_plot(
                gammas,
                results,
                **ANNOTATE_SWEEP_PLOT[stem],
                ylabel=label,
                fname=stem,
                experiment=self.name,
                vlines=f1_marks,
            )

            # F2
            results = {name: np.full((points, 1, 2), np.nan) for name in models}
            for name, model in models.items():
                if not hasattr(model, "gamma_z"):
                    results[name][:, 0] = interval(model, self.gamma)
                    continue
                declared = model.gamma_z
                try:
                    for i, radius in enumerate(radii):
                        model.gamma_z = float(radius**2 / (model.sigma_sq / model.rho))
                        results[name][i, 0] = interval(model, self.gamma)
                finally:
                    model.gamma_z = declared
            stem = f"beta_{coefficient}_budget"
            save(radii, f"{stem}_values", self.name, "pkl", subdir=SUBDIR_QUERY)
            save(results, f"{stem}_outcomes", self.name, "pkl", subdir=SUBDIR_QUERY)
            save(np.array(f2_marks), f"{stem}_vlines", self.name, "pkl", subdir=SUBDIR_QUERY)
            create_query_sweep_plot(
                radii,
                results,
                **ANNOTATE_SWEEP_PLOT[stem],
                ylabel=label,
                fname=stem,
                experiment=self.name,
                vlines=f2_marks,
            )

    def _write_benchmarks(self):
        """T2: the SS6.2 table, both gamma columns printed since their agreement
        is the argument, the rejected proxies included."""
        rows = self.benchmarks()
        lines = [
            rf"% cigarette benchmark budgets at spec {self.spec}: gamma for dropping W, in the paper's units.",
            r"% Controls re-residualised inside each covariate's own sample; plain OLS, no clustering.",
            r"% gamma_OVB = c^2 ||P_X W||^2 / (n sigma^2); gamma_CH = r2(W~X) r2(Y~W|X) / (1 - r2(W~X)).",
            r"\begin{tabular}{lrrrrr}",
            r"\toprule",
            r"benchmark $W$ & $n$ & $r^2(Y \sim W \mid X, C)$ & $r^2(W \sim X \mid C)$ "
            r"& $\gamma_{\mathrm{OVB}}$ & $\gamma_{\mathrm{CH}}$ \\",
            r"\midrule",
        ]
        for key, label in BENCHMARK_NAMES.items():
            n, r2y, r2w, gamma_ovb, gamma_ch = rows[key]
            lines.append(f"{label} & {n:d} & {r2y:.4f} & {r2w:.4f} & {gamma_ovb:.4f} & {gamma_ch:.4f} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        save("\n".join(lines) + "\n", "benchmarks", self.name, "tex", subdir=SUBDIR_QUERY)

    def _write_benchmarks_iv(self, runner):
        """T3: the gamma_z rows of the same W's against the configured set, the
        ratio to T2's gamma beside each (Nevo and Rosen's premise), then the
        direct-effect rows: what a direct tax elasticity delta outside the price
        channel costs in gamma_z, with the declared budget read back as a delta."""
        design = runner.sem.design
        rows, gammas = self.benchmarks_iv(), self.benchmarks()
        z_var = float(np.mean(design.Z[:, 0] ** 2))
        declared_delta = float(np.sqrt(self.gamma_z / z_var) * design.sigma)
        lines = [
            rf"% cigarette IV benchmark budgets at spec {design.spec}, instrument set {list(self.iv_columns)}:",
            r"% gamma_z for dropping W, in the paper's units. Same sample, controls and c as benchmarks.tex;",
            r"% gamma_z = c^2 ||P_Z W||^2 / (n sigma^2), r2(W~tau_s) the projection on the external excise alone.",
            rf"% declared gamma_z = {self.gamma_z:g} (r_Z {np.sqrt(self.gamma_z):.4f}) reads as a direct tax "
            rf"elasticity delta = {declared_delta:.4f}: gamma_z = (delta / {design.sigma:.4f})^2 * {z_var:.5f}.",
            r"\begin{tabular}{lrrrrrr}",
            r"\toprule",
            r"benchmark $W$ & $n$ & $r^2(W \sim Z \mid C)$ & $r^2(W \sim \tau_s \mid C)$ "
            r"& $\gamma_z$ & $r_Z = \sqrt{\gamma_z}$ & $\gamma_z / \gamma$ \\",
            r"\midrule",
        ]
        for key, label in BENCHMARK_NAMES.items():
            n, r2z, r2t, gamma_z = rows[key]
            ratio = gamma_z / gammas[key][3] if gammas[key][3] > 0 else float("nan")
            lines.append(
                f"{label} & {n:d} & {r2z:.4f} & {r2t:.4f} & {gamma_z:.4f} & {np.sqrt(gamma_z):.4f} & {ratio:.3f} \\\\"
            )
        lines.append(r"\midrule")
        for delta in DIRECT_EFFECTS:
            gamma_z = direct_effect_gamma_z(design, delta)
            lines.append(
                rf"direct tax elasticity $\delta = {delta:g}$ & {design.n:d} & & & {gamma_z:.4f} "
                rf"& {np.sqrt(gamma_z):.4f} & \\"
            )
        lines += [r"\bottomrule", r"\end{tabular}"]
        save("\n".join(lines) + "\n", "benchmarks_iv", self.name, "tex", subdir=SUBDIR_QUERY)

    def _plot_width_ratio(self, runner, panel):
        """Width over PI's width against |cos(x, v)| in the Sigma^-1 metric.

        For a linear h and eps = 0 the ratio of PI+INV's half-width to PI's is
        exactly sqrt(1 - cos^2(x, v)) in that metric, because
        N(N' Sigma N)^-1 N' = Sigma^-1 - Sigma^-1 v v' Sigma^-1 / (v' Sigma^-1 v).
        So one picture carries what the four per-dimension figures cannot: the trim
        is a function of the query's angle to the homogeneity direction and of
        nothing else. The four coefficient queries are the reference lines.

        `create_sweep_plot` is the right shape for it (an x grid and one curve per
        method) but writes into the sweep folder, hence `subdir`. It also swallows
        every exception and only logs it, so a55(i) checks the file exists.
        """
        X = runner.X
        rows = np.linspace(0, len(X) - 1, min(RATIO_QUERIES, len(X)), dtype=int)
        queries = X[rows]
        precision = np.linalg.inv(X.T @ X / len(X))
        results, _ = panel.predict(queries)

        widths = {name: record[:, 0, 1] - record[:, 0, 0] for name, record in results.items() if record.ndim == 3}
        if "PI" not in widths:
            logger.warning("width-ratio figure needs PI in `methods`; skipping.")
            return
        ratios = {name: (width / widths["PI"])[:, None] for name, width in widths.items()}

        cosines = _cosines(queries, precision)
        # the four coefficient queries, marked on the axis and saved beside it: the
        # figure's whole reading is that each one's trim is fixed by its own angle
        marks = _cosines(np.eye(X.shape[1]), precision)
        save(cosines, "ratio_cos_values", self.name, "pkl", subdir=SUBDIR_QUERY)
        save(ratios, "ratio_cos_outcomes", self.name, "pkl", subdir=SUBDIR_QUERY)
        save(marks, "ratio_cos_marks", self.name, "pkl", subdir=SUBDIR_QUERY)
        create_sweep_plot(
            cosines,
            ratios,
            experiment=self.name,
            fname="ratio_cos",
            subdir=SUBDIR_QUERY,
            xlabel=r"$|\cos({\bm{x}}, {\bm{v}})|$",
            ylabel=r"width / PI width",
            bootstrapped=False,
            vlines=tuple(marks),
        )

    # ------------------------------------------------------------------ tables

    def _write_coefficients(self, runner, panel):
        """T1: every method's interval on the four coefficient queries h(e_j).

        In RAW log units (the intervals are multiplied back by the OLS residual sd),
        so the numbers read as elasticities; the width ratios are unit free. The
        restricted point b_r is the target, the unrestricted 2SLS fit and its
        clustered intervals are context and never a target (SS3).
        """
        design = runner.sem.design
        scale = design.sigma
        queries = np.eye(design.k)
        results, _ = panel.predict(queries)
        precision = np.linalg.inv(design.Sigma)
        cosines = _cosines(queries, precision)

        b_r = runner.sem.solution.ravel()
        b_u, _ = two_stage_fit(design)
        widths = {name: record[:, 0, 1] - record[:, 0, 0] for name, record in results.items() if record.ndim == 3}
        reference = widths.get("PI")

        # under a configured set one more column: beta_pn solved at the primary
        # benchmark budget (the addiction stock, SS6.2) rather than the declared
        # query budget, so the headline coefficient is read at a benchmarked gamma
        benchmark = self.benchmarks()["lag_q"][3] if self.iv_columns else None
        extra_head, extra_cells = [], {}
        if benchmark is not None:
            extra_head = [rf"$\beta_{{p_n}}$ at $\gamma_{{\mathrm{{lag}}}} = {benchmark:.3f}$"]
            query = queries[PN][None, :]
            for name in results:
                model = panel.fitted_models.get(name)
                if model is None:
                    extra_cells[name] = f"{scale * results[name][PN, 0]:.3f}"
                elif results[name].ndim == 3:
                    low, high = scale * model.predict(query, gamma=benchmark)[0]
                    extra_cells[name] = f"$[{low:.3f}, {high:.3f}]$"
                else:
                    extra_cells[name] = f"{scale * results[name][PN, 0]:.3f}"

        head = " & ".join([""] + [ANNOTATE_SWEEP_PLOT[name]["xlabel"] for name in DIM_IDS] + extra_head)
        lines = [
            r"% cigarette coefficient queries h(e_j) = beta_j, raw log units.",
            rf"% spec {design.spec}, anchor {design.anchor}, query budget gamma = {QUERY_GAMMA[self.spec]:g}.",
        ]
        if benchmark is not None:
            lines.append(
                rf"% instrument set {list(self.iv_columns)}, gamma_z = {self.gamma_z:g}; benchmarks (SS6.2): "
                rf"1x tax-diff {self.benchmarks()['tax_diff'][3]:.4f}, 1x lag-q {benchmark:.4f}."
            )
        lines += [
            r"\begin{tabular}{l" + "r" * (design.k + len(extra_head)) + "}",
            r"\toprule",
            head + r" \\",
            r"\midrule",
        ]
        for name, record in results.items():
            if record.ndim != 3:
                cells = [f"{scale * record[j, 0]:.3f}" for j in range(design.k)]
            else:
                cells = []
                for j in range(design.k):
                    low, high = scale * record[j, 0, 0], scale * record[j, 0, 1]
                    ratio = "" if reference is None else f" ({widths[name][j] / reference[j]:.3f})"
                    cells.append(f"$[{low:.3f}, {high:.3f}]${ratio}")
            if name in extra_cells:
                cells.append(extra_cells[name])
            lines.append(TEX_MAPPER.get(name, name) + " & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")
        # the target's own name: the anchor set's restricted point under `iv`,
        # the restricted fit against the configured set or the synthetic
        # homogeneous coefficient otherwise
        label = r"$b_r$" if self.target == "iv" and not self.iv_columns else r"$b_*$"
        lines.append(label + " & " + " & ".join(f"{scale * value:.3f}" for value in b_r) + r" \\")
        lines.append(r"$b_u$ (2SLS) & " + " & ".join(f"{scale * value:.3f}" for value in b_u) + r" \\")
        for by in ("state", "year"):
            low, high = _two_stage_interval(design, by)
            cells = [f"$[{scale * low[j]:.3f}, {scale * high[j]:.3f}]$" for j in range(design.k)]
            lines.append(rf"95\% CI ({by}) & " + " & ".join(cells) + r" \\")
        lines.append(r"$|\cos(e_j, v)|$ & " + " & ".join(f"{value:.3f}" for value in cosines) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        save("\n".join(lines) + "\n", "coefficients", self.name, "tex", subdir=SUBDIR_QUERY)

    def _write_ladder(self):
        """T2: the trend ladder and its diagnostics, one row per spec.

        The two statistics on v'b are kept apart on purpose: `W` is the classical
        restriction F on the OLS ladder (iid by construction) and `J2` the efficient
        two-step restricted GMM, which is the statistic of the program actually
        being solved. t1 and t2 are the pre-registered falsification rows.
        """
        panel_data = SEM.panel()
        lines = [
            rf"% cigarette trend ladder; anchor {self.anchor}, sliver guard gamma_z = {CIGARETTE_CONFIG.gamma_z:g}.",
            r"% A property of the panel and the anchor, not of the target: it is the same under `plasmode`.",
            r"% W is the classical restriction F (iid by construction); J2 is the two-step restricted GMM.",
            r"\begin{tabular}{lrrrrrrrrr}",
            r"\toprule",
            r"spec & $K$ & $\rho_{\max}$ & $W$ & $J_2$ (state) & $J_2$ (year) & $J_2$ (iid) "
            r"& SW $F$ & $\gamma^*$ & sliver \\",
            r"\midrule",
        ]
        for spec in SPECS:
            design = build_design(panel_data, spec=spec, anchor=self.anchor)
            b_r, misfit = restricted_fit(design)
            gap = design.b_ols - b_r
            row = [
                spec,
                f"{design.K:d}",
                f"{rho_max(design):.4f}",
                f"{homogeneity_f(design):.1f}",
                *(f"{restricted_gmm(design, by=by)[1]:.2f}" for by in ("state", "year", "iid")),
                f"{first_stage_f(design, by='state')[0]:.1f}",
                f"{float(gap @ design.Sigma @ gap):.4f}",
                "ok" if misfit**2 < CIGARETTE_CONFIG.gamma_z else "empty",
            ]
            lines.append(" & ".join(row) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        save("\n".join(lines) + "\n", "ladder", self.name, "tex", subdir=SUBDIR_QUERY)
