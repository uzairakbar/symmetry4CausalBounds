"""
US state cigarette demand experiment using generic runners.
"""

from functools import partial

import numpy as np
from loguru import logger

from src.data_augmentors.cigarettes import ScaleTranslation as DA
from src.experiments.base import ExperimentOrchestrator
from src.experiments.configs import (
    ANNOTATE_SWEEP_PLOT,
    CIGARETTE_CONFIG,
    EPS_TOL,
    QUERY_GAMMA,
    MethodRegistry,
)
from src.experiments.generic_runner import STRATEGIES, GenericQuerySweep
from src.experiments.utils import PanelBuilder, create_query_sweep_plot, create_sweep_plot, save
from src.experiments.utils.constants import SUBDIR_QUERY, TEX_MAPPER
from src.oracle import epsilon_star, preserve_rng
from src.sem.cigarettes import (
    SPECS,
    V,
    build_design,
    clustered_vcov,
    clusters,
    first_stage_f,
    homogeneity_f,
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
        **kwargs,
    ):
        """
        Args:
            target: `iv` = the restricted-IV point (the tax-moment program solved
                subject to v'b = 0, so Z can be discarded); `plasmode` = the real
                design with a synthetic homogeneous h_* and known confounding.
            spec: state FE plus a trend of this order.
            anchor: which excise columns instrument which regressors.
            da_amplitude: DA amplitude along v, in units of sd(X . v-hat).
            sliver: target the leaky-IV set rather than the restricted point.
        """
        self.target, self.spec, self.anchor = target, spec, anchor
        self.sliver = bool(sliver)
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

        class CigaretteRegistry(MethodRegistry):
            @staticmethod
            def build_methods(names):
                return MethodRegistry.build_methods(names, gamma=QUERY_GAMMA[spec], epsilon=epsilon, **toggles)

        super().__init__(EXPERIMENT_NAME, CigaretteRegistry(), **kwargs)

    # ---------------------------------------------------------------- factories

    def _sem_factory(self, bootstrap: bool = False):
        """Factory for creating SEM instances.

        `bootstrap` is the replicate mechanism and is bound PER RUNNER (SS6): the
        query figures and the coefficient table are fit on the full panel, the
        sweeps on state-cluster bootstrap replicates of it. `pool` is the whole
        panel either way, so h_*, gamma* and eps* do not move between them.
        """
        return SEM(
            spec=self.spec,
            target=self.target,
            anchor=self.anchor,
            bootstrap=bootstrap,
            sliver=self.sliver,
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
                    **kwargs,
                )

        return CigaretteQuerySweep

    def build_methods(self, gamma: float, epsilon: float, epsilon_iv=None, n_jobs=None, rho=1.0):
        """Methods at explicit (per-experiment) budgets. `n_jobs` overrides the
        toggle -- perf needs serial models to time methods, not the harness."""
        toggles = self.toggles if n_jobs is None else {**self.toggles, "n_jobs": n_jobs}
        return MethodRegistry.build_methods(
            self.kwargs["methods"], gamma=gamma, epsilon=epsilon, epsilon_iv=epsilon_iv, rho=rho, **toggles
        )

    def get_sweep_runner_cls(self, param: str) -> type:
        """Configured strategy for one sweep parameter."""
        outer, Strategy = self, STRATEGIES[param]
        # the sweep replicates: a state-cluster bootstrap under `iv` (a resampling
        # rate against a fixed target), the fresh outcome draw under `plasmode`
        sem_factory = partial(outer._sem_factory, bootstrap=(outer.target == "iv"))

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
        self._write_ladder()

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
        save(cosines, "ratio_cos_values", self.name, "pkl", subdir=SUBDIR_QUERY)
        save(ratios, "ratio_cos_outcomes", self.name, "pkl", subdir=SUBDIR_QUERY)
        create_sweep_plot(
            cosines,
            ratios,
            experiment=self.name,
            fname="ratio_cos",
            subdir=SUBDIR_QUERY,
            xlabel=r"$|\cos({\bm{x}}, {\bm{v}})|$",
            ylabel=r"width / PI width",
            bootstrapped=False,
            vlines=tuple(_cosines(np.eye(X.shape[1]), precision)),
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

        head = " & ".join([""] + [ANNOTATE_SWEEP_PLOT[name]["xlabel"] for name in DIM_IDS])
        lines = [
            r"% cigarette coefficient queries h(e_j) = beta_j, raw log units.",
            rf"% spec {design.spec}, anchor {design.anchor}, query budget gamma = {QUERY_GAMMA[self.spec]:g}.",
            r"\begin{tabular}{l" + "r" * design.k + "}",
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
            lines.append(TEX_MAPPER.get(name, name) + " & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")
        lines.append(r"$b_r$ & " + " & ".join(f"{scale * value:.3f}" for value in b_r) + r" \\")
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
