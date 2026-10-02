"""
Base classes for experiment orchestration with unified runner logic.
Updated to use simplified fit_model signature and OPTIMIZED LOOP ORDER.
"""

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

import enlighten
import numpy as np
from loguru import logger

from src.experiments.configs import (
    ANNOTATE_SWEEP_PLOT,
    DATASET_DEFAULTS,
    EPS_TOL,
    METRIC_SPECS,
    PARAM_SPECS,
)
from src.experiments.perf import perf_sweeps
from src.experiments.utils import fit_model, save, set_seed
from src.experiments.utils.constants import (
    SUBDIR_PERF,
    SUBDIR_QUERY,
    SUBDIR_SWEEP,
)
from src.experiments.utils.metrics import STATUS_CATEGORIES, evaluate_queries, rho_hat
from src.experiments.utils.model_fitting import instrument_columns
from src.experiments.utils.plotting import (
    create_query_sweep_plot,
    create_sweep_plot,
)
from src.methods.abstract import pointEstimator as Regressor
from src.methods.sensitivity_models import constraint_floor

# QueryEval scalar fields recorded at every (method, step, experiment)
METRIC_FIELDS: tuple[str, ...] = (
    "approximation_error",
    "worst_error",
    "interval_width",
    "coverage",
    "wall_clock",
)

ModelBuilder = Callable[[], Regressor]
MANAGER = enlighten.get_manager()


# =============================================================================
# DATA CONTEXT
# =============================================================================


@dataclass
class ExperimentDataContext:
    """Container for experiment data."""

    sem: Any
    da: Any
    X: np.ndarray
    y: np.ndarray
    GX: np.ndarray
    G: np.ndarray
    X_raw: np.ndarray | None = None
    GX_raw: np.ndarray | None = None
    oracle: Any | None = None  # OracleParameters; unused by default
    # the real instrument, (n, m); None is spelled (n, 0) on construction (SS2.5)
    Z: np.ndarray | None = None
    # a second augmented copy of the same rows for the invariance pairs, where the
    # DA measure the DA+ methods fit on is not the pairs' measure (do-MNIST mixes
    # observed rows into `GX`; the PI+INV pairs must stay unmixed). None everywhere
    # else: `fit_model` then pairs X with `GX`
    GX_inv: np.ndarray | None = None

    def __post_init__(self):
        self.Z = instrument_columns(self.Z, len(self.X))


@dataclass
class SweepData:
    """One sweep step's data. Unpacks as the legacy 6-tuple."""

    X: np.ndarray
    y: np.ndarray
    GX: np.ndarray
    G: np.ndarray
    X_test: np.ndarray
    estimand: np.ndarray
    # untiled copies for baselines that are exactly tiling-invariant (m-sweep)
    X_base: np.ndarray | None = None
    y_base: np.ndarray | None = None
    # half-width of the target SET per query (`SEM.extent`); None is a point target
    extent: np.ndarray | None = None
    # the real instrument beside X, (n, m), and its untiled copy beside X_base.
    # None is spelled (n, 0) here, the one place a None is converted, so no
    # consumer ever sees one (SS2.5)
    Z: np.ndarray | None = None
    Z_base: np.ndarray | None = None
    # the unmixed pairs for PI+INV, see `ExperimentDataContext.GX_inv`
    GX_inv: np.ndarray | None = None

    def __post_init__(self):
        self.Z = instrument_columns(self.Z, len(self.X))
        if self.X_base is not None:
            # a tiled Z starts with its untiled block, so that is the default
            base = self.Z[: len(self.X_base)] if self.Z_base is None else self.Z_base
            self.Z_base = instrument_columns(base, len(self.X_base))

    def __iter__(self):
        return iter((self.X, self.y, self.GX, self.G, self.X_test, self.estimand))

    @classmethod
    def coerce(cls, data) -> "SweepData":
        return data if isinstance(data, cls) else cls(*data)

    @property
    def fit_arrays(self) -> dict[str, Any]:
        return dict(
            X=self.X,
            y=self.y,
            GX=self.GX,
            G=self.G,
            X_base=self.X_base,
            y_base=self.y_base,
            Z=self.Z,
            Z_base=self.Z_base,
            GX_inv=self.GX_inv,
        )

    @property
    def metric_extent(self) -> float | np.ndarray:
        """`extent` as the metrics take it: 0.0 when the target is a point."""
        return 0.0 if self.extent is None else self.extent


# =============================================================================
# BASE RUNNER
# =============================================================================


class BaseExperimentRunner(ABC):
    """Base class for experiment runners."""

    # controls partialled out of the design per observation, charged to every
    # sigma-hat as `absorbed_rate * n_obs` dof; set per dataset by the
    # orchestrator (the cigarette FWL), 0 everywhere else
    absorbed_rate: float = 0.0

    def __init__(
        self,
        seed: int,
        n_samples: int,
        n_experiments: int,
        sweep_samples: int,
        methods: dict[str, ModelBuilder],
        hyperparameters: dict[str, Any] | None = None,
        recalibrate: bool = True,
        pad: bool = False,
        clipy: bool = True,
        mean_match: bool = True,
        gamma_n_alpha: float = 0.0,
        unit_cap: int | None = None,
        n_jobs: int = 1,
        **kwargs,
    ):
        if seed >= 0:
            set_seed(seed)

        self.seed = seed
        self.n_samples = n_samples
        self.n_experiments = n_experiments
        self.sweep_samples = sweep_samples
        self.methods = methods
        self.hyperparameters = hyperparameters
        # toggles: `recalibrate` and `mean_match` are EXPLICIT, not swallowed by
        # **kwargs: the floor report must measure the ball the solver actually uses
        # (recalibrated budget, Lem. 2 geometry), and a silent default would be a
        # lie the gates cannot see.
        self.recalibrate = recalibrate
        self.pad = pad
        self.clipy = clipy
        self.mean_match = mean_match
        # the finite-sample pads the models solve with (`PartialR2.ball_budget`),
        # explicit for the floor reports' sake as the toggles above: 0.0 is raw
        self.gamma_n_alpha = float(gamma_n_alpha)
        self.unit_cap = unit_cap
        self.n_jobs = n_jobs

    @abstractmethod
    def run(self, desc: str):
        """Run the experiment."""
        pass

    def _iv_floor_report(self, name: str, model, GX, y, G, Z, gamma, n_obs=None) -> None:
        """Log each DA IV row's radius against that row's own floor on the DA ball;
        never change it. A radius under the floor makes every query INFEASIBLE, as
        any empty constraint set does. The floor is taken at the mean row's
        delta = 0 (`constraint_floor`), an upper reference for the padded program.
        A model without a DA IV row (or an intersection's baseline) is skipped."""
        for part in (model, getattr(model, "augmented", None)):
            if part is None or not getattr(part, "_da_fit", False) or not getattr(part, "rows", ()):
                continue
            T = None if G is None else np.asarray(G, dtype=float).reshape(len(GX), -1)
            Z = None if Z is None else np.asarray(Z, dtype=float).reshape(len(GX), -1)
            blocks = {"t": T, "z": Z, "tz": None if T is None or Z is None else np.hstack([T, Z])}
            for row in part.rows:
                block = blocks[row]
                if block is None or np.size(block) == 0:
                    continue
                radius = part.iv_radius(row, gamma)
                try:
                    floor = constraint_floor(
                        GX,
                        y,
                        gamma,
                        kind="iv",
                        Z=np.asarray(block, dtype=float).reshape(len(GX), -1),
                        mean_match=self.mean_match,
                        rho=part.rho,
                        recalibrate=self.recalibrate,
                        n_obs=n_obs,
                        absorbed_rate=self.absorbed_rate,
                        gamma_n_alpha=self.gamma_n_alpha,
                        unit_cap=self.unit_cap,
                    )
                except Exception as error:  # never let a diagnostic break a run
                    logger.warning(f"{name} IV row {row}: constraint floor unavailable ({error}).")
                    continue
                if radius**2 < floor:
                    logger.info(
                        f"{name} IV row {row}: radius {radius:.6g} (r^2 {radius**2:.4g}) is BELOW the row's own "
                        f"floor {floor:.4g}; left as is, every query will read INFEASIBLE, never raised."
                    )


# =============================================================================
# QUERY SWEEP RUNNER (for visualization)
# =============================================================================


class QuerySweepRunner(BaseExperimentRunner):
    """Runner for query sweep experiments (radial/PC sweeps for visualization)."""

    def run(self, desc: str = "Query Sweep") -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """
        Run query sweep across treatment space.

        Returns:
            Tuple of (query_points, predictions_dict)
        """
        context = self.setup_data()
        self.context_ = context
        queries = self.get_sweep_values()
        results = {}
        # the fitted models, by name: a runner that scores them elsewhere than on the
        # sweep queries (do-MNIST's population metrics) reads them off here, beside
        # each one's `fit_model` seconds
        self.models_ = {}
        self.fit_seconds_ = {}

        with MANAGER.counter(total=len(self.methods), desc=desc, unit="methods") as pbar:
            for name, builder in self.methods.items():
                if name == "ATE":
                    predictions = self.estimand(queries)
                else:
                    model = builder()

                    self.before_fit(name)
                    # Pass method name so fit_model knows which data to use
                    start = time.perf_counter()
                    fit_model(
                        model=model,
                        method_name=name,
                        X=context.X,
                        y=context.y,
                        GX=context.GX,
                        G=context.G,
                        Z=context.Z,
                        GX_inv=context.GX_inv,
                        hyperparameters=self.hyperparameters,
                        da=context.da,
                    )
                    self.fit_seconds_[name] = time.perf_counter() - start
                    self.after_fit(name, model)

                    predictions = model.predict(queries)
                    self.models_[name] = model

                # Reshape for consistent output format
                if "PI" in name:
                    results[name] = predictions[:, np.newaxis, :]
                else:
                    results[name] = predictions.reshape(len(queries), 1)

                pbar.update()

        return queries, results

    def before_fit(self, name: str) -> None:
        """Called right before each method's `fit_model`, outside its timer. A no-op
        here; do-MNIST records the load average in it."""

    def after_fit(self, name: str, model) -> None:
        """Called right after each method's `fit_model`, before its first predict.
        A no-op here; do-MNIST times its constraint floors in it."""

    def estimand(self, queries) -> np.ndarray:
        """The causal target at the queries, `sem.f` by default. A SEM whose target
        is not a function of the query features (do-MNIST's h_* is a function of
        the digit label) overrides this beside `get_sweep_values`."""
        return self.context_.sem.f(queries)

    @abstractmethod
    def setup_data(self) -> ExperimentDataContext:
        """Setup experiment data. Must return ExperimentDataContext."""
        pass

    @abstractmethod
    def get_sweep_values(self) -> np.ndarray:
        """Get query points for sweep."""
        pass


# =============================================================================
# PARAMETER SWEEP RUNNER (for metrics)
# =============================================================================


class ParamSweepRunner(BaseExperimentRunner):
    """
    Core sweep loop, shared by every param strategy.

    Emits one full record per (method, step, experiment):
        results[method][metric] -> (n_steps, n_experiments)
        statuses[method]        -> (n_steps, n_experiments, 4)
    """

    param_key: str = None  # indexes PARAM_SPECS

    def __init__(
        self,
        method_factory: Callable | None = None,
        experiment_name: str = "simulation",
        param_grid_override: Any | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # builds methods at experiment-specific budgets (ParamPolicy, PLAN 7)
        self.method_factory = method_factory
        self.experiment_name = experiment_name
        # perf collapses the grid to a single default operating point
        self.param_grid_override = param_grid_override
        # one cell's budgets, ((j, i), data, budgets): the floor report inside
        # `fit_epsilon` fires once per cell as it always has
        self._budgets = None
        self.setup_sems_and_das()

    @property
    def spec(self):
        return PARAM_SPECS[self.param_key]

    @property
    def data_depends_on_param(self) -> bool:
        """False => fit once per experiment and only re-solve (DPP cache)."""
        return not self.spec.data_constant

    def get_param_range(self) -> np.ndarray:
        if self.param_grid_override is not None:
            return np.asarray(self.param_grid_override)
        return self.spec.grid_fn(self.experiment_name, self.sweep_samples)

    def observed_x(self, param_values: np.ndarray) -> np.ndarray:
        """x-axis actually plotted; strategies with a measured x override."""
        return param_values

    @property
    def xlabel(self) -> str:
        """Label of the plotted x; strategies whose axis depends on a toggle override."""
        return self.spec.xlabel

    @property
    def vlines(self) -> tuple[float, ...]:
        """Reference x positions; strategies may add measured thresholds."""
        return self.spec.vlines

    @property
    def xticks(self) -> tuple[float, ...]:
        """Labelled major ticks of the plotted x; () = the scale's own locator."""
        return self.spec.xticks

    def axis_record(self) -> dict[str, Any] | None:
        """The factors behind a MEASURED x-axis, in knob order; None for a designed grid."""
        return None

    # ------------------------------------------------------ per-experiment policy

    def fit_gamma(self, experiment_index: int) -> float:
        """Budget baked into the built methods (PLAN 7); gamma* by default."""
        return self._finite(self.get_oracle(experiment_index).gamma_star, self.default_gamma, "gamma*")

    def fit_epsilon(self, experiment_index: int, step_index: int = 0, data=None) -> float:
        """
        Assumed invariance error: oracle eps* = ||W|| over the full augmentation
        (the q0.95 of |W| where the SEM carries `epsilon_quantile`, i.e. on every
        sweep), just large enough to admit h_* in PI+INV, off the knife edge.

        The oracle quantity is the budget, as is: where it lands under what the
        constraint can attain on this ball it is reported (`_floor_report`) and
        every query reads INFEASIBLE.
        """
        budget = self._finite(self.get_oracle(experiment_index).epsilon_star, self.default_epsilon, "eps*") + EPS_TOL
        self._floor_report(budget, data, "inv", experiment_index, "epsilon")
        return budget

    def fit_rho(self, experiment_index: int, data=None) -> float:
        """Information-loss factor of this step's DA draw, rho_hat = sigma~^2/sigma^2
        on the fit sample (the ratio of the two in-class MMSEs, SS4.2). The DA+
        methods solve at gamma~ = gamma ((1 - t) + t / rho) with t = `recalibrate`;
        the intersections compute the same ratio from their own two branches."""
        if data is None or getattr(data, "GX", None) is None:
            return 1.0
        rho = self._finite(rho_hat(data.X, data.GX, data.y, intercept=self.mean_match), 1.0, "rho_hat")
        if rho < 1.0:
            logger.warning(f"rho_hat {rho:.4f} < 1 (DPI says >= 1): sampling noise; the solvers read it as 1.")
        return rho

    def _finite(self, value, fallback, name: str) -> float:
        if value is None or not np.isfinite(value):
            logger.warning(f"{name} not computable; falling back to {float(fallback):g}.")
            return float(fallback)
        return float(value)

    def _floor_report(self, budget, data, kind: str, experiment_index: int, label: str) -> None:
        """Log a budget against the constraint's own attainable floor; never change it.

        A budget under the floor (`budget^2 < floor`) is no bound at all: `_prepare`
        returns all-INFEASIBLE, the cell's width and coverage are NaN, and the method
        simply does not show up at that step. That is the rule PI+INV has always
        followed on the epsilon sweep, and it is now the rule everywhere:
        there is no floor guard any more, and no budget is ever raised. Where the
        oracle budget lands under the floor (small n, the m grid, a few omega
        knobs): PLAN v16 SS2.2.

        `kind` 'iv' measures the T row's own floor, with the translation amounts
        alone, on the DA ball; IV rows can still be jointly infeasible with every
        floor cleared, and that too reads INFEASIBLE, unlogged. Only a budget BELOW
        the floor is logged.

        No `data` means no measurement: that is the epsilon sweep's per-step call,
        where the budget is an ASSUMPTION the figure exists to test (decision 16).
        """
        if data is None:
            return
        design = getattr(data, "X" if kind == "inv" else "GX", None)
        # 'iv' measures the T constraint's own geometry: the translation amounts alone
        extra = {"GX": getattr(data, "GX", None)} if kind == "inv" else {"Z": getattr(data, "G", None)}
        if design is None or next(iter(extra.values())) is None:
            return
        # the DA ball ('iv': DA+PI+IV fits on GX) is recalibrated; the baseline
        # ball ('inv': PI+INV fits on X) is not
        if kind == "iv":
            extra.update(rho=self.fit_rho(experiment_index, data), recalibrate=self.recalibrate)
        try:
            floor = constraint_floor(
                design,
                data.y,
                self.fit_gamma(experiment_index),
                kind=kind,
                mean_match=self.mean_match,
                # the ball's own dof: the m sweep's rows are an m-fold tiling
                n_obs=None if getattr(data, "X_base", None) is None else len(data.X_base),
                absorbed_rate=self.absorbed_rate,
                gamma_n_alpha=self.gamma_n_alpha,
                unit_cap=self.unit_cap,
                **extra,
            )
        except Exception as error:  # never let a diagnostic break a run
            logger.warning(f"{label}: constraint floor unavailable ({error}); budget left at the oracle value.")
            return

        if budget**2 < floor:
            logger.info(
                f"{label}: oracle {budget:.6g} is BELOW the constraint's own floor (budget^2 "
                f"{budget**2:.4g} < floor {floor:.4g}); left as is, every query will read "
                "INFEASIBLE, never raised."
            )

    def method_kwargs(self, experiment_index: int) -> dict[str, Any]:
        """Extra builder kwargs. Override when methods need per-experiment state
        the budgets do not carry (e.g. prefit outcome models)."""
        return {}

    def fit_budgets(self, experiment_index: int, step_index: int, data) -> dict[str, Any]:
        """The builder kwargs of one cell, memoised on the cell and its data (the
        same object). A runner attribute changed between two calls on the same cell and data is not
        seen: the second call returns the first's budgets."""
        key = (experiment_index, step_index)
        if self._budgets is None or self._budgets[0] != key or self._budgets[1] is not data:
            budgets = dict(
                gamma=self.fit_gamma(experiment_index),
                epsilon=self.fit_epsilon(experiment_index, step_index, data),
                rho=self.fit_rho(experiment_index, data),
                **self.method_kwargs(experiment_index),
            )
            self._budgets = (key, data, budgets)
        return self._budgets[2]

    def build_models(self, experiment_index: int, step_index: int, data) -> dict[str, Any]:
        """Fresh, fitted models at this experiment's budgets."""
        budgets = self.fit_budgets(experiment_index, step_index, data)
        builders = self.method_factory(**budgets) if self.method_factory else self.methods

        models = {}
        for name in self.methods:
            if name == "ATE":
                continue
            model = builders[name]()
            fit_model(
                model=model,
                method_name=name,
                hyperparameters=self.hyperparameters,
                da=self.get_da(experiment_index),
                **data.fit_arrays,
            )
            # the DA IV rows against their own floors, as `fit_epsilon` reports eps
            n_obs = None if getattr(data, "X_base", None) is None else len(data.X_base)
            self._iv_floor_report(name, model, data.GX, data.y, data.G, data.Z, budgets["gamma"], n_obs=n_obs)
            models[name] = model
        return models

    # ------------------------------------------------------------------ loop

    def run(self, desc: str = "Param Sweep") -> tuple[np.ndarray, dict, dict]:
        """Sweep the parameter, recording every metric at every step."""
        param_values = self.get_param_range()
        n_steps = len(param_values)
        shape = (n_steps, self.n_experiments)

        results = {name: {metric: np.full(shape, np.nan) for metric in METRIC_FIELDS} for name in self.methods}
        statuses = {name: np.zeros(shape + (len(STATUS_CATEGORIES),), dtype=int) for name in self.methods}

        with MANAGER.counter(total=self.n_experiments * n_steps, desc=desc, unit="runs") as pbar:
            for j in range(self.n_experiments):
                # data-constant sweeps (gamma, epsilon) fit ONCE per experiment
                cached_data, cached_models = None, None
                if not self.data_depends_on_param:
                    cached_data = SweepData.coerce(self.generate_data(j, param_values[0]))
                    cached_models = self.build_models(j, 0, cached_data)

                for i, param in enumerate(param_values):
                    if self.data_depends_on_param:
                        data = SweepData.coerce(self.generate_data(j, param))
                        models = self.build_models(j, i, data)
                    else:
                        data, models = cached_data, cached_models

                    for name in self.methods:
                        if name == "ATE":
                            estimate, query_status, elapsed = data.estimand, None, 0.0
                        else:
                            model = models[name]
                            # query evaluation ONLY -- DA transform lives in
                            # generate_data and fitting in build_models, both
                            # outside this timer. Keep it that way.
                            start = time.perf_counter()
                            estimate = model.predict(data.X_test, **self.get_predict_kwargs(param, j))
                            elapsed = time.perf_counter() - start
                            query_status = getattr(model, "query_status", None)

                        record = evaluate_queries(
                            data.estimand, estimate, query_status, elapsed, extent=data.metric_extent
                        )
                        for metric in METRIC_FIELDS:
                            results[name][metric][i, j] = getattr(record, metric)
                        statuses[name][i, j] = record.status_counts

                    pbar.update()

        return self.observed_x(param_values), results, statuses

    @abstractmethod
    def setup_sems_and_das(self):
        """Setup SEMs and data augmentors for all experiments."""
        pass

    @abstractmethod
    def get_da(self, experiment_index: int):
        """Get data augmentor for specific experiment."""
        pass

    @abstractmethod
    def get_oracle(self, experiment_index: int):
        """Get oracle parameters for specific experiment."""
        pass

    def get_predict_kwargs(self, param, experiment_index: int) -> dict[str, Any]:
        """Additional kwargs for model.predict(). Override to sweep a budget."""
        return {}

    @abstractmethod
    def generate_data(self, experiment_index: int, param) -> "SweepData":
        """
        Generate data for one experiment at one parameter value.

        Returns:
            SweepData, or the legacy (X, y, GX, G, X_test, estimand) tuple.
        """
        pass


# =============================================================================
# ORCHESTRATOR
# =============================================================================


class ExperimentOrchestrator(ABC):
    """Orchestrates experiment execution: Setup -> Run -> Save -> Plot."""

    # PC panel + radial sweep. False => the query runner supplies its own x-axis.
    build_panel: bool = True

    def __init__(self, experiment_name: str, method_registry, **kwargs):
        self.name = experiment_name
        self.registry = method_registry
        self.kwargs = kwargs
        # the m sweep's n fallback as `resolve_dataset_block` fills it, for an
        # orchestrator built without the yaml
        defaults = DATASET_DEFAULTS.get(experiment_name)
        if defaults is not None and defaults.m_sweep_n_percent is not None:
            self.kwargs.setdefault("m_sweep_n_percent", defaults.m_sweep_n_percent)
        self._sweep_cache = {}  # (param) -> (x, results, statuses), memo per param
        self._sweep_vlines = {}  # (param) -> measured reference x positions
        self._sweep_axis = {}  # (param) -> factors behind a measured x, or None
        self._sweep_xlabel = {}  # (param) -> the runner's label for the plotted x
        self._sweep_xticks = {}  # (param) -> the runner's fixed x ticks, or ()

    @abstractmethod
    def get_query_runner_cls(self) -> type[QuerySweepRunner]:
        """Return the QuerySweepRunner class for this experiment."""
        pass

    @abstractmethod
    def get_sweep_runner_cls(self, param: str) -> type[ParamSweepRunner]:
        """Return the configured strategy class for one sweep parameter."""
        pass

    @abstractmethod
    def build_methods(
        self,
        gamma: float,
        epsilon: float,
        n_jobs: int | None = None,
        rho: float = 1.0,
    ) -> dict[str, Any]:
        """Build methods at explicit budgets (per-experiment ParamPolicy); `rho`
        is the step's information-loss factor for the DA+ balls. The IV rows read
        `epsilon` and the orchestrator's declared `gamma_z`."""
        pass

    @property
    def methods(self):
        """Build methods for this experiment."""
        return self.registry.build_methods(self.kwargs["methods"])

    @property
    def has_z(self) -> bool:
        """Whether the experiment has a real Z (a non-empty `iv`): every label, hue
        and line style is a function of (method, has_z). No Z here; the simulation
        and cigarette orchestrators override it on their `iv`."""
        return False

    def run(self, plan):
        """Run the experiment types the `experiment:` block asked for."""
        if getattr(plan, "tint", None) is not None:
            raise ValueError(f"experiment.query.tint is a do-MNIST sweep; {self.name} has none.")
        # beside query/, sweep/ and perf/: the aggregate reads `has_z` off it per
        # column; `methods` is informational, the stored spellings
        save({"has_z": self.has_z, "methods": list(self.kwargs["methods"])}, "labels", self.name, "json")
        if plan.query:
            self._run_query_sweep()
        if plan.sweep:
            self._run_sweeps(plan.sweep)
        if plan.perf:
            self._run_perf(plan.perf)

    def _get_clean_kwargs(self) -> dict[str, Any]:
        """Remove arguments that cause collision with explicit runner args."""
        clean_kwargs = self.kwargs.copy()
        clean_kwargs.pop("methods", None)
        return clean_kwargs

    def sweep_record(self, param: str):
        """Run (or reuse) one param sweep, memoised per parameter."""
        if param in self._sweep_cache:
            return self._sweep_cache[param]

        spec = PARAM_SPECS[param]
        methods = self.methods
        if not spec.include_ate:
            methods = {k: v for k, v in methods.items() if k != "ATE"}

        runner = self.get_sweep_runner_cls(param)(
            methods=methods, method_factory=self.build_methods, **self._get_clean_kwargs()
        )
        record = runner.run(f"{param} sweep")
        self._sweep_cache[param] = record
        self._sweep_vlines[param] = runner.vlines
        self._sweep_axis[param] = runner.axis_record()
        self._sweep_xlabel[param] = runner.xlabel
        self._sweep_xticks[param] = runner.xticks
        return record

    def _run_sweeps(self, sweep_spec):
        """One figure per (param, metric); one pkl per param."""
        for param in sweep_spec.param:
            x_values, results, statuses = self.sweep_record(param)

            save(x_values, f"{param}_values", self.name, "pkl", subdir=SUBDIR_SWEEP)
            save(results, f"{param}_results", self.name, "pkl", subdir=SUBDIR_SWEEP)
            save(statuses, f"{param}_statuses", self.name, "pkl", subdir=SUBDIR_SWEEP)
            # a measured axis also records its factors, so the figure can be
            # re-rendered under the other budget convention without a rerun
            if self._sweep_axis.get(param) is not None:
                save(self._sweep_axis[param], f"{param}_axis", self.name, "pkl", subdir=SUBDIR_SWEEP)
            for metric in sweep_spec.metric:
                metric_spec = METRIC_SPECS[metric]
                create_sweep_plot(
                    x_values,
                    {
                        name: record[metric_spec.key]
                        for name, record in results.items()
                        if metric_spec.include_ate or name != "ATE"
                    },
                    experiment=self.name,
                    fname=f"{param}_{metric}",
                    xlabel=self._sweep_xlabel[param],
                    ylabel=metric_spec.ylabel,
                    xscale=PARAM_SPECS[param].xscale,
                    yscale=metric_spec.yscale,
                    vlines=self._sweep_vlines.get(param, PARAM_SPECS[param].vlines),
                    xticks=self._sweep_xticks.get(param, ()),
                    # the global toggle of SS10.1; absent from the shipped yaml
                    normalize=self.kwargs.get("normalize", False),
                    has_z=self.has_z,
                )

    def _run_perf(self, perf_spec):
        """The epsilon sweeps that time the solves, cross-check the backends and count
        the backends that return a usable bound, on the robustness sweep's own grid,
        data and models (`get_sweep_runner_cls("epsilon")`): one experiment, serial,
        never the cached sweep record."""
        # Always serial: n_jobs speeds up only the SOCP methods, so a parallel
        # record would compare harnesses, not methods. The runner kwarg alone does
        # NOT reach the models: build_methods reads the orchestrator's toggles, so
        # force it on the factory too. ATE is the truth, not an estimator.
        methods = {k: v for k, v in self.methods.items() if k != "ATE"}
        runner = self.get_sweep_runner_cls("epsilon")(
            methods=methods,
            method_factory=partial(self.build_methods, n_jobs=1),
            **{**self._get_clean_kwargs(), "n_jobs": 1, "n_experiments": 1},
        )
        record = perf_sweeps(runner, perf_spec.metric, repeats=perf_spec.repeats)

        save(record.x, "epsilon_values", self.name, "pkl", subdir=SUBDIR_PERF)
        save(record.meta, "epsilon_perf_meta", self.name, "pkl", subdir=SUBDIR_PERF)
        for metric in perf_spec.metric:
            save(record.results[metric], f"epsilon_{metric}_results", self.name, "pkl", subdir=SUBDIR_PERF)
            if metric == "seed_var":
                save(record.failures, "epsilon_seed_var_failures", self.name, "pkl", subdir=SUBDIR_PERF)
                save(record.statuses, "epsilon_seed_var_statuses", self.name, "pkl", subdir=SUBDIR_PERF)
            spec = METRIC_SPECS[metric]
            create_sweep_plot(
                record.x,
                record.results[metric],
                experiment=self.name,
                fname=f"epsilon_{metric}",
                subdir=SUBDIR_PERF,
                xlabel=runner.xlabel,
                ylabel=spec.ylabel,
                xscale=PARAM_SPECS["epsilon"].xscale,
                yscale=spec.yscale,
                vlines=runner.vlines,
                # the wall clock is one median line per method, the seed var and
                # the feasibility a mean over queries with the bootstrap band over
                # them; none is clipped (the slowest method is the result) nor
                # promoted to log (D(eps) spans decades near zero); the
                # feasibility frame is clamped like coverage (create_sweep_plot)
                bootstrapped=(metric in ("seed_var", "feasibility")),
                clip_y=False,
                promote_y=False,
                has_z=self.has_z,
            )

    def _run_query_sweep(self):
        """Query sweep + panel. The panel is a query-space view, so it is
        always built here and never for param sweeps."""
        from src.experiments.utils import PanelBuilder

        # Create runner ONCE - used for both panel and radial sweep
        runner = self.get_query_runner_cls()(methods=self.methods, **{**self._get_clean_kwargs(), "n_experiments": 1})

        if self.build_panel:
            # Optical uses augmented geometry, simulation uses raw
            panel_builder = PanelBuilder(runner, self.name, "optical" in self.name, has_z=self.has_z)
            panel_builder.build(self.kwargs["sweep_samples"])
            # Radial sweep plot reuses the panel's cached results
            results = panel_builder.get_radial_results()
        else:
            _, results = runner.run("Query Sweep")

        self._plot_query_sweep(runner, results)

    def _plot_query_sweep(self, runner, results):
        """Save + plot the query sweep. Panel experiments plot the radial angle;
        override when the queries are not a PC sweep."""
        angles = np.linspace(0, 2 * np.pi, self.kwargs["sweep_samples"], endpoint=False)

        save(angles, "treatment_values", self.name, "pkl", subdir=SUBDIR_QUERY)
        save(results, "outcome_values", self.name, "pkl", subdir=SUBDIR_QUERY)

        create_query_sweep_plot(angles, results, **ANNOTATE_SWEEP_PLOT["pc12"], experiment=self.name, has_z=self.has_z)
