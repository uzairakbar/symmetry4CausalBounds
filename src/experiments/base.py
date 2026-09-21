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
    EPS_TOL,
    FLOOR_GUARD_R,
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
        declared_iv: bool = False,
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
        # **kwargs: the floor guard must measure the ball the solver actually uses
        # (recalibrated budget, Lem. 2 geometry), and a silent default would be a
        # lie the gates cannot see.
        self.recalibrate = recalibrate
        self.pad = pad
        self.clipy = clipy
        self.mean_match = mean_match
        # the IV budget rule of SS2.6, set per dataset by the orchestrator like
        # `raw_gamma` and `eps_tol` are. False: oracle, the T piece from
        # `eps_iv_star` and the Z piece from `eps_iv_z_star`, each guarding its own
        # constraint. True: declared, a non-empty `iv:` asserting near-perfect
        # instruments; the Z radius is then exactly s sqrt(gamma_z) and the T
        # budget is logged against its floor and never raised (decision 8)
        self.declared_iv = bool(declared_iv)

    @abstractmethod
    def run(self, desc: str):
        """Run the experiment."""
        pass


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
        queries = self.get_sweep_values()
        results = {}

        with MANAGER.counter(total=len(self.methods), desc=desc, unit="methods") as pbar:
            for name, builder in self.methods.items():
                if name == "ATE":
                    predictions = context.sem.f(queries)
                else:
                    model = builder()

                    # Pass method name so fit_model knows which data to use
                    fit_model(
                        model=model,
                        method_name=name,
                        X=context.X,
                        y=context.y,
                        GX=context.GX,
                        G=context.G,
                        Z=context.Z,
                        hyperparameters=self.hyperparameters,
                        da=context.da,
                    )

                    predictions = model.predict(queries)

                # Reshape for consistent output format
                if "PI" in name:
                    results[name] = predictions[:, np.newaxis, :]
                else:
                    results[name] = predictions.reshape(len(queries), 1)

                pbar.update()

        return queries, results

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

    def axis_record(self) -> dict[str, Any] | None:
        """The factors behind a MEASURED x-axis, in knob order; None for a designed grid."""
        return None

    # ------------------------------------------------------ per-experiment policy

    def fit_gamma(self, experiment_index: int) -> float:
        """Budget baked into the built methods (PLAN 7); gamma* by default."""
        return self._finite(self.get_oracle(experiment_index).gamma_star, self.default_gamma, "gamma*")

    def fit_epsilon(self, experiment_index: int, step_index: int = 0, data=None) -> float:
        """
        Assumed invariance error: oracle eps* = ||W|| over the full augmentation,
        just large enough to admit h_* in PI+INV, off the knife edge.

        Floor-guarded (see `_floor_guard`): the oracle quantity is the budget, but
        never below what the constraint can actually attain on this ball.
        """
        budget = self._finite(self.get_oracle(experiment_index).epsilon_star, self.default_epsilon, "eps*") + EPS_TOL
        return self._floor_guard(budget, data, "inv", experiment_index, "epsilon")

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

    def _floor_guard(self, budget, data, kind: str, experiment_index: int, label: str, declared: bool = False) -> float:
        """Rescue a budget that is INFEASIBLE, and only such a budget.

        A budget under the constraint's own attainable floor is not a tighter
        bound, it is NO bound: `_prepare` returns all-INFEASIBLE and the method
        drops out of the sweep entirely. Measured on the simulation omega grid, the
        oracle IV budget (EPS_TOL) was below the floor at 5 of 12 steps.

        The trigger is `budget^2 < floor`, i.e. actual infeasibility -- NOT
        `budget^2 < FLOOR_GUARD_R * floor`. Those differ: a budget in
        `[sqrt(floor), sqrt(r*floor))` is feasible and doing its job, and an
        unconditional lower bound would loosen it for nothing. Measured: on
        optical at n=200 the INV floor is 0.0189 against an oracle eps of 0.2477,
        already feasible, and the unconditional form moved it to 0.4123.

        Once a budget IS infeasible the oracle value carries no information about
        where to put it, so it goes to `sqrt(FLOOR_GUARD_R * floor)` -- far enough
        off the knife edge to cover (`configs.py` has the calibration).

        The IV floor is the T CONSTRAINT'S OWN, measured with the translation
        amounts alone, as if the observed instrument was never there; the Z radius
        is never guarded (SS2.6). A T and a Z constraint can still be jointly
        infeasible with both floors cleared, and that is left to read INFEASIBLE.
        `declared` (a non-empty `iv:`): the floor is measured and logged beside the
        budget, and the budget is NEVER raised -- a declared budget that turns out
        infeasible is information, not something to inflate away (decision 8).

        No `data` means no measurement and no guard: that is the epsilon sweep's
        per-step call, where the budget is an ASSUMPTION the figure exists to test
        (decision 16), exactly as the swept epsilon is.
        """
        if data is None:
            return budget
        design = getattr(data, "X" if kind == "inv" else "GX", None)
        # 'iv' measures the T constraint's own geometry: the translation amounts alone
        extra = {"GX": getattr(data, "GX", None)} if kind == "inv" else {"Z": getattr(data, "G", None)}
        if design is None or next(iter(extra.values())) is None:
            return budget
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
                **extra,
            )
        except Exception as error:  # never let a diagnostic break a run
            logger.warning(f"{label}: constraint floor unavailable ({error}); budget left at the oracle value.")
            return budget

        if declared:
            side = "above" if budget**2 >= floor else "BELOW"
            logger.info(
                f"{label}: declared path, r_T {budget:.6g} (r_T^2 {budget**2:.4g}) is {side} the T "
                f"constraint's own floor {floor:.4g}; left as declared, never raised. The observed "
                "instrument has its own constraint at r_Z and its own budget."
            )
            return budget

        if budget**2 >= floor:  # feasible: leave it exactly as it was
            return budget

        guarded = float(np.sqrt(FLOOR_GUARD_R * max(floor, 0.0)))
        logger.info(
            f"{label}: oracle {budget:.6g} is INFEASIBLE (budget^2 {budget**2:.4g} "
            f"< floor {floor:.4g}); raising to {guarded:.6g} = sqrt({FLOOR_GUARD_R} "
            f"* floor). Every query would have come back INFEASIBLE."
        )
        return guarded

    def fit_epsilon_iv(self, experiment_index: int, step_index: int = 0, data=None, ratio: float = 1.0) -> float | None:
        """The ASSUMED T-as-IV budget r_T: `ratio` times the oracle T piece
        `eps_iv_star`, plus EPS_TOL. The observed instrument has its own constraint
        and its own budget (`fit_epsilon_iv_z`), so nothing is pooled here.

        `ratio` is 1 at FIT, where `data` is present and the floor guard applies: a
        fitted model must not be born infeasible. The epsilon sweep passes its grid
        ratio at PREDICT, with no data, and the budget is then raw -- exactly what
        the code does for the swept epsilon (`fit_epsilon` guards,
        `EpsilonRatioStrategy` passes `r eps* + EPS_TOL` unguarded). Guarding per
        step would pin the budget at sqrt(FLOOR_GUARD_R * floor) wherever the ratio
        is small, i.e. at a constant, and the sweep would show nothing. Where the
        fit-time guard DOES fire, the r = 1 column is the raw budget and not the
        fitted one, as it already is for epsilon.
        """
        budget = getattr(self.get_oracle(experiment_index), "eps_iv_star", None)
        if budget is None or not np.isfinite(budget):
            return None
        return self._floor_guard(
            float(ratio) * float(budget) + EPS_TOL,
            data,
            "iv",
            experiment_index,
            "epsilon_iv",
            declared=self.declared_iv,
        )

    def fit_epsilon_iv_z(self, experiment_index: int, data=None) -> float:
        """The observed instrument's own budget, one number for every Z constraint
        (SS2.6): 0.0 under an empty instrument (inert) and on the declared path (the
        radius is then exactly r_Z = s sqrt(gamma_z)); on the oracle path the measured
        piece off the knife edge, `eps_iv_z_star + EPS_TOL`, never floor-guarded."""
        Z = getattr(data, "Z", None)
        if self.declared_iv or Z is None or np.shape(Z)[1] == 0:
            return 0.0
        z_piece = getattr(self.get_oracle(experiment_index), "eps_iv_z_star", None)
        if z_piece is None or not np.isfinite(z_piece):
            return 0.0
        return float(z_piece) + EPS_TOL

    def method_kwargs(self, experiment_index: int) -> dict[str, Any]:
        """Extra builder kwargs. Override when methods need per-experiment state
        the budgets do not carry (e.g. prefit outcome models)."""
        return {}

    def build_models(self, experiment_index: int, step_index: int, data) -> dict[str, Any]:
        """Fresh, fitted models at this experiment's budgets."""
        gamma = self.fit_gamma(experiment_index)
        epsilon = self.fit_epsilon(experiment_index, step_index, data)
        builders = (
            self.method_factory(
                gamma=gamma,
                epsilon=epsilon,
                epsilon_iv=self.fit_epsilon_iv(experiment_index, step_index, data),
                epsilon_iv_z=self.fit_epsilon_iv_z(experiment_index, data),
                rho=self.fit_rho(experiment_index, data),
                **self.method_kwargs(experiment_index),
            )
            if self.method_factory
            else self.methods
        )

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
        self._sweep_cache = {}  # (param) -> (x, results, statuses), memo per param
        self._sweep_vlines = {}  # (param) -> measured reference x positions
        self._sweep_axis = {}  # (param) -> factors behind a measured x, or None
        self._sweep_xlabel = {}  # (param) -> the runner's label for the plotted x

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
        epsilon_iv: float | None = None,
        n_jobs: int | None = None,
        rho: float = 1.0,
        epsilon_iv_z: float = 0.0,
    ) -> dict[str, Any]:
        """Build methods at explicit budgets (per-experiment ParamPolicy); `rho`
        is the step's information-loss factor for the DA+ balls, `epsilon_iv` the
        DA+ methods' T-side IV term and `epsilon_iv_z` the non-DA +IV methods' own."""
        pass

    @property
    def methods(self):
        """Build methods for this experiment."""
        return self.registry.build_methods(self.kwargs["methods"])

    def run(self, plan):
        """Run the experiment types the `experiment:` block asked for."""
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
                    # the global toggle of SS10.1; absent from the shipped yaml
                    normalize=self.kwargs.get("normalize", False),
                )

    def _run_perf(self, perf_spec):
        """Two epsilon sweeps that time the solves and cross-check the backends, on the
        robustness sweep's own grid, data and models (`get_sweep_runner_cls("epsilon")`):
        one experiment, serial, never the cached sweep record."""
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
                # the wall clock is one median line per method, the seed var a
                # mean over queries with the bootstrap band over them; neither
                # is clipped (the slowest method is the result) nor promoted to
                # log (D(eps) spans decades near zero)
                bootstrapped=(metric == "seed_var"),
                clip_y=False,
                promote_y=False,
                failures=record.failures if metric == "seed_var" else None,
            )

    def _run_query_sweep(self):
        """Query sweep + panel. The panel is a query-space view, so it is
        always built here and never for param sweeps."""
        from src.experiments.utils import PanelBuilder

        # Create runner ONCE - used for both panel and radial sweep
        runner = self.get_query_runner_cls()(methods=self.methods, **{**self._get_clean_kwargs(), "n_experiments": 1})

        if self.build_panel:
            # Optical uses augmented geometry, simulation uses raw
            panel_builder = PanelBuilder(runner, self.name, "optical" in self.name)
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

        create_query_sweep_plot(angles, results, **ANNOTATE_SWEEP_PLOT["pc12"], experiment=self.name)
