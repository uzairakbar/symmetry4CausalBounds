"""
Base classes for experiment orchestration with unified runner logic.
Updated to use simplified fit_model signature and OPTIMIZED LOOP ORDER.
"""
import time
import enlighten
from functools import partial
import numpy as np
from loguru import logger
from abc import ABC, abstractmethod
from typing import Dict, Callable, Optional, Any, Tuple, Type
from dataclasses import dataclass

from src.methods.abstract import pointEstimator as Regressor
from src.experiments.utils import fit_model, set_seed, save
from src.experiments.utils.metrics import evaluate_queries, STATUS_CATEGORIES
from src.experiments.utils.plotting import (
    create_sweep_plot, create_scatter_plot, create_perf_plot,
    create_query_sweep_plot,
)
from src.experiments.configs import (
    PARAM_SPECS, METRIC_SPECS, EPS_TOL, ANNOTATE_SWEEP_PLOT,
    SCATTER_SE_CROSSHAIRS,
)
from src.experiments.utils.constants import (
    SUBDIR_QUERY, SUBDIR_SWEEP, SUBDIR_PERF,
)

# QueryEval scalar fields recorded at every (method, step, experiment)
METRIC_FIELDS: Tuple[str, ...] = (
    'approximation_error', 'worst_error', 'interval_width', 'coverage', 'wall_clock',
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
    X_raw: Optional[np.ndarray] = None
    GX_raw: Optional[np.ndarray] = None
    oracle: Optional[Any] = None    # OracleParameters; unused by default


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
    X_base: Optional[np.ndarray] = None
    y_base: Optional[np.ndarray] = None

    def __iter__(self):
        return iter((self.X, self.y, self.GX, self.G, self.X_test, self.estimand))

    @classmethod
    def coerce(cls, data) -> 'SweepData':
        return data if isinstance(data, cls) else cls(*data)

    @property
    def fit_arrays(self) -> Dict[str, Any]:
        return dict(X=self.X, y=self.y, GX=self.GX, G=self.G,
                    X_base=self.X_base, y_base=self.y_base)


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
        methods: Dict[str, ModelBuilder],
        hyperparameters: Optional[Dict[str, Any]] = None,
        calibrate: bool = False,
        pad: bool = False,
        clipy: bool = True,
        **kwargs
    ):
        if seed >= 0:
            set_seed(seed)

        self.seed = seed
        self.n_samples = n_samples
        self.n_experiments = n_experiments
        self.sweep_samples = sweep_samples
        self.methods = methods
        self.hyperparameters = hyperparameters
        # toggles: needed here only to report oracle parameters in matching units
        self.calibrate = calibrate
        self.pad = pad
        self.clipy = clipy
    
    @abstractmethod
    def run(self, desc: str):
        """Run the experiment."""
        pass


# =============================================================================
# QUERY SWEEP RUNNER (for visualization)
# =============================================================================

class QuerySweepRunner(BaseExperimentRunner):
    """Runner for query sweep experiments (radial/PC sweeps for visualization)."""
    
    def run(self, desc: str = "Query Sweep") -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Run query sweep across treatment space.
        
        Returns:
            Tuple of (query_points, predictions_dict)
        """
        context = self.setup_data()
        queries = self.get_sweep_values()
        results = {}
        
        with MANAGER.counter(total=len(self.methods), desc=desc, unit='methods') as pbar:
            for name, builder in self.methods.items():
                if name == 'ATE':
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
                        hyperparameters=self.hyperparameters,
                        da=context.da
                    )
                    
                    predictions = model.predict(queries)
                
                # Reshape for consistent output format
                if 'PI' in name:
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

    param_key: str = None       # indexes PARAM_SPECS

    def __init__(
        self,
        method_factory: Optional[Callable] = None,
        experiment_name: str = 'simulation',
        param_grid_override: Optional[Any] = None,
        **kwargs
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
    def vlines(self) -> Tuple[float, ...]:
        """Reference x positions; strategies may add measured thresholds."""
        return self.spec.vlines

    # ------------------------------------------------------ per-experiment policy

    def fit_gamma(self, experiment_index: int) -> float:
        """Budget baked into the built methods (PLAN 7); gamma* by default."""
        return self._finite(self.get_oracle(experiment_index).gamma_star,
                            self.default_gamma, 'gamma*')

    def fit_epsilon(self, experiment_index: int, step_index: int = 0) -> float:
        """
        Assumed invariance error: oracle eps* = ||W|| over the full augmentation,
        just large enough to admit h_* in PI_INV, off the knife edge.
        """
        return self._finite(self.get_oracle(experiment_index).epsilon_star,
                            self.default_epsilon, 'eps*') + EPS_TOL

    def _finite(self, value, fallback, name: str) -> float:
        if value is None or not np.isfinite(value):
            logger.warning(f'{name} not computable; falling back to yaml default.')
            return float(fallback)
        return float(value)

    def fit_epsilon_iv(self, experiment_index: int) -> Optional[float]:
        """IV budget for this experiment: oracle eps_iv_star, off the knife edge."""
        eps_iv_star = getattr(self.get_oracle(experiment_index), 'eps_iv_star', None)
        if eps_iv_star is None or not np.isfinite(eps_iv_star):
            return None
        return float(eps_iv_star) + EPS_TOL

    def method_kwargs(self, experiment_index: int) -> Dict[str, Any]:
        """Extra builder kwargs. Override when methods need per-experiment state
        the budgets do not carry (e.g. prefit outcome models)."""
        return {}

    def build_models(self, experiment_index: int, step_index: int, data) -> Dict[str, Any]:
        """Fresh, fitted models at this experiment's budgets."""
        gamma = self.fit_gamma(experiment_index)
        epsilon = self.fit_epsilon(experiment_index, step_index)
        builders = (
            self.method_factory(
                gamma=gamma, epsilon=epsilon,
                epsilon_iv=self.fit_epsilon_iv(experiment_index),
                **self.method_kwargs(experiment_index),
            )
            if self.method_factory else self.methods
        )

        models = {}
        for name in self.methods:
            if name == 'ATE':
                continue
            model = builders[name]()
            fit_model(
                model=model,
                method_name=name,
                hyperparameters=self.hyperparameters,
                da=self.get_da(experiment_index),
                **data.fit_arrays
            )
            models[name] = model
        return models

    # ------------------------------------------------------------------ loop

    def run(self, desc: str = "Param Sweep") -> Tuple[np.ndarray, Dict, Dict]:
        """Sweep the parameter, recording every metric at every step."""
        param_values = self.get_param_range()
        n_steps = len(param_values)
        shape = (n_steps, self.n_experiments)

        results = {
            name: {metric: np.full(shape, np.nan) for metric in METRIC_FIELDS}
            for name in self.methods
        }
        statuses = {
            name: np.zeros(shape + (len(STATUS_CATEGORIES),), dtype=int)
            for name in self.methods
        }

        with MANAGER.counter(total=self.n_experiments * n_steps, desc=desc, unit='runs') as pbar:
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
                        if name == 'ATE':
                            estimate, query_status, elapsed = data.estimand, None, 0.0
                        else:
                            model = models[name]
                            # query evaluation ONLY -- DA transform lives in
                            # generate_data and fitting in build_models, both
                            # outside this timer. Keep it that way.
                            start = time.perf_counter()
                            estimate = model.predict(
                                data.X_test, **self.get_predict_kwargs(param, j)
                            )
                            elapsed = time.perf_counter() - start
                            query_status = getattr(model, 'query_status', None)

                        record = evaluate_queries(
                            data.estimand, estimate, query_status, elapsed
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

    def get_predict_kwargs(self, param, experiment_index: int) -> Dict[str, Any]:
        """Additional kwargs for model.predict(). Override to sweep a budget."""
        return {}
    
    @abstractmethod
    def generate_data(self, experiment_index: int, param) -> 'SweepData':
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
        self._sweep_cache = {}      # (param) -> (x, results, statuses), reused by scatter
        self._sweep_vlines = {}     # (param) -> measured reference x positions

    @abstractmethod
    def get_query_runner_cls(self) -> Type[QuerySweepRunner]:
        """Return the QuerySweepRunner class for this experiment."""
        pass

    @abstractmethod
    def get_sweep_runner_cls(self, param: str) -> Type[ParamSweepRunner]:
        """Return the configured strategy class for one sweep parameter."""
        pass

    @abstractmethod
    def build_methods(self, gamma: float, epsilon: float,
                      epsilon_iv: Optional[float] = None,
                      n_jobs: Optional[int] = None) -> Dict[str, Any]:
        """Build methods at explicit budgets (per-experiment ParamPolicy)."""
        pass

    @property
    def methods(self):
        """Build methods for this experiment."""
        return self.registry.build_methods(self.kwargs['methods'])

    def run(self, plan):
        """Run the experiment types the `experiment:` block asked for."""
        if plan.query:
            self._run_query_sweep()
        if plan.sweep:
            self._run_sweeps(plan.sweep)
        if plan.scatter:
            self._run_scatter(plan.scatter)
        if plan.perf:
            self._run_perf(plan.perf)

    def _get_clean_kwargs(self) -> Dict[str, Any]:
        """Remove arguments that cause collision with explicit runner args."""
        clean_kwargs = self.kwargs.copy()
        clean_kwargs.pop('methods', None)
        return clean_kwargs

    def sweep_record(self, param: str):
        """Run (or reuse) one param sweep. Cached so scatter shares the compute."""
        if param in self._sweep_cache:
            return self._sweep_cache[param]

        spec = PARAM_SPECS[param]
        methods = self.methods
        if not spec.include_ate:
            methods = {k: v for k, v in methods.items() if k != 'ATE'}

        runner = self.get_sweep_runner_cls(param)(
            methods=methods,
            method_factory=self.build_methods,
            **self._get_clean_kwargs()
        )
        record = runner.run(f'{param} sweep')
        self._sweep_cache[param] = record
        self._sweep_vlines[param] = runner.vlines
        return record

    def _run_sweeps(self, sweep_spec):
        """One figure per (param, metric); one pkl per param."""
        for param in sweep_spec.param:
            x_values, results, statuses = self.sweep_record(param)

            save(x_values, f'{param}_values', self.name, 'pkl', subdir=SUBDIR_SWEEP)
            save(results, f'{param}_results', self.name, 'pkl', subdir=SUBDIR_SWEEP)
            save(statuses, f'{param}_statuses', self.name, 'pkl', subdir=SUBDIR_SWEEP)

            for metric in sweep_spec.metric:
                metric_spec = METRIC_SPECS[metric]
                create_sweep_plot(
                    x_values,
                    {name: record[metric_spec.key]
                     for name, record in results.items()
                     if metric_spec.include_ate or name != 'ATE'},
                    experiment=self.name,
                    fname=f'{param}_{metric}',
                    xlabel=PARAM_SPECS[param].xlabel,
                    ylabel=metric_spec.ylabel,
                    xscale=PARAM_SPECS[param].xscale,
                    yscale=metric_spec.yscale,
                    vlines=self._sweep_vlines.get(param, PARAM_SPECS[param].vlines),
                )


    def _run_perf(self, perf_spec):
        """
        Per-method reliability + cost, at the dataset defaults.

        A degenerate 1-point m-sweep (m=1) gives exactly the default operating
        point, so this reuses the sweep machinery rather than a second path.
        """
        n_experiments = self.kwargs['n_experiments']

        # Always serial, never the cached sweep: n_jobs speeds up only the SOCP
        # methods, so a parallel record would compare harnesses, not methods
        # (measured: the PI/ERM wall_clock ratio moves 9.4x with the knob).
        # ATE is the truth, not an estimator: no cost, no failure modes.
        methods = {k: v for k, v in self.methods.items() if k != 'ATE'}
        runner = self.get_sweep_runner_cls('m')(
            methods=methods,
            # the runner kwarg alone does NOT reach the models: build_methods
            # reads the orchestrator's toggles. Force it on the factory too.
            method_factory=partial(self.build_methods, n_jobs=1),
            param_grid_override=[1],
            **{**self._get_clean_kwargs(), 'n_jobs': 1}
        )
        _, results, statuses = runner.run('perf')

        results = {k: v for k, v in results.items() if k != 'ATE'}
        overlay = list(perf_spec.metric)
        if 'seed_var' in overlay and n_experiments < 2:
            logger.warning(
                f'n_experiments={n_experiments}: seed SD is undefined, '
                'dropping the seed_var series.'
            )
            overlay.remove('seed_var')

        record = {}
        for name, metrics in results.items():
            counts = statuses[name][0]                  # first step = m=1
            total = max(int(counts.sum()), 1)
            per_experiment_width = metrics['interval_width'][0]
            record[name] = {
                'rates': counts.sum(axis=0) / total,    # 4-way split, sums to 1
                'wall_clock': float(np.nanmean(metrics['wall_clock'][0])),
                # SD across seeds of the per-experiment means (PLAN 3)
                'seed_var': float(np.nanstd(per_experiment_width)),
                'coverage_sd': float(np.nanstd(metrics['coverage'][0])),
                'midpoint_sd': float(np.nanstd(per_experiment_width) / 2.0),
                'n_experiments': n_experiments,
            }

        save(record, 'perf', self.name, 'pkl', subdir=SUBDIR_PERF)
        create_perf_plot(record, overlay_metrics=overlay, experiment=self.name)

    def _run_scatter(self, scatter_spec):
        """Trade-off scatters. Reuses the cached sweep records; no extra compute."""
        n_experiments = self.kwargs['n_experiments']
        crosshairs = SCATTER_SE_CROSSHAIRS and n_experiments >= 2
        if SCATTER_SE_CROSSHAIRS and not crosshairs:
            logger.warning(
                f'n_experiments={n_experiments}: SE is undefined, '
                'scatter crosshairs hidden.'
            )

        for param in scatter_spec.param:
            x_values, results, _ = self.sweep_record(param)

            for metric_x, metric_y in scatter_spec.metric:
                spec_x, spec_y = METRIC_SPECS[metric_x], METRIC_SPECS[metric_y]
                # a degenerate ATE on either axis pins a corner and squashes the rest
                keep = spec_x.include_ate and spec_y.include_ate
                create_scatter_plot(
                    {n: r for n, r in results.items() if keep or n != 'ATE'},
                    x_values,
                    metric_x=spec_x.key, metric_y=spec_y.key,
                    xlabel=spec_x.ylabel, ylabel=spec_y.ylabel,
                    param_label=PARAM_SPECS[param].xlabel,
                    experiment=self.name,
                    fname=f'scatter_{param}_{metric_x}_vs_{metric_y}',
                    crosshairs=crosshairs,
                )

    def _run_query_sweep(self):
        """Query sweep + panel. The panel is a query-space view, so it is
        always built here and never for param sweeps."""
        from src.experiments.utils import PanelBuilder

        # Create runner ONCE - used for both panel and radial sweep
        runner = self.get_query_runner_cls()(
            methods=self.methods,
            **{**self._get_clean_kwargs(), 'n_experiments': 1}
        )

        if self.build_panel:
            # Optical uses augmented geometry, simulation uses raw
            panel_builder = PanelBuilder(runner, self.name, 'optical' in self.name)
            panel_builder.build(self.kwargs['sweep_samples'])
            # Radial sweep plot reuses the panel's cached results
            results = panel_builder.get_radial_results()
        else:
            _, results = runner.run('Query Sweep')

        self._plot_query_sweep(runner, results)

    def _plot_query_sweep(self, runner, results):
        """Save + plot the query sweep. Panel experiments plot the radial angle;
        override when the queries are not a PC sweep."""
        angles = np.linspace(0, 2*np.pi, self.kwargs['sweep_samples'], endpoint=False)

        save(angles, 'treatment_values', self.name, 'pkl', subdir=SUBDIR_QUERY)
        save(results, 'outcome_values', self.name, 'pkl', subdir=SUBDIR_QUERY)

        create_query_sweep_plot(
            angles, results,
            **ANNOTATE_SWEEP_PLOT['pc12'],
            experiment=self.name
        )