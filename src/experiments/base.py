"""
Base classes for experiment orchestration with unified runner logic.
Updated to use simplified fit_model signature and OPTIMIZED LOOP ORDER.
"""
import enlighten
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Callable, Optional, Any, List, Tuple, Type
from dataclasses import dataclass

from src.methods.abstract import pointEstimator as Regressor
from src.experiments.utils import fit_model, set_seed, save
from src.experiments.utils.plotting import create_param_sweep_plot, create_query_sweep_plot
from src.experiments.configs import METRIC_CONFIGS, ANNOTATE_POPULATION_PLOT, ANNOTATE_SWEEP_PLOT

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
                    predictions = queries @ context.sem.solution
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
    """Runner for parameter sweep experiments (computing metrics across parameter ranges)."""
    
    def __init__(self, metric: str = 'approximation_error', **kwargs):
        super().__init__(**kwargs)
        self.metric_name = metric
        
        # Import metric function dynamically
        from src.experiments.utils import metrics as metrics_module
        try:
            self.metric_fn = getattr(metrics_module, metric)
        except AttributeError:
            raise ValueError(f"Metric '{metric}' not found in metrics module")
        
        self.setup_sems_and_das()

    @property
    def data_depends_on_param(self) -> bool:
        """
        Optimization flag:
        If False (e.g. Gamma sweep), we fit once and sweep params.
        If True (e.g. Kappa/Alpha sweep), we must regenerate data and refit every step.
        """
        return True

    def run(self, desc: str = "Param Sweep") -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Run parameter sweep computing metrics.
        OPTIMIZED: Inverted loop structure to maximize DPP cache reuse.
        """
        # FIX: Get actual params first to determine size
        param_values = self.get_param_range()
        n_steps = len(param_values)
        
        results = {
            name: np.zeros((n_steps, self.n_experiments))
            for name in self.methods
        }
        
        # Iterate Experiments Outer Loop (Allows reusing fitted models)
        with MANAGER.counter(total=self.n_experiments * n_steps, desc=desc, unit='runs') as pbar:
            for j in range(self.n_experiments):
                
                # OPTIMIZATION:
                # If data does NOT depend on the parameter (e.g., Gamma Sweep),
                # we generate data and fit the models ONCE per experiment.
                cached_models = {}
                cached_data = None
                
                if not self.data_depends_on_param:
                    # Generate data once using the first param value (dummy)
                    cached_data = SweepData.coerce(
                        self.generate_data(j, param_values[0])
                    )

                    # Fit all models once
                    for name, builder in self.methods.items():
                        if name == 'ATE': continue
                        model = builder()
                        fit_model(
                            model=model,
                            method_name=name,
                            hyperparameters=self.hyperparameters,
                            da=self.get_da(j),
                            **cached_data.fit_arrays
                        )
                        cached_models[name] = model

                # Iterate Parameters Inner Loop
                for i, param in enumerate(param_values):

                    if self.data_depends_on_param:
                        # Full regeneration for data-dependent sweeps
                        data = SweepData.coerce(self.generate_data(j, param))
                    else:
                        # Reuse data for Gamma sweeps
                        data = cached_data
                    X_test, estimand = data.X_test, data.estimand

                    for name, builder in self.methods.items():
                        if name == 'ATE':
                            estimate = estimand
                        else:
                            # Use cached model or fit new one
                            if self.data_depends_on_param:
                                model = builder()
                                fit_model(
                                    model=model,
                                    method_name=name,
                                    hyperparameters=self.hyperparameters,
                                    da=self.get_da(j),
                                    **data.fit_arrays
                                )
                            else:
                                model = cached_models[name]
                            
                            # Predict (DPP makes this fast if model is cached)
                            estimate = model.predict(X_test, **self.get_predict_kwargs(param))
                        
                        results[name][i, j] = self.metric_fn(estimand, estimate)
                    
                    pbar.update()
        
        return param_values, results
    
    @abstractmethod
    def setup_sems_and_das(self):
        """Setup SEMs and data augmentors for all experiments."""
        pass
    
    @abstractmethod
    def get_da(self, experiment_index: int):
        """Get data augmentor for specific experiment."""
        pass
    
    def get_predict_kwargs(self, param) -> Dict[str, Any]:
        """Get additional kwargs for model.predict(). Override if needed."""
        return {}
    
    @abstractmethod
    def get_param_range(self) -> np.ndarray:
        """Get parameter values to sweep over."""
        pass
    
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
    
    def __init__(self, experiment_name: str, method_registry, **kwargs):
        self.name = experiment_name
        self.registry = method_registry
        self.kwargs = kwargs
        self.metric = kwargs.get('metric', 'approximation_error')
        self.ylabel = METRIC_CONFIGS[self.metric]['ylabel']
    
    @abstractmethod
    def get_query_runner_cls(self) -> Type[QuerySweepRunner]:
        """Return the QuerySweepRunner class for this experiment."""
        pass
    
    @abstractmethod
    def get_param_sweeps(self) -> List[Tuple[Type[ParamSweepRunner], str]]:
        """
        Return list of parameter sweeps to run.
        
        Returns:
            List of (RunnerClass, parameter_name) tuples
        """
        pass
    
    @property
    def methods(self):
        """Build methods for this experiment."""
        return self.registry.build_methods(self.kwargs['methods'])
    
    @property
    def gamma_only_methods(self):
        """Filter gamma-dependent methods."""
        return self.registry.filter_gamma_methods(self.methods)
    
    def run(self, sweep_mode: str, plot_panel: bool = False, panel_only: bool = False):
        """
        Run experiments based on sweep mode.
        
        Args:
            sweep_mode: 'param' for parameter sweeps, 'query' for query sweeps
            plot_panel: Whether to generate panel plots
            panel_only: Only generate panel, skip standard plots
        """
        if sweep_mode == 'param':
            self._run_param_sweeps()
        else:
            self._run_query_sweep(plot_panel, panel_only)
    
    def _get_clean_kwargs(self) -> Dict[str, Any]:
        """Remove arguments that cause collision with explicit runner args."""
        clean_kwargs = self.kwargs.copy()
        clean_kwargs.pop('methods', None)
        return clean_kwargs
    
    def _run_param_sweeps(self):
        """Execute all parameter sweeps."""
        clean_kwargs = self._get_clean_kwargs()
        
        for RunnerCls, param_name in self.get_param_sweeps():
            # gamma* sweeps need all methods (including ATE), others use gamma-dependent only
            methods = self.methods if param_name == 'gamma_star' else self.gamma_only_methods
            
            runner = RunnerCls(methods=methods, **clean_kwargs)
            param_values, metric_results = runner.run(f"{param_name.title()} Sweep")
            
            # Save results
            save(param_values, f'{param_name}_values', self.name, 'pkl')
            save(metric_results, f'{param_name}_{self.metric}', self.name, 'pkl')
            
            # Plot
            plot_config = ANNOTATE_POPULATION_PLOT.get(param_name, {}).copy()
            plot_config['ylabel'] = self.ylabel
            create_param_sweep_plot(
                param_values, metric_results,
                experiment=self.name, fname=param_name, **plot_config
            )
    
    def _run_query_sweep(self, plot_panel: bool, panel_only: bool):
        """Execute query sweep and generate visualizations."""
        from src.experiments.utils import PanelBuilder
        
        RunnerCls = self.get_query_runner_cls()
        
        # Prepare kwargs
        run_kwargs = self._get_clean_kwargs()
        run_kwargs['n_experiments'] = 1
        
        # Create runner ONCE - used for both panel and radial sweep
        runner = RunnerCls(methods=self.methods, **run_kwargs)
        
        # Generate panel plot if requested
        panel_builder = None
        if plot_panel or panel_only:
            # Optical uses augmented geometry, simulation uses raw
            use_augmented_geometry = 'optical' in self.name
            panel_builder = PanelBuilder(runner, self.name, use_augmented_geometry)
            panel_builder.build(self.kwargs['sweep_samples'])
            
            if panel_only:
                return
        
        # Generate standard radial sweep plot using CACHED results from panel
        if panel_builder is not None:
            # Reuse results from panel builder
            results = panel_builder.get_radial_results()
        else:
            # No panel, run fresh
            _, results = runner.run("Radial Sweep")
        
        # Create angle values for plotting (x-axis)
        angles = np.linspace(0, 2*np.pi, self.kwargs['sweep_samples'], endpoint=False)
        
        save(angles, 'treatment_values', self.name, 'pkl')
        save(results, 'outcome_values', self.name, 'pkl')
        
        create_query_sweep_plot(
            angles, results,
            **ANNOTATE_SWEEP_PLOT['pc12'],
            experiment=self.name
        )