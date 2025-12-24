"""
Base classes for experiment orchestration with unified runner logic.
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
                    fit_model(
                        model, name,
                        context.X, context.y, context.GX,
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
    
    def run(self, desc: str = "Param Sweep") -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Run parameter sweep computing metrics.
        
        Returns:
            Tuple of (parameter_values, metric_results_dict)
        """
        param_values = self.get_param_range()
        results = {
            name: np.zeros((self.sweep_samples, self.n_experiments))
            for name in self.methods
        }
        
        with MANAGER.counter(total=self.sweep_samples, desc=desc, unit='params') as pbar:
            for i, param in enumerate(param_values):
                for j in range(self.n_experiments):
                    X, y, GX, G, X_test, estimand = self.generate_data(j, param)
                    
                    for name, builder in self.methods.items():
                        if name == 'ATE':
                            estimate = estimand
                        else:
                            model = builder()
                            fit_model(
                                model, name, X, y, GX,
                                G=G,
                                hyperparameters=self.hyperparameters,
                                da=self.get_da(j)
                            )
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
    def generate_data(self, experiment_index: int, param) -> Tuple:
        """
        Generate data for one experiment at one parameter value.
        
        Returns:
            Tuple of (X, y, GX, G, X_test, estimand)
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
            # Kappa sweeps need all methods (including ATE), others use gamma-dependent only
            methods = self.methods if param_name == 'kappa' else self.gamma_only_methods
            
            runner = RunnerCls(methods=methods, **clean_kwargs)
            param_values, metric_results = runner.run(f"{param_name.title()} Sweep")
            
            # Save results
            save(param_values, f'{param_name}_values', self.name, 'pkl')
            save(metric_results, f'{param_name}_{self.metric}', self.name, 'pkl')
            
            # Plot
            plot_config = ANNOTATE_POPULATION_PLOT.get(param_name, {}).copy()
            plot_config['ylabel'] = self.ylabel
            create_param_sweep_plot(param_values, metric_results, **plot_config)
    
    def _run_query_sweep(self, plot_panel: bool, panel_only: bool):
        """Execute query sweep and generate visualizations."""
        from src.experiments.panel_builder import PanelBuilder
        
        RunnerCls = self.get_query_runner_cls()
        
        # Prepare kwargs
        run_kwargs = self._get_clean_kwargs()
        run_kwargs['n_experiments'] = 1
        
        runner = RunnerCls(methods=self.methods, **run_kwargs)
        
        # Generate panel plot if requested
        if plot_panel or panel_only:
            # Optical uses augmented geometry, simulation uses raw
            use_augmented_geometry = 'optical' in self.name
            PanelBuilder(runner, self.name, use_augmented_geometry).build(
                self.kwargs['sweep_samples']
            )
            if panel_only:
                return
        
        # Generate standard radial sweep plot
        _, results = runner.run("Radial Sweep")
        angles = np.linspace(0, 2*np.pi, self.kwargs['sweep_samples'], endpoint=False)
        
        save(angles, 'treatment_values', self.name, 'pkl')
        save(results, 'outcome_values', self.name, 'pkl')
        
        create_query_sweep_plot(
            angles, results,
            **ANNOTATE_SWEEP_PLOT['pc12'],
            experiment=self.name
        )