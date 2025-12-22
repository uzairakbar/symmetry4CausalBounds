"""
src/experiments/base.py - Core logic and Orchestration
"""
import enlighten
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Callable, Optional, Any, List, Tuple, Type
from src.methods.abstract import pointEstimator as Regressor
import src.experiments.utils as experiment_utils
from src.experiments.utils import (
    fit_model, set_seed, save, param_sweep_plot, query_sweep_plot
)
from src.experiments.configs import METRIC_CONFIGS, ANNOTATE_POPULATION_PLOT, ANNOTATE_SWEEP_PLOT
from src.experiments.panels import PanelBuilder

ModelBuilder = Callable[[], Regressor]
MANAGER = enlighten.get_manager()


class BaseExperimentRunner(ABC):
    """Base class for specific sweep logic (Data generation & Model fitting)"""
    def __init__(
            self,
            seed: int,
            n_samples: int,
            n_experiments: int,
            sweep_samples: int,
            methods: Dict[str, ModelBuilder],
            hyperparameters: Optional[Dict[str, Any]] = None,
            **kwargs  # Absorb extra config args (sweep_mode, plot_panel, etc.)
    ):
        if seed >= 0: set_seed(seed)
        self.seed = seed
        self.n_samples = n_samples
        self.n_experiments = n_experiments
        self.sweep_samples = sweep_samples
        self.methods = methods
        self.hyperparameters = hyperparameters

    @abstractmethod
    def run(self, desc: str): pass


class DataSetupMixin:
    """Mixin for common data setup patterns"""
    def setup_experiment_data(self, sem, da, transform_fn=None):
        self.sem = sem; self.da = da
        self.X_raw, self.y = self.sem(N=self.n_samples)
        self.GX_raw, self.G_raw = self.da(self.X_raw)
        
        if transform_fn:
            self.X, self.GX = transform_fn(self.X_raw), transform_fn(self.GX_raw)
        else:
            self.X, self.GX = self.X_raw, self.GX_raw
        
        return {'sem': self.sem, 'da': self.da, 'X': self.X, 'y': self.y, 'GX': self.GX, 'G': self.G_raw}


class QuerySweepRunner(DataSetupMixin, BaseExperimentRunner):
    """Runner for query sweep experiments (Visualization)"""
    def run(self, desc: str = "Query Sweep"):
        context = self.setup_data()
        queries = self.get_sweep_values()
        results = {}
        
        with MANAGER.counter(total=len(self.methods), desc=desc, unit='methods') as pbar:
            for name, builder in self.methods.items():
                if name == 'ATE':
                    preds = queries @ context['sem'].solution
                else:
                    model = builder()
                    fit_model(model, name, context['X'], context['y'], context['GX'], 
                              G=context.get('G'), hyperparameters=self.hyperparameters, da=context['da'])
                    preds = model.predict(queries)

                results[name] = preds[:, np.newaxis, :] if 'PI' in name else preds.reshape(len(queries), 1)
                pbar.update()
        return queries, results

    @abstractmethod
    def get_sweep_values(self): pass
    @abstractmethod
    def setup_data(self) -> Dict[str, Any]: pass


class ParamSweepRunner(DataSetupMixin, BaseExperimentRunner):
    """Runner for parameter sweep experiments (Metrics)"""
    def __init__(self, metric: str = 'approximation_error', **kwargs):
        super().__init__(**kwargs)
        self.metric_name = metric
        try:
            self.metric_fn = getattr(experiment_utils, metric)
        except AttributeError:
            raise ValueError(f"Metric '{metric}' not found in src.experiments.utils")
        self.setup_sems_and_das()

    def run(self, desc: str = "Param Sweep"):
        param_values = self.get_param_range()
        results = {name: np.zeros((self.sweep_samples, self.n_experiments)) for name in self.methods}
        
        with MANAGER.counter(total=self.sweep_samples, desc=desc, unit='params') as pbar:
            for i, param in enumerate(param_values):
                for j in range(self.n_experiments):
                    X, y, GX, G, X_test, estimand = self.generate_data(j, param)
                    for name, builder in self.methods.items():
                        if name == 'ATE': estimate = estimand 
                        else:
                            model = builder()
                            fit_model(model, name, X, y, GX, G=G, 
                                      hyperparameters=self.hyperparameters, da=self.get_da(j))
                            estimate = model.predict(X_test, **self.get_predict_kwargs(param))
                        results[name][i, j] = self.metric_fn(estimand, estimate)
                pbar.update()
        return param_values, results

    @abstractmethod
    def setup_sems_and_das(self): pass
    @abstractmethod
    def get_da(self, experiment_index): pass
    def get_predict_kwargs(self, param): return {}
    @abstractmethod
    def get_param_range(self): pass
    @abstractmethod
    def generate_data(self, experiment_index, param): pass


# ============================================================================
# ORCHESTRATOR
# ============================================================================

class ExperimentOrchestrator(ABC):
    """Handles the execution workflow: Setup -> Run -> Save -> Plot"""
    
    def __init__(self, experiment_name: str, method_registry, **kwargs):
        self.name = experiment_name
        self.registry = method_registry
        self.kwargs = kwargs
        self.metric = kwargs.get('metric', 'approximation_error')
        self.ylabel = METRIC_CONFIGS[self.metric]['ylabel']

    @abstractmethod
    def get_query_runner_cls(self) -> Type[QuerySweepRunner]: pass
    
    @abstractmethod
    def get_param_sweeps(self) -> List[Tuple[Type[ParamSweepRunner], str]]: 
        """Returns list of (RunnerClass, param_name_for_file_saving)"""
        pass
    
    @property
    def methods(self):
        return self.registry.build_methods(self.kwargs['methods'])

    @property
    def gamma_only_methods(self):
        return self.registry.filter_gamma_methods(self.methods)

    def run(self, sweep_mode: str, plot_panel: bool = False, panel_only: bool = False):
        if sweep_mode == 'param':
            self._run_param_sweeps()
        else:
            self._run_query_sweep(plot_panel, panel_only)

    def _get_clean_kwargs(self):
        """Removes arguments that cause collision with explicit args passed to Runners"""
        clean_kwargs = self.kwargs.copy()
        clean_kwargs.pop('methods', None)
        return clean_kwargs

    def _run_param_sweeps(self):
        clean_kwargs = self._get_clean_kwargs()
        
        for RunnerCls, param_name in self.get_param_sweeps():
            # Filter methods: Kappa usually needs all (inc ATE), others usually just gamma-dependent
            methods = self.methods if param_name == 'kappa' else self.gamma_only_methods
            
            runner = RunnerCls(methods=methods, **clean_kwargs)
            x, res = runner.run(f"{param_name.title()} Sweep")
            
            # Save & Plot
            save(x, f'{param_name}_values', self.name, 'pkl')
            save(res, f'{param_name}_{self.metric}', self.name, 'pkl')
            
            plot_cfg = ANNOTATE_POPULATION_PLOT.get(param_name, {}).copy()
            plot_cfg['ylabel'] = self.ylabel
            param_sweep_plot(x, res, **plot_cfg)

    def _run_query_sweep(self, plot_panel, panel_only):
        RunnerCls = self.get_query_runner_cls()
        
        # Prepare kwargs
        run_kwargs = self._get_clean_kwargs()
        run_kwargs['n_experiments'] = 1
        
        runner = RunnerCls(methods=self.methods, **run_kwargs)

        if plot_panel or panel_only:
            # Determine if we use augmented geometry (Optical) or Raw (Simulation)
            use_aug = 'optical' in self.name
            PanelBuilder(runner, self.name, use_augmented_geometry=use_aug).build(self.kwargs['sweep_samples'])
            if panel_only: return

        # Standard Plot
        _, results = runner.run("Radial Sweep")
        angles = np.linspace(0, 2*np.pi, self.kwargs['sweep_samples'], endpoint=False)
        
        save(angles, 'treatment_values', self.name, 'pkl')
        save(results, 'outcome_values', self.name, 'pkl')
        query_sweep_plot(angles, results, **ANNOTATE_SWEEP_PLOT['pc12'], experiment=self.name)