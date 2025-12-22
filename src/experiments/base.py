"""
src/experiments/base.py - Refactored base classes with reduced duplication
"""
import enlighten
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Callable, Optional, Any, Tuple
from src.methods.abstract import pointEstimator as Regressor
from src.experiments.utils import fit_model, set_seed, approximation_error

ModelBuilder = Callable[[], Regressor]
MANAGER = enlighten.get_manager()


class BaseExperimentRunner(ABC):
    """Base class for all experiment runners"""
    def __init__(
            self,
            seed: int,
            n_samples: int,
            n_experiments: int,
            sweep_samples: int,
            methods: Dict[str, ModelBuilder],
            hyperparameters: Optional[Dict[str, Any]] = None
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
    def run(self, desc: str): pass


class DataSetupMixin:
    """Mixin for common data setup patterns"""
    def setup_experiment_data(self, sem, da, transform_fn=None):
        """
        Generic data setup for experiments.
        
        Args:
            sem: Structural equation model instance
            da: Data augmenter instance
            transform_fn: Optional transformation (e.g., PolynomialFeatures)
        """
        self.sem = sem
        self.da = da
        
        # Generate raw data
        self.X_raw, self.y = self.sem(N=self.n_samples)
        self.GX_raw, self.G_raw = self.da(self.X_raw)
        
        # Apply transformation if provided
        if transform_fn:
            self.X = transform_fn(self.X_raw)
            self.GX = transform_fn(self.GX_raw)
        else:
            self.X = self.X_raw
            self.GX = self.GX_raw
        
        return {
            'sem': self.sem, 
            'da': self.da, 
            'X': self.X, 
            'y': self.y, 
            'GX': self.GX, 
            'G': self.G_raw
        }


class QuerySweepRunner(DataSetupMixin, BaseExperimentRunner):
    """Runner for query sweep experiments"""
    
    def run(self, desc: str = "Query Sweep"):
        context = self.setup_data()
        queries = self.get_sweep_values()
        
        results = {}
        pbar = MANAGER.counter(total=len(self.methods), desc=desc, unit='methods')
        
        for name, builder in self.methods.items():
            if name == 'ATE':
                preds = queries @ context['sem'].solution
            else:
                model = builder()
                fit_model(
                    model, name, 
                    context['X'], context['y'], context['GX'], 
                    G=context.get('G'),
                    hyperparameters=self.hyperparameters, 
                    da=context['da']
                )
                preds = model.predict(queries)

            if 'PI' in name:
                results[name] = preds[:, np.newaxis, :]
            else:
                results[name] = preds.reshape(len(queries), 1)
            
            pbar.update()
        pbar.close()
        return queries, results

    @abstractmethod
    def get_sweep_values(self): 
        """Override to define sweep geometry"""
        pass

    @abstractmethod
    def setup_data(self) -> Dict[str, Any]: 
        """Override to define data setup"""
        pass


class ParamSweepRunner(DataSetupMixin, BaseExperimentRunner):
    """Runner for parameter sweep experiments"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.setup_sems_and_das()

    def run(self, desc: str = "Param Sweep"):
        param_values = self.get_param_range()
        results = {name: np.zeros((self.sweep_samples, self.n_experiments)) 
                   for name in self.methods}
        
        pbar_exp = MANAGER.counter(total=self.sweep_samples, desc=desc, unit='params')

        for i, param in enumerate(param_values):
            for j in range(self.n_experiments):
                X, y, GX, G, X_test, estimand = self.generate_data(
                    experiment_index=j, param=param
                )
                
                for name, builder in self.methods.items():
                    if name == 'ATE':
                        estimate = estimand 
                    else:
                        model = builder()
                        fit_model(
                            model, name, X, y, GX, G=G, 
                            hyperparameters=self.hyperparameters, 
                            da=self.get_da(j)
                        )
                        
                        kwargs = self.get_predict_kwargs(param)
                        estimate = model.predict(X_test, **kwargs)

                    results[name][i, j] = approximation_error(estimand, estimate)
            
            pbar_exp.update()
        pbar_exp.close()
        return param_values, results

    @abstractmethod
    def setup_sems_and_das(self):
        """Initialize SEMs and DAs for all experiments"""
        pass
    
    @abstractmethod
    def get_da(self, experiment_index): 
        """Get DA for specific experiment"""
        pass
    
    def get_predict_kwargs(self, param): 
        """Override if predict needs param-specific kwargs"""
        return {}
    
    @abstractmethod
    def get_param_range(self): 
        """Define parameter sweep range"""
        pass
    
    @abstractmethod
    def generate_data(self, experiment_index, param): 
        """Generate data for specific experiment and parameter"""
        pass