import enlighten
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Callable, Optional, Any
from src.methods.abstract import pointEstimator as Regressor
from src.experiments.utils import fit_model, set_seed, approximation_error

ModelBuilder = Callable[[], Regressor]
MANAGER = enlighten.get_manager()

class BaseExperimentRunner(ABC):
    def __init__(
            self,
            seed: int,
            n_samples: int,
            n_experiments: int,
            sweep_samples: int,
            methods: Dict[str, ModelBuilder],
            hyperparameters: Optional[Dict[str, Any]] = None
    ):
        # 1. SET SEED IMMEDIATELY
        # This ensures SEMs created in subclass __init__ are deterministic
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

class QuerySweepRunner(BaseExperimentRunner):
    def run(self, desc: str = "Query Sweep"):
        # 1. Setup Data
        context = self.setup_data()
        
        # 2. Get Queries
        queries = self.get_sweep_values() 

        results = {}
        pbar = MANAGER.counter(total=len(self.methods), desc=desc, unit='methods')
        
        for name, builder in self.methods.items():
            if name == 'ATE':
                preds = (queries @ context['sem'].solution)
            else:
                model = builder()
                fit_model(model, name, context['X'], context['y'], context['GX'], G=context.get('G'),
                          hyperparameters=self.hyperparameters, da=context['da'])
                preds = model.predict(queries)

            if 'PI' in name:
                results[name] = preds[:, np.newaxis, :]
            else:
                results[name] = preds.reshape(len(queries), 1) # Use actual query len
            
            pbar.update()
        pbar.close()
        return queries, results

    @abstractmethod
    def get_sweep_values(self): pass

    @abstractmethod
    def setup_data(self) -> Dict[str, Any]: pass

class ParamSweepRunner(BaseExperimentRunner):
    def run(self, desc: str = "Param Sweep"):
        # Param sweep re-sets seed inside the loop usually to ensure consistency across params,
        # but setting it in init is a good baseline.
        
        param_values = self.get_param_range()
        results = {name: np.zeros((self.sweep_samples, self.n_experiments)) for name in self.methods}
        
        self.setup_runner() 

        pbar_exp = MANAGER.counter(total=self.sweep_samples, desc=desc, unit='params')

        for i, param in enumerate(param_values):
            for j in range(self.n_experiments):
                X, y, GX, G, X_test, estimand = self.generate_data(experiment_index=j, param=param)
                
                for name, builder in self.methods.items():
                    if name == 'ATE':
                        estimate = estimand 
                    else:
                        model = builder()
                        fit_model(model, name, X, y, GX, G=G, 
                                  hyperparameters=self.hyperparameters, da=self.get_da(j))
                        
                        kwargs = self.get_predict_kwargs(param)
                        estimate = model.predict(X_test, **kwargs)

                    results[name][i, j] = approximation_error(estimand, estimate)
            
            pbar_exp.update()
        pbar_exp.close()
        return param_values, results

    def setup_runner(self): pass
    def get_da(self, experiment_index): return None
    def get_predict_kwargs(self, param): return {}
    @abstractmethod
    def get_param_range(self): pass
    @abstractmethod
    def generate_data(self, experiment_index, param): pass