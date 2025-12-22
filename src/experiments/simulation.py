"""
src/experiments/simulation.py
"""
from src.data_augmentors.simulation import NullSpaceTranslation as DA
from src.sem.simulation import LinearSimulationSEM as SEM, TREATMENT_DIMENSION
from src.experiments.base import QuerySweepRunner, ParamSweepRunner, ExperimentOrchestrator
from src.experiments.configs import MethodRegistry, SIMULATION_PARAMS, SWEEP_CONFIGS
from src.experiments.utils import radial_sweep_pcs

EXPERIMENT = 'simulation'

# --- Runners ---

class SimQuerySweep(QuerySweepRunner):
    def __init__(self, kernel_dim, **kwargs):
        super().__init__(**kwargs)
        self.kernel_dim = kernel_dim
        self.sem = SEM()
        self.da = DA(self.sem.W_XY, kernel_dim=self.kernel_dim)
        self.X, self.y = self.sem(N=self.n_samples, kappa=SIMULATION_PARAMS['kappa'])
        self.GX, self.G = self.da(self.X)
        self.X_raw, self.GX_raw = self.X, self.GX # Linear sim needs no transform

    def get_sweep_values(self): return radial_sweep_pcs(self.X, self.sweep_samples)
    def setup_data(self): return {'sem': self.sem, 'da': self.da, 'X': self.X, 'y': self.y, 'GX': self.GX, 'G': self.G}

class SimParamSweepBase(ParamSweepRunner):
    def __init__(self, kernel_dim, **kwargs):
        self.kernel_dim = kernel_dim
        super().__init__(**kwargs)
    
    def setup_sems_and_das(self):
        self.sems = [SEM() for _ in range(self.n_experiments)]
        self.das = [DA(sem.W_XY, kernel_dim=self.kernel_dim) for sem in self.sems]
    
    def get_da(self, idx): return self.das[idx]

class KappaSweep(SimParamSweepBase):
    def get_param_range(self): return SWEEP_CONFIGS['simulation']['kappa']['range_fn'](self.sweep_samples)
    def generate_data(self, idx, param):
        sem, da = self.sems[idx], self.das[idx]
        X, y = sem(N=self.n_samples, kappa=param)
        X_test, _ = sem(N=int(SIMULATION_PARAMS['test_frac'] * self.n_samples), intervention=True, kappa=param)
        GX, G = da(X, gamma=1.0)
        return X, y, GX, G, X_test, X_test @ sem.solution

class AlphaSweep(SimParamSweepBase):
    def get_param_range(self): return SWEEP_CONFIGS['simulation']['alpha']['range_fn'](self.sweep_samples)
    def generate_data(self, idx, param):
        sem, da = self.sems[idx], self.das[idx]
        X, y = sem(N=self.n_samples)
        X_test, _ = sem(N=int(SIMULATION_PARAMS['test_frac'] * self.n_samples), intervention=True)
        GX, G = da(X, gamma=param)
        return X, y, GX, G, X_test, X_test @ sem.solution

class GammaSweep(SimParamSweepBase):
    def get_param_range(self): return SWEEP_CONFIGS['simulation']['gamma']['range_fn'](self.sweep_samples)
    def get_predict_kwargs(self, param): return {'gamma': param, 'gamma0': SIMULATION_PARAMS['gamma0']}
    def generate_data(self, idx, param):
        sem, da = self.sems[idx], self.das[idx]
        X, y = sem(N=self.n_samples)
        X_test, _ = sem(N=int(SIMULATION_PARAMS['test_frac'] * self.n_samples), intervention=True)
        GX, G = da(X, gamma=1.0)
        return X, y, GX, G, X_test, X_test @ sem.solution

# --- Orchestration ---

class SimulationOrchestrator(ExperimentOrchestrator):
    def __init__(self, **kwargs):
        # Bind simulation specific params to MethodRegistry
        class SimRegistry(MethodRegistry):
            @staticmethod
            def build_methods(names):
                return MethodRegistry.build_methods(names, gamma=SIMULATION_PARAMS['gamma'], 
                                                    gamma0=SIMULATION_PARAMS['gamma0'], 
                                                    epsilon=SIMULATION_PARAMS['epsilon'])
        super().__init__(EXPERIMENT, SimRegistry(), **kwargs)

    def get_query_runner_cls(self): return SimQuerySweep
    def get_param_sweeps(self):
        return [
            (KappaSweep, 'kappa'),
            (GammaSweep, 'gamma'),
            (AlphaSweep, 'alpha')
        ]