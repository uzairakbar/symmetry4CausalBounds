"""
Simulation experiment with unified runner implementation.
"""
import numpy as np
from sklearn.model_selection import train_test_split

from src.data_augmentors.simulation import NullSpaceTranslation as DA
from src.sem.simulation import LinearSimulationSEM as SEM
from src.experiments.base import (
    QuerySweepRunner, ParamSweepRunner, ExperimentOrchestrator, ExperimentDataContext
)
from src.experiments.configs import (
    MethodRegistry, SIMULATION_CONFIG, SWEEP_CONFIGS
)
from src.experiments.utils import radial_sweep_pcs

EXPERIMENT_NAME = 'simulation'


# =============================================================================
# QUERY SWEEP RUNNER
# =============================================================================

class SimulationQuerySweep(QuerySweepRunner):
    """Query sweep for simulation experiment."""
    
    def __init__(self, kernel_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.kernel_dim = kernel_dim
        
        # Setup data
        self.sem = SEM()
        self.da = DA(self.sem.W_XY, kernel_dim=self.kernel_dim)
        
        X, y = self.sem(N=self.n_samples, kappa=SIMULATION_CONFIG.kappa)
        GX, G = self.da(X)
        
        # Store data (no transformation needed for linear simulation)
        self.X, self.y = X, y
        self.GX = GX
        self.G = G
        self.X_raw = X
        self.GX_raw = GX
    
    def get_sweep_values(self) -> np.ndarray:
        """Generate radial sweep points."""
        return radial_sweep_pcs(self.X, self.sweep_samples)
    
    def setup_data(self) -> ExperimentDataContext:
        """Return data context."""
        return ExperimentDataContext(
            sem=self.sem,
            da=self.da,
            X=self.X,
            y=self.y,
            GX=self.GX,
            G=self.G,
            X_raw=self.X_raw,
            GX_raw=self.GX_raw
        )


# =============================================================================
# PARAMETER SWEEP RUNNERS
# =============================================================================

class SimulationParamSweepBase(ParamSweepRunner):
    """Base class for simulation parameter sweeps."""
    
    def __init__(self, kernel_dim: int, **kwargs):
        self.kernel_dim = kernel_dim
        super().__init__(**kwargs)
    
    def setup_sems_and_das(self):
        """Setup SEMs and DAs for all experiments."""
        self.sems = [SEM() for _ in range(self.n_experiments)]
        self.das = [
            DA(sem.W_XY, kernel_dim=self.kernel_dim)
            for sem in self.sems
        ]
    
    def get_da(self, experiment_index: int):
        """Get DA for specific experiment."""
        return self.das[experiment_index]


class KappaSweep(SimulationParamSweepBase):
    """Sweep over kappa (confounding strength)."""
    
    def get_param_range(self) -> np.ndarray:
        return SWEEP_CONFIGS['simulation']['kappa'].range_fn(self.sweep_samples)
    
    def generate_data(self, experiment_index: int, kappa: float):
        sem = self.sems[experiment_index]
        da = self.das[experiment_index]
        
        # Training data
        X, y = sem(N=self.n_samples, kappa=kappa)
        GX, G = da(X, gamma=1.0)
        
        # Test data
        test_size = int(SIMULATION_CONFIG.test_fraction * self.n_samples)
        X_test, _ = sem(N=test_size, intervention=True, kappa=kappa)
        estimand = X_test @ sem.solution
        
        return X, y, GX, G, X_test, estimand


class AlphaSweep(SimulationParamSweepBase):
    """Sweep over alpha (augmentation strength)."""
    
    def get_param_range(self) -> np.ndarray:
        return SWEEP_CONFIGS['simulation']['alpha'].range_fn(self.sweep_samples)
    
    def generate_data(self, experiment_index: int, alpha: float):
        sem = self.sems[experiment_index]
        da = self.das[experiment_index]
        
        # Training data with augmentation strength alpha
        X, y = sem(N=self.n_samples)
        GX, G = da(X, gamma=alpha)
        
        # Test data
        test_size = int(SIMULATION_CONFIG.test_fraction * self.n_samples)
        X_test, _ = sem(N=test_size, intervention=True)
        estimand = X_test @ sem.solution
        
        return X, y, GX, G, X_test, estimand


class GammaSweep(SimulationParamSweepBase):
    """Sweep over gamma (sensitivity parameter)."""
    
    def get_param_range(self) -> np.ndarray:
        return SWEEP_CONFIGS['simulation']['gamma'].range_fn(self.sweep_samples)
    
    def get_predict_kwargs(self, gamma: float):
        """Pass gamma to predict method."""
        return {'gamma': gamma, 'gamma0': SIMULATION_CONFIG.gamma0}
    
    def generate_data(self, experiment_index: int, gamma: float):
        sem = self.sems[experiment_index]
        da = self.das[experiment_index]
        
        # Training data
        X, y = sem(N=self.n_samples)
        GX, G = da(X, gamma=1.0)
        
        # Test data
        test_size = int(SIMULATION_CONFIG.test_fraction * self.n_samples)
        X_test, _ = sem(N=test_size, intervention=True)
        estimand = X_test @ sem.solution
        
        return X, y, GX, G, X_test, estimand


# =============================================================================
# ORCHESTRATOR
# =============================================================================

class SimulationOrchestrator(ExperimentOrchestrator):
    """Orchestrator for simulation experiments."""
    
    def __init__(self, **kwargs):
        # Create registry with simulation-specific parameters
        class SimulationRegistry(MethodRegistry):
            @staticmethod
            def build_methods(names):
                return MethodRegistry.build_methods(
                    names,
                    gamma=SIMULATION_CONFIG.gamma,
                    gamma0=SIMULATION_CONFIG.gamma0,
                    epsilon=SIMULATION_CONFIG.epsilon
                )
        
        super().__init__(EXPERIMENT_NAME, SimulationRegistry(), **kwargs)
    
    def get_query_runner_cls(self):
        return SimulationQuerySweep
    
    def get_param_sweeps(self):
        return [
            (KappaSweep, 'kappa'),
            (GammaSweep, 'gamma'),
            (AlphaSweep, 'alpha'),
        ]