"""
Optical device experiment with unified runner implementation.
"""
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

from src.data_augmentors.optical_device import OpticalDeviceDA as DA
from src.sem.optical_device import OpticalDeviceSEM as SEM
from src.experiments.base import (
    QuerySweepRunner, ParamSweepRunner, ExperimentOrchestrator, ExperimentDataContext
)
from src.experiments.configs import (
    MethodRegistry, OPTICAL_CONFIG, SWEEP_CONFIGS
)
from src.experiments.utils import radial_sweep_pcs

EXPERIMENT_NAME = 'optical_device'


# =============================================================================
# QUERY SWEEP RUNNER
# =============================================================================

class OpticalQuerySweep(QuerySweepRunner):
    """Query sweep for optical device experiment."""
    
    def __init__(self, augmentation: str, **kwargs):
        super().__init__(**kwargs)
        
        # Setup SEM and DA
        self.sem = SEM(
            experiment=OPTICAL_CONFIG.dataset_index,
            ground_truth=OPTICAL_CONFIG.ground_truth_model
        )
        self.da = DA(augmentation)
        self.poly = PolynomialFeatures(self.sem.poly_degree, include_bias=False)
        
        # Load and transform data
        X_raw, y = self.sem(N=self.n_samples)
        GX_raw, G = self.da(X_raw)
        
        self.X_raw = X_raw
        self.GX_raw = GX_raw
        self.X = self.poly.fit_transform(X_raw)
        self.GX = self.poly.fit_transform(GX_raw)
        self.y = y
        self.G = G
    
    def get_sweep_values(self) -> np.ndarray:
        """Generate radial sweep points with polynomial features."""
        raw_sweep = radial_sweep_pcs(self.GX_raw, self.sweep_samples)
        return self.poly.fit_transform(raw_sweep)
    
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
# PARAMETER SWEEP RUNNER
# =============================================================================

class OpticalGammaSweep(ParamSweepRunner):
    """Gamma sweep for optical device."""
    
    def __init__(self, augmentation: str, **kwargs):
        self.augmentation = augmentation
        super().__init__(**kwargs)
    
    def setup_sems_and_das(self):
        """Setup SEM and DA (single instance for optical device)."""
        self.sem = SEM(
            experiment=OPTICAL_CONFIG.dataset_index,
            ground_truth=OPTICAL_CONFIG.ground_truth_model
        )
        self.da = DA(self.augmentation)
        self.poly = PolynomialFeatures(self.sem.poly_degree, include_bias=False)
        
        # Load and transform all data once
        X_raw, y = self.sem(N=self.n_samples)
        GX_raw, G = self.da(X_raw)
        
        self.X = self.poly.fit_transform(X_raw)
        self.GX = self.poly.fit_transform(GX_raw)
        self.y = y
        self.G = G
    
    def get_param_range(self) -> np.ndarray:
        return SWEEP_CONFIGS['optical_device']['gamma'].range_fn(self.sweep_samples)
    
    def get_da(self, experiment_index: int):
        return self.da
    
    def get_predict_kwargs(self, gamma: float):
        """Pass gamma to predict method."""
        return {'gamma': gamma, 'gamma0': OPTICAL_CONFIG.gamma0}
    
    def generate_data(self, experiment_index: int, gamma: float):
        """Split data for this experiment."""
        X_train, X_test, y_train, y_test, GX_train, _, G_train, _ = train_test_split(
            self.X, self.y, self.GX, self.G,
            test_size=OPTICAL_CONFIG.test_fraction,
            random_state=self.seed + experiment_index
        )
        
        estimand = X_test @ self.sem.solution
        return X_train, y_train, GX_train, G_train, X_test, estimand


# =============================================================================
# ORCHESTRATOR
# =============================================================================

class OpticalOrchestrator(ExperimentOrchestrator):
    """Orchestrator for optical device experiments."""
    
    def __init__(self, **kwargs):
        # Create registry with optical-specific parameters
        class OpticalRegistry(MethodRegistry):
            @staticmethod
            def build_methods(names):
                return MethodRegistry.build_methods(
                    names,
                    gamma=OPTICAL_CONFIG.gamma,
                    gamma0=OPTICAL_CONFIG.gamma0,
                    epsilon=OPTICAL_CONFIG.epsilon
                )
        
        super().__init__(EXPERIMENT_NAME, OpticalRegistry(), **kwargs)
    
    def get_query_runner_cls(self):
        return OpticalQuerySweep
    
    def get_param_sweeps(self):
        return [(OpticalGammaSweep, 'gamma')]