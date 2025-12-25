"""
Generic experiment runners that work with any SEM/DA combination.
Eliminates duplication between simulation and optical device experiments.
"""
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, Callable, Optional, Any, Dict

from src.experiments.base import (
    QuerySweepRunner, ParamSweepRunner, ExperimentDataContext
)
from src.experiments.utils import fit_model, radial_sweep_pcs


# =============================================================================
# GENERIC QUERY SWEEP RUNNER
# =============================================================================

class GenericQuerySweep(QuerySweepRunner):
    """
    Generic query sweep runner that works with any SEM/DA.
    
    Eliminates duplication between simulation and optical experiments.
    """
    
    def __init__(
        self,
        sem_factory: Callable,
        da_factory: Callable,
        poly_transform: Optional[Callable] = None,
        **kwargs
    ):
        """
        Initialize generic query sweep.
        
        Args:
            sem_factory: Function that returns a SEM instance
            da_factory: Function that returns a DA instance
            poly_transform: Optional polynomial feature transformer
            **kwargs: Base runner arguments (seed, n_samples, etc.)
        """
        super().__init__(**kwargs)
        
        # Create SEM and DA
        self.sem = sem_factory()
        self.da = da_factory()
        self.poly = poly_transform
        
        # Load and transform data
        X_raw, y = self.sem(N=self.n_samples)
        GX_raw, G = self.da(X_raw)
        
        # Store raw data
        self.X_raw = X_raw
        self.GX_raw = GX_raw
        self.y = y
        self.G = G
        
        # Apply polynomial transformation if provided
        if self.poly:
            self.X = self.poly.fit_transform(X_raw)
            self.GX = self.poly.fit_transform(GX_raw)
        else:
            self.X = X_raw
            self.GX = GX_raw
    
    def get_sweep_values(self) -> np.ndarray:
        """Generate radial sweep points with optional polynomial features."""
        geometry = self.X_raw if not hasattr(self, 'poly') else self.GX_raw
        raw_sweep = radial_sweep_pcs(geometry, self.sweep_samples)
        
        if self.poly:
            return self.poly.fit_transform(raw_sweep)
        return raw_sweep
    
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
# GENERIC PARAMETER SWEEP RUNNERS
# =============================================================================

class GenericParamSweep(ParamSweepRunner):
    """
    Generic parameter sweep base class.
    
    Subclasses only need to implement:
    - get_param_range()
    - generate_single_experiment_data()
    """
    
    def __init__(
        self,
        sem_factory: Callable,
        da_factory: Callable,
        poly_transform: Optional[Callable] = None,
        test_fraction: float = 0.1,
        **kwargs
    ):
        """
        Initialize generic parameter sweep.
        
        Args:
            sem_factory: Function that returns a SEM instance
            da_factory: Function that takes a SEM and returns a DA instance
            poly_transform: Optional polynomial feature transformer
            test_fraction: Fraction of data for testing
            **kwargs: Base runner arguments
        """
        self.sem_factory = sem_factory
        self.da_factory = da_factory
        self.poly = poly_transform
        self.test_fraction = test_fraction
        
        super().__init__(**kwargs)
    
    def setup_sems_and_das(self):
        """
        Setup SEMs and DAs for all experiments.
        
        CRITICAL: Each DA must be created from its corresponding SEM,
        since DA computes null space of that specific SEM's W_XY.
        """
        self.sems = [self.sem_factory() for _ in range(self.n_experiments)]
        self.das = [self.da_factory(sem) for sem in self.sems]
    
    def get_da(self, experiment_index: int):
        """Get DA for specific experiment."""
        return self.das[experiment_index]
    
    def apply_transform(self, X: np.ndarray) -> np.ndarray:
        """Apply polynomial transformation if available."""
        if self.poly:
            return self.poly.fit_transform(X)
        return X


# =============================================================================
# SPECIFIC SWEEP TYPES
# =============================================================================

class KappaSweep(GenericParamSweep):
    """Sweep over kappa (confounding strength)."""
    
    def __init__(self, sweep_config, **kwargs):
        self.sweep_config = sweep_config
        super().__init__(**kwargs)
    
    def get_param_range(self) -> np.ndarray:
        return self.sweep_config.range_fn(self.sweep_samples)
    
    def generate_data(self, experiment_index: int, kappa: float):
        """Generate data for one experiment at given kappa."""
        sem = self.sems[experiment_index]
        da = self.das[experiment_index]
        
        # Training data
        X_raw, y = sem(N=self.n_samples, kappa=kappa)
        GX_raw, G = da(X_raw, gamma=1.0)
        
        X = self.apply_transform(X_raw)
        GX = self.apply_transform(GX_raw)
        
        # Test data (interventional)
        test_size = int(self.test_fraction * self.n_samples)
        X_test_raw, _ = sem(N=test_size, intervention=True, kappa=kappa)
        X_test = self.apply_transform(X_test_raw)
        
        estimand = X_test @ sem.solution
        
        return X, y, GX, G, X_test, estimand


class AlphaSweep(GenericParamSweep):
    """Sweep over alpha (augmentation strength)."""
    
    def __init__(self, sweep_config, **kwargs):
        self.sweep_config = sweep_config
        super().__init__(**kwargs)
    
    def get_param_range(self) -> np.ndarray:
        return self.sweep_config.range_fn(self.sweep_samples)
    
    def generate_data(self, experiment_index: int, alpha: float):
        """Generate data for one experiment at given alpha."""
        sem = self.sems[experiment_index]
        da = self.das[experiment_index]
        
        # Training data with augmentation strength alpha
        X_raw, y = sem(N=self.n_samples)
        GX_raw, G = da(X_raw, gamma=alpha)
        
        X = self.apply_transform(X_raw)
        GX = self.apply_transform(GX_raw)
        
        # Test data
        test_size = int(self.test_fraction * self.n_samples)
        X_test_raw, _ = sem(N=test_size, intervention=True)
        X_test = self.apply_transform(X_test_raw)
        
        estimand = X_test @ sem.solution
        
        return X, y, GX, G, X_test, estimand


class GammaSweep(GenericParamSweep):
    """Sweep over gamma (sensitivity parameter)."""
    
    def __init__(
        self,
        sweep_config,
        gamma0: float,
        use_train_test_split: bool = False,
        **kwargs
    ):
        self.sweep_config = sweep_config
        self.gamma0 = gamma0
        self.use_split = use_train_test_split
        super().__init__(**kwargs)
    
    def get_param_range(self) -> np.ndarray:
        return self.sweep_config.range_fn(self.sweep_samples)
    
    def get_predict_kwargs(self, gamma: float):
        """Pass gamma to predict method."""
        return {'gamma': gamma, 'gamma0': self.gamma0}
    
    def generate_data(self, experiment_index: int, gamma: float):
        """Generate data for one experiment at given gamma."""
        sem = self.sems[experiment_index]
        da = self.das[experiment_index]
        
        if self.use_split:
            # Optical device: use train/test split
            X_raw, y = sem(N=self.n_samples)
            GX_raw, G = da(X_raw)
            
            X = self.apply_transform(X_raw)
            GX = self.apply_transform(GX_raw)
            
            X_train, X_test, y_train, _, GX_train, _, G_train, _ = train_test_split(
                X, y, GX, G,
                test_size=self.test_fraction,
                random_state=self.seed + experiment_index
            )
            
            estimand = X_test @ sem.solution
            return X_train, y_train, GX_train, G_train, X_test, estimand
        
        else:
            # Simulation: use interventional test data
            X_raw, y = sem(N=self.n_samples)
            GX_raw, G = da(X_raw, gamma=1.0)
            
            X = self.apply_transform(X_raw)
            GX = self.apply_transform(GX_raw)
            
            test_size = int(self.test_fraction * self.n_samples)
            X_test_raw, _ = sem(N=test_size, intervention=True)
            X_test = self.apply_transform(X_test_raw)
            
            estimand = X_test @ sem.solution
            
            return X, y, GX, G, X_test, estimand