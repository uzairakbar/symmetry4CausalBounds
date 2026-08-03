"""
Generic experiment runners that work with any SEM/DA combination.
Eliminates duplication between simulation and optical device experiments.
"""
import numpy as np
from loguru import logger
from sklearn.model_selection import train_test_split
from typing import Tuple, Callable, Optional, Any, Dict

from src.experiments.base import (
    QuerySweepRunner, ParamSweepRunner, ExperimentDataContext, SweepData
)
from src.experiments.utils import fit_model, radial_sweep_pcs
from src.experiments.utils.metrics import trace_S_over_k
from src.oracle import calibrate_da_epsilon, compute_oracle_parameters


# =============================================================================
# ORACLE MIXIN
# =============================================================================

class OracleMixin:
    """
    Prepares an (SEM, DA) pair: optionally sets the DA to a target epsilon,
    then records the oracle parameters. Nothing consumes them yet.
    """

    epsilon_true: Optional[float] = None

    def prepare_pair(self, sem, da, features: Optional[Callable] = None):
        if self.epsilon_true is not None:
            calibrate_da_epsilon(
                sem=sem, da=da,
                epsilon_target=self.epsilon_true,
                features=features,
            )

        oracle = compute_oracle_parameters(
            sem=sem, da=da, features=features, calibrate=self.calibrate
        )
        logger.info(f'Oracle parameters: {oracle}')
        return oracle


# =============================================================================
# GENERIC QUERY SWEEP RUNNER
# =============================================================================

class GenericQuerySweep(OracleMixin, QuerySweepRunner):
    """
    Generic query sweep runner that works with any SEM/DA.
    """

    def __init__(
        self,
        sem_factory: Callable,
        da_factory: Callable,
        poly_transform: Optional[Callable] = None,
        epsilon_true: Optional[float] = None,
        **kwargs
    ):
        super().__init__(**kwargs)

        # Create SEM and DA
        self.sem = sem_factory()
        self.da = da_factory()
        self.poly = poly_transform
        self.epsilon_true = epsilon_true
        self.oracle = self.prepare_pair(self.sem, self.da, features=self._features)

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
    
    @property
    def _features(self) -> Optional[Callable]:
        return self.poly.fit_transform if self.poly else None

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
            GX_raw=self.GX_raw,
            oracle=self.oracle
        )


# =============================================================================
# GENERIC PARAMETER SWEEP RUNNERS
# =============================================================================

class GenericParamSweep(OracleMixin, ParamSweepRunner):
    """Generic parameter sweep base class."""

    def __init__(
        self,
        sem_factory: Callable,
        da_factory: Callable,
        poly_transform: Optional[Callable] = None,
        test_fraction: float = 0.1,
        epsilon_true: Optional[float] = None,
        **kwargs
    ):
        self.sem_factory = sem_factory
        self.da_factory = da_factory
        self.poly = poly_transform
        self.test_fraction = test_fraction
        self.epsilon_true = epsilon_true
        super().__init__(**kwargs)

    @property
    def _features(self) -> Optional[Callable]:
        return self.poly.fit_transform if self.poly else None

    def setup_sems_and_das(self):
        """Setup SEMs and DAs for all experiments."""
        self.sems = [self.sem_factory() for _ in range(self.n_experiments)]
        self.das = [self.da_factory(sem) for sem in self.sems]
        self.oracles = [
            self.prepare_pair(sem, da, features=self._features)
            for sem, da in zip(self.sems, self.das)
        ]

    def get_da(self, experiment_index: int):
        return self.das[experiment_index]

    def get_oracle(self, experiment_index: int):
        return self.oracles[experiment_index]

    def apply_transform(self, X: np.ndarray) -> np.ndarray:
        if self.poly:
            return self.poly.fit_transform(X)
        return X


# =============================================================================
# SPECIFIC SWEEP TYPES
# =============================================================================

class GammaStarSweep(GenericParamSweep):
    """Sweep over the true confounding budget gamma*. Data changes every step."""

    def __init__(self, sweep_config, **kwargs):
        self.sweep_config = sweep_config
        super().__init__(**kwargs)

    @property
    def data_depends_on_param(self) -> bool:
        return True  # gamma* changes data generation

    def get_param_range(self) -> np.ndarray:
        return self.sweep_config.range_fn(self.sweep_samples)

    def generate_data(self, experiment_index: int, gamma_star: float):
        sem = self.sems[experiment_index]
        da = self.das[experiment_index]

        # Training data
        X_raw, y = sem(N=self.n_samples, gamma=gamma_star)
        GX_raw, G = da(X_raw, scale=1.0)

        X = self.apply_transform(X_raw)
        GX = self.apply_transform(GX_raw)
        
        # Test data (interventional)
        test_size = int(self.test_fraction * self.n_samples)
        X_test_raw, _ = sem(N=test_size, intervention=True)
        X_test = self.apply_transform(X_test_raw)
        
        estimand = X_test @ sem.solution
        
        return X, y, GX, G, X_test, estimand


class TraceSSweep(GenericParamSweep):
    """
    Sweep over DA scale.

    The x-axis returned by run() is the measured tr(S)/k (Prop. 2), not the
    input scale.
    """

    def __init__(self, sweep_config, **kwargs):
        self.sweep_config = sweep_config
        super().__init__(**kwargs)
        self.strength_values = []

    @property
    def data_depends_on_param(self) -> bool:
        return True  # scale changes G matrix, so re-fit is required

    def get_param_range(self) -> np.ndarray:
        return self.sweep_config.range_fn(self.sweep_samples)

    def run(self, desc: str = "tr(S)/k Sweep") -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Run sweep and return measured strength values as x-axis."""
        self.strength_values = []
        
        # Run standard sweep logic
        _, results = super().run(desc)
        
        # Process strength values collected during generation
        # Shape: (n_experiments, n_steps) due to loop order in ParamSweepRunner
        strength_array = np.array(self.strength_values).reshape(self.n_experiments, self.sweep_samples)
        
        # Use average strength across experiments for the x-axis
        mean_strength = strength_array.mean(axis=0)
        
        return mean_strength, results

    def generate_data(self, experiment_index: int, scale: float):
        sem = self.sems[experiment_index]
        da = self.das[experiment_index]

        # Training data with augmentation scale
        X_raw, y = sem(N=self.n_samples)
        GX_raw, G = da(X_raw, scale=scale)

        # Transform
        X = self.apply_transform(X_raw)
        GX = self.apply_transform(GX_raw)

        # tr(S)/k in the feature space the methods use
        self.strength_values.append(trace_S_over_k(X, GX))
        
        # Test data
        test_size = int(self.test_fraction * self.n_samples)
        X_test_raw, _ = sem(N=test_size, intervention=True)
        X_test = self.apply_transform(X_test_raw)
        
        estimand = X_test @ sem.solution
        
        return X, y, GX, G, X_test, estimand


class GammaSweep(GenericParamSweep):
    """Sweep over gamma (sensitivity parameter). Data is CONSTANT."""
    
    def __init__(
        self,
        sweep_config,
        use_train_test_split: bool = False,
        **kwargs
    ):
        self.sweep_config = sweep_config
        self.use_split = use_train_test_split
        super().__init__(**kwargs)
        
    @property
    def data_depends_on_param(self) -> bool:
        return False  # Gamma only changes constraint RHS, not the data
    
    def get_param_range(self) -> np.ndarray:
        return self.sweep_config.range_fn(self.sweep_samples)
    
    def get_predict_kwargs(self, gamma: float):
        return {'gamma': gamma}
    
    def generate_data(self, experiment_index: int, gamma: float):
        """Generate data. 'gamma' param is ignored for generation."""
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
            GX_raw, G = da(X_raw, scale=1.0)
            
            X = self.apply_transform(X_raw)
            GX = self.apply_transform(GX_raw)
            
            test_size = int(self.test_fraction * self.n_samples)
            X_test_raw, _ = sem(N=test_size, intervention=True)
            X_test = self.apply_transform(X_test_raw)
            
            estimand = X_test @ sem.solution
            
            return X, y, GX, G, X_test, estimand


class SampleSweep(GenericParamSweep):
    """
    Sweep over sample size N.
    """
    def __init__(self, sweep_config, **kwargs):
        self.sweep_config = sweep_config
        super().__init__(**kwargs)
    
    @property
    def data_depends_on_param(self) -> bool:
        return True # Data size changes, must refit
    
    def get_param_range(self) -> np.ndarray:
        return self.sweep_config.range_fn(self.sweep_samples)
    
    def generate_data(self, experiment_index: int, n_samples_param):
        """
        Generate data with specific sample size N.
        """
        N = int(n_samples_param)
        sem = self.sems[experiment_index]
        da = self.das[experiment_index]
        
        # Training data
        X_raw, y = sem(N=N)
        GX_raw, G = da(X_raw)
        
        X = self.apply_transform(X_raw)
        GX = self.apply_transform(GX_raw)
        
        # Test data (Keep test set large and constant for fair comparison if possible, 
        # or scale with N. Here scaling with N via test_fraction)
        test_size = max(50, int(self.test_fraction * N)) 
        
        if 'OpticalDeviceSEM' in str(type(sem)):
            # Optical uses train/test split from finite pool
            X_train, X_test, y_train, _, GX_train, _, G_train, _ = train_test_split(
                X, y, GX, G,
                test_size=self.test_fraction,
                random_state=self.seed + experiment_index
            )
            # Recalculate estimand on specific test set
            estimand = X_test @ sem.solution
            return X_train, y_train, GX_train, G_train, X_test, estimand
        else:
            # Simulation uses fresh interventional generation
            X_test_raw, _ = sem(N=test_size, intervention=True)
            X_test = self.apply_transform(X_test_raw)
            estimand = X_test @ sem.solution
            return X, y, GX, G, X_test, estimand


class AugmentationFoldSweep(GenericParamSweep):
    """
    Sweep over the number of augmentation folds m, at FIXED base data.

    Base sample and train/test split are drawn ONCE per experiment, so m is the
    only thing that varies: GX/G stack m distinct DA passes over the same train
    X, and X/y are tiled to match. The augmented train set thus grows m-fold --
    the one sanctioned departure from the paper's fixed-size convention, kept
    deliberately to study the finite-n benefit of m.
    """
    def __init__(self, sweep_config, **kwargs):
        self.sweep_config = sweep_config
        self._base = {}
        super().__init__(**kwargs)

    @property
    def data_depends_on_param(self) -> bool:
        return True # Changing folds changes dataset size

    def get_param_range(self) -> np.ndarray:
        return self.sweep_config.range_fn(self.sweep_samples)

    def _base_data(self, experiment_index: int):
        """(X_train_raw, X_train, y_train, X_test, estimand); drawn once per experiment."""
        if experiment_index not in self._base:
            self._base[experiment_index] = self._draw_base(experiment_index)
        return self._base[experiment_index]

    def _draw_base(self, experiment_index: int):
        sem = self.sems[experiment_index]

        if 'OpticalDeviceSEM' in str(type(sem)):
            # finite pool: split unique instances, test set then fixed across m
            X_all, y_all = sem(N=self.n_samples)
            X_train_raw, X_test_raw, y_train, _ = train_test_split(
                X_all, y_all,
                test_size=self.test_fraction,
                random_state=self.seed + experiment_index
            )
        else:
            # generator: interventional test set
            X_train_raw, y_train = sem(N=self.n_samples)
            X_test_raw, _ = sem(
                N=int(self.test_fraction * self.n_samples), intervention=True
            )

        X_test = self.apply_transform(X_test_raw)
        return (X_train_raw, self.apply_transform(X_train_raw), y_train,
                X_test, X_test @ sem.solution)

    def generate_data(self, experiment_index: int, n_folds_param):
        """m distinct augmentations of the same fixed train set."""
        m = int(n_folds_param)
        da = self.das[experiment_index]
        X_raw, X, y, X_test, estimand = self._base_data(experiment_index)

        GX_raws, Gs = zip(*(da(X_raw) for _ in range(m)))

        return SweepData(
            # transform is row-wise: transform(tile(.)) == tile(transform(.))
            X=np.tile(X, (m, 1)),
            y=np.tile(y, (m, 1)),
            GX=self.apply_transform(np.vstack(GX_raws)),
            G=np.vstack(Gs),
            X_test=X_test,
            estimand=estimand,
            X_base=X,
            y_base=y,
        )