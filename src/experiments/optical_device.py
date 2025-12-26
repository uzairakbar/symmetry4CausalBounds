"""
Optical device experiment using generic runners.
Dramatically reduced code duplication.
"""
from typing import Type, List, Tuple
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

from src.data_augmentors.optical_device import OpticalDeviceDA as DA
from src.sem.optical_device import OpticalDeviceSEM as SEM
from src.experiments.base import ExperimentOrchestrator
from src.experiments.generic_runner import (
    GenericQuerySweep, 
    GammaSweep, 
    AugmentationFoldSweep,
    GenericParamSweep
)
from src.experiments.configs import (
    MethodRegistry,
    OPTICAL_CONFIG,
    SWEEP_CONFIGS
)

EXPERIMENT_NAME = 'optical_device'


# =============================================================================
# ORCHESTRATOR
# =============================================================================

class OpticalOrchestrator(ExperimentOrchestrator):
    """Orchestrator for optical device experiments."""
    
    def __init__(self, augmentation: str, **kwargs):
        """
        Initialize optical orchestrator.
        """
        self.augmentation = augmentation
        
        # Create registry with optical-specific parameters
        class OpticalRegistry(MethodRegistry):
            @staticmethod
            def build_methods(names):
                return MethodRegistry.build_methods(
                    names,
                    gamma=OPTICAL_CONFIG.gamma,
                    gamma0=OPTICAL_CONFIG.gamma0,
                    delta=OPTICAL_CONFIG.delta,
                    epsilon=OPTICAL_CONFIG.epsilon
                )
        
        super().__init__(EXPERIMENT_NAME, OpticalRegistry(), **kwargs)
    
    def _sem_factory(self):
        """Factory for creating SEM instances."""
        return SEM(
            experiment=OPTICAL_CONFIG.dataset_index,
            ground_truth=OPTICAL_CONFIG.ground_truth_model
        )
    
    def _da_factory(self, sem=None):
        """Factory for creating DA instances."""
        return DA(self.augmentation)
    
    def _poly_factory(self):
        """Factory for creating polynomial transformer."""
        # Get degree from a sample SEM
        sem = self._sem_factory()
        return PolynomialFeatures(sem.poly_degree, include_bias=False)
    
    def get_query_runner_cls(self) -> Type[GenericQuerySweep]:
        """Return query sweep runner."""
        class OpticalQuerySweep(GenericQuerySweep):
            def __init__(inner_self, **kwargs):
                super().__init__(
                    sem_factory=self._sem_factory,
                    da_factory=self._da_factory,
                    poly_transform=self._poly_factory(),
                    **kwargs
                )
        
        return OpticalQuerySweep
    
    def get_param_sweeps(self) -> List[Tuple[Type, str]]:
        """Return parameter sweeps to run."""
        
        # 1. Standard Gamma Sweep
        class OpticalGammaSweep(GammaSweep):
            def __init__(inner_self, **kwargs):
                super().__init__(
                    sem_factory=self._sem_factory,
                    da_factory=self._da_factory,
                    poly_transform=self._poly_factory(),
                    test_fraction=OPTICAL_CONFIG.test_fraction,
                    sweep_config=SWEEP_CONFIGS['optical_device']['gamma'],
                    gamma0=OPTICAL_CONFIG.gamma0,
                    use_train_test_split=True,
                    **kwargs
                )

        # 2. Augmentation Folds Sweep
        class OpticalFoldSweep(AugmentationFoldSweep):
            def __init__(inner_self, **kwargs):
                super().__init__(
                    sem_factory=self._sem_factory,
                    da_factory=self._da_factory,
                    poly_transform=self._poly_factory(),
                    test_fraction=OPTICAL_CONFIG.test_fraction,
                    sweep_config=SWEEP_CONFIGS['optical_device']['folds'],
                    **kwargs
                )

        # 3. Kappa (Dataset) Sweep
        class OpticalKappaSweep(GenericParamSweep):
            def __init__(inner_self, **kwargs):
                # Manually extract config
                config = SWEEP_CONFIGS['optical_device']['kappa']
                
                super().__init__(
                    sem_factory=self._sem_factory,
                    da_factory=self._da_factory,
                    poly_transform=self._poly_factory(),
                    test_fraction=OPTICAL_CONFIG.test_fraction,
                    sweep_config=config,
                    **kwargs
                )
                
                # Attach to THIS runner instance
                inner_self.sweep_config = config 
                inner_self.kappas = [] 

            @property
            def data_depends_on_param(inner_self) -> bool:
                return True 

            def get_param_range(inner_self) -> np.ndarray:
                return inner_self.sweep_config.range_fn(12) 

            def generate_data(inner_self, experiment_index: int, dataset_idx):
                dataset_idx = int(dataset_idx)
                
                # FRESH SEM for this specific dataset
                sem = SEM(experiment=dataset_idx, ground_truth=OPTICAL_CONFIG.ground_truth_model)
                
                # Estimate Kappa
                kappa_est = np.sqrt(sem.varEXiX / (sem.varXi + 1e-9))
                
                # Store kappa (once per dataset index)
                if experiment_index == 0:
                     inner_self.kappas.append((dataset_idx, kappa_est))

                # Load Data (using parent factory methods via outer self)
                da = self._da_factory()
                X_raw, y = sem(N=inner_self.n_samples) 
                GX_raw, G = da(X_raw)
                
                # Transform
                # FIX: Use the specific degree from the current SEM, not the global default
                poly = PolynomialFeatures(sem.poly_degree, include_bias=False)
                X = poly.fit_transform(X_raw)
                GX = poly.fit_transform(GX_raw)
                
                X_train, X_test, y_train, _, GX_train, _, G_train, _ = train_test_split(
                    X, y, GX, G,
                    test_size=inner_self.test_fraction,
                    random_state=inner_self.seed + experiment_index
                )
                
                estimand = X_test @ sem.solution
                return X_train, y_train, GX_train, G_train, X_test, estimand

            def run(inner_self, desc: str):
                """Sort results by Kappa after running."""
                # Run standard sweep
                indices, raw_results = super().run(desc)
                
                # Sort by calculated kappa (value is at index 1)
                inner_self.kappas.sort(key=lambda x: x[1]) 
                
                # Extract sorted values and dataset indices
                sorted_k_values = np.array([k for _, k in inner_self.kappas])
                sorted_dataset_indices = [idx for idx, _ in inner_self.kappas]
                
                sorted_results = {}
                for method, matrix in raw_results.items():
                    # We map row i to sorted position
                    new_matrix = np.zeros_like(matrix)
                    
                    # The param range was 0..11
                    for new_row_idx, original_dataset_idx in enumerate(sorted_dataset_indices):
                         new_matrix[new_row_idx] = matrix[original_dataset_idx]
                    
                    sorted_results[method] = new_matrix
                
                return sorted_k_values, sorted_results

        return [
            (OpticalGammaSweep, 'gamma'),
            (OpticalFoldSweep, 'folds'),
            (OpticalKappaSweep, 'kappa')
        ]