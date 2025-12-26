"""
Optical device experiment using generic runners.
Dramatically reduced code duplication.
"""
from typing import Type, List, Tuple
from sklearn.preprocessing import PolynomialFeatures

from src.data_augmentors.optical_device import OpticalDeviceDA as DA
from src.sem.optical_device import OpticalDeviceSEM as SEM
from src.experiments.base import ExperimentOrchestrator
from src.experiments.generic_runner import GenericQuerySweep, GammaSweep
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
        
        Args:
            augmentation: Augmentation type ('all', 'rotation', etc.)
            **kwargs: Other experiment parameters
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
        """
        Factory for creating DA instances.
        
        Args:
            sem: SEM instance (not used for optical device, but kept for consistency)
        """
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
        class OpticalGammaSweep(GammaSweep):
            def __init__(inner_self, **kwargs):
                super().__init__(
                    sem_factory=self._sem_factory,
                    da_factory=self._da_factory,
                    poly_transform=self._poly_factory(),
                    test_fraction=OPTICAL_CONFIG.test_fraction,
                    sweep_config=SWEEP_CONFIGS['optical_device']['gamma'],
                    gamma0=OPTICAL_CONFIG.gamma0,
                    use_train_test_split=True,  # Optical uses train/test split
                    **kwargs
                )
        
        return [(OpticalGammaSweep, 'gamma')]