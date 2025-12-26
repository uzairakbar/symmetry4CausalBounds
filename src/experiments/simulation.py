"""
Simulation experiment using generic runners.
Dramatically reduced code duplication.
"""
from typing import Type, List, Tuple

from src.data_augmentors.simulation import NullSpaceTranslation as DA
from src.sem.simulation import LinearSimulationSEM as SEM
from src.experiments.base import ExperimentOrchestrator
from src.experiments.generic_runner import (
    GenericQuerySweep,
    KappaSweep,
    AlphaSweep,
    GammaSweep,
    AugmentationFoldSweep,
)
from src.experiments.configs import (
    MethodRegistry,
    SIMULATION_CONFIG,
    SWEEP_CONFIGS
)

EXPERIMENT_NAME = 'simulation'


# =============================================================================
# ORCHESTRATOR
# =============================================================================

class SimulationOrchestrator(ExperimentOrchestrator):
    """Orchestrator for simulation experiments."""
    
    def __init__(self, kernel_dim: int, **kwargs):
        """
        Initialize simulation orchestrator.
        
        Args:
            kernel_dim: Dimensionality of DA kernel
            **kwargs: Other experiment parameters
        """
        self.kernel_dim = kernel_dim
        
        # Create registry with simulation-specific parameters
        class SimulationRegistry(MethodRegistry):
            @staticmethod
            def build_methods(names):
                return MethodRegistry.build_methods(
                    names,
                    gamma=SIMULATION_CONFIG.gamma,
                    gamma0=SIMULATION_CONFIG.gamma0,
                    delta=SIMULATION_CONFIG.delta,
                    epsilon=SIMULATION_CONFIG.epsilon
                )
        
        super().__init__(EXPERIMENT_NAME, SimulationRegistry(), **kwargs)
    
    def _sem_factory(self):
        """Factory for creating SEM instances."""
        return SEM()
    
    def _da_factory(self, sem):
        """
        Factory for creating DA instances.
        
        CRITICAL: DA must be created from the SAME SEM instance,
        since it computes null space of that specific SEM's W_XY.
        """
        return DA(sem.W_XY, kernel_dim=self.kernel_dim)
    
    def get_query_runner_cls(self) -> Type[GenericQuerySweep]:
        """Return query sweep runner."""
        # Create a configured class
        class SimulationQuerySweep(GenericQuerySweep):
            def __init__(inner_self, **kwargs):
                # Create SEM first
                sem = self._sem_factory()
                
                # Create DA from SAME SEM
                def da_factory_from_sem():
                    return self._da_factory(sem)
                
                super().__init__(
                    sem_factory=lambda: sem,  # Return same instance
                    da_factory=da_factory_from_sem,
                    poly_transform=None,
                    **kwargs
                )
        
        return SimulationQuerySweep
    
    def get_param_sweeps(self) -> List[Tuple[Type, str]]:
        """Return parameter sweeps to run."""
        # Helper to create configured sweep classes
        def make_sweep_cls(base_cls, sweep_name):
            class ConfiguredSweep(base_cls):
                def __init__(inner_self, **kwargs):
                    super().__init__(
                        sem_factory=self._sem_factory,
                        da_factory=self._da_factory,  # Now expects SEM as argument
                        poly_transform=None,
                        test_fraction=SIMULATION_CONFIG.test_fraction,
                        sweep_config=SWEEP_CONFIGS['simulation'][sweep_name],
                        **kwargs
                    )
            return ConfiguredSweep
        
        # Gamma sweep needs gamma0
        class ConfiguredGammaSweep(GammaSweep):
            def __init__(inner_self, **kwargs):
                super().__init__(
                    sem_factory=self._sem_factory,
                    da_factory=self._da_factory,  # Now expects SEM as argument
                    poly_transform=None,
                    test_fraction=SIMULATION_CONFIG.test_fraction,
                    sweep_config=SWEEP_CONFIGS['simulation']['gamma'],
                    gamma0=SIMULATION_CONFIG.gamma0,
                    use_train_test_split=False,
                    **kwargs
                )
        
        return [
            (make_sweep_cls(KappaSweep, 'kappa'), 'kappa'),
            (ConfiguredGammaSweep, 'gamma'),
            (make_sweep_cls(AlphaSweep, 'alpha'), 'alpha'),
            (make_sweep_cls(AugmentationFoldSweep, 'folds'), 'folds'),
        ]