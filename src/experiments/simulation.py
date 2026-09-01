"""
Simulation experiment using generic runners.
Dramatically reduced code duplication.
"""

from typing import Type

from src.data_augmentors.simulation import NullSpaceTranslation as DA
from src.sem.simulation import LinearSimulationSEM as SEM
from src.experiments.base import ExperimentOrchestrator
from src.experiments.generic_runner import GenericQuerySweep, STRATEGIES
from src.experiments.configs import MethodRegistry, SIMULATION_CONFIG

EXPERIMENT_NAME = "simulation"

# m-sweep holds n fixed here (PLAN 5.5)
FOLD_SWEEP_SAMPLES: int = 512


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
        self.toggles = dict(
            calibrate=kwargs.get("calibrate", False),
            pad=kwargs.get("pad", False),
            clipy=kwargs.get("clipy", True),
            n_jobs=kwargs.get("n_jobs", 1),
        )
        toggles = self.toggles

        # Create registry with simulation-specific parameters
        class SimulationRegistry(MethodRegistry):
            @staticmethod
            def build_methods(names):
                return MethodRegistry.build_methods(
                    names, gamma=SIMULATION_CONFIG.gamma, epsilon=SIMULATION_CONFIG.epsilon, **toggles
                )

        super().__init__(EXPERIMENT_NAME, SimulationRegistry(), **kwargs)

    def _sem_factory(self):
        """Factory for creating SEM instances."""
        return SEM(gamma=SIMULATION_CONFIG.gamma_true)

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
                    method_factory=self.build_methods,
                    default_gamma=SIMULATION_CONFIG.gamma,
                    default_epsilon=SIMULATION_CONFIG.epsilon,
                    **kwargs,
                )

        return SimulationQuerySweep

    def build_methods(self, gamma: float, epsilon: float, epsilon_iv=None, n_jobs=None):
        """Methods at explicit (per-experiment) budgets. `n_jobs` overrides the
        toggle -- perf needs serial models to time methods, not the harness."""
        toggles = self.toggles if n_jobs is None else {**self.toggles, "n_jobs": n_jobs}
        return MethodRegistry.build_methods(
            self.kwargs["methods"], gamma=gamma, epsilon=epsilon, epsilon_iv=epsilon_iv, **toggles
        )

    def get_sweep_runner_cls(self, param: str) -> Type:
        """Configured strategy for one sweep parameter."""
        outer, Strategy = self, STRATEGIES[param]

        class ConfiguredSweep(Strategy):
            def __init__(inner_self, **kwargs):
                extra = {}
                if param == "m":
                    extra["n_samples_override"] = FOLD_SWEEP_SAMPLES
                super().__init__(
                    sem_factory=outer._sem_factory,
                    da_factory=outer._da_factory,  # expects the SEM as argument
                    poly_transform=None,
                    test_fraction=SIMULATION_CONFIG.test_fraction,
                    default_gamma=SIMULATION_CONFIG.gamma,
                    default_epsilon=SIMULATION_CONFIG.epsilon,
                    experiment_name=EXPERIMENT_NAME,
                    **extra,
                    **kwargs,
                )

        return ConfiguredSweep
