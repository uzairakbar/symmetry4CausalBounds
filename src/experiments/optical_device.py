"""
Optical device experiment using generic runners.
Dramatically reduced code duplication.
"""

from typing import Type, Dict
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

from src.data_augmentors.optical_device import OpticalDeviceDA as DA
from src.sem.optical_device import OpticalDeviceSEM as SEM
from src.experiments.base import ExperimentOrchestrator
from src.experiments.generic_runner import GenericQuerySweep, STRATEGIES
from src.experiments.configs import MethodRegistry, OPTICAL_CONFIG

EXPERIMENT_NAME = "optical_device"

# m-sweep holds n fixed here (PLAN 5.5)
FOLD_SWEEP_SAMPLES: int = 512


def _knob_to_augment_kwargs(strength: float) -> Dict[str, float]:
    """Map the [0, 1] strength knob onto the optical DA's own parameters."""
    return {
        "p": 0.5 * float(strength),  # flip/rotation probability
        "noise_coeff": float(np.sqrt(0.1 * strength)),  # up to 0.1 Var(X)
    }


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
        self.toggles = dict(
            calibrate=kwargs.get("calibrate", False),
            pad=kwargs.get("pad", False),
            clipy=kwargs.get("clipy", True),
            n_jobs=kwargs.get("n_jobs", 1),
        )
        toggles = self.toggles

        # Create registry with optical-specific parameters
        class OpticalRegistry(MethodRegistry):
            @staticmethod
            def build_methods(names):
                return MethodRegistry.build_methods(
                    names, gamma=OPTICAL_CONFIG.gamma, epsilon=OPTICAL_CONFIG.epsilon, **toggles
                )

        super().__init__(EXPERIMENT_NAME, OpticalRegistry(), **kwargs)

    def _sem_factory(self):
        """Factory for creating SEM instances."""
        return SEM(experiment=OPTICAL_CONFIG.dataset_index, ground_truth=OPTICAL_CONFIG.ground_truth_model)

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
                    epsilon_true=OPTICAL_CONFIG.epsilon_true,
                    method_factory=self.build_methods,
                    default_gamma=OPTICAL_CONFIG.gamma,
                    default_epsilon=OPTICAL_CONFIG.query_epsilon,
                    **kwargs,
                )

        return OpticalQuerySweep

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
                if param == "trS":
                    extra["augment_kwargs_fn"] = _knob_to_augment_kwargs
                super().__init__(
                    sem_factory=outer._sem_factory,
                    da_factory=outer._da_factory,
                    poly_transform=outer._poly_factory(),
                    test_fraction=OPTICAL_CONFIG.test_fraction,
                    epsilon_true=OPTICAL_CONFIG.epsilon_true,
                    default_gamma=OPTICAL_CONFIG.gamma,
                    default_epsilon=OPTICAL_CONFIG.epsilon,
                    experiment_name=EXPERIMENT_NAME,
                    **extra,
                    **kwargs,
                )

        return ConfiguredSweep
