"""
Optical device experiment using generic runners.
Dramatically reduced code duplication.
"""

import numpy as np
from loguru import logger
from sklearn.preprocessing import PolynomialFeatures

from src.data_augmentors.optical_device import OpticalDeviceDA as DA
from src.experiments.base import ExperimentOrchestrator
from src.experiments.configs import EPS_TOL, OPTICAL_CONFIG, MethodRegistry
from src.experiments.generic_runner import STRATEGIES, GenericQuerySweep
from src.oracle import PAD_QUANTILE, epsilon_pad_star, epsilon_star, preserve_rng
from src.sem.optical_device import OpticalDeviceSEM as SEM

EXPERIMENT_NAME = "optical_device"

# m-sweep holds n fixed here (PLAN 5.5)
FOLD_SWEEP_SAMPLES: int = 512
# eps* is an RMS over one DA draw, so it carries draw noise; pool this many draws
# (in the SQUARE, which is what an RMS averages) so the budget does not wobble
# between runs. The X it is evaluated on is the WHOLE pool, not a resample of it:
# the optical pool has 1000 rows and `CALIBRATION_SAMPLES` is 2048, so the default
# path would draw them WITH REPLACEMENT and add resampling noise for nothing.
EPSILON_STAR_DRAWS: int = 8
# Drawn from a FIXED stream, seeded PER DRAW. Per draw is not a flourish:
# `_invariance_signal` restores the global RNG on exit, so seeding once and looping
# evaluates the SAME realisation `EPSILON_STAR_DRAWS` times -- pooling nothing at
# `EPSILON_STAR_DRAWS` times the cost. Fixing the stream at all is what makes the
# budget a function of (SEM, DA, features) alone rather than of whatever state the
# caller happened to be in; `preserve_rng` hands that state back untouched.
EPSILON_STAR_SEED: int = 0


def _knob_to_augment_kwargs(strength: float) -> dict[str, float]:
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
        self._epsilon_star = None
        self._epsilon_pad = None
        self.toggles = dict(
            calibrate=kwargs.get("calibrate", False),
            pad=kwargs.get("pad", False),
            clipy=kwargs.get("clipy", True),
            n_jobs=kwargs.get("n_jobs", 1),
            mean_match=kwargs.get("mean_match", True),
        )
        toggles = self.toggles
        epsilon = self._epsilon_budget(OPTICAL_CONFIG.epsilon)
        pad_epsilon = self._pad_budget(OPTICAL_CONFIG.pad_epsilon)

        # Create registry with optical-specific parameters
        class OpticalRegistry(MethodRegistry):
            @staticmethod
            def build_methods(names):
                return MethodRegistry.build_methods(
                    names, gamma=OPTICAL_CONFIG.gamma, epsilon=epsilon, pad_epsilon=pad_epsilon, **toggles
                )

        super().__init__(EXPERIMENT_NAME, OpticalRegistry(), **kwargs)

    def _sem_factory(self):
        """Factory for creating SEM instances."""
        return SEM(experiment=OPTICAL_CONFIG.dataset_index, ground_truth=OPTICAL_CONFIG.ground_truth_model)

    def _oracle_pieces(self):
        """(sem, da, features) for the budget estimators, built once."""
        sem, da = self._sem_factory(), self._da_factory()
        return sem, da, self._poly_factory().fit(sem.X).transform

    def measured_epsilon_star(self) -> float:
        """eps* for THIS (SEM, DA, features): the L2 defect, pooled over
        `EPSILON_STAR_DRAWS` independent draws on the full pool. Cached."""
        if self._epsilon_star is None:
            sem, da, features = self._oracle_pieces()
            with preserve_rng():
                squares = []
                for draw in range(EPSILON_STAR_DRAWS):
                    np.random.seed(EPSILON_STAR_SEED + draw)  # per draw; see the constant
                    squares.append(epsilon_star(sem, da, X=sem.X, features=features) ** 2)
            self._epsilon_star = float(np.sqrt(np.mean(squares)))
            logger.info(f"Optical eps* over {EPSILON_STAR_DRAWS} draws: {self._epsilon_star:.6f}")
        return self._epsilon_star

    def measured_epsilon_pad(self) -> float:
        """Thm. 3.A's epsilon for this pair: the POINTWISE defect, not the L2 one.
        See `oracle.epsilon_pad_star` for why it is a high quantile of |W| and not
        a sup. Cached."""
        if self._epsilon_pad is None:
            sem, da, features = self._oracle_pieces()
            self._epsilon_pad = float(epsilon_pad_star(sem, da, X=sem.X, features=features))
            logger.info(f"Optical padding eps (q{PAD_QUANTILE:g} of |W|): {self._epsilon_pad:.6f}")
        return self._epsilon_pad

    def _epsilon_budget(self, configured: float | None) -> float:
        """PI+INV's ASSUMED invariance bound -- the SS3.1 constraint budget.

        `None` means take the measured one. SS3.1 constrains E_inv(h) <= eps^2, and
        `epsilon_star` is exactly that functional evaluated at h_*, so eps* +
        EPS_TOL admits h_* by construction while any smaller budget excludes it --
        an invalid interval, not a tight one. Whether the published 2**-2 was
        smaller depends on the augmentation in force (measured: 0.2121 for
        `rotation > gaussian-noise`, which config.yaml ships, but 0.2600 for
        `all`), which is exactly why this is measured rather than declared. A
        configured float is passed through unchanged so a number can still be
        pinned; `a30` gates the relation either way.
        """
        if configured is not None:
            return float(configured)
        return self.measured_epsilon_star() + EPS_TOL

    def _pad_budget(self, configured: float | None) -> float:
        """Thm. 3.A's epsilon. `None` takes the measured pointwise budget; a float
        pins it. NOTE this is ~3x the constraint budget on this device (0.691 vs
        0.212): the two are different norms of the same W, and padding by the L2
        one is not the guarantee Thm. 3.A states."""
        if configured is not None:
            return float(configured)
        return self.measured_epsilon_pad() + EPS_TOL

    def _da_factory(self, sem=None):
        """Factory for creating DA instances."""
        return DA(self.augmentation)

    def _poly_factory(self):
        """Factory for creating polynomial transformer."""
        # Get degree from a sample SEM
        sem = self._sem_factory()
        return PolynomialFeatures(sem.poly_degree, include_bias=False)

    def get_query_runner_cls(self) -> type[GenericQuerySweep]:
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
                    default_epsilon=self._epsilon_budget(OPTICAL_CONFIG.query_epsilon),
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

    def get_sweep_runner_cls(self, param: str) -> type:
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
                    default_epsilon=outer._epsilon_budget(OPTICAL_CONFIG.epsilon),
                    experiment_name=EXPERIMENT_NAME,
                    **extra,
                    **kwargs,
                )

        return ConfiguredSweep
