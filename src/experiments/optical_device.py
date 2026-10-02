"""
Optical device experiment using generic runners.
Dramatically reduced code duplication.
"""

from functools import partial

import numpy as np
from loguru import logger
from sklearn.preprocessing import PolynomialFeatures

from src.data_augmentors.optical_device import ALL_AUGMENTATIONS, P
from src.data_augmentors.optical_device import OpticalDeviceDA as DA
from src.experiments.base import ExperimentOrchestrator
from src.experiments.configs import EPS_TOL, OPTICAL_CONFIG, MethodRegistry
from src.experiments.generic_runner import STRATEGIES, GenericQuerySweep
from src.oracle import epsilon_star, epsilon_star_q95, preserve_rng
from src.sem.optical_device import OpticalDeviceSEM as SEM

EXPERIMENT_NAME = "optical_device"

# the query panel's eps* is an RMS over one DA draw, so it carries draw noise; pool
# this many draws
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
    """Map the [0, 1] strength knob onto the optical DA's own parameters.

    s is the permutation probability of every component (flips, rotation and
    random-permutation alike), overriding the block's `augmentation_p`, and the
    multiple of translate's step (0.5 s std(X)).
    `noise_coeff` reaches a chain that still names gaussian-noise and is inert
    otherwise."""
    return {
        "p": float(strength),
        "noise_coeff": float(np.sqrt(0.1 * strength)),  # up to 0.1 Var(X)
    }


def _with_component(chain: str, component: str) -> str:
    """`chain` with `component` appended, unless the chain already carries it."""
    present = list(ALL_AUGMENTATIONS) if chain == "all" else chain.replace(" ", "").split(">")
    return chain if component in present else f"{chain} > {component}"


# =============================================================================
# ORCHESTRATOR
# =============================================================================


class OpticalOrchestrator(ExperimentOrchestrator):
    """Orchestrator for optical device experiments."""

    def __init__(self, augmentation: str, augmentation_p: float = P, **kwargs):
        """
        Initialize optical orchestrator. `augmentation_p` is the chain's rotation /
        flip probability (the dataset block's key; `P` when omitted).
        """
        self.augmentation = augmentation
        self.augmentation_p = float(augmentation_p)
        self._epsilon_star = {}  # epsilon_quantile -> eps*
        self.toggles = dict(
            recalibrate=kwargs.get("recalibrate", True),
            pad=kwargs.get("pad", False),
            clipy=kwargs.get("clipy", True),
            n_jobs=kwargs.get("n_jobs", 1),
            mean_match=kwargs.get("mean_match", True),
            gamma_n_alpha=kwargs.get("gamma_n_alpha", 0.0),
        )
        toggles = self.toggles
        epsilon = self._epsilon_budget(OPTICAL_CONFIG.epsilon, quantile=OPTICAL_CONFIG.epsilon_quantile)

        # Create registry with optical-specific parameters
        class OpticalRegistry(MethodRegistry):
            @staticmethod
            def build_methods(names):
                return MethodRegistry.build_methods(names, gamma=OPTICAL_CONFIG.gamma, epsilon=epsilon, **toggles)

        super().__init__(EXPERIMENT_NAME, OpticalRegistry(), **kwargs)

    def _sem_factory(self, epsilon_quantile: float | None = None, device: int | None = None):
        """Factory for creating SEM instances; `dataset_index` unless a device is
        named. `epsilon_quantile` rides on the SEM to `oracle.epsilon_star`
        (None = RMS); the sweeps and the query each bind their own from OPTICAL_CONFIG."""
        sem = SEM(
            experiment=OPTICAL_CONFIG.dataset_index if device is None else int(device),
            ground_truth=OPTICAL_CONFIG.ground_truth_model,
            intercept=self.toggles["mean_match"],
        )
        sem.epsilon_quantile = epsilon_quantile
        return sem

    @staticmethod
    def sweep_devices() -> tuple[int, ...]:
        """The device of each sweep experiment (`OpticalDeviceConfig.sweep_devices`):
        `dataset_index` first, then every loaded device not in `excluded_devices`,
        ascending."""
        if OPTICAL_CONFIG.sweep_devices is not None:
            return tuple(int(d) for d in OPTICAL_CONFIG.sweep_devices)
        first = OPTICAL_CONFIG.dataset_index
        excluded = {int(d) for d in OPTICAL_CONFIG.excluded_devices}
        return (first, *(d for d in sorted(SEM.dataset()) if d != first and d not in excluded))

    def _oracle_pieces(self, epsilon_quantile: float | None = None):
        """(sem, da, features) for the budget estimators, built once."""
        sem, da = self._sem_factory(epsilon_quantile), self._da_factory()
        return sem, da, self._poly_factory().fit(sem.X).transform

    def measured_epsilon_star(self, epsilon_quantile: float | None = None) -> float:
        """eps* for THIS (SEM, DA, features) on the full pool: the defect's RMS
        pooled over `EPSILON_STAR_DRAWS` independent draws (the query panel), or
        with an `epsilon_quantile` the sweeps' shared q0.95 reading
        (`oracle.epsilon_star_q95`). Cached per quantile."""
        if epsilon_quantile not in self._epsilon_star:
            sem, da, features = self._oracle_pieces(epsilon_quantile)
            if epsilon_quantile is not None:
                self._epsilon_star[epsilon_quantile] = epsilon_star_q95(sem, da, X=sem.X, features=features)
                logger.info(f"Optical eps* (the q0.95 reading of |W|): {self._epsilon_star[epsilon_quantile]:.6f}")
                return self._epsilon_star[epsilon_quantile]
            with preserve_rng():
                squares = []
                for draw in range(EPSILON_STAR_DRAWS):
                    np.random.seed(EPSILON_STAR_SEED + draw)  # per draw; see the constant
                    squares.append(epsilon_star(sem, da, X=sem.X, features=features) ** 2)
            self._epsilon_star[epsilon_quantile] = float(np.sqrt(np.mean(squares)))
            logger.info(
                f"Optical eps* (RMS of |W|) over {EPSILON_STAR_DRAWS} draws: {self._epsilon_star[epsilon_quantile]:.6f}"
            )
        return self._epsilon_star[epsilon_quantile]

    def _epsilon_budget(
        self, configured: float | None, tol: float = EPS_TOL, *, quantile: float | None = None
    ) -> float:
        """PI+INV's ASSUMED invariance bound -- the SS3.1 constraint budget.

        `None` means take the measured one. SS3.1 constrains E_inv(h) <= eps^2, and
        `epsilon_star` is exactly that functional evaluated at h_*, so eps* +
        `tol` (EPS_TOL) admits h_* by construction while any smaller budget excludes it --
        an invalid interval, not a tight one. Whether the published 2**-2 was
        smaller depends on the augmentation in force (measured: 0.2121 for
        `rotation > gaussian-noise`, which config.yaml ships, but 0.2600 for
        `all`), which is exactly why this is measured rather than declared. A
        configured float is passed through unchanged so a number can still be
        pinned; `a30` gates the relation either way. `quantile` picks the norm of
        W the measured eps* is (`OpticalDeviceConfig.epsilon_quantile`).
        """
        if configured is not None:
            return float(configured)
        return self.measured_epsilon_star(quantile) + tol

    def _da_factory(self, sem=None, append: str | None = None):
        """Factory for creating DA instances. `append` adds one component to the
        configured chain unless it is already there (`all` carries every one);
        only the robustness sweep passes it."""
        chain = self.augmentation if append is None else _with_component(self.augmentation, append)
        return DA(chain, p=self.augmentation_p)

    def _poly_factory(self):
        """Factory for creating polynomial transformer."""
        # Get degree from a sample SEM
        sem = self._sem_factory()
        return PolynomialFeatures(sem.poly_degree, include_bias=False)

    def get_query_runner_cls(self) -> type[GenericQuerySweep]:
        """Return query sweep runner."""

        class OpticalQuerySweep(GenericQuerySweep):
            def __init__(inner_self, **kwargs):
                quantile = OPTICAL_CONFIG.query_epsilon_quantile
                super().__init__(
                    sem_factory=partial(self._sem_factory, epsilon_quantile=quantile),
                    da_factory=self._da_factory,
                    poly_transform=self._poly_factory(),
                    epsilon_true=OPTICAL_CONFIG.epsilon_true,
                    method_factory=self.build_methods,
                    default_gamma=OPTICAL_CONFIG.gamma,
                    default_epsilon=self._epsilon_budget(OPTICAL_CONFIG.query_epsilon, quantile=quantile),
                    raw_gamma=True,
                    **kwargs,
                )

        return OpticalQuerySweep

    def build_methods(
        self,
        gamma: float,
        epsilon: float,
        n_jobs=None,
        rho=1.0,
    ):
        """Methods at explicit (per-experiment) budgets. `n_jobs` overrides the
        toggle -- perf needs serial models to time methods, not the harness."""
        toggles = self.toggles if n_jobs is None else {**self.toggles, "n_jobs": n_jobs}
        return MethodRegistry.build_methods(
            self.kwargs["methods"],
            gamma=gamma,
            epsilon=epsilon,
            rho=rho,
            **toggles,
        )

    def get_sweep_runner_cls(self, param: str) -> type:
        """Configured strategy for one sweep parameter."""
        outer, Strategy = self, STRATEGIES[param]
        quantile = OPTICAL_CONFIG.epsilon_quantile

        class ConfiguredSweep(Strategy):
            def __init__(inner_self, **kwargs):
                extra = {}
                if param == "omega":
                    extra["augment_kwargs_fn"] = _knob_to_augment_kwargs
                super().__init__(
                    sem_factory=partial(outer._sem_factory, epsilon_quantile=quantile),
                    da_factory=outer._da_factory,
                    poly_transform=outer._poly_factory(),
                    test_fraction=OPTICAL_CONFIG.test_fraction,
                    epsilon_true=OPTICAL_CONFIG.epsilon_true,
                    default_gamma=OPTICAL_CONFIG.gamma,
                    default_epsilon=outer._epsilon_budget(OPTICAL_CONFIG.epsilon, quantile=quantile),
                    experiment_name=EXPERIMENT_NAME,
                    **extra,
                    **kwargs,
                )

            def make_sem(inner_self, experiment_index: int):
                """Experiment j runs on its own device: replicates span the
                recorded devices rather than redraw one of them."""
                devices = outer.sweep_devices()
                if inner_self.n_experiments > len(devices):
                    raise ValueError(f"n_experiments {inner_self.n_experiments} exceeds the {len(devices)} devices")
                return outer._sem_factory(epsilon_quantile=quantile, device=devices[experiment_index])

            def poly_for_sem(inner_self, sem):
                """The selected degree differs by device (1 or 2)."""
                return PolynomialFeatures(sem.poly_degree, include_bias=False)

        return ConfiguredSweep
