"""
US state cigarette demand experiment using generic runners.
"""

from functools import partial

import numpy as np
from loguru import logger

from src.data_augmentors.cigarettes import ScaleTranslation as DA
from src.experiments.base import ExperimentOrchestrator
from src.experiments.configs import CIGARETTE_CONFIG, EPS_TOL, QUERY_GAMMA, MethodRegistry
from src.experiments.generic_runner import STRATEGIES, GenericQuerySweep
from src.oracle import epsilon_star, preserve_rng
from src.sem.cigarettes import CigaretteSEM as SEM
from src.sem.cigarettes import V, build_design

EXPERIMENT_NAME = "cigarettes"

# m-sweep holds n fixed here (PLAN 5.5). 10 whole state histories: a multiple of
# the 50-year history length, so the folds get whole states like every other draw.
FOLD_SWEEP_SAMPLES: int = 500

# eps* is a defect of the TARGET under the DA, and under the translation the defect
# is w = -c (v-hat' b) with b homogeneous on both paths, so it is 0 whatever c comes
# out of the draw. One seeded draw is therefore the population value, not a sample
# of it -- unlike the optical device, where eps* is an RMS over image draws and has
# to be pooled. Fixing the seed keeps the budget a function of (SEM, DA) alone.
EPSILON_STAR_SEED: int = 0


# =============================================================================
# ORCHESTRATOR
# =============================================================================


class CigaretteOrchestrator(ExperimentOrchestrator):
    """Orchestrator for the cigarette panel experiments."""

    def __init__(
        self,
        target: str = "iv",
        spec: str = "t3",
        anchor: str = "own-tax",
        da_amplitude: float = 1.0,
        sliver: bool = False,
        **kwargs,
    ):
        """
        Args:
            target: `iv` = the restricted-IV point (the tax-moment program solved
                subject to v'b = 0, so Z can be discarded); `plasmode` = the real
                design with a synthetic homogeneous h_* and known confounding.
            spec: state FE plus a trend of this order.
            anchor: which excise columns instrument which regressors.
            da_amplitude: DA amplitude along v, in units of sd(X . v-hat).
            sliver: target the leaky-IV set rather than the restricted point.
        """
        self.target, self.spec, self.anchor = target, spec, anchor
        self.sliver = bool(sliver)
        self._epsilon_star = None
        self.toggles = dict(
            recalibrate=kwargs.get("recalibrate", True),
            pad=kwargs.get("pad", False),
            clipy=kwargs.get("clipy", True),
            n_jobs=kwargs.get("n_jobs", 1),
            mean_match=kwargs.get("mean_match", True),
        )
        toggles = self.toggles

        # The DA amplitude is a DATASET constant and the query runner calls
        # `da_factory()` with no arguments, so it cannot come off a SEM: the panel
        # is read once here, exactly as the optical orchestrator builds its feature
        # transform from a throwaway SEM.
        design = build_design(SEM.panel(), spec=spec, anchor=anchor)
        self._amplitude = float(da_amplitude) * float(np.std(design.X @ (V / np.linalg.norm(V))))
        logger.info(f"DA amplitude {da_amplitude:g} x sd(X . v-hat) = {self._amplitude:.6f} at {spec}.")

        # DECLARED, per spec, and only the query panel ever sees it: the sweeps
        # solve at gamma*(target) through `ParamSweepRunner.fit_gamma`. See
        # `QUERY_GAMMA` for why the query budget is declared rather than measured.
        self.gamma = QUERY_GAMMA[spec]
        epsilon = self._epsilon_budget(CIGARETTE_CONFIG.epsilon)

        class CigaretteRegistry(MethodRegistry):
            @staticmethod
            def build_methods(names):
                return MethodRegistry.build_methods(names, gamma=QUERY_GAMMA[spec], epsilon=epsilon, **toggles)

        super().__init__(EXPERIMENT_NAME, CigaretteRegistry(), **kwargs)

    # ---------------------------------------------------------------- factories

    def _sem_factory(self, bootstrap: bool = False):
        """Factory for creating SEM instances.

        `bootstrap` is the replicate mechanism and is bound PER RUNNER (SS6): the
        query figures and the coefficient table are fit on the full panel, the
        sweeps on state-cluster bootstrap replicates of it. `pool` is the whole
        panel either way, so h_*, gamma* and eps* do not move between them.
        """
        return SEM(
            spec=self.spec,
            target=self.target,
            anchor=self.anchor,
            bootstrap=bootstrap,
            sliver=self.sliver,
            gamma_z=CIGARETTE_CONFIG.gamma_z,
            gamma_true=CIGARETTE_CONFIG.gamma_true,
            confound_direction=CIGARETTE_CONFIG.confound_direction,
            outcome_noise_std=CIGARETTE_CONFIG.outcome_noise_std,
        )

    def _da_factory(self, sem=None, append: str | None = None):
        """Factory for creating DA instances. The direction and the amplitude are
        dataset constants, so neither argument is needed; `append` is accepted
        because the robustness sweep passes it where a chain has components, and
        this DA has none."""
        return DA(V, std=self._amplitude)

    def _oracle_pieces(self):
        """(sem, da, features) for the budget estimators, built once."""
        return self._sem_factory(), self._da_factory(), None

    def measured_epsilon_star(self) -> float:
        """eps* for THIS (SEM, DA): the L2 defect on the panel. Cached."""
        if self._epsilon_star is None:
            sem, da, features = self._oracle_pieces()
            with preserve_rng():
                np.random.seed(EPSILON_STAR_SEED)
                self._epsilon_star = float(epsilon_star(sem, da, X=sem.X, features=features))
            logger.info(f"Cigarette eps*: {self._epsilon_star:.3e}")
        return self._epsilon_star

    def _epsilon_budget(self, configured: float | None, tol: float = EPS_TOL) -> float:
        """PI+INV's ASSUMED invariance bound -- the SS3.1 constraint budget.

        `None` means the measured eps* plus `tol`, which is what the optical
        orchestrator does and for the same reason: a budget under eps* excludes
        h_* from the PI+INV set, which costs validity rather than width. Here eps*
        is 0 by construction on both targets (h_* is exactly homogeneous and the DA
        translates along v), so the budget IS the tolerance and nothing is padded
        to cover a defect that does not exist.
        """
        if configured is not None:
            return float(configured)
        return self.measured_epsilon_star() + tol

    # ------------------------------------------------------------------ runners

    def get_query_runner_cls(self) -> type[GenericQuerySweep]:
        """Return query sweep runner."""
        outer = self

        class CigaretteQuerySweep(GenericQuerySweep):
            def __init__(inner_self, **kwargs):
                super().__init__(
                    sem_factory=outer._sem_factory,  # bootstrap=False: the panel itself
                    da_factory=outer._da_factory,
                    poly_transform=None,
                    method_factory=outer.build_methods,
                    default_gamma=QUERY_GAMMA[outer.spec],
                    default_epsilon=outer._epsilon_budget(CIGARETTE_CONFIG.query_epsilon, tol=CIGARETTE_CONFIG.eps_tol),
                    eps_tol=CIGARETTE_CONFIG.eps_tol,
                    # the panel is sigma-normalised, so gamma is already in the
                    # paper's units and sigma-hat^2 is 1: nothing to rescale
                    raw_gamma=False,
                    **kwargs,
                )

        return CigaretteQuerySweep

    def build_methods(self, gamma: float, epsilon: float, epsilon_iv=None, n_jobs=None, rho=1.0):
        """Methods at explicit (per-experiment) budgets. `n_jobs` overrides the
        toggle -- perf needs serial models to time methods, not the harness."""
        toggles = self.toggles if n_jobs is None else {**self.toggles, "n_jobs": n_jobs}
        return MethodRegistry.build_methods(
            self.kwargs["methods"], gamma=gamma, epsilon=epsilon, epsilon_iv=epsilon_iv, rho=rho, **toggles
        )

    def get_sweep_runner_cls(self, param: str) -> type:
        """Configured strategy for one sweep parameter."""
        outer, Strategy = self, STRATEGIES[param]
        # the sweep replicates: a state-cluster bootstrap under `iv` (a resampling
        # rate against a fixed target), the fresh outcome draw under `plasmode`
        sem_factory = partial(outer._sem_factory, bootstrap=(outer.target == "iv"))

        class ConfiguredSweep(Strategy):
            def __init__(inner_self, **kwargs):
                extra = {}
                if param == "m":
                    extra["n_samples_override"] = FOLD_SWEEP_SAMPLES
                super().__init__(
                    sem_factory=sem_factory,
                    da_factory=outer._da_factory,
                    poly_transform=None,
                    test_fraction=CIGARETTE_CONFIG.test_fraction,
                    default_gamma=QUERY_GAMMA[outer.spec],
                    default_epsilon=outer._epsilon_budget(CIGARETTE_CONFIG.epsilon),
                    experiment_name=EXPERIMENT_NAME,
                    **extra,
                    **kwargs,
                )

        return ConfiguredSweep
