"""
do-MNIST experiment: CopSens partial identification on raw pixels.

Two things differ from the linear experiments and shape everything here.

1. **The estimand is a trained net.** h_* is not linear in pixels, so `sem.solution`
   does not exist and `sem.f` is a CNN fitted on interventional draws. Sound because
   E[Y | X=x, do] = h_*(x) exactly. It is COUPLED to the ERM -- same images, same
   init_seed -- so the estimation error the two share cancels in |mu_erm - h^|,
   whose p95 is what sets the PI width.

2. **The outcome nets are PREFIT.** SOURCE trains one ERM on X and one on GX per
   replicate and hands the appropriate one to every PI variant. Refitting inside
   each `_fit` would cost four extra CNN fits, train on the n_pi subset, and break
   the ERM/PI matching. That inverts the usual build order: methods can only be
   built AFTER the data exists.

Phase 1 is the query sweep + perf. The sweep grids are Phase 2 -- see
`get_sweep_runner_cls` for what each one still needs.
"""
import numpy as np
from loguru import logger
from typing import Type, Dict, Optional

from src.sem.do_mnist import DoMNISTSEM as SEM
from src.data_augmentors.do_mnist import DoMNISTDA as DA
from src.methods.regression import GradientDescentERM
from src.experiments.base import ExperimentOrchestrator
from src.experiments.generic_runner import GenericQuerySweep, GenericParamSweep, STRATEGIES
from src.experiments.configs import MethodRegistry, DOMNIST_CONFIG
from src.experiments.utils import save
from src.experiments.utils.constants import SUBDIR_QUERY
from src.experiments.utils.plotting import create_digit_sweep_plot

EXPERIMENT_NAME = 'do_mnist'

# test draws must not collide with the train stream
TEST_SEED_OFFSET: int = 77_000


class Flatten:
    """Duck-types PolynomialFeatures, so images reach the methods through the
    existing `poly_transform` hook and no base class has to learn about pixels."""

    def fit_transform(self, X):
        X = np.asarray(X)
        # reshape on a C-contiguous array is a VIEW, which is what keeps X_raw + X
        # at ~2.8 GB rather than 5.6 GB at 1.2M draws. Assert it, so a future
        # slicing change cannot silently double the footprint.
        assert X.flags['C_CONTIGUOUS'], 'Flatten needs C-contiguous input to stay a view'
        return X.reshape(len(X), -1)


def train_pair(X, GX, y, init_seed, net='domnist-fast', **train_kw):
    """The matched ERM / DA+ERM pair: same architecture, same init_seed, same batch
    order, differing only in their inputs."""
    logger.info(f'do-mnist: training the ERM pair on {len(X):,} draws')
    return {'X': GradientDescentERM(net).fit(X, y, init_seed=init_seed, **train_kw),
            'GX': GradientDescentERM(net).fit(GX, y, init_seed=init_seed, **train_kw)}


def pi_subset(n_total: int, n_pi: int, seed: int) -> np.ndarray:
    """Rows the latent model and constraints are fit on. Drawn once per experiment,
    so PI and DA+PI see the same rows."""
    return np.random.default_rng(seed).choice(n_total, min(n_pi, n_total), replace=False)


def log_vacuous(bounds: Dict[str, np.ndarray]):
    """An all-INFEASIBLE method draws nothing, so the figure shows only a legend
    entry. Say so, or it reads as a plotting bug."""
    for name, prediction in bounds.items():
        prediction = np.asarray(prediction)
        if prediction.ndim == 3 and not np.isfinite(prediction).any():
            logger.warning(f'{name}: INFEASIBLE at every query -- it contributes a '
                           'legend entry and no band. Check the floor/budget pair '
                           'logged above.')


def log_nesting(models: Dict[str, object], bounds: Dict[str, np.ndarray]):
    """R5: the SLSQP multi-start is non-convex, so PI_INV subset PI is not
    guaranteed. Logged, never raised -- a violation is information about the
    optimiser, not a reason to discard the run."""
    if not {'PI', 'PI_INV'} <= set(bounds):
        return
    width = {k: bounds[k][..., 1] - bounds[k][..., 0] for k in ('PI', 'PI_INV')}
    violations = int(np.nansum(width['PI_INV'] > width['PI'] + 1e-9))
    if violations:
        logger.warning(f'PI_INV wider than PI at {violations}/{len(width["PI"])} '
                       'queries: the non-convex multi-start missed an optimum.')


# =============================================================================
# QUERY SWEEP
# =============================================================================

class DoMNISTQuerySweep(GenericQuerySweep):
    """GenericQuerySweep builds methods BEFORE data exists; prefit nets need the
    reverse order, so the early build is suppressed and redone once nets exist."""

    def __init__(self, method_factory=None, sem_test_factory=None, n_pi=60_000,
                 default_gamma=0.1, default_epsilon=0.05, **kwargs):
        # GenericQuerySweep takes these as LOCALS and never stores them, unlike
        # GenericParamSweep. Keep our own copies.
        self.default_gamma, self.default_epsilon = default_gamma, default_epsilon
        self.sem_test_factory, self.n_pi = sem_test_factory, n_pi
        self.nets = None
        super().__init__(method_factory=None, **kwargs)      # suppress the early build

        if method_factory is not None:
            self.methods = method_factory(
                gamma=self.default_gamma, epsilon=self.default_epsilon,
                epsilon_iv=self.default_epsilon, outcome_models=self.nets,
            )

    def _load_data(self):
        """ONE paired draw, which the SEM's target net drew identically (same N,
        same seed) -- that is what couples the estimand to the ERM."""
        X_raw, y, _ = self.sem.sample_paired(self.n_samples, seed=self.sem.seed)
        GX_raw, G = self.da(X_raw)

        # nets see the FULL draw; the PI machinery only the n_pi subset
        self.nets = train_pair(self.poly.fit_transform(X_raw),
                               self.poly.fit_transform(GX_raw), y,
                               init_seed=self.sem.seed, net=self.sem.net,
                               **(self.hyperparameters or {}))

        keep = pi_subset(len(X_raw), self.n_pi, self.seed)
        return X_raw[keep], GX_raw[keep], y[keep], G[keep]

    def get_sweep_values(self) -> np.ndarray:
        """The 10 frozen digit exemplars, from the TEST SEM -- the nets never saw
        these images. GenericQuerySweep would run a PCA over (N,3,14,14) here,
        which is a crash rather than a bad plot."""
        sem_test = self.sem_test_factory()
        self.exemplars_, self.digits_ = sem_test.exemplars(
            DOMNIST_CONFIG.exemplar_seed)
        # same digits and tints at full resolution: `subsample` is a modelling
        # choice, and the figure has no reason to inherit it
        self.exemplar_images_, _ = sem_test.exemplars(
            DOMNIST_CONFIG.exemplar_seed, subsample=1)
        return self.poly.fit_transform(self.exemplars_)


# =============================================================================
# PARAMETER SWEEP MIXIN
# =============================================================================

class DoMNISTMixin:
    """Prefit nets + the n_pi subset, composed OVER a sweep strategy.

    A mixin and not a base class: perf routes through `STRATEGIES['m']`, which has
    its own `generate_data`, so this has to sit ahead of it in the MRO rather than
    replace it.
    """

    def __init__(self, sem_test_factory=None, n_pi=60_000, n_queries=512, **kwargs):
        self.sem_test_factory = sem_test_factory
        self.n_pi, self.n_queries = n_pi, n_queries
        self._nets = {}
        super().__init__(**kwargs)

    def setup_sems_and_das(self):
        super().setup_sems_and_das()
        # the test SEM holds the held-out MNIST images; queries are drawn from it
        self.sems_test = [self.sem_test_factory() for _ in range(self.n_experiments)]

    # -------------------------------------------------------- per-experiment policy

    def fit_gamma(self, experiment_index: int) -> float:
        """gamma* (= bias_sq) is a Lemma-2 radius in FUNCTION space; CopSens's budget
        lives in a 32-d latent index space. Not comparable, so the config value is
        used as-is and gamma_star is recorded unconsumed."""
        return self.default_gamma

    def fit_epsilon(self, experiment_index: int, step_index: int = 0) -> float:
        """A MODELLING ASSUMPTION, not estimable. The oracle eps* here measures the
        TARGET NET's approximation error, not h_*'s invariance defect -- which is 0
        for these ops, since none of them can change E[Y|do(x)]."""
        return self.default_epsilon

    def fit_epsilon_iv(self, experiment_index: int) -> Optional[float]:
        """Same reasoning as fit_epsilon: eps_iv_star is not h_*'s defect either."""
        return self.default_epsilon

    def method_kwargs(self, experiment_index: int) -> Dict[str, object]:
        return {'outcome_models': self._nets[experiment_index]}

    # ------------------------------------------------------------- base sample

    def _draw_base(self, experiment_index: int, n_samples=None):
        """Paired draw -> train the pair on the FULL draw -> subset for the PI.

        The strategy re-augments the subset for the PI fits, so DA+ERM saw a
        different DA realisation of the same images than DA+PI's GX. Statistically
        the same draw; ~1 s over 60k images, against a full second pass over 1.2M.
        """
        sem = self.sems[experiment_index]
        n_total = self.n_samples if n_samples is None else int(n_samples)

        X_raw, y, _ = sem.sample_paired(n_total, seed=sem.seed)
        GX_raw, _ = self.das[experiment_index](X_raw)
        self._nets[experiment_index] = train_pair(
            self.apply_transform(X_raw), self.apply_transform(GX_raw), y,
            init_seed=sem.seed, net=sem.net, **(self.hyperparameters or {}))
        del GX_raw

        keep = pi_subset(len(X_raw), self.n_pi, self.seed + experiment_index)
        X_raw, y = X_raw[keep], y[keep]

        # queries: interventional draws from the HELD-OUT images. n_queries, not
        # test_fraction * n: at 1.2M that would be 120k queries x ~0.1 s = days.
        sem_test = self.sems_test[experiment_index]
        X_test_raw, _ = sem_test(
            N=self.n_queries, intervention=True,
            seed=self.seed + TEST_SEED_OFFSET + experiment_index)
        X_test = self.apply_transform(X_test_raw)

        return (X_raw, self.apply_transform(X_raw), y, X_test, sem.f(X_test))


# =============================================================================
# ORCHESTRATOR
# =============================================================================

class DoMNISTOrchestrator(ExperimentOrchestrator):
    """Orchestrator for do-MNIST experiments."""

    build_panel = False          # queries are digit exemplars, not a PC sweep

    def __init__(self, augmentation: str, gamma: float, epsilon: float,
                 n_pi: int = 60_000, n_queries: int = 512, n_components: int = 32,
                 net: str = 'domnist-fast', **kwargs):
        self.augmentation = augmentation
        self.gamma, self.epsilon = gamma, epsilon
        self.n_pi, self.n_queries = n_pi, n_queries
        self.n_components, self.net = n_components, net
        self.toggles = dict(
            calibrate=kwargs.get('calibrate', False),
            pad=kwargs.get('pad', False),
            clipy=kwargs.get('clipy', True),
            n_jobs=kwargs.get('n_jobs', 1),
        )
        toggles = self.toggles
        outer = self

        class DoMNISTRegistry(MethodRegistry):
            @staticmethod
            def build_methods(names):
                return MethodRegistry.build_methods(
                    names, gamma=gamma, epsilon=epsilon, backend='copsens',
                    n_components=outer.n_components, **toggles
                )

        super().__init__(EXPERIMENT_NAME, DoMNISTRegistry(), **kwargs)

    # ---------------------------------------------------------------- factories

    def _sem_factory(self, train: bool = True):
        return SEM(
            seed=self.kwargs['seed'], train=train, net=self.net,
            alpha=DOMNIST_CONFIG.alpha, beta=DOMNIST_CONFIG.beta,
            eta=DOMNIST_CONFIG.eta, subsample=DOMNIST_CONFIG.subsample,
            # the target net draws at exactly (n_samples, seed), which is what the
            # runners draw too -- that identity IS the coupling
            target_samples=self.kwargs['n_samples'],
            target_kw=dict(self.kwargs.get('hyperparameters') or {}),
        )

    def _sem_test_factory(self):
        return self._sem_factory(train=False)

    def _da_factory(self, sem=None):
        return DA(self.augmentation)

    def _poly_factory(self):
        return Flatten()

    def build_methods(self, gamma: float, epsilon: float, epsilon_iv=None,
                      n_jobs=None, outcome_models=None):
        """Methods at explicit budgets. `n_jobs` overrides the toggle -- perf needs
        serial models to time methods, not the harness."""
        toggles = self.toggles if n_jobs is None else {**self.toggles, 'n_jobs': n_jobs}
        return MethodRegistry.build_methods(
            self.kwargs['methods'], gamma=gamma, epsilon=epsilon,
            epsilon_iv=epsilon_iv, backend='copsens', outcome_models=outcome_models,
            n_components=self.n_components, **toggles
        )

    # ------------------------------------------------------------------ runners

    def get_query_runner_cls(self) -> Type[GenericQuerySweep]:
        outer = self

        class ConfiguredQuerySweep(DoMNISTQuerySweep):
            def __init__(inner, **kwargs):
                super().__init__(
                    sem_factory=outer._sem_factory,
                    sem_test_factory=outer._sem_test_factory,
                    da_factory=outer._da_factory,
                    poly_transform=outer._poly_factory(),
                    method_factory=outer.build_methods,
                    default_gamma=outer.gamma,
                    default_epsilon=outer.epsilon,
                    n_pi=outer.n_pi,
                    **kwargs
                )

        return ConfiguredQuerySweep

    def get_sweep_runner_cls(self, param: str) -> Type:
        if param != 'm':
            raise NotImplementedError(
                f'do_mnist {param} sweep is Phase 2. Blockers: '
                '(a) gamma -- needs a CopSens-scale gamma*, i.e. the population '
                'coverage bisection, since oracle.gamma_star is in partial-r2 units; '
                '(b) epsilon -- needs a trusted eps*, which the estimated target does '
                'not give (see fit_epsilon); (c) trS -- needs augment_kwargs_fn '
                'wired to DA.strength, and _augment_once must pass its seed through '
                'to the DA so common random numbers reach the torch draws; '
                '(d) n -- needs the nets retrained per step.')

        outer, Strategy = self, STRATEGIES[param]

        class ConfiguredSweep(DoMNISTMixin, Strategy):     # MRO: mixin first
            def __init__(inner, **kwargs):
                super().__init__(
                    sem_factory=outer._sem_factory,
                    sem_test_factory=outer._sem_test_factory,
                    da_factory=outer._da_factory,
                    poly_transform=outer._poly_factory(),
                    test_fraction=DOMNIST_CONFIG.test_fraction,
                    default_gamma=outer.gamma,
                    default_epsilon=outer.epsilon,
                    experiment_name=EXPERIMENT_NAME,
                    n_pi=outer.n_pi,
                    n_queries=outer.n_queries,
                    **kwargs
                )

        return ConfiguredSweep

    # --------------------------------------------------------------------- plot

    def _plot_query_sweep(self, runner, results):
        """x-axis is 10 digit exemplars, so thumbnails replace a numeric axis."""
        log_nesting(runner.methods, results)
        log_vacuous(results)
        digits = list(runner.digits_)

        save(np.asarray(digits), 'treatment_values', self.name, 'pkl',
             subdir=SUBDIR_QUERY)
        save(results, 'outcome_values', self.name, 'pkl', subdir=SUBDIR_QUERY)

        create_digit_sweep_plot(runner.exemplar_images_, results, labels=digits,
                                experiment=self.name)
