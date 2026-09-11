"""
Generic experiment runners that work with any SEM/DA combination.
Eliminates duplication between simulation and optical device experiments.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
from loguru import logger
from sklearn.model_selection import train_test_split

from src.experiments.base import ExperimentDataContext, ParamSweepRunner, QuerySweepRunner, SweepData
from src.experiments.configs import EPS_TOL, FLOOR_GUARD_R, ROBUSTNESS_EPSILON_TRUE, SPECTRUM_KEEP
from src.experiments.utils import radial_sweep_pcs
from src.experiments.utils.metrics import rho_hat, trace_S_over_k
from src.methods.sensitivity_models import constraint_floor
from src.oracle import (
    compute_oracle_parameters,
    epsilon_star,
    pool_oracles,
    preserve_rng,
    recalibrated_da_epsilon,
)

# per-experiment DA seed offset: common random numbers across a knob grid
CRN_OFFSET: int = 10_000
# `prepare_pair` pools this many seeded augmentation draws for a SEM with a fixed
# pool -- see the comment there, and `oracle.pool_oracles` for why rho in
# particular must not be a single draw.
ORACLE_POOL_DRAWS: int = 8
ORACLE_POOL_SEED: int = 0


# =============================================================================
# ORACLE MIXIN
# =============================================================================


class OracleMixin:
    """
    Prepares an (SEM, DA) pair: optionally sets the DA to a target epsilon,
    then records the oracle parameters. Nothing consumes them yet.
    """

    epsilon_true: float | None = None

    def prepare_pair(self, sem, da, features: Callable | None = None):
        pool = getattr(sem, "pool", None)
        if self.epsilon_true is not None:
            recalibrated_da_epsilon(
                sem=sem,
                da=da,
                epsilon_target=self.epsilon_true,
                X=None if pool is None else pool[0],  # the device's own rows, as below
                features=features,
            )

        # A recorded SEM is estimated on its OWN rows, not on a bootstrap of them
        # (see `SEM.pool`): the optical pool is 1000 rows and CALIBRATION_SAMPLES
        # is 2048, so the default draw resampled with replacement and made every
        # oracle number wobble with the RNG state for nothing.
        #
        # Fixing the rows leaves the AUGMENTATION draw as the only randomness, and
        # one draw is not the population quantity the figures annotate -- so pool
        # several, seeded (`_invariance_signal` restores the RNG on exit, hence a
        # seed per draw rather than consecutive calls). A SEM that draws fresh rows
        # every time already averages over replicates and is left alone.
        X, y = pool if pool is not None else (None, None)
        draws = ORACLE_POOL_DRAWS if pool is not None else 1
        with preserve_rng():
            oracles = []
            for draw in range(draws):
                if pool is not None:
                    np.random.seed(ORACLE_POOL_SEED + draw)
                oracles.append(
                    compute_oracle_parameters(
                        sem=sem,
                        da=da,
                        X=X,
                        y=y,
                        features=features,
                        mean_match=self.mean_match,
                    )
                )
        oracle = pool_oracles(oracles)
        logger.info(f"Oracle parameters: {oracle}")
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
        poly_transform: Callable | None = None,
        epsilon_true: float | None = None,
        method_factory: Callable | None = None,
        default_gamma: float = 1.0,
        default_epsilon: float = 2**-8,
        eps_tol: float = EPS_TOL,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # knife-edge tolerance on the oracle IV budget; the dataset configs set
        # it for the query sweep, the param sweeps keep EPS_TOL
        self.eps_tol = float(eps_tol)

        # Create SEM and DA
        self.sem = sem_factory()
        self.da = da_factory()
        self.poly = poly_transform
        self.epsilon_true = epsilon_true
        self.oracle = self.prepare_pair(self.sem, self.da, features=self._features)

        # stored, not local: the IV floor guard needs the ball these are used with.
        # Subclasses must FORWARD their budgets here rather than assigning before
        # super().__init__, or these defaults silently overwrite them.
        self.default_gamma = default_gamma
        self.default_epsilon = default_epsilon

        # Data FIRST, methods second. The IV budget is oracle-derived AND
        # floor-guarded, and the floor is a property of (GX, G, gamma), so it does
        # not exist until the draw does. Building methods first would silently skip
        # the guard. (`DoMNISTQuerySweep` passes method_factory=None and rebuilds
        # after its nets exist; that still works.)
        self.X_raw, self.GX_raw, self.y, self.G = self._load_data()

        # Apply polynomial transformation if provided
        if self.poly:
            self.X = self.poly.fit_transform(self.X_raw)
            self.GX = self.poly.fit_transform(self.GX_raw)
        else:
            self.X = self.X_raw
            self.GX = self.GX_raw

        # gamma/epsilon stay at the yaml defaults here (PLAN 7: the query sweep
        # never auto-sets them).
        if method_factory is not None:
            self.methods = method_factory(
                gamma=default_gamma,
                epsilon=default_epsilon,
                epsilon_iv=self.epsilon_iv,
                rho=self.fit_rho(),
            )

    def _load_data(self):
        """(X_raw, GX_raw, y, G). Override when the draw needs its own protocol."""
        X_raw, y = self.sem(N=self.n_samples)
        GX_raw, G = self.da(X_raw)
        return X_raw, GX_raw, y, G

    def fit_rho(self) -> float:
        """rho_hat of the one draw, as `ParamSweepRunner.fit_rho` (SS4.2)."""
        rho = rho_hat(self.X, self.GX, self.y, intercept=self.mean_match)
        if not np.isfinite(rho):
            logger.warning("rho_hat not computable; falling back to 1.")
            return 1.0
        if rho < 1.0:
            logger.warning(f"rho_hat {rho:.4f} < 1 (DPI says >= 1): sampling noise; the solvers read it as 1.")
        return float(rho)

    @property
    def epsilon_iv(self) -> float:
        """Oracle IV budget, off the knife edge (same guard as PI+INV), then raised
        to the constraint's own floor if it lands under it (`FLOOR_GUARD_R`)."""
        budget = getattr(self.oracle, "eps_iv_star", None)
        if budget is None or not np.isfinite(budget):
            logger.warning("eps_iv_star unavailable; IV budget falls back to the tolerance.")
            budget = 0.0
        budget = float(budget) + self.eps_tol

        if getattr(self, "G", None) is None or np.size(self.G) == 0:
            return budget
        try:
            floor = constraint_floor(
                self.GX,
                self.y,
                self.default_gamma,
                kind="iv",
                Z=self.G,
                mean_match=self.mean_match,
                rho=self.fit_rho(),
                recalibrate=self.recalibrate,
            )
        except Exception as error:
            logger.warning(f"epsilon_iv: constraint floor unavailable ({error}).")
            return budget

        if budget**2 >= floor:  # feasible: leave it exactly as it was
            return budget

        guarded = float(np.sqrt(FLOOR_GUARD_R * max(floor, 0.0)))
        logger.info(f"epsilon_iv: oracle {budget:.6g} is INFEASIBLE (floor {floor:.4g}); raising to {guarded:.6g}.")
        return guarded

    @property
    def _features(self) -> Callable | None:
        return self.poly.fit_transform if self.poly else None

    def get_sweep_values(self) -> np.ndarray:
        """Generate radial sweep points with optional polynomial features."""
        geometry = self.X_raw if not hasattr(self, "poly") else self.GX_raw
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
            oracle=self.oracle,
        )


# =============================================================================
# GENERIC PARAMETER SWEEP BASE
# =============================================================================


class GenericParamSweep(OracleMixin, ParamSweepRunner):
    """
    Shared machinery for every sweep strategy: per-experiment SEM/DA/oracle,
    the feature transform, and a base sample drawn ONCE per experiment so that
    only the swept parameter varies.
    """

    def __init__(
        self,
        sem_factory: Callable,
        da_factory: Callable,
        poly_transform: Callable | None = None,
        test_fraction: float = 0.1,
        epsilon_true: float | None = None,
        default_gamma: float = 1.0,
        default_epsilon: float = 2**-8,
        **kwargs,
    ):
        self.sem_factory = sem_factory
        self.da_factory = da_factory
        self.poly = poly_transform
        self.test_fraction = test_fraction
        self.epsilon_true = epsilon_true
        self.default_gamma = default_gamma
        self.default_epsilon = default_epsilon
        self._base = {}
        super().__init__(**kwargs)

    @property
    def _features(self) -> Callable | None:
        return self.poly.fit_transform if self.poly else None

    def setup_sems_and_das(self):
        """Setup SEMs and DAs for all experiments."""
        self.sems = [self.sem_factory() for _ in range(self.n_experiments)]
        self.das = [self.da_factory(sem) for sem in self.sems]
        self.oracles = [
            self.prepare_pair(sem, da, features=self._features) for sem, da in zip(self.sems, self.das, strict=False)
        ]

    def get_da(self, experiment_index: int):
        return self.das[experiment_index]

    def get_oracle(self, experiment_index: int):
        return self.oracles[experiment_index]

    def apply_transform(self, X: np.ndarray) -> np.ndarray:
        if self.poly:
            return self.poly.fit_transform(X)
        return X

    @property
    def finite_pool(self) -> bool:
        """Optical draws from a fixed pool and needs a train/test split."""
        return "OpticalDeviceSEM" in str(type(self.sems[0]))

    # ------------------------------------------------------- fixed base sample

    def _base_data(self, experiment_index: int, n_samples: int | None = None):
        """(X_train_raw, X_train, y_train, X_test, estimand), drawn once per j."""
        key = (experiment_index, n_samples)
        if key not in self._base:
            self._base[key] = self._draw_base(experiment_index, n_samples)
        return self._base[key]

    def _draw_base(self, experiment_index: int, n_samples: int | None = None):
        sem = self.sems[experiment_index]
        n_total = self.n_samples if n_samples is None else int(n_samples)

        if self.finite_pool:
            # finite pool: split unique instances; test set fixed across steps
            X_all, y_all = sem(N=n_total)
            X_train_raw, X_test_raw, y_train, _ = train_test_split(
                X_all, y_all, test_size=self.test_fraction, random_state=self.seed + experiment_index
            )
        else:
            # generator: interventional test set
            X_train_raw, y_train = sem(N=n_total)
            X_test_raw, _ = sem(N=int(self.test_fraction * n_total), intervention=True)

        X_test = self.apply_transform(X_test_raw)
        return (X_train_raw, self.apply_transform(X_train_raw), y_train, X_test, sem.f(X_test))

    def _augment_once(self, experiment_index: int, X_raw, **augment_kwargs):
        """One DA pass; common random numbers across a knob grid within an experiment."""
        with preserve_rng():
            np.random.seed(self.seed + CRN_OFFSET + experiment_index)
            return self.das[experiment_index](X_raw, **augment_kwargs)

    def _sweep_data(
        self, experiment_index: int, n_samples: int | None = None, common_random: bool = False, **augment_kwargs
    ) -> SweepData:
        """Default single-fold SweepData on the fixed base sample."""
        X_raw, X, y, X_test, estimand = self._base_data(experiment_index, n_samples)

        if common_random:
            GX_raw, G = self._augment_once(experiment_index, X_raw, **augment_kwargs)
        else:
            GX_raw, G = self.das[experiment_index](X_raw, **augment_kwargs)

        return SweepData(X=X, y=y, GX=self.apply_transform(GX_raw), G=G, X_test=X_test, estimand=estimand)


# =============================================================================
# SWEEP STRATEGIES (one per PARAM_SPECS key)
# =============================================================================


class GammaRatioStrategy(GenericParamSweep):
    """
    Validity: sweep the ASSUMED budget as a ratio of the oracle, gamma = r gamma*.

    x is the ratio, identical across experiments, so no sorting and no wiggle.
    Data is constant, so models are fit once per experiment and only re-solved.
    """

    param_key = "gamma"

    def generate_data(self, experiment_index: int, param) -> SweepData:
        return self._sweep_data(experiment_index)

    def get_predict_kwargs(self, param, experiment_index: int):
        return {"gamma": float(param) * self.fit_gamma(experiment_index)}


class EpsilonRatioStrategy(GenericParamSweep):
    """
    Robustness: sweep the ASSUMED invariance budget, epsilon = r eps* + EPS_TOL.

    gamma stays at gamma*_j, as everywhere outside the validity sweep, so the
    baseline is valid and the plot isolates the effect of misstating epsilon.
    (Pinning gamma to the Thm. 1 threshold instead would make the baseline
    invalid by construction, leaving PI+INV infeasible throughout and the
    intersections inheriting that invalidity -- see PLAN 7.)

    This is the ONLY sweep that recalibrates the DA (to ROBUSTNESS_EPSILON_TRUE),
    so that eps* > 0 makes the ratio axis meaningful.
    """

    param_key = "epsilon"

    def __init__(self, **kwargs):
        # scoped to this sweep only; never leaks into trS/n/m/perf
        kwargs["epsilon_true"] = ROBUSTNESS_EPSILON_TRUE
        super().__init__(**kwargs)

        if not self.pad:
            logger.warning(
                "Robustness sweep with pad=false: the DA+ validity claim needs "
                "eps-padding (Thm. 3.A). Run under `pad: true`."
            )

    def prepare_pair(self, sem, da, features=None):
        if da.strength is None:
            logger.warning(
                f"{type(da).__name__} has no strength knob; robustness sweep "
                "keeps the DA as configured (eps* may be 0)."
            )
            self.epsilon_true = None
        return super().prepare_pair(sem, da, features=features)

    def generate_data(self, experiment_index: int, param) -> SweepData:
        return self._sweep_data(experiment_index)

    def get_predict_kwargs(self, param, experiment_index: int):
        eps_star = self._finite(self.get_oracle(experiment_index).epsilon_star, self.default_epsilon, "eps*")
        return {"epsilon": float(param) * eps_star + EPS_TOL}


class ExpansionStrategy(GenericParamSweep):
    """
    Informativeness: sweep the DA strength knob; the x-axis is the MEASURED
    relative expansion of Prop. 2, post-poly, averaged over experiments: the
    factor `rho if recalibrate else 1.0` times tr(S)/k, and the label stays
    `rho tr(S)/k`. That keeps the axis of the shipped toggle; the factor and
    label that follow the recalibrated mechanism are the next change. Base
    data is fixed per experiment and the DA draws use common random numbers.
    On both datasets tr(S)/k was measured to fall with the knob while rho rises,
    so the product can fold back (it does on optical at full scale and on the
    4-step sim fixture of a31), which is why `create_sweep_plot` sorts the
    (x, y) pairs before drawing.
    """

    param_key = "trS"

    def __init__(self, augment_kwargs_fn: Callable | None = None, **kwargs):
        # knob -> DA call kwargs; dataset-specific
        self.augment_kwargs_fn = augment_kwargs_fn or (lambda s: {"scale": float(s)})
        self._measured = {}
        self._factors = {}  # (experiment, knob) -> (rho, tr(S)/k)
        self._step_epsilon = {}
        super().__init__(**kwargs)

    def generate_data(self, experiment_index: int, param) -> SweepData:
        data = self._sweep_data(experiment_index, common_random=True, **self.augment_kwargs_fn(param))

        # measured expansion in the space the methods fit in. The spectrum is
        # TRUNCATED: this axis is exactly where Sigma_GX goes ill-conditioned, so
        # the raw pinv reads ~23% high at the top of the grid (see SPECTRUM_KEEP).
        rho = rho_hat(data.X, data.GX, data.y, intercept=self.mean_match)
        trace_S = trace_S_over_k(data.X, data.GX, keep=SPECTRUM_KEEP)
        factor = rho if self.recalibrate else 1.0
        x = factor * trace_S
        self._measured[(experiment_index, float(param))] = x
        self._factors[(experiment_index, float(param))] = (rho, trace_S)
        convention = "recalibrated: x = rho tr(S)/k" if self.recalibrate else "inherited gamma: rho := 1, x = tr(S)/k"
        logger.info(
            f"trS step {float(param):.4g}: rho {rho:.4f} tr(S)/k {trace_S:.5f} "
            f"(untruncated {trace_S_over_k(data.X, data.GX):.5f}) "
            f"x {x:.5f} ({convention})"
        )

        # The knob IS the invariance-error driver, so a setup-time eps* is
        # stale. Pass the same augment kwargs: sim's `scale` is call-time and
        # does NOT persist on the DA, so omitting them would silently measure
        # eps* at scale=1.0 for every step.
        X_raw = self._base_data(experiment_index)[0]
        self._step_epsilon[experiment_index] = (
            epsilon_star(
                self.sems[experiment_index],
                self.das[experiment_index],
                X=X_raw,
                features=self._features,
                **self.augment_kwargs_fn(param),
            )
            + EPS_TOL
        )

        return data

    def fit_epsilon(self, experiment_index: int, step_index: int = 0, data=None) -> float:
        per_step = self._step_epsilon.get(experiment_index)
        if per_step is None:
            return super().fit_epsilon(experiment_index, step_index, data)
        # the knob drives eps*, but it can still land under the floor -- guard it
        # exactly as the base does, or this sweep alone bypasses the guard
        return self._floor_guard(per_step, data, "inv", experiment_index, "epsilon")

    def axis_record(self) -> dict[str, Any]:
        """
        Both factors of the measured x, per (knob, experiment), in KNOB order like
        the values pkl: `x == nanmean(rho * trS, 1)` when recalibrated, `nanmean(trS, 1)`
        otherwise, so the other convention is `nanmean` of the other product.
        """
        knob = np.asarray(self.get_param_range(), dtype=float)
        nan_pair = (np.nan, np.nan)
        factors = np.array(
            [[self._factors.get((j, float(s)), nan_pair) for j in range(self.n_experiments)] for s in knob]
        )
        return {
            "knob": knob,
            "rho": factors[:, :, 0],
            "trS": factors[:, :, 1],
            "x": self.observed_x(knob),
            "recalibrate": bool(self.recalibrate),
        }

    def observed_x(self, param_values: np.ndarray) -> np.ndarray:
        """Mean over experiments of the measured expansion."""
        return np.array(
            [
                np.nanmean([self._measured.get((j, float(s)), np.nan) for j in range(self.n_experiments)])
                for s in param_values
            ]
        )


class SampleSizeStrategy(GenericParamSweep):
    """
    Sweep n. The base sample and the test set are drawn ONCE per experiment and
    only the train side is subsampled, so the test set is identical across n.

    Optical: n is the PRE-SPLIT total from the 1000-row pool, so the train set
    is round(0.9 n) = {115, 230, 461, 900}. Simulation: the train set is exactly
    n, taken from the (larger) default draw.
    """

    param_key = "n"

    def generate_data(self, experiment_index: int, param) -> SweepData:
        n = int(param)
        X_raw, X, y, X_test, estimand = self._base_data(experiment_index)

        # optical n counts pre-split rows; sim n is the train size itself
        n_train = int(round((1.0 - self.test_fraction) * n)) if self.finite_pool else n
        if n_train > len(X):
            logger.warning(f"n={n} needs {n_train} train rows, only {len(X)} drawn.")
            n_train = len(X)
        X_raw, X, y = X_raw[:n_train], X[:n_train], y[:n_train]

        GX_raw, G = self.das[experiment_index](X_raw)
        return SweepData(X=X, y=y, GX=self.apply_transform(GX_raw), G=G, X_test=X_test, estimand=estimand)


class FoldStrategy(GenericParamSweep):
    """
    Sweep the number of augmentation folds m, at FIXED base data.

    GX/G stack m distinct DA passes over the same train X, and X/y are tiled to
    match. The augmented train set thus grows m-fold -- the one sanctioned
    departure from the paper's fixed-size convention, kept deliberately to study
    the finite-n benefit of m. Baselines fit the untiled copy (exactly
    equivalent, and avoids a QR on the m-fold matrix).
    """

    param_key = "m"

    def __init__(self, n_samples_override: int | None = None, **kwargs):
        self.n_samples_override = n_samples_override
        super().__init__(**kwargs)

    def generate_data(self, experiment_index: int, param) -> SweepData:
        m = int(param)
        da = self.das[experiment_index]
        X_raw, X, y, X_test, estimand = self._base_data(experiment_index, self.n_samples_override)

        GX_raws, Gs = zip(*(da(X_raw) for _ in range(m)), strict=False)

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


STRATEGIES: dict[str, type] = {
    "gamma": GammaRatioStrategy,
    "epsilon": EpsilonRatioStrategy,
    "trS": ExpansionStrategy,
    "n": SampleSizeStrategy,
    "m": FoldStrategy,
}
