"""
Generic experiment runners that work with any SEM/DA combination.
Eliminates duplication between simulation and optical device experiments.
"""

from collections.abc import Callable

import numpy as np
from loguru import logger
from sklearn.model_selection import train_test_split

from src.experiments.base import ExperimentDataContext, ParamSweepRunner, QuerySweepRunner, SweepData
from src.experiments.configs import EPS_TOL, FLOOR_GUARD_R, ROBUSTNESS_EPSILON_TRUE, SPECTRUM_KEEP
from src.experiments.utils import radial_sweep_pcs
from src.experiments.utils.metrics import rho_hat, trace_S_over_k
from src.methods.sensitivity_models import constraint_floor
from src.oracle import (
    calibrate_da_epsilon,
    compute_oracle_parameters,
    epsilon_star,
    preserve_rng,
    thm1_gamma_min,
)

# per-experiment DA seed offset: common random numbers across a knob grid
CRN_OFFSET: int = 10_000


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
        if self.epsilon_true is not None:
            calibrate_da_epsilon(
                sem=sem,
                da=da,
                epsilon_target=self.epsilon_true,
                features=features,
            )

        oracle = compute_oracle_parameters(
            sem=sem, da=da, features=features, calibrate=self.calibrate, mean_match=self.mean_match
        )
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
        **kwargs,
    ):
        super().__init__(**kwargs)

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
            )

    def _load_data(self):
        """(X_raw, GX_raw, y, G). Override when the draw needs its own protocol."""
        X_raw, y = self.sem(N=self.n_samples)
        GX_raw, G = self.da(X_raw)
        return X_raw, GX_raw, y, G

    @property
    def epsilon_iv(self) -> float:
        """Oracle IV budget, off the knife edge (same guard as PI+INV), then raised
        to the constraint's own floor if it lands under it (`FLOOR_GUARD_R`)."""
        budget = getattr(self.oracle, "eps_iv_star", None)
        if budget is None or not np.isfinite(budget):
            logger.warning("eps_iv_star unavailable; IV budget falls back to EPS_TOL.")
            budget = 0.0
        budget = float(budget) + EPS_TOL

        if getattr(self, "G", None) is None or np.size(self.G) == 0:
            return budget
        try:
            floor = constraint_floor(
                self.GX,
                self.y,
                self.default_gamma,
                kind="iv",
                Z=self.G,
                calibrate=self.calibrate,
                mean_match=self.mean_match,
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

    @property
    def vlines(self):
        """Also mark the Thm. 1 threshold, as a ratio of gamma*.

        `thm1_gamma_min` inverts the TIGHT ceiling of App. F.1, so the line is
        the exact budget below which DA can lose h_*, not a sufficient bound.

        Thm. 1 assumes a T-invariant h_*. Where the DA is only approximately
        invariant (eps* > 0, e.g. optical) the governing statement is Thm. 3.A's
        eps-padding instead, and this line is a reference, not a prediction.
        """
        ratios = []
        for j in range(self.n_experiments):
            oracle = self.get_oracle(j)
            gamma_star = self._finite(oracle.gamma_star, self.default_gamma, "gamma*")
            if gamma_star > 0:
                ratios.append(thm1_gamma_min(oracle, self.calibrate) / gamma_star)

        if not ratios:
            return self.spec.vlines
        return tuple(self.spec.vlines) + (float(np.mean(ratios)),)


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
    relative expansion rho * tr(S)/k (Prop. 2), post-poly, averaged over
    experiments. Base data is fixed per experiment and the DA draws use common
    random numbers, so the measured x moves monotonically with the knob.
    """

    param_key = "trS"

    def __init__(self, augment_kwargs_fn: Callable | None = None, **kwargs):
        # knob -> DA call kwargs; dataset-specific
        self.augment_kwargs_fn = augment_kwargs_fn or (lambda s: {"scale": float(s)})
        self._measured = {}
        self._step_epsilon = {}
        super().__init__(**kwargs)

    def generate_data(self, experiment_index: int, param) -> SweepData:
        data = self._sweep_data(experiment_index, common_random=True, **self.augment_kwargs_fn(param))

        # measured expansion in the space the methods fit in. The spectrum is
        # TRUNCATED: this axis is exactly where Sigma_GX goes ill-conditioned, so
        # the raw pinv reads ~23% high at the top of the grid (see SPECTRUM_KEEP).
        rho = rho_hat(data.X, data.GX, data.y, intercept=self.mean_match)
        trace_S = trace_S_over_k(data.X, data.GX, keep=SPECTRUM_KEEP)
        self._measured[(experiment_index, float(param))] = rho * trace_S
        logger.info(
            f"trS step {float(param):.4g}: rho {rho:.4f} tr(S)/k {trace_S:.5f} "
            f"(untruncated {trace_S_over_k(data.X, data.GX):.5f}) "
            f"x {rho * trace_S:.5f}"
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
