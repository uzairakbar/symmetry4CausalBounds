"""
Generic experiment runners that work with any SEM/DA combination.
Eliminates duplication between simulation and optical device experiments.
"""

from collections.abc import Callable
from dataclasses import replace
from functools import partial
from typing import Any

import numpy as np
from loguru import logger
from sklearn.model_selection import train_test_split

from src.experiments.base import ExperimentDataContext, ParamSweepRunner, QuerySweepRunner, SweepData
from src.experiments.configs import (
    EPS_TOL,
    OMEGA_XLABEL,
    ROBUSTNESS_AUGMENTATION,
    ROBUSTNESS_EPSILON_RADII,
    SPECTRUM_KEEP,
    percent_of,
)
from src.experiments.utils import radial_sweep_pcs
from src.experiments.utils.metrics import rho_hat, sigma_sq_hat, trace_S_over_k
from src.experiments.utils.model_fitting import instrument_columns
from src.methods.sensitivity_models import recalibrated_gamma
from src.oracle import (
    compute_oracle_parameters,
    epsilon_star_q95,
    gamma_star,
    pool_oracles,
    preserve_rng,
    recalibrated_da_epsilon,
    sweep_sem,
    t_leak_star,
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
        # a sweep's SEM (`epsilon_quantile` set) reads eps* once per (SEM, DA), as
        # the shared q0.95 of |W| over `W_DRAWS` concatenated draws; the query
        # panel's SEM keeps the pooled RMS above
        if sweep_sem(sem):
            oracle = replace(oracle, epsilon_star=epsilon_star_q95(sem, da, X=X, features=features))
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
        default_epsilon: float | None = 2**-8,
        raw_gamma: bool = False,
        oracle_t_leak: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # `raw_gamma`: the declared gamma is a RAW squared radius, the units the
        # query panels were drawn in before sigma-hat entered every ball: the PI
        # radius is sqrt(gamma), not sigma-hat sqrt(gamma). It is rescaled by
        # 1 / sigma-hat^2 of the draw once the data exist, so the panels keep
        # their old radii (1.0 on sim, 0.5 on optical). Query sweep only; the
        # param sweeps solve at gamma* in the paper's units and never see this.
        # This branch exists so the two panels can be compared side by side;
        # merge or drop it on that comparison.
        self.raw_gamma = bool(raw_gamma)

        # Create SEM and DA
        self.sem = sem_factory()
        self.da = da_factory()
        self.poly = poly_transform
        self.epsilon_true = epsilon_true
        self.oracle = self.prepare_pair(self.sem, self.da, features=self._features)
        # `oracle_t_leak`: the DA T row at its population radius, the oracle T leak
        # (`oracle.t_leak_star`, pooled as `prepare_pair` pools) + EPS_TOL, in place
        # of the row formula at gamma~_z(eps). Opt-in, the optical query only
        # (`resolve_dataset_block`); None leaves every T row on the formula
        self.t_leak_star = self.pooled_t_leak_star() if oracle_t_leak else None
        t_radius = None if self.t_leak_star is None else self.t_leak_star + EPS_TOL
        if t_radius is not None:
            logger.info(f"oracle T leak {self.t_leak_star:.6g}; the T row's radius {t_radius:.6g} (+ EPS_TOL)")

        # Subclasses must FORWARD their budgets here rather than assigning before
        # super().__init__, or these defaults silently overwrite them. A None
        # epsilon is the panel's own measured budget: the oracle's eps* (the RMS
        # of |W|, the query SEM's norm) + EPS_TOL.
        self.default_gamma = default_gamma
        if default_epsilon is None:
            default_epsilon = float(self.oracle.epsilon_star) + EPS_TOL
            logger.info(f"query eps = eps* {self.oracle.epsilon_star:.6g} + EPS_TOL = {default_epsilon:.6g}")
        self.default_epsilon = default_epsilon

        # Data FIRST, methods second: rho and the raw gamma's sigma-hat are
        # properties of the draw. (`DoMNISTQuerySweep` passes method_factory=None
        # and rebuilds after its nets exist.)
        loaded = self._load_data()
        self.X_raw, self.GX_raw, self.y, self.G = loaded[:4]
        # a 4-tuple (do-MNIST's override) carries no instrument
        self.Z = instrument_columns(loaded[4] if len(loaded) > 4 else None, len(self.X_raw))

        # Apply polynomial transformation if provided
        if self.poly:
            self.X = self.poly.fit_transform(self.X_raw)
            self.GX = self.poly.fit_transform(self.GX_raw)
        else:
            self.X = self.X_raw
            self.GX = self.GX_raw

        # a raw declared gamma into the paper's units: the solver's radius
        # sigma-hat sqrt(gamma / sigma-hat^2) is back at sqrt(gamma)
        if self.raw_gamma:
            sigma_sq = sigma_sq_hat(self.X, self.y, intercept=self.mean_match, absorbed_rate=self.absorbed_rate)
            if not np.isfinite(sigma_sq) or sigma_sq <= 0.0:
                raise ValueError(f"raw_gamma: sigma-hat^2 of the draw is {sigma_sq!r}; cannot rescale gamma.")
            self.default_gamma = default_gamma / sigma_sq
            logger.info(f"query gamma {default_gamma:.4g} / sigma-hat^2 {sigma_sq:.4g} = {self.default_gamma:.4g}")

        # gamma/epsilon stay at the yaml defaults here (PLAN 7: the query sweep
        # never auto-sets them).
        if method_factory is not None:
            self.methods = method_factory(
                gamma=self.default_gamma,
                epsilon=default_epsilon,
                rho=self.fit_rho(),
                **({} if t_radius is None else {"t_radius": t_radius}),
            )

    def pooled_t_leak_star(self) -> float:
        """The oracle T leak of this panel's (SEM, DA, features), RMS-pooled over the
        same seeded draws on the same rows as `prepare_pair`'s oracle."""
        pool = getattr(self.sem, "pool", None)
        draws = ORACLE_POOL_DRAWS if pool is not None else 1
        with preserve_rng():
            leaks = []
            for draw in range(draws):
                np.random.seed(ORACLE_POOL_SEED + draw)
                leaks.append(
                    t_leak_star(
                        self.sem,
                        self.da,
                        X=None if pool is None else pool[0],
                        features=self._features,
                        mean_match=self.mean_match,
                    )
                )
        return float(np.sqrt(np.mean(np.square(leaks))))

    def extent(self, queries) -> np.ndarray:
        """Target-set half-width at each query, beside `self.sem.f(queries)`. Zeros
        for a point target; the query figures of a set-valued target read it."""
        return self.sem.extent(queries)

    def _load_data(self):
        """(X_raw, GX_raw, y, G, Z). Override when the draw needs its own protocol.

        Split site: a SEM with instruments emits them as the trailing columns of
        the draw, and they come off here so the DA, the features and every query
        see treatment columns only."""
        X_raw, y = self.sem(N=self.n_samples)
        X_raw, Z = self.sem.split_instruments(X_raw)
        GX_raw, G = self.da(X_raw)
        return X_raw, GX_raw, y, G, Z

    def after_fit(self, name: str, model) -> None:
        """The DA IV rows against their own floors on this draw, as the sweeps
        report them (`_iv_floor_report`); never raised."""
        self._iv_floor_report(name, model, self.GX, self.y, self.G, self.Z, self.default_gamma)

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
            Z=self.Z,
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

    def make_sem(self, experiment_index: int):
        """Experiment j's SEM. A fresh factory call by default; a dataset with
        several recorded SEMs picks one per experiment instead."""
        return self.sem_factory()

    def poly_for_sem(self, sem) -> Callable | None:
        """The feature map of one SEM; the shared transform by default."""
        return self.poly

    def _poly_at(self, experiment_index: int) -> Callable | None:
        polys = getattr(self, "polys", None)
        return self.poly if polys is None else polys[experiment_index]

    def _features_at(self, experiment_index: int) -> Callable | None:
        poly = self._poly_at(experiment_index)
        return poly.fit_transform if poly else None

    def setup_sems_and_das(self):
        """Setup SEMs and DAs for all experiments."""
        self.sems = [self.make_sem(j) for j in range(self.n_experiments)]
        self.polys = [self.poly_for_sem(sem) for sem in self.sems]
        self.das = [self.da_factory(sem) for sem in self.sems]
        self.oracles = [
            self.prepare_pair(sem, da, features=self._features_at(j))
            for j, (sem, da) in enumerate(zip(self.sems, self.das, strict=True))
        ]

    def get_da(self, experiment_index: int):
        return self.das[experiment_index]

    def get_oracle(self, experiment_index: int):
        return self.oracles[experiment_index]

    def apply_transform(self, X: np.ndarray, experiment_index: int) -> np.ndarray:
        poly = self._poly_at(experiment_index)
        if poly:
            return poly.fit_transform(X)
        return X

    @property
    def finite_pool(self) -> bool:
        """A SEM that exposes a pool draws from a fixed set of rows and needs a
        train/test split; a generator gets an interventional test set instead.
        `SEM.pool` is the property that says which, so ask it rather than the class
        name (the two agree on both shipped SEMs)."""
        return getattr(self.sems[0], "pool", None) is not None

    # ------------------------------------------------------- fixed base sample

    def _base_data(self, experiment_index: int, n_samples: int | None = None):
        """(X_train_raw, X_train, y_train, X_test, estimand, Z_train), drawn once per j.

        An override that returns the old 5-tuple (do-MNIST's `_draw_base`) carries
        no instrument and is padded with an (n, 0) Z here."""
        key = (experiment_index, n_samples)
        if key not in self._base:
            drawn = tuple(self._draw_base(experiment_index, n_samples))
            if len(drawn) == 5:
                drawn = (*drawn, None)
            X_raw, X, y, X_test, estimand, Z = drawn
            self._base[key] = (X_raw, X, y, X_test, estimand, instrument_columns(Z, len(X_raw)))
        return self._base[key]

    def _draw_base(self, experiment_index: int, n_samples: int | None = None):
        """Split site: a SEM with instruments emits them as the trailing columns of
        every draw. Rows are split BEFORE columns on a finite pool, so Z stays
        aligned with X under the cigarette state-cluster bootstrap; the DA, the
        features, `f` and `extent` all see treatment columns only."""
        sem = self.sems[experiment_index]
        n_total = self.n_samples if n_samples is None else int(n_samples)

        if self.finite_pool:
            # finite pool: split unique instances; test set fixed across steps
            XZ_all, y_all = sem(N=n_total)
            XZ_train, XZ_test, y_train, _ = train_test_split(
                XZ_all, y_all, test_size=self.test_fraction, random_state=self.seed + experiment_index
            )
            X_train_raw, Z_train = sem.split_instruments(XZ_train)
            X_test_raw, _ = sem.split_instruments(XZ_test)
        else:
            # generator: interventional test set
            XZ_train, y_train = sem(N=n_total)
            X_train_raw, Z_train = sem.split_instruments(XZ_train)
            XZ_test, _ = sem(N=int(self.test_fraction * n_total), intervention=True)
            X_test_raw, _ = sem.split_instruments(XZ_test)

        X_test = self.apply_transform(X_test_raw, experiment_index)
        return (
            X_train_raw,
            self.apply_transform(X_train_raw, experiment_index),
            y_train,
            X_test,
            sem.f(X_test),
            Z_train,
        )

    def _extent(self, experiment_index: int, X_test) -> np.ndarray:
        """Target-set half-width at each query, beside `sem.f(X_test)`. Zeros for a
        point target, which is every shipped SEM. Not folded into `_draw_base`'s
        tuple: do-MNIST overrides that method, and a sixth element would break it.
        """
        return self.sems[experiment_index].extent(X_test)

    def _augment_once(self, experiment_index: int, X_raw, **augment_kwargs):
        """One DA pass; common random numbers across a knob grid within an experiment."""
        with preserve_rng():
            np.random.seed(self.seed + CRN_OFFSET + experiment_index)
            return self.das[experiment_index](X_raw, **augment_kwargs)

    def _sweep_data(
        self, experiment_index: int, n_samples: int | None = None, common_random: bool = False, **augment_kwargs
    ) -> SweepData:
        """Default single-fold SweepData on the fixed base sample."""
        X_raw, X, y, X_test, estimand, Z = self._base_data(experiment_index, n_samples)

        if common_random:
            GX_raw, G = self._augment_once(experiment_index, X_raw, **augment_kwargs)
        else:
            GX_raw, G = self.das[experiment_index](X_raw, **augment_kwargs)

        return SweepData(
            X=X,
            y=y,
            GX=self.apply_transform(GX_raw, experiment_index),
            G=G,
            X_test=X_test,
            estimand=estimand,
            extent=self._extent(experiment_index, X_test),
            Z=Z,
        )


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

    This is the ONLY sweep that recalibrates the DA (to eps* =
    ROBUSTNESS_EPSILON_RADII x sigma sqrt(gamma*) of each SEM, `robustness_target`),
    so that eps* > 0 makes the ratio
    axis meaningful, and the only one that appends
    ROBUSTNESS_AUGMENTATION[experiment_name] to the configured DA chain where
    that is set (optical: the configured chain has no knob to tune).

    Every DA IV row reads the swept epsilon through gamma~_z(eps), the same
    misspecification on the same DA draw; the declared gamma_z does not move.
    Below r = 1 the assumed budgets can fall under what the constraints can attain on the ball, and the
    queries then read INFEASIBLE -- which is what an under-budget ratio means,
    what `PI+INV` has always done at the left of this grid, and what every sweep
    now does wherever a budget lands under its floor (no budget is ever raised).
    The grid itself is centred on r = 1 (`_EPSILON_RATIO_GRID`), so both halves
    are on the figure.
    """

    param_key = "epsilon"

    def __init__(self, **kwargs):
        # scoped to this sweep only; never leaks into omega/n/m/perf or the query panel
        name = kwargs.get("experiment_name", "simulation")
        # the target is per SEM (`robustness_target`), set in `prepare_pair`
        self.epsilon_radii = float(ROBUSTNESS_EPSILON_RADII)
        component = ROBUSTNESS_AUGMENTATION[name]
        if component is not None:
            kwargs["da_factory"] = partial(kwargs["da_factory"], append=component)
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
        else:
            self.epsilon_true = self.robustness_target(sem)
        return super().prepare_pair(sem, da, features=features)

    def robustness_target(self, sem) -> float:
        """The tuned DA's eps* (in the sweeps' norm of W): `epsilon_radii` confounding
        radii R = sigma sqrt(gamma*), with the oracle's sigma and gamma* of this SEM
        -- the units the DA+ ball is drawn in (see ROBUSTNESS_EPSILON_RADII)."""
        radius = float(np.sqrt(sem.sigma_sq * gamma_star(sem)))
        target = self.epsilon_radii * radius
        logger.info(f"robustness eps* target {target:.6g} = {self.epsilon_radii:g} x sigma sqrt(gamma*) {radius:.6g}")
        return target

    def generate_data(self, experiment_index: int, param) -> SweepData:
        return self._sweep_data(experiment_index)

    def get_predict_kwargs(self, param, experiment_index: int):
        eps_star = self._finite(self.get_oracle(experiment_index).epsilon_star, self.default_epsilon, "eps*")
        # a DA+ IV method's rows re-solve at it (gamma~_z(eps)); the rest pad by it
        return {"epsilon": float(param) * eps_star + EPS_TOL}


class ExpansionStrategy(GenericParamSweep):
    """
    Informativeness: sweep the DA strength knob; the x-axis is the MEASURED
    relative expansion of Prop. 2, post-poly, averaged over experiments, for
    the ball in force. Recalibrated (`recalibrate: true`) the DA+ radius is
    sigma~ sqrt(gamma/rho) = sigma sqrt(gamma), so the ratio is tr(S)/k and the
    axis plots x = tr(S)/k under that label; at the inherited gamma the radius
    carries sqrt(rho), so x = rho tr(S)/k under `rho tr(S)/k`. The label follows
    the factor (`xlabel`, OMEGA_XLABEL) and both go into the axis pkl. Base data
    is fixed per experiment and the DA draws use common random numbers. On sim
    tr(S)/k falls with the knob while rho rises, so the product can fold back
    (it does on the 4-step sim fixture of a31), which is why `create_sweep_plot`
    sorts the (x, y) pairs it is given, i.e. the plotted quantity, before drawing.
    """

    param_key = "omega"

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
        # the ball in force: recalibrated the sqrt(rho) cancels, so x = tr(S)/k;
        # at the inherited gamma it does not, so x = rho tr(S)/k
        factor = 1.0 if self.recalibrate else rho
        x = factor * trace_S
        self._measured[(experiment_index, float(param))] = x
        self._factors[(experiment_index, float(param))] = (rho, trace_S)
        convention = "recalibrated: x = tr(S)/k" if self.recalibrate else "inherited gamma: x = rho tr(S)/k"
        logger.info(
            f"omega step {float(param):.4g}: rho {rho:.4f} tr(S)/k {trace_S:.5f} "
            f"(untruncated {trace_S_over_k(data.X, data.GX):.5f}) "
            f"x {x:.5f} ({convention})"
        )

        # The knob IS the invariance-error driver, so a setup-time eps* is
        # stale. Pass the same augment kwargs: sim's `scale` is call-time and
        # does NOT persist on the DA, so omitting them would silently measure
        # eps* at scale=1.0 for every step.
        X_raw = self._base_data(experiment_index)[0]
        augment_kwargs = self.augment_kwargs_fn(param)
        self._step_epsilon[experiment_index] = (
            epsilon_star_q95(
                self.sems[experiment_index],
                self.das[experiment_index],
                X=X_raw,
                features=self._features_at(experiment_index),
                **augment_kwargs,
            )
            + EPS_TOL
        )
        # the shared q0.95 reading, as the setup oracle's: same draws, same stream

        return data

    def fit_epsilon(self, experiment_index: int, step_index: int = 0, data=None) -> float:
        per_step = self._step_epsilon.get(experiment_index)
        if per_step is None:
            return super().fit_epsilon(experiment_index, step_index, data)
        # the knob drives eps*, and it can still land under the floor -- report it
        # exactly as the base does, and leave it as is
        self._floor_report(per_step, data, "inv", experiment_index, "epsilon")
        return per_step

    @property
    def xlabel(self) -> str:
        """The label of the factor in force (OMEGA_XLABEL), not the spec's static one."""
        return OMEGA_XLABEL[bool(self.recalibrate)]

    def axis_record(self) -> dict[str, Any]:
        """
        Both factors of the measured x, per (knob, experiment), in KNOB order like
        the values pkl: `x == nanmean(trS, 1)` when recalibrated, `nanmean(rho * trS, 1)`
        otherwise, so the other convention is `nanmean` of the other product. The
        key `trS` names tr(S)/k itself, not the sweep, so it keeps its name under
        the omega sweep. The toggle and the label the figure was drawn with ride along.
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
            "xlabel": self.xlabel,
        }

    def observed_x(self, param_values: np.ndarray) -> np.ndarray:
        """Mean over experiments of the measured expansion."""
        return np.array(
            [
                np.nanmean([self._measured.get((j, float(s)), np.nan) for j in range(self.n_experiments)])
                for s in param_values
            ]
        )


class RecalibrationStrategy(GenericParamSweep):
    """
    Recalibration: sweep t in [0, 1], the post-DA budget of SS4.2
    gamma~ = gamma ((1 - t) + t / rho), from the inherited gamma (t = 0) to
    gamma/rho (t = 1); the continuous version of the `recalibrate` toggle.

    x is the MEASURED ratio gamma~/gamma = (1 - t) + t/rho_j averaged over
    experiments (each has its own rho_hat of its fit sample), so it runs from
    1/rho (fully recalibrated) up to 1 and `create_sweep_plot` sorts it. Data
    is constant: models are fit ONCE per experiment at the configured toggle
    and only re-solved through the predict-time knob, like the gamma sweep.
    Baselines have rho = 1 and ignore it.
    """

    param_key = "recalibrate"

    def __init__(self, **kwargs):
        self._rho = {}  # experiment -> rho_hat of the fit sample
        super().__init__(**kwargs)

    def generate_data(self, experiment_index: int, param) -> SweepData:
        data = self._sweep_data(experiment_index)
        self._rho[experiment_index] = self._finite(
            rho_hat(data.X, data.GX, data.y, intercept=self.mean_match), 1.0, "rho_hat"
        )
        return data

    def fit_rho(self, experiment_index: int, data=None) -> float:
        # the models solve at the rho the axis is drawn with
        if experiment_index in self._rho:
            return self._rho[experiment_index]
        return super().fit_rho(experiment_index, data)

    def get_predict_kwargs(self, param, experiment_index: int):
        return {"recalibrate": float(param)}

    def _rho_array(self) -> np.ndarray:
        return np.array([self._rho.get(j, np.nan) for j in range(self.n_experiments)], dtype=float)

    def observed_x(self, param_values: np.ndarray) -> np.ndarray:
        """Mean over experiments of gamma~/gamma = (1 - t) + t/rho_j, through the
        solver's own `recalibrated_gamma` so a sample rho_j < 1 reads as 1 here too."""
        rho = self._rho_array()
        return np.array([np.nanmean([recalibrated_gamma(1.0, r, t) for r in rho]) for t in param_values])

    def axis_record(self) -> dict[str, Any]:
        """The knob, the per-experiment rho behind x, x in knob order, and the label."""
        knob = np.asarray(self.get_param_range(), dtype=float)
        return {"knob": knob, "rho": self._rho_array(), "x": self.observed_x(knob), "xlabel": self.xlabel}


class SampleSizeStrategy(GenericParamSweep):
    """
    Sweep n. The base sample and the test set are drawn ONCE per experiment and
    only the train side is subsampled, so the test set is identical across n.

    The grid is `sweep_samples` percentages of `n_samples` log-spaced on
    N_PERCENT_RANGE (10 .. 100); the knob is the row count `percent_of(n_samples,
    p)` (rounded half up) and the x plotted is p itself. Two percentages that
    round to one row count raise. Optical and cigarettes: n is the PRE-SPLIT
    total, so the train set is round(0.9 n), 90 .. 900 of the 1000-row optical
    pool. Simulation: the train set is exactly n, taken from the default draw of
    `n_samples` train rows.
    """

    param_key = "n"

    def __init__(self, **kwargs):
        self._n_train = {}  # (experiment, n) -> train rows used
        self._percent = {}  # n -> the grid percentage it was rounded from
        super().__init__(**kwargs)

    def get_param_range(self) -> np.ndarray:
        """Row counts, one per grid percentage; an override is taken as rows."""
        if self.param_grid_override is not None:
            return np.asarray(self.param_grid_override)
        percents = self.spec.grid_fn(self.experiment_name, self.sweep_samples)
        counts = [percent_of(self.n_samples, p) for p in percents]
        if len(set(counts)) < len(counts):
            raise ValueError(
                f"n sweep: {len(percents)} percentages of n_samples = {self.n_samples} round to "
                f"repeated row counts {counts}; lower sweep_samples or raise n_samples."
            )
        self._percent.update(zip(counts, (float(p) for p in percents), strict=True))
        return np.asarray(counts, dtype=int)

    def observed_x(self, param_values: np.ndarray) -> np.ndarray:
        """n as a percentage of `n_samples`: the grid value it came from, exactly."""
        return np.array(
            [self._percent.get(int(n), 100.0 * float(n) / self.n_samples) for n in param_values], dtype=float
        )

    def axis_record(self) -> dict[str, Any]:
        """The rows behind each percentage, and the label and ticks the figure used;
        its presence is what tells the aggregate the x is a percentage."""
        knob = np.asarray(self.get_param_range(), dtype=int)
        n_train = np.array(
            [[self._n_train.get((j, int(n)), -1) for j in range(self.n_experiments)] for n in knob], dtype=int
        )
        return {
            "knob": knob,
            "n_samples": int(self.n_samples),
            "n_train": n_train,
            "x": self.observed_x(knob),
            "xlabel": self.xlabel,
            "xticks": tuple(self.xticks),
        }

    def generate_data(self, experiment_index: int, param) -> SweepData:
        n = int(param)
        X_raw, X, y, X_test, estimand, Z = self._base_data(experiment_index)

        # optical/cigarettes n counts pre-split rows; sim n is the train size itself
        n_train = int(round((1.0 - self.test_fraction) * n)) if self.finite_pool else n
        if n_train > len(X):
            logger.warning(f"n={n} needs {n_train} train rows, only {len(X)} drawn.")
            n_train = len(X)
        self._n_train[(experiment_index, n)] = n_train
        if experiment_index == 0:
            percent = float(self.observed_x([n])[0])
            logger.info(f"n step {percent:g}% of {self.n_samples}: n {n}, {n_train} train rows")
        X_raw, X, y, Z = X_raw[:n_train], X[:n_train], y[:n_train], Z[:n_train]

        GX_raw, G = self.das[experiment_index](X_raw)
        return SweepData(
            X=X,
            y=y,
            GX=self.apply_transform(GX_raw, experiment_index),
            G=G,
            X_test=X_test,
            estimand=estimand,
            extent=self._extent(experiment_index, X_test),
            Z=Z,
        )


class FoldStrategy(GenericParamSweep):
    """
    Sweep the number of augmentation folds m, at FIXED base data.

    GX/G stack m distinct DA passes over the same train X, and X/y are tiled to
    match. The augmented train set thus grows m-fold -- the one sanctioned
    departure from the paper's fixed-size convention, kept deliberately to study
    the finite-n benefit of m. Baselines fit the untiled copy (exactly
    equivalent, and avoids a QR on the m-fold matrix).

    The base draw is `m_sweep_n_percent` % of `n_samples` (`percent_of`), in the
    unit `n_samples` counts; the orchestrators pass the dataset's (6.25 unless
    the yaml says otherwise). None draws `n_samples` itself; an explicit
    `n_samples_override` (rows) wins over both.
    """

    param_key = "m"

    def __init__(self, n_samples_override: int | None = None, m_sweep_n_percent: float | None = None, **kwargs):
        # set before the base constructor, which draws the SEMs and DAs
        self.m_sweep_n_percent = None if m_sweep_n_percent is None else float(m_sweep_n_percent)
        if n_samples_override is None and self.m_sweep_n_percent is not None:
            n_samples_override = percent_of(kwargs["n_samples"], self.m_sweep_n_percent)
            logger.info(f"m sweep at {self.m_sweep_n_percent:g}% of {kwargs['n_samples']}: n {n_samples_override}")
        self.n_samples_override = n_samples_override
        super().__init__(**kwargs)

    def axis_record(self) -> dict[str, Any]:
        """The fold grid and the n the sweep drew, beside its label."""
        knob = np.asarray(self.get_param_range())
        return {
            "knob": knob,
            "x": self.observed_x(knob),
            # the REQUESTED rows: cigarettes' `iv` bootstrap draws whole state
            # histories, so it rounds this up to a multiple of 50 (153 -> 200)
            "n": self.n_samples if self.n_samples_override is None else int(self.n_samples_override),
            "n_samples": int(self.n_samples),
            "m_sweep_n_percent": self.m_sweep_n_percent,
            "xlabel": self.xlabel,
        }

    def generate_data(self, experiment_index: int, param) -> SweepData:
        m = int(param)
        da = self.das[experiment_index]
        X_raw, X, y, X_test, estimand, Z = self._base_data(experiment_index, self.n_samples_override)

        GX_raws, Gs = zip(*(da(X_raw) for _ in range(m)), strict=False)

        return SweepData(
            # transform is row-wise: transform(tile(.)) == tile(transform(.))
            X=np.tile(X, (m, 1)),
            y=np.tile(y, (m, 1)),
            GX=self.apply_transform(np.vstack(GX_raws), experiment_index),
            G=np.vstack(Gs),
            X_test=X_test,
            estimand=estimand,
            extent=self._extent(experiment_index, X_test),
            X_base=X,
            y_base=y,
            # tiled beside X and y, the untiled copy beside X_base
            Z=np.tile(Z, (m, 1)),
            Z_base=Z,
        )


STRATEGIES: dict[str, type] = {
    "gamma": GammaRatioStrategy,
    "epsilon": EpsilonRatioStrategy,
    "omega": ExpansionStrategy,
    "n": SampleSizeStrategy,
    "m": FoldStrategy,
    "recalibrate": RecalibrationStrategy,
}
