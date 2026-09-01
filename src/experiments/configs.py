"""
Unified configuration management for experiments.
All parameters defined here - no defaults in method classes.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from src.methods.copsens import (
    CopSensPI,
    RecentredInvCopSens,
)
from src.methods.copsens import (
    IntersectedCopSens as IntCopSens,
)
from src.methods.copsens import (
    IntersectedIVCopSens as IntIVCopSens,
)
from src.methods.copsens import (
    IVConstrainedCopSens as IVCopSens,
)
from src.methods.regression import (
    LeastSquaresClosedForm as ERM,
)
from src.methods.regression import (
    TwoStageLeastSquaresIV as IV,
)
from src.methods.sensitivity_models import (
    InstrumentalVariablePartialR2 as IVPartialR2,
)
from src.methods.sensitivity_models import (
    IntersectedInstrumentalVariablePartialR2 as IntIVPartialR2,
)
from src.methods.sensitivity_models import (
    IntersectedPartialR2 as IntPartialR2,
)
from src.methods.sensitivity_models import (
    InvarianceConstrainedPartialR2 as InvPartialR2,
)
from src.methods.sensitivity_models import (
    PartialR2,
)

# =============================================================================
# EXPERIMENT PARAMETERS
# =============================================================================


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for simulation experiments."""

    gamma: float = 1.0
    epsilon: float = 2**-8
    # SEM confounding. None = fully confounded, which drives sigma^2 to the
    # outcome-noise floor and rho to ~57: Prop. 2 can then never hold.
    gamma_true: float | None = 1.0
    test_fraction: float = 0.1


@dataclass(frozen=True)
class OpticalDeviceConfig:
    """Configuration for optical device experiments."""

    gamma: float = 2**-1.5
    epsilon: float = 2**-2
    query_epsilon: float = 2**-2
    epsilon_true: float | None = None
    test_fraction: float = 0.1
    dataset_index: int = 8
    ground_truth_model: Literal["linear", "polynomial"] = "polynomial"


@dataclass(frozen=True)
class DoMNISTConfig:
    """do-MNIST constants. Everything a run should never have to restate."""

    # SEM (see src/sem/do_mnist.py). Every derived quantity is a round number:
    # bias = beta(1/2-eta) = 0.10, h* in {0.2, 0.8}, h_erm cells {.1,.3,.7,.9},
    # ATE contrast 0.6, attainable [0.1, 0.9].
    alpha: float = 0.0
    beta: float = 0.4
    eta: float = 0.25
    subsample: int = 2  # 1 = 28x28 (d=2352), 2 = 14x14 (d=588)
    exemplar_seed: int = 1  # digit exemplars, frozen across replicates
    # CopSens. n_anchors is SOURCE's EFFECTIVE value: CopSensPI's own default is
    # 256, but every published do-MNIST number was measured at 128.
    link: Literal["probit", "gaussian"] = "probit"
    n_anchors: int = 128
    n_anchors_c: int = 48  # coarse set, for the feasibility constraint
    n_constraint_inv: int = 256  # 192
    n_constraint_iv: int = 256  # 384
    mu_clip: bool = False  # True                # clip mu_y to `attainable`; see §2.5
    jax_grad: bool = True  # analytic gradients for the SLSQP hot path
    test_fraction: float = 0.1

    @property
    def attainable(self) -> tuple[float, float]:
        """Range h_erm can occupy. mu_y outside it is impossible."""
        lo = (1 - self.beta) * self.alpha + self.beta * self.eta
        return float(lo), float(1.0 - lo)


# Default configurations
SIMULATION_CONFIG = SimulationConfig()
OPTICAL_CONFIG = OpticalDeviceConfig()
DOMNIST_CONFIG = DoMNISTConfig()


# =============================================================================
# DATASET DEFAULTS (root-yaml fallbacks)
# =============================================================================


@dataclass(frozen=True)
class DatasetDefaults:
    """Filled in when the root yaml omits the key."""

    n_samples: int
    n_experiments: int
    sweep_samples: int


DATASET_DEFAULTS: dict[str, DatasetDefaults] = {
    "simulation": DatasetDefaults(n_samples=2048, n_experiments=1, sweep_samples=32),
    "optical_device": DatasetDefaults(n_samples=1000, n_experiments=8, sweep_samples=32),
    # sweep_samples = the 10 digit exemplars on the query x-axis
    "do_mnist": DatasetDefaults(n_samples=1_200_000, n_experiments=1, sweep_samples=10),
}


# =============================================================================
# SWEEP PARAMETER / METRIC SPECS
# =============================================================================

# keeps auto-set epsilon off the PI_INV feasibility knife edge (eps=0 forces h~0)
EPS_TOL: float = 2**-8

# Floor guard. An auto-set budget below the constraint's own attainable floor is not
# a tighter bound, it is NO bound: every query comes back INFEASIBLE and the method
# vanishes from the sweep. `budget^2 >= FLOOR_GUARD_R * floor` is enforced in
# `ParamSweepRunner._floor_guard`, which only ever RAISES a budget, and only where the
# oracle value was already unusable.
#
# Calibrated on the simulation trS grid, where the oracle IV budget (EPS_TOL, 0.0039)
# sits below the floor at the 5 lowest knobs and DA+PI_IV was all-NaN there. Measured
# DA+PI_IV coverage / width at those 5 steps, at budget = sqrt(r * floor):
#   r=2.25  0.82-0.88            under-covers
#   r=4     0.931-1.000  2.9-4.3 two steps under nominal
#   r=9     0.985-1.000  4.2-6.0 <- CHOSEN: covers everywhere, still 24-47% inside
#                                  DA+PI (7.9-8.1), i.e. the constraint is alive
#   r=16    1.000                DA+PI parity: inert, a duplicate column
# The floor moves with gamma, n and the DA draw, so this is a RATIO, never an epsilon.
# It stays ~3 orders of magnitude below eps_rms, so "guard" is not "loose".
FLOOR_GUARD_R: float = 2  # 9.0

# the robustness sweep -- and ONLY it -- recalibrates a strength-knob DA to this
# true invariance error, so that eps/eps* is a meaningful ratio axis.
# MUST clear the optical eps* floor: the strength knob drives only the gaussian
# noise, so permutations hold eps* at ~0.239 even at strength 0. 0.5 sits at
# strength ~0.87 there (2.1x the floor) and ~0.084 on simulation, whose floor
# is 0. Below the floor, calibrate_da_epsilon clamps and the sweep goes flat.
ROBUSTNESS_EPSILON_TRUE: float = 2**-1

# SE crosshairs on scatter plots (needs n_experiments >= 2)
SCATTER_SE_CROSSHAIRS: bool = True

# Fraction of Sigma_GX's variance kept before inverting it for tr(S)/k.
# The near-null eigendirections of Sigma_GX are noise and 1/w blows them up, so the
# untruncated estimate is inflated exactly where the DA is strongest. Measured on the
# simulation trS sweep: at the top of the knob grid tr(S)/k reads 0.22889 untruncated
# vs 0.17706 here -- a 23% error, at the end of the axis the sweep is about.
SPECTRUM_KEEP: float = 0.999


# budget-ratio grid: centred on 1, i.e. on the oracle value
def _RATIO_GRID(dataset, n):
    return np.geomspace(2**-3, 2**3, num=n)


@dataclass(frozen=True)
class ParamSpec:
    """Axis + policy metadata for one sweepable parameter."""

    xlabel: str
    grid_fn: Callable[[str, int], np.ndarray]
    xscale: Literal["linear", "log"] = "log"
    vlines: tuple[float, ...] = ()  # reference values annotated on the x-axis
    include_ate: bool = True  # ATE is flat, useless on budget-ratio axes
    data_constant: bool = False  # False => data regenerated every step


PARAM_SPECS: dict[str, ParamSpec] = {
    "gamma": ParamSpec(
        xlabel=r"$\gamma / \gamma^\star$",
        grid_fn=_RATIO_GRID,
        vlines=(1.0,),
        include_ate=False,
        data_constant=True,
    ),
    "epsilon": ParamSpec(
        xlabel=r"$\varepsilon / \varepsilon^\star$",
        grid_fn=_RATIO_GRID,
        vlines=(1.0,),
        include_ate=False,
        data_constant=True,
    ),
    "trS": ParamSpec(
        # knob grid; the x-axis actually plotted is the MEASURED expansion
        xlabel=r"$\rho \operatorname{tr}(\mathcal{S})/k$",
        # tuned to the informative range: past it both DAs saturate and the
        # measured x moves by less than the across-seed SD (PLAN 5.3).
        # Optical never reaches x < 1 -- its permutations symmetrise rather
        # than inflate the covariance, so Prop. 2 never holds for it.
        grid_fn=lambda dataset, n: (
            np.logspace(-1.5, 1.0, num=n) if dataset == "simulation" else np.linspace(0.01, 0.3, num=n)
        ),
        xscale="linear",
        vlines=(1.0,),
    ),
    "n": ParamSpec(
        xlabel=r"$n$",
        grid_fn=lambda dataset, n: np.array(
            [128, 256, 512, 1024] if dataset == "simulation" else [128, 256, 512, 1000]  # 1000 = optical pool max
        ),
    ),
    "m": ParamSpec(
        xlabel=r"Augmentation Folds ($m$)",
        grid_fn=lambda dataset, n: np.array([1, 2, 4, 8, 16, 32]),
    ),
}


@dataclass(frozen=True)
class MetricSpec:
    """`key` names the QueryEval field / metrics.py function."""

    key: str
    ylabel: str
    yscale: Literal["linear", "log", "asinh"] = "linear"
    perf_only: bool = False
    # ATE is the truth: zero width, unit coverage. Plotting it on those axes
    # only drags the limits out and squashes the range the methods live in.
    include_ate: bool = True


METRIC_SPECS: dict[str, MetricSpec] = {
    "approx_error": MetricSpec("approximation_error", r"average $E^-_{{\bm{x}}}$", "asinh"),
    "worst_error": MetricSpec("worst_error", r"average $E^+_{{\bm{x}}}$", "asinh"),
    "width": MetricSpec("interval_width", r"average interval width", include_ate=False),
    "coverage": MetricSpec("coverage", r"coverage rate", include_ate=False),
    "wall_clock": MetricSpec("wall_clock", r"seconds per query", "log", perf_only=True),
    "seed_var": MetricSpec("seed_var", r"SD across seeds", perf_only=True),
}


# =============================================================================
# EXPERIMENT PLAN (root-yaml `experiment:` block)
# =============================================================================


@dataclass(frozen=True)
class SweepSpec:
    param: tuple[str, ...]
    metric: tuple[str, ...]


@dataclass(frozen=True)
class ScatterSpec:
    param: tuple[str, ...]
    metric: tuple[tuple[str, str], ...]  # (x-metric, y-metric) pairs


@dataclass(frozen=True)
class PerfSpec:
    metric: tuple[str, ...]  # overlay series; bar always drawn


@dataclass(frozen=True)
class ExperimentPlan:
    """Which experiment types to run. Panel is bound to `query`."""

    query: bool = False
    sweep: SweepSpec | None = None
    scatter: ScatterSpec | None = None
    perf: PerfSpec | None = None


def _reject_unknown(got, allowed, where: str):
    unknown = sorted(set(got) - set(allowed))
    if unknown:
        raise ValueError(f"Unknown key(s) {unknown} in {where}; expected {sorted(allowed)}.")


def _check_values(values, allowed, where: str) -> tuple:
    values = tuple(values)
    _reject_unknown(values, allowed, where)
    return values


def parse_experiment_plan(block: dict[str, Any] | None) -> ExperimentPlan:
    """Parse+validate the `experiment:` block. Unknown keys are a hard error."""
    block = dict(block or {})
    _reject_unknown(block, {"query", "sweep", "scatter", "perf"}, "experiment")

    sweep_metrics = {k for k, v in METRIC_SPECS.items() if not v.perf_only}
    perf_metrics = {k for k, v in METRIC_SPECS.items() if v.perf_only}

    sweep = block.get("sweep")
    if sweep is not None:
        _reject_unknown(sweep, {"param", "metric"}, "experiment.sweep")
        sweep = SweepSpec(
            param=_check_values(sweep.get("param", ()), PARAM_SPECS, "experiment.sweep.param"),
            metric=_check_values(sweep.get("metric", ()), sweep_metrics, "experiment.sweep.metric"),
        )

    scatter = block.get("scatter")
    if scatter is not None:
        _reject_unknown(scatter, {"param", "metric"}, "experiment.scatter")
        pairs = []
        for pair in scatter.get("metric", ()):
            if len(pair) != 2:
                raise ValueError(f"experiment.scatter.metric entries must be pairs, got {pair}.")
            pairs.append(_check_values(pair, sweep_metrics, "experiment.scatter.metric"))
        scatter = ScatterSpec(
            param=_check_values(scatter.get("param", ()), PARAM_SPECS, "experiment.scatter.param"),
            metric=tuple(pairs),
        )

    perf = block.get("perf")
    if perf is not None:
        # `param` is meaningless for perf (1-point sweep); accepted and ignored
        _reject_unknown(perf, {"param", "metric"}, "experiment.perf")
        perf = PerfSpec(metric=_check_values(perf.get("metric", ()), perf_metrics, "experiment.perf.metric"))

    return ExperimentPlan(
        query=bool(block.get("query", False)),
        sweep=sweep,
        scatter=scatter,
        perf=perf,
    )


# =============================================================================
# PLOT ANNOTATIONS
# =============================================================================

ANNOTATE_SWEEP_PLOT: dict[str, dict[str, Any]] = {
    "pc1": {
        "xlabel": r"$t$",
        "xscale": "linear",
    },
    "pc2": {
        "xlabel": r"$t$",
        "xscale": "linear",
    },
    "pc12": {
        "xlabel": r"$\vartheta$",
        "xscale": "linear",
    },
}

# =============================================================================
# METHOD REGISTRY
# =============================================================================

ALL_METHODS: tuple[str, ...] = (
    "ATE",
    "ERM",
    "DA+ERM",
    "DA+IV",
    "PI_INV",
    "PI",
    "PI_IV",
    "DA+PI",
    "DA+PI_IV",
    "PI&DA+PI",
    "PI&DA+PI_IV",
)

# the copsens backend defines a strict subset: no 2SLS and no baseline-IV. It DOES
# define the intersections -- Cor. 1 needs h_*(x) inside both intervals, which is a
# membership fact, not a claim that the two balls share a parameterisation.
COPSENS_METHODS: tuple[str, ...] = (
    "ATE",
    "ERM",
    "DA+ERM",
    "PI_INV",
    "PI",
    "DA+PI",
    "DA+PI_IV",
    "PI&DA+PI",
    "PI&DA+PI_IV",
)


def _copsens_builders(
    method_names, gamma, epsilon, epsilon_iv, calibrate, pad, clipy, n_jobs, outcome_models, n_components
):
    """CopSens backend. Every method takes a PREFIT outcome net, so only the PI
    machinery differs between them."""
    config = DOMNIST_CONFIG
    common = dict(
        n_components=n_components,
        link=config.link,
        calibrate=calibrate,
        clipy=clipy,
        n_jobs=n_jobs,
        n_anchors=config.n_anchors,
        n_anchors_c=config.n_anchors_c,
        jax_grad=config.jax_grad,
        mu_clip=config.attainable if config.mu_clip else None,
    )

    def net(key):
        if outcome_models is None:
            raise ValueError(
                "copsens methods need the prefit outcome nets. "
                "`ExperimentOrchestrator.methods` names them only -- the runner "
                "must rebuild via method_factory(..., outcome_models=...) once the "
                "nets exist."
            )
        return outcome_models[key]

    all_builders = {
        "ATE": lambda: None,  # computed via sem.f
        "ERM": lambda: net("X"),  # the prefit net, not a fresh one
        "DA+ERM": lambda: net("GX"),
        "PI": lambda: CopSensPI(gamma=gamma, epsilon=epsilon, pad=False, outcome_model=net("X"), **common),
        "DA+PI": lambda: CopSensPI(gamma=gamma, epsilon=epsilon, pad=pad, outcome_model=net("GX"), **common),
        # recentred on the post-DA measure: from an X-centred ball PI_INV is empty
        # at any reasonable eps (see RecentredInvCopSens)
        "PI_INV": lambda: RecentredInvCopSens(
            gamma=gamma,
            epsilon=epsilon,
            pad=False,
            outcome_model=net("GX"),
            n_constraint=config.n_constraint_inv,
            **common,
        ),
        "DA+PI_IV": lambda: IVCopSens(
            gamma=gamma,
            epsilon=epsilon,
            epsilon_iv=epsilon_iv,
            pad=pad,
            outcome_model=net("GX"),
            n_constraint=config.n_constraint_iv,
            **common,
        ),
        # `pad` reaches the DA branch only, so pad=false gives H_pi n H_pi~ (Thm. 1
        # under exact invariance) and pad=true gives H_pi n (H_pi~ +- eps) (Cor. 1)
        "PI&DA+PI": lambda: IntCopSens(
            gamma=gamma, epsilon=epsilon, pad=pad, outcome_models={"X": net("X"), "GX": net("GX")}, **common
        ),
        "PI&DA+PI_IV": lambda: IntIVCopSens(
            gamma=gamma,
            epsilon=epsilon,
            epsilon_iv=epsilon_iv,
            pad=pad,
            outcome_models={"X": net("X"), "GX": net("GX")},
            n_constraint=config.n_constraint_iv,
            **common,
        ),
    }
    if set(all_builders) != set(COPSENS_METHODS):
        raise ValueError("COPSENS_METHODS out of sync.")

    # a HARD error, not a silent filter: quietly dropping 4 of 11 requested methods
    # is how a run comes back missing columns with nothing in the log
    unknown = sorted(set(method_names) - set(all_builders))
    if unknown:
        raise ValueError(f"the copsens backend does not define {unknown}; valid: {sorted(all_builders)}.")
    return {name: all_builders[name] for name in method_names}


class MethodRegistry:
    """Registry for building method instances with proper configuration."""

    @staticmethod
    def build_methods(
        method_names: list[str],
        gamma: float,
        epsilon: float,
        calibrate: bool = False,
        pad: bool = False,
        clipy: bool = True,
        epsilon_iv: float | None = None,
        n_jobs: int = 1,
        backend: Literal["partial_r2", "copsens"] = "partial_r2",
        outcome_models: dict[str, Any] | None = None,
        n_components: int = 32,
    ) -> dict[str, Callable]:
        """
        Build only requested methods with given hyperparameters.

        `pad` is applied to DA+ methods only; baseline PI/PI_INV/PI_IV never pad.
        `gamma_z` is never set: no experiment uses instruments.

        Args:
            method_names: List of method names to build
            gamma: Confounding budget gamma (Asm. 2)
            epsilon: Invariance error epsilon = ||W|| over the full
                augmentation (§3.1, Thm. 3.A); oracle `epsilon_star`
            calibrate: Scale budgets by the noise level sigma (paper)
            pad: eps-pad DA+ intervals (Thm. 3.A)
            clipy: Clip intervals to the observed outcome range
            epsilon_iv: IV budget ||E[W#|Z-tilde]||, i.e. oracle `eps_iv_star`
                + EPS_TOL. Reaches the IV constraint ONLY -- padding keeps the
                pointwise eps that Thm. 3.A requires.
            n_jobs: query-solve workers; 1 = serial, -1 = all cores
            backend: which PI machinery. 'copsens' swaps in the latent-factor
                model (do-MNIST); its gamma is NOT comparable to partial-r2's.
            outcome_models: {'X': net, 'GX': net}, prefit. copsens only.
            n_components: latent dimension. copsens only.

        Returns:
            Dictionary mapping method names to builder functions
        """
        if backend == "copsens":
            return _copsens_builders(
                method_names,
                gamma=gamma,
                epsilon=epsilon,
                epsilon_iv=epsilon_iv,
                calibrate=calibrate,
                pad=pad,
                clipy=clipy,
                n_jobs=n_jobs,
                outcome_models=outcome_models,
                n_components=n_components,
            )

        common = dict(epsilon=epsilon, calibrate=calibrate, clipy=clipy, n_jobs=n_jobs)
        iv_common = dict(common, epsilon_iv=epsilon_iv)

        all_builders = {
            "ATE": lambda: None,  # ATE computed analytically
            "ERM": lambda: ERM(),
            "DA+ERM": lambda: ERM(),
            "DA+IV": lambda: IV(),
            "PI_INV": lambda: InvPartialR2(gamma=gamma, pad=False, **common),
            "PI": lambda: PartialR2(gamma=gamma, pad=False, **common),
            # baseline PI_IV has a null instrument, so it reduces to PI and
            # never reads the IV budget
            "PI_IV": lambda: IVPartialR2(gamma=gamma, pad=False, **common),
            "DA+PI": lambda: PartialR2(gamma=gamma, pad=pad, **common),
            "DA+PI_IV": lambda: IVPartialR2(gamma=gamma, pad=pad, **iv_common),
            "PI&DA+PI": lambda: IntPartialR2(gamma=gamma, pad=pad, **common),
            "PI&DA+PI_IV": lambda: IntIVPartialR2(gamma=gamma, pad=pad, **iv_common),
        }

        if set(all_builders) != set(ALL_METHODS):
            raise ValueError("ALL_METHODS out of sync.")

        return {name: all_builders[name] for name in method_names if name in all_builders}


# =============================================================================
# ROOT-YAML DATASET BLOCK
# =============================================================================

# keys a dataset block may carry besides `experiment` and the global toggles
DATASET_KEYS: dict[str, set] = {
    "simulation": {"seed", "n_samples", "n_experiments", "sweep_samples", "methods", "augmentation", "kernel_dim"},
    "optical_device": {"seed", "n_samples", "n_experiments", "sweep_samples", "methods", "augmentation"},
    "do_mnist": {
        "seed",
        "n_samples",
        "n_experiments",
        "sweep_samples",
        "methods",
        "augmentation",
        "gamma",
        "epsilon",
        "n_pi",
        "n_queries",
        "n_components",
        "net",
    },
}

TOGGLE_KEYS: set = {"calibrate", "pad", "clipy", "n_jobs"}

# no sensible default: the run is not reproducible / constructible without them
REQUIRED_KEYS: dict[str, set] = {
    "simulation": {"seed", "kernel_dim"},
    "optical_device": {"seed", "augmentation"},
    # `methods` is required HERE and nowhere else: the fallback below is all 11 of
    # ALL_METHODS, and the copsens backend defines only 7. Omitting it would be a
    # hard error mid-run rather than a config error up front.
    "do_mnist": {"seed", "augmentation", "gamma", "epsilon", "methods"},
}


def resolve_dataset_block(name: str, block: dict[str, Any]) -> dict[str, Any]:
    """Validate a dataset block and fill omitted keys from `DatasetDefaults`."""
    block = dict(block)
    block.pop("experiment", None)

    allowed = DATASET_KEYS[name] | TOGGLE_KEYS
    _reject_unknown(block, allowed, f"config.{name}")

    missing = sorted(REQUIRED_KEYS[name] - set(block))
    if missing:
        raise ValueError(f"config.{name} is missing required key(s) {missing}.")

    # catch it here, not minutes into a run inside np.array_split.
    # bool is an int subclass; `n_jobs: true` must not silently mean 1.
    n_jobs = block.get("n_jobs", 1)
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, int) or n_jobs == 0:
        raise ValueError(f"config.{name}.n_jobs must be a non-zero int (1 = serial, -1 = all cores); got {n_jobs!r}.")

    defaults = DATASET_DEFAULTS[name]
    for key in ("n_samples", "n_experiments", "sweep_samples"):
        block.setdefault(key, getattr(defaults, key))
    block.setdefault("methods", list(ALL_METHODS))

    return block
