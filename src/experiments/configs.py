"""
Unified configuration management for experiments.
All parameters defined here - no defaults in method classes.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from src.experiments.utils.constants import _STYLE_KEYS, validate_plot_keys
from src.methods.partial_r2_net import (
    IntersectedIVPartialR2Net,
    IntersectedPartialR2Net,
    IVConstrainedPartialR2Net,
    PartialR2Net,
    RecentredInvPartialR2Net,
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
from src.sem.simulation import TREATMENT_DIMENSION

# =============================================================================
# EXPERIMENT PARAMETERS
# =============================================================================


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for simulation experiments."""

    # query sweep: a RAW squared radius (the PI radius is sqrt(gamma) = 1.0); the
    # query runner divides it by sigma-hat^2 of the draw (`raw_gamma`)
    gamma: float = 1.0
    epsilon: float = 2**-8
    # query sweep only; the sweeps and floor guards use EPS_TOL (2**-5), which is
    # the more favourable setting there
    eps_tol: float = 2**-8
    # SEM confounding. None = fully confounded, which drives sigma^2 to the
    # outcome-noise floor and rho to ~57: Prop. 2 can then never hold.
    gamma_true: float | None = 1.0
    test_fraction: float = 0.1


# @dataclass(frozen=True)
# class OpticalDeviceConfig:
#     """Configuration for optical device experiments."""

#     gamma: float = 2**-2
#     epsilon: float = 2**-2
#     query_epsilon: float = 2**-1.8
#     pad_epsilon: float | None = 0.0
#     epsilon_true: float | None = None
#     test_fraction: float = 0.1
#     dataset_index: int = 8
#     ground_truth_model: Literal["linear", "polynomial"] = "polynomial"


@dataclass(frozen=True)
class OpticalDeviceConfig:
    """Configuration for optical device experiments."""

    # DECLARED, not measured -- deliberately unlike `epsilon` below. gamma is the
    # sensitivity parameter: an assumption about UNOBSERVED confounding, which is
    # the one thing the data cannot report. Reading it off the oracle would make
    # the analysis circular and would erase the point of the gamma sweep, whose
    # whole content is how the interval behaves as the assumed budget crosses the
    # true gamma*. epsilon is different in kind: it bounds |W| for a KNOWN
    # augmentation, a quantity the analyst can simply compute.
    # Consequence to read the query panel with: this gamma is a RAW squared
    # radius (PI radius sqrt(gamma) = 0.5; the query runner divides it by
    # sigma-hat^2 of the draw, `raw_gamma`), about 0.41 in the paper's units
    # against a measured gamma* of 0.662, so h_* is NOT in the identified set
    # there and PI+INV misses it on a minority of queries. That is the
    # assumption being violated, not the solver failing.
    gamma: float = 2**-2
    # None = take the HONEST bound: the measured eps* (+ EPS_TOL), which is what
    # SS3.1's epsilon is -- the constraint E_inv(h) <= eps^2 evaluated at h_*, for
    # the DA in force. Whether the published 2**-2 clears it DEPENDS on that DA
    # (measured: eps* = 0.2121 under `rotation > gaussian-noise`, which config.yaml
    # ships, but 0.2600 under `all`), and a budget below eps* excludes h_* from the
    # PI+INV set, which costs VALIDITY, not just width. Measuring it removes the
    # dependence on which augmentation someone uncomments. A float pins it instead.
    epsilon: float | None = None
    query_epsilon: float | None = None
    # Thm. 3.A's epsilon, a POINTWISE budget on the same defect (SS2.4 states it as
    # a sup; SS3.1's `epsilon` above is an L2 budget on it). None = measured. These
    # are NOT interchangeable: on this device the L2 budget is 0.212 and the
    # pointwise one 0.691 (the q0.99; the raw sup is 1.23), so padding by the
    # former understates Thm. 3.A's own requirement ~3x. See `epsilon_pad_star`.
    pad_epsilon: float | None = None
    epsilon_true: float | None = None
    # query sweep only; the sweeps and floor guards use EPS_TOL (2**-5), which is
    # the more favourable setting there
    eps_tol: float = 2**-8
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
    link: Literal["probit", "logistic"] = "probit"
    # how many trailing layers of the prefit net are refit. l=1 is the 257-param
    # head; l=2 adds the ~74k-param fc1 (slow, AL solver).
    unfrozen_layers: int = 1
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
    # simulation only: the SEM's treatment dimension. None = the dataset has no such key.
    treatment_dim: int | None = None


DATASET_DEFAULTS: dict[str, DatasetDefaults] = {
    "simulation": DatasetDefaults(n_samples=2048, n_experiments=1, sweep_samples=32, treatment_dim=TREATMENT_DIMENSION),
    "optical_device": DatasetDefaults(n_samples=1000, n_experiments=8, sweep_samples=32),
    # sweep_samples = the 10 digit exemplars on the query x-axis
    "do_mnist": DatasetDefaults(n_samples=1_200_000, n_experiments=1, sweep_samples=10),
}


# =============================================================================
# SWEEP PARAMETER / METRIC SPECS
# =============================================================================

# keeps auto-set epsilon off the PI+INV feasibility knife edge (eps=0 forces h~0)
EPS_TOL: float = 2**-5

# Floor guard. An auto-set budget below the constraint's own attainable floor is not
# a tighter bound, it is NO bound: every query comes back INFEASIBLE and the method
# vanishes from the sweep. `budget^2 >= FLOOR_GUARD_R * floor` is enforced in
# `ParamSweepRunner._floor_guard`, which only ever RAISES a budget, and only where the
# oracle value was already unusable.
#
# Calibrated on the simulation trS grid, where the oracle IV budget (EPS_TOL, then 2**-8)
# sits below the floor at the 5 lowest knobs and DA+PI+IV was all-NaN there. Measured
# DA+PI+IV coverage / width at those 5 steps, at budget = sqrt(r * floor):
#   r=2.25  0.82-0.88            under-covers
#   r=4     0.931-1.000  2.9-4.3 two steps under nominal
#   r=9     0.985-1.000  4.2-6.0 <- CHOSEN: covers everywhere, still 24-47% inside
#                                  DA+PI (7.9-8.1), i.e. the constraint is alive
#   r=16    1.000                DA+PI parity: inert, a duplicate column
# The floor moves with gamma, n and the DA draw, so this is a RATIO, never an epsilon.
# It stays ~3 orders of magnitude below eps_rms, so "guard" is not "loose".
FLOOR_GUARD_R: float = 9.0

# the robustness sweep -- and ONLY it -- recalibrates a strength-knob DA to this
# true invariance error, so that eps/eps* is a meaningful ratio axis. Keyed by
# EXPERIMENT_NAME (simulation.py, optical_device.py).
# Why the two differ. The DA+ ball keeps the radius sigma sqrt(gamma*) after
# the DA (0.71 on sim: bias^2 0.505), so h* only leaves it below eps* once the
# DA+ERM centre drifts by that much, and the drift grows with eps*. At the old
# 0.5 nothing ever left the ball and every DA+ coverage curve sat at 1.0.
# Measured on sim (seed 42, d 32, n 2048, recalibrate/pad/mean_match on, clipy
# off, r = eps/eps* from 2^-6 to 1; logs ~/scratch/tmp/impl_v7/logs/runs/):
#   eps*  strength   DA+PI at 2^-6   min DA+PI+IV   DA+PI/PI width at r = 1
#   1     0.09-0.10  1.000           1.000          0.91
#   2     0.18-0.20  1.000           0.983          1.16
#   3     0.26-0.30  0.917           0.885          1.37   (4 exp, 16 steps)
#   4     0.35-0.40  0.757           0.737          1.62   (4 exp, 16 steps)
#   6     0.54-0.61  0.576           0.576          2.13
# All are back at 1.0 by r = 1. 3 is the smallest with a dip the experiment
# band does not swallow (per-experiment 0.84-0.97 at 2^-6), a slope and not a
# cliff; 4 slips under 0.8 and 6 is a failure mode. The price is DA+PI 1.37x
# PI wide at eps = eps*.
# The optical device has no knob under the configured permutation chain, so
# the sweep appends gaussian-noise (ROBUSTNESS_AUGMENTATION) and tunes its
# strength. Measured there (seed 42, same toggles, 2 exp, 8 steps unless
# noted; std(h*) 0.72 on the pool; logs ~/scratch/tmp/impl_v7/logs/optical/runs/):
#   eps*   strength   DA+PI at 2^-6   min DA+PI+IV   DA+PI/PI width at r = 1
#   0.5-3  0.74-2.65  1.000           1.000          0.72-1.05
#   4      3.10       0.980           0.962          1.28   (4 exp, 16 steps; flat on seed 7)
#   5      3.49       0.962           0.952          1.50   (4 exp, 16 steps; seeds 7 / 1: 0.98 / 0.97)
#   6      3.85       0.958           0.952          1.73   (4 exp, 16 steps)
#   8      4.47       0.960           0.955          2.17
# The dip is capped by the device and the toggles, not by eps*. Under
# `recalibrate: true` the DA+PI radius is sigma~ sqrt(gamma*/rho_hat) =
# sigma_X sqrt(gamma*) = 0.635, the baseline PI radius, whatever the noise
# does: rho_hat keeps rising with the strength (2.1 -> 2.3) and is cancelled.
# What the noise moves is the representer norm of the centred clean query
# against the noise-dominated Sigma_GX, and that floors at 2.2 (a second
# moment over its variance for the squared features), so the raw DA+PI
# half-width floors at 0.635 x 2.2 = 1.41 while the DA+ERM centre collapses
# to the train mean (std 0.20 -> 0.06). Coverage then tends to
# P(|h* - mean| < 1.41) = 0.947 on the pool: h* has a 5% right tail (2.1 to
# 3.9) and no left tail, and those same queries miss at every eps* from 5 up
# (0.955-0.96 in the mean, 0.94-0.98 per 100-query split, at 5, 8, 12, 20).
# 5 is the smallest value whose dip shows on every draw seen (4 is flat on
# seed 7) and already sits on that floor; 6 and 8 add width, nothing else.
# It is 7 std of h* and noise 3.5x the pixel std: a tail count of 4-5
# queries in 100, not the sim's slope. Every DA+ line stays above 0.95. The
# tuner solves on one draw; the pooled oracle (8 draws) reads 4.88 for it.
ROBUSTNESS_EPSILON_TRUE: dict[str, float] = {
    "simulation": 3.0,
    "optical_device": 5.0,
}

# The component the robustness sweep APPENDS to the configured DA chain, where
# that chain has no strength knob. None = the configured chain as is. Optical:
# config.yaml ships a permutation-only chain (eps* pinned at 0.254, every DA+
# coverage line flat at 1.0), so the sweep, and ONLY it, runs config.yaml's
# chain plus gaussian-noise (skipped when the chain already carries it, `all`
# included) and retunes its strength to the constant above; the query panel and
# the other sweeps read config.yaml alone. Applied in EpsilonRatioStrategy,
# derived in OpticalOrchestrator._da_factory.
ROBUSTNESS_AUGMENTATION: dict[str, str | None] = {
    "simulation": None,
    "optical_device": "gaussian-noise",
}

# Fraction of Sigma_GX's variance kept before inverting it for tr(S)/k.
# The near-null eigendirections of Sigma_GX are noise and 1/w blows them up, so the
# untruncated estimate is inflated exactly where the DA is strongest. Measured on the
# simulation trS sweep: at the top of the knob grid tr(S)/k reads 0.22889 untruncated
# vs 0.17706 here -- a 23% error, at the end of the axis the sweep is about.
SPECTRUM_KEEP: float = 0.999


# budget-ratio grid: centred on 1, i.e. on the oracle value
def _RATIO_GRID(dataset, n):
    return np.geomspace(2**-6, 2**0, num=n)


# trS x-axis label by the `recalibrate` toggle: the ball in force decides which
# Prop. 2 ratio the axis is (see ExpansionStrategy)
TRS_XLABEL: dict[bool, str] = {
    True: r"$\operatorname{tr}(\mathcal{S})/k$",
    False: r"$\rho \operatorname{tr}(\mathcal{S})/k$",
}


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
        # knob grid; the x-axis actually plotted is the MEASURED expansion of
        # Prop. 2 for the ball in force: tr(S)/k under `recalibrate: true`
        # (the DA+ radius is sigma sqrt(gamma)), rho tr(S)/k under `false` (the
        # radius carries sqrt(rho)). The runner picks the factor and the label
        # (TRS_XLABEL); this static xlabel is the `false` one.
        xlabel=TRS_XLABEL[False],
        # sim: tuned to the informative range: past it both DAs saturate and the
        # measured x moves by less than the across-seed SD (PLAN 5.3).
        # optical: s is the permutation probability of every component
        # (_knob_to_augment_kwargs). Measured on the shipped chain (seed 42):
        # tr(S)/k 0.79 -> 1.09 and rho 1.55 -> 1.67 from s = 0.2 to 0.99, both
        # monotone, so the recalibrated axis tr(S)/k crosses Prop. 2's 1.0 and
        # the inherited-gamma axis rho tr(S)/k (1.23 -> 1.82) sits above it. Below 0.2
        # tr(S)/k folds back (0.94 at s = 0.01, minimum at 0.2 on every seed
        # measured), which would put two knobs on one x and zigzag the sorted
        # line, so the grid starts at 0.2. p = 1 is excluded: the Bernoulli
        # scaler divides by sqrt(p(1-p)) = 0 there (NaN instrument for DA+PI+IV).
        grid_fn=lambda dataset, n: (
            np.logspace(-1.5, 1.0, num=n) if dataset == "simulation" else np.linspace(0.2, 0.99, num=n)
        ),
        xscale="linear",
        vlines=(1.0,),
    ),
    "n": ParamSpec(
        xlabel=r"$n$",
        # grid_fn=lambda dataset, n: np.array(
        #     [128, 256, 512, 1024] if dataset == "simulation" else [128, 256, 512, 1000]  # 1000 = optical pool max
        # ),
        grid_fn=lambda dataset, n: np.linspace(
            128,
            1024 if dataset == "simulation" else 1000,
            16,
            dtype=int,
        ),
    ),
    "m": ParamSpec(
        xlabel=r"Augmentation Folds ($m$)",
        # grid_fn=lambda dataset, n: np.array([1, 2, 4, 8, 16]),
        grid_fn=lambda dataset, n: np.arange(1, 16 + 1),
    ),
    "recalibrate": ParamSpec(
        # the continuous `recalibrate` knob t in [0, 1] (SS4.2): the DA+ methods
        # solve at gamma~ = gamma ((1 - t) + t / rho), from the inherited gamma
        # (t = 0) to gamma/rho (t = 1). The x-axis actually plotted is the
        # MEASURED ratio gamma~/gamma averaged over experiments (each has its own
        # rho_hat), so it runs from 1/rho up to 1 (see RecalibrationStrategy).
        # Same name as the toggle in `defaults:`, different yaml namespace, as
        # `gamma` is.
        xlabel=r"$\tilde{\gamma} / \gamma$",
        grid_fn=lambda dataset, n: np.linspace(0.0, 1.0, num=n),
        xscale="linear",
        include_ate=False,
        data_constant=True,
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
class PerfSpec:
    metric: tuple[str, ...]  # overlay series; bar always drawn


@dataclass(frozen=True)
class ExperimentPlan:
    """Which experiment types to run. Panel is bound to `query`."""

    query: bool = False
    sweep: SweepSpec | None = None
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
    _reject_unknown(block, {"query", "sweep", "perf"}, "experiment")

    sweep_metrics = {k for k, v in METRIC_SPECS.items() if not v.perf_only}
    perf_metrics = {k for k, v in METRIC_SPECS.items() if v.perf_only}

    sweep = block.get("sweep")
    if sweep is not None:
        _reject_unknown(sweep, {"param", "metric"}, "experiment.sweep")
        sweep = SweepSpec(
            param=_check_values(sweep.get("param", ()), PARAM_SPECS, "experiment.sweep.param"),
            metric=_check_values(sweep.get("metric", ()), sweep_metrics, "experiment.sweep.metric"),
        )

    perf = block.get("perf")
    if perf is not None:
        # `param` is meaningless for perf (1-point sweep); accepted and ignored
        _reject_unknown(perf, {"param", "metric"}, "experiment.perf")
        perf = PerfSpec(metric=_check_values(perf.get("metric", ()), perf_metrics, "experiment.perf.metric"))

    return ExperimentPlan(
        query=bool(block.get("query", False)),
        sweep=sweep,
        perf=perf,
    )


# =============================================================================
# PLOT ANNOTATIONS
# =============================================================================

# Keyword arguments of `create_query_sweep_plot`, per query-sweep id; 'pc12' is
# the radial sweep the orchestrator plots (base.py `_plot_query_sweep`). Besides
# `xlabel` and `xscale` every style key of constants._STYLE_KEYS is accepted:
# `legend` (False / True / a matplotlib loc), `x_color`, `y_color`, `title`,
# `title_color`. These apply to BOTH experiments; a per-experiment entry
# `PLOT_CONFIGS[experiment]["query"]` (constants.py) wins over them key by key.
# Example:
#   "pc12": {"xlabel": r"$\vartheta$", "xscale": "linear",
#            "title": r"radial sweep", "title_color": "tab:blue", "legend": "upper left"}
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

validate_plot_keys("ANNOTATE_SWEEP_PLOT", ANNOTATE_SWEEP_PLOT, {"xlabel", "xscale"} | _STYLE_KEYS)

# =============================================================================
# METHOD REGISTRY
# =============================================================================

ALL_METHODS: tuple[str, ...] = (
    "ATE",
    "ERM",
    "DA+ERM",
    "DA+IV",
    "PI+INV",
    "PI",
    "PI+IV",
    "DA+PI",
    "DA+PI+IV",
    "PI&DA+PI",
    "PI&DA+PI+IV",
)

# a strict subset of ALL_METHODS: no 2SLS and no baseline-IV. It DOES define the
# intersections -- Cor. 1 needs h_*(x) inside both intervals, which is a
# membership fact, not a claim that the two balls share a parameterisation.
PARTIAL_R2_NET_METHODS: tuple[str, ...] = (
    "ATE",
    "ERM",
    "DA+ERM",
    "PI+INV",
    "PI",
    "DA+PI",
    "DA+PI+IV",
    "PI&DA+PI",
    "PI&DA+PI+IV",
)


def _partial_r2_net_builders(
    method_names,
    gamma,
    epsilon,
    epsilon_iv,
    recalibrate,
    pad,
    clipy,
    n_jobs,
    mean_match,
    rho,
    outcome_models,
    unfrozen_layers,
):
    """partial_r2_net backend (App. D, (P2)). Every method refits the last
    `unfrozen_layers` layers of a PREFIT outcome net, so only the constraint set
    differs between them."""
    common = dict(
        link=DOMNIST_CONFIG.link,
        unfrozen_layers=unfrozen_layers,
        recalibrate=recalibrate,
        clipy=clipy,
        n_jobs=n_jobs,
        mean_match=mean_match,
    )
    # the standalone DA+ balls carry the step's rho; the intersections read
    # theirs off their two branches (`IntersectedPartialR2Net.rho`)
    da_common = dict(common, rho=rho)

    def net(key):
        if outcome_models is None:
            raise ValueError(
                "partial_r2_net methods need the prefit outcome nets. "
                "`ExperimentOrchestrator.methods` names them only -- the runner "
                "must rebuild via method_factory(..., outcome_models=...) once the "
                "nets exist."
            )
        return outcome_models[key]

    all_builders = {
        "ATE": lambda: None,  # computed via sem.f
        "ERM": lambda: net("X"),  # the prefit net, not a fresh one
        "DA+ERM": lambda: net("GX"),
        "PI": lambda: PartialR2Net(gamma=gamma, epsilon=epsilon, pad=False, outcome_model=net("X"), **common),
        "DA+PI": lambda: PartialR2Net(gamma=gamma, epsilon=epsilon, pad=pad, outcome_model=net("GX"), **da_common),
        # recentred on the post-DA measure: from an X-centred ball the invariant
        # slice is out of reach at any reasonable eps
        "PI+INV": lambda: RecentredInvPartialR2Net(
            gamma=gamma, epsilon=epsilon, pad=False, outcome_model=net("GX"), **common
        ),
        "DA+PI+IV": lambda: IVConstrainedPartialR2Net(
            gamma=gamma, epsilon=epsilon, epsilon_iv=epsilon_iv, pad=pad, outcome_model=net("GX"), **da_common
        ),
        # `pad` reaches the DA branch only (Cor. 1)
        "PI&DA+PI": lambda: IntersectedPartialR2Net(
            gamma=gamma, epsilon=epsilon, pad=pad, outcome_models={"X": net("X"), "GX": net("GX")}, **common
        ),
        "PI&DA+PI+IV": lambda: IntersectedIVPartialR2Net(
            gamma=gamma,
            epsilon=epsilon,
            epsilon_iv=epsilon_iv,
            pad=pad,
            outcome_models={"X": net("X"), "GX": net("GX")},
            **common,
        ),
    }
    if set(all_builders) != set(PARTIAL_R2_NET_METHODS):
        raise ValueError("PARTIAL_R2_NET_METHODS out of sync.")

    unknown = sorted(set(method_names) - set(all_builders))
    if unknown:
        raise ValueError(f"the partial_r2_net backend does not define {unknown}; valid: {sorted(all_builders)}.")
    return {name: all_builders[name] for name in method_names}


class MethodRegistry:
    """Registry for building method instances with proper configuration."""

    @staticmethod
    def build_methods(
        method_names: list[str],
        gamma: float,
        epsilon: float,
        recalibrate: bool = True,
        pad: bool = False,
        pad_epsilon: float | None = None,
        clipy: bool = True,
        epsilon_iv: float | None = None,
        n_jobs: int = 1,
        mean_match: bool = True,
        rho: float = 1.0,
        backend: Literal["partial_r2", "partial_r2_net"] = "partial_r2",
        outcome_models: dict[str, Any] | None = None,
        unfrozen_layers: int = 1,
    ) -> dict[str, Callable]:
        """
        Build only requested methods with given hyperparameters.

        `pad` is applied to DA+ methods only; baseline PI/PI+INV/PI+IV never pad.
        `gamma_z` is never set: no experiment uses instruments.

        Args:
            method_names: List of method names to build
            gamma: Confounding budget gamma (Asm. 2), in the paper's sigma-scaled
                units: every ball has the radius sigma-hat sqrt(gamma)
            epsilon: Invariance error epsilon = ||W|| over the full
                augmentation; the §3.1 CONSTRAINT budget, oracle `epsilon_star`
            pad_epsilon: Thm. 3.A's epsilon, a POINTWISE budget on the same W
                (§2.4 states it as a sup). None pads by `epsilon` instead, which
                is an L2 quantity standing in for a sup -- see `BoundedSA`.
                Oracle `epsilon_pad_star`.
            recalibrate: solve the DA+ balls at the recalibrated budget gamma/rho
                (SS4.2); False keeps the inherited gamma. A float in [0, 1]
                interpolates linearly (the recalibrate sweep).
            rho: the DA draw's information-loss factor sigma~^2/sigma^2, reaching
                the standalone DA+ balls only; the intersections measure their own
            pad: eps-pad DA+ intervals (Thm. 3.A)
            clipy: Clip intervals to the observed outcome range
            epsilon_iv: IV budget ||E[W#|Z-tilde]||, i.e. oracle `eps_iv_star`
                + EPS_TOL. Reaches the IV constraint ONLY -- padding keeps the
                pointwise eps that Thm. 3.A requires.
            n_jobs: query-solve workers; 1 = serial, -1 = all cores
            mean_match: solve on the mean-matched slice E_n[h(X)] = E_n[Y]
                (Lem. 2). False keeps the pre-2026-09 uncentred geometry.
            backend: which PI machinery. 'partial_r2' is the linear SOCP;
                'partial_r2_net' the do-MNIST last-l-layer refit (same Lemma-2
                gamma units, so oracle gamma* is principled).
            outcome_models: {'X': net, 'GX': net}, prefit. partial_r2_net only.
            unfrozen_layers: refit depth `l`. partial_r2_net only.

        Returns:
            Dictionary mapping method names to builder functions
        """
        if backend == "partial_r2_net":
            return _partial_r2_net_builders(
                method_names,
                gamma=gamma,
                epsilon=epsilon,
                epsilon_iv=epsilon_iv,
                recalibrate=recalibrate,
                pad=pad,
                clipy=clipy,
                n_jobs=n_jobs,
                mean_match=mean_match,
                rho=rho,
                outcome_models=outcome_models,
                unfrozen_layers=unfrozen_layers,
            )

        if backend != "partial_r2":
            # a hard error: an unknown backend must not fall through to the linear
            # SOCP and quietly report numbers from a model nobody asked for
            raise ValueError(f"unknown backend {backend!r}; valid: 'partial_r2', 'partial_r2_net'.")

        common = dict(
            epsilon=epsilon,
            pad_epsilon=pad_epsilon,
            recalibrate=recalibrate,
            clipy=clipy,
            n_jobs=n_jobs,
            mean_match=mean_match,
        )
        iv_common = dict(common, epsilon_iv=epsilon_iv)
        # the standalone DA+ balls carry the step's rho; the intersections read
        # theirs off their two branches (`IntersectedPartialR2.rho`)
        da_common = dict(common, rho=rho)
        da_iv_common = dict(iv_common, rho=rho)

        all_builders = {
            "ATE": lambda: None,  # ATE computed analytically
            # Lem. 2's centre is the ERM over a class with a free intercept, so
            # under `mean_match` the plotted point estimators carry one too
            "ERM": lambda: ERM(fit_intercept=mean_match),
            "DA+ERM": lambda: ERM(fit_intercept=mean_match),
            "DA+IV": lambda: IV(fit_intercept=mean_match),
            "PI+INV": lambda: InvPartialR2(gamma=gamma, pad=False, **common),
            "PI": lambda: PartialR2(gamma=gamma, pad=False, **common),
            # baseline PI+IV has a null instrument, so it reduces to PI and
            # never reads the IV budget
            "PI+IV": lambda: IVPartialR2(gamma=gamma, pad=False, **common),
            "DA+PI": lambda: PartialR2(gamma=gamma, pad=pad, **da_common),
            "DA+PI+IV": lambda: IVPartialR2(gamma=gamma, pad=pad, **da_iv_common),
            "PI&DA+PI": lambda: IntPartialR2(gamma=gamma, pad=pad, **common),
            "PI&DA+PI+IV": lambda: IntIVPartialR2(gamma=gamma, pad=pad, **iv_common),
        }

        if set(all_builders) != set(ALL_METHODS):
            raise ValueError("ALL_METHODS out of sync.")

        return {name: all_builders[name] for name in method_names if name in all_builders}


# =============================================================================
# ROOT-YAML DATASET BLOCK
# =============================================================================

# keys a dataset block may carry besides `experiment` and the global toggles
DATASET_KEYS: dict[str, set] = {
    "simulation": {
        "seed",
        "n_samples",
        "n_experiments",
        "sweep_samples",
        "methods",
        "augmentation",
        "kernel_dim",
        "treatment_dim",
    },
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
        "net",
        "unfrozen_layers",
    },
}

TOGGLE_KEYS: set = {"recalibrate", "pad", "clipy", "n_jobs", "mean_match"}

# no sensible default: the run is not reproducible / constructible without them
REQUIRED_KEYS: dict[str, set] = {
    "simulation": {"seed", "kernel_dim"},
    "optical_device": {"seed", "augmentation"},
    # `methods` is required HERE and nowhere else: the fallback below is all 11 of
    # ALL_METHODS, and the partial_r2_net backend defines only 9. Omitting it would
    # be a hard error mid-run rather than a config error up front.
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

    # the toggle is bool-only: the trS factor and label key on it, and a float
    # here would solve the ball in between under the recalibrated label. The
    # solver's predict-time knob (`recalibrate=t`, the sweep) is a different thing.
    recalibrate = block.get("recalibrate", True)
    if not isinstance(recalibrate, bool):
        raise ValueError(
            f"config.{name}.recalibrate must be a bool (true = gamma/rho, false = gamma); got {recalibrate!r}."
        )

    defaults = DATASET_DEFAULTS[name]
    for key in ("n_samples", "n_experiments", "sweep_samples"):
        block.setdefault(key, getattr(defaults, key))
    if defaults.treatment_dim is not None:
        block.setdefault("treatment_dim", defaults.treatment_dim)
        dim = block["treatment_dim"]
        if isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0:
            raise ValueError(f"config.{name}.treatment_dim must be a positive int; got {dim!r}.")
    block.setdefault("methods", list(ALL_METHODS))
    # a stale method name (e.g. an old underscore spelling) must be a config
    # error here, not silently filtered out of the run by the registry
    _reject_unknown(block["methods"], ALL_METHODS, f"config.{name}.methods")

    return block
