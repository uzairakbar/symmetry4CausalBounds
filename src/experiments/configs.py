"""
Unified configuration management for experiments.
All parameters defined here - no defaults in method classes.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from loguru import logger

from src.experiments.utils.constants import (
    _STYLE_KEYS,
    IV_MODE_METHODS,
    parse_method,
    spelled_method,
    validate_plot_keys,
)
from src.methods.copsens import (
    CopSensPI,
    IntersectedCopSens,
    IntersectedIVCopSens,
    InvarianceConstrainedCopSens,
    IVConstrainedCopSens,
    RecentredInvCopSens,
)
from src.methods.regression import (
    LeastSquaresClosedForm as ERM,
)
from src.methods.regression import (
    MomentConstrainedLeastSquares as ERMIV,
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
    InvarianceConstrainedInstrumentalVariablePartialR2 as InvIVPartialR2,
)
from src.methods.sensitivity_models import (
    InvarianceConstrainedPartialR2 as InvPartialR2,
)
from src.methods.sensitivity_models import (
    PartialR2,
)
from src.sem.cigarettes import TREATMENTS as CIGARETTE_TREATMENTS
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
    # query sweep only; the sweeps use EPS_TOL (2**-5), which is
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
    # query sweep only; the sweeps use EPS_TOL (2**-5), which is
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
    # the CopSens outcome link: probit is the configured one (the closed-form h is
    # a Gaussian cdf of the latent shift), gaussian the identity link the closed
    # form verifier of the selection script uses. Not a block key: a run never
    # restates it, and the selection script's verifier reads it
    link: Literal["probit", "gaussian"] = "probit"
    # the CopSens plumbing (src/methods/copsens.py): anchors of the marginalising
    # average, the smaller anchor set of the constraint, the constraint rows of the
    # INV and IV cones, analytic JAX gradients for the SLSQP solves, and the clip
    # of the outcome net's mean to the attainable range
    n_anchors: int = 128
    n_anchors_c: int = 48
    n_constraint_inv: int = 192
    n_constraint_iv: int = 384
    jax_grad: bool = True
    mu_clip: bool = True
    # BLAS threads of each CopSens fit (the factor analysis). The full pool of a
    # loaded 32-core node oversubscribes: the X fit took 26 s at 32 threads, 1 s at 8
    blas_threads: int | None = 8
    # the exemplar tints on the query axis, alternating red/blue down the digits
    exemplar_colors: Literal["alternating", "random"] = "alternating"
    # the tint sweep's image per digit: `test` draws it from MNIST test (an image no
    # net trained on), `exemplar` reuses the exemplar figure's training image; both
    # at `exemplar_seed` (`DoMNISTSEM.tinted`)
    tint_image_source: Literal["test", "exemplar"] = "test"
    # the held-out probe of the ERM report: obs/do paired draws from MNIST test
    probe_seed: int = 7
    probe_samples: int = 10_000
    # the gamma selection (scripts/select_domnist_gamma.py): rows drawn from
    # split C, the bisection bracket on log gamma, its tolerance, the iteration
    # cap and the number of grid points of the coverage figure
    n_select: int = 5_000
    gamma_lo: float = 1e-4
    gamma_hi: float = 10.0
    gamma_tol: float = 0.05
    max_iter: int = 20
    plot_points: int = 8
    # the ERM+INV centre (`InvariantGradientDescentERM`). `erm_inv_tau` is the
    # invariance target on the squared scale, E[(h(X) - h(GX))^2], the fallback of
    # the block key of the same name (4e-4 = 0.02^2 in epsilon units; eps^2 =
    # 0.0016). The rest is the augmented Lagrangian's plumbing: the initial
    # penalty, its growth factor and cap, the multiplier updates per epoch, and the
    # epoch count (None = the block's `hyperparameters.epochs`). The penalty starts
    # small so the net learns the digit before the invariance penalty tightens: at
    # mu0 = 1 the stiff quadratic held the near-constant initial net in place.
    # `InvariantGradientDescentERM`'s own defaults match these
    erm_inv_tau: float = 4e-4
    erm_inv_mu0: float = 1e-4
    erm_inv_growth: float = 2.0
    erm_inv_updates_per_epoch: int = 20
    erm_inv_mu_max: float = 1e4
    erm_inv_epochs: int | None = 2
    # the shift-operator spectrum cut of the prescreen (`diagnostics.shift_operators`)
    spectrum_keep: float = 0.999
    test_fraction: float = 0.1

    @property
    def attainable(self) -> tuple[float, float]:
        """Range h_erm can occupy. mu_y outside it is impossible."""
        lo = (1 - self.beta) * self.alpha + self.beta * self.eta
        return float(lo), float(1.0 - lo)


@dataclass(frozen=True)
class CigaretteConfig:
    """Cigarette panel constants. READ THIS FIRST, on where the budget comes from.

    The SWEEPS never read a declared gamma: `ParamSweepRunner.fit_gamma` returns
    `oracle.gamma_star`, so all six solve at gamma*(target), recomputed per target
    and per spec -- that is where the budget is tuned to the oracle. `QUERY_GAMMA`
    below reaches the query runner alone, where it is DECLARED, as
    `OpticalDeviceConfig.gamma` is and for the reason its comment gives: coverage
    at gamma* exactly is a tautology, since h_* then sits ON the Lem. 2 sphere.
    """

    # None = the measured eps* + EPS_TOL, which is EPS_TOL here: h_* is exactly
    # homogeneous on both targets and the DA translates along v, so the defect is
    # 0 by construction and the budget is pure knife-edge tolerance.
    epsilon: float | None = None
    query_epsilon: float | None = None
    # None pads DA+ intervals by `epsilon` (Thm. 3.A); with eps* = 0 that is
    # 2 x EPS_TOL of width, about 3% of the PI interval, and there is no defect
    # for it to repair.
    pad_epsilon: float | None = None
    epsilon_true: float | None = None
    # query sweep only; the sweeps use EPS_TOL (2**-5)
    eps_tol: float = 2**-8
    test_fraction: float = 0.1
    # the leaky-IV misspecification guard (SS5): FIXED, never swept. The sliver is
    # empty at every spec but t3 with the own-tax anchor, which is what makes its
    # non-emptiness a falsification rather than a p-value.
    gamma_z: float = 2**-8
    # plasmode: the confounding is drawn along `confound_direction` and calibrated
    # so gamma* == gamma_true exactly. v is the spec-S story made explicit, a taste
    # drift co-moving with the price level.
    gamma_true: float = 0.25
    confound_direction: Literal["v", "own_price", "worst_case"] = "v"
    outcome_noise_std: float = 0.1


# Default configurations
SIMULATION_CONFIG = SimulationConfig()
OPTICAL_CONFIG = OpticalDeviceConfig()
DOMNIST_CONFIG = DoMNISTConfig()
CIGARETTE_CONFIG = CigaretteConfig()


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
    # n_samples = the whole balanced panel; ask for more and the SEM resamples
    # whole state histories WITH replacement
    "cigarettes": DatasetDefaults(n_samples=2450, n_experiments=8, sweep_samples=32),
}


# =============================================================================
# SWEEP PARAMETER / METRIC SPECS
# =============================================================================

# keeps auto-set epsilon off the PI+INV feasibility knife edge (eps=0 forces h~0)
# Below the constraint's own floor a budget is left as is and reads INFEASIBLE; it is never
# raised. Where that happens: PLAN v16 SS2.2 (`_floor_report` logs every such cell).
EPS_TOL: float = 2**-5
# Under the IM-CI (`im-ci` > 0) the sweeps pad by eps* alone and keep EPS_TOL on the
# constraints only: the CI is the sampling allowance on the interval that the
# tolerance used to add to the pad (a73, on config.yaml's six methods: coverage
# unchanged in every cell without it), while an empty set has no CI to widen, so the
# constraint's knife edge still needs it.

# the Imbens-Manski CI of the `im-ci` toggle (SS3): B nonparametric bootstrap refits
# of the base units per sweep cell, a unit being a row and, on the fold sweep, its m
# augmented copies; the replicate spread gives s_L, s_U
IM_CI_REPLICATES: int = 100
# a log threshold and nothing else: a cell whose mean valid-replicate fraction is
# under it is logged as a WARNING; with < 2 valid replicates a query keeps its raw bounds
IM_CI_VALID_WARN: float = 0.5
# seeds the bootstrap stream as [offset, seed, j, i]; distinct from CRN_OFFSET
# (generic_runner.py), so a resample never coincides with a DA draw stream
IM_CI_SEED_OFFSET: int = 20_000

# the IV leakiness budget of a non-empty `iv:` (SS2.6). The observed instrument's
# radius is s sqrt(gamma_z), a fraction of the residual sd. DECLARED on every
# path, never oracle: listing instruments asserts they are near perfect. Read only
# when the instrument set is non-empty; an absent key means this.
#
# the DECLARED leak budget of a configured instrument set: Conley's direct-effect
# scale at delta = 0.05, i.e. doubling the excise moves taxed sales by at most 5%
# outside the four prices (r_Z = s sqrt(gamma_z) = 0.133). 2^-8 shipped before and
# sat below the bootstrap noise floor 0.104 of the moment it bounds.
GAMMA_Z_DEFAULT: float = 0.0177

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
# Cigarettes: eps* is LINEAR in the translation strength (0.096 at 0, 0.509 at 0.5,
# 1.007 at 1.0 against the unrestricted target), and at 0.5 the DA+PI coverage runs
# 0.797 at r = 2^-6 up to 1.000 at r = 1 for 1.486x the width -- the same profile
# the simulation's 3.0 was chosen for. a53(v) is where it is re-read.
ROBUSTNESS_EPSILON_TRUE: dict[str, float] = {
    "simulation": 3.0,
    "optical_device": 5.0,
    "cigarettes": 0.5,
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
    # the translation carries its own `strength` knob, so nothing is appended
    "cigarettes": None,
}


# Per-spec QUERY budget for the cigarette panel, a module constant beside the two
# ROBUSTNESS tables, which is this file's pattern for a per-dataset lookup. It is
# kept out of `CigaretteConfig` so that dataclass stays flat and scalar like the
# other three: a dict field under `frozen=True` makes the generated __hash__ raise.
# The rule: the smallest power of two STRICTLY ABOVE the measured gamma*(restricted)
# at the spec, so h_* stays strictly interior on the falsification rows too. A flat
# 2**-2 would sit BELOW gamma* at s (0.2953) and t1 (0.4136) and PI itself could
# miss there. Measured coverage at these values is 1.000 for PI and PI+INV at every
# spec, and the trim barely moves: PI+INV/PI reads 0.868 at t3 here against 0.869
# at gamma*. The SWEEPS are unaffected -- they solve at gamma*(target) and sweep
# the ratio around it, which is where the validity reading lives.
QUERY_GAMMA: dict[str, float] = {"s": 2**-1, "t1": 2**-1, "t2": 2**-2, "t3": 2**-2, "t4": 2**-2}

# Fraction of Sigma_GX's variance kept before inverting it for tr(S)/k.
# The near-null eigendirections of Sigma_GX are noise and 1/w blows them up, so the
# untruncated estimate is inflated exactly where the DA is strongest. Measured on the
# simulation omega sweep: at the top of the knob grid tr(S)/k reads 0.22889 untruncated
# vs 0.17706 here -- a 23% error, at the end of the axis the sweep is about.
SPECTRUM_KEEP: float = 0.999


# budget-ratio grid for the VALIDITY sweep: ratios from 2^-6 up to the oracle
# value, which is the right edge. `gamma` keeps this; the robustness sweep has its
# own, centred on 1 (below)
def _RATIO_GRID(dataset, n):
    return np.geomspace(2**-6, 2**0, num=n)


# how far either side of the oracle budget the ROBUSTNESS grid runs, in octaves.
# 1 puts four under-budget points at 0.500, 0.595, 0.707, 0.841 and four over at
# 1.189, 1.414, 1.682, 2.000. Below 1 that reaches the refutation cliff, where a
# misstated budget excludes h_* and every query reads INFEASIBLE, without spending
# most of the grid there; above 1 it reaches far enough for the over-budget half
# to be visibly monotone. At two octaves four of nine points sit in the
# all-infeasible region.
EPSILON_RATIO_OCTAVES: float = 1.0


def _EPSILON_RATIO_GRID(dataset, n):
    """Budget-ratio grid for the robustness sweep: log-symmetric about 1, with 1
    exactly ON it. An even number of points straddles 1 instead of landing on it
    and `sweep_samples` is even both shipped and in the live config, so the count
    is forced odd: the oracle budget is where the vline sits, where the fitted
    models' budgets are, and where every "r = 1 is the fitted budget" check reads."""
    points = int(n) | 1
    grid = np.geomspace(2.0**-EPSILON_RATIO_OCTAVES, 2.0**EPSILON_RATIO_OCTAVES, num=points)
    grid[points // 2] = 1.0  # exact, whatever the log round trip leaves
    return grid


# omega x-axis label by the `recalibrate` toggle. Both branches ARE Prop. 2's Omega:
# Prop. 2 has Omega := rho [tr(K) + D_B^2] / k and Lem. 3 has S = K + delta* delta, so
# tr(S) = tr(K) + D_B^2 and the `false` branch rho tr(S)/k is Omega as written;
# recalibrating to gamma/rho reduces the same term to tr(S)/k, which the paper again
# calls Omega. The dict stays: the toggle is load-bearing at ExpansionStrategy.xlabel
OMEGA_XLABEL: dict[bool, str] = {
    True: r"$\Omega$",
    False: r"$\Omega$",
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
        grid_fn=_EPSILON_RATIO_GRID,
        vlines=(1.0,),
        include_ate=False,
        data_constant=True,
    ),
    "omega": ParamSpec(
        # knob grid; the x-axis actually plotted is the MEASURED expansion of
        # Prop. 2 for the ball in force: tr(S)/k under `recalibrate: true`
        # (the DA+ radius is sigma sqrt(gamma)), rho tr(S)/k under `false` (the
        # radius carries sqrt(rho)). The runner picks the factor and the label
        # (OMEGA_XLABEL); this static xlabel is the `false` one.
        xlabel=OMEGA_XLABEL[False],
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
        # cigarettes: s multiplies the DA amplitude along v, whose unit is
        # sd(X . v-hat) after FWL, so the grid is the decades either side of it.
        grid_fn=lambda dataset, n: (
            np.logspace(-1.5, 1.0, num=n)
            if dataset == "simulation"
            else np.logspace(-3, 3, num=n, base=2)
            if dataset == "cigarettes"
            else np.linspace(0.2, 0.99, num=n)
        ),
        xscale="linear",
        vlines=(1.0,),
    ),
    "n": ParamSpec(
        xlabel=r"$n$",
        # grid_fn=lambda dataset, n: np.array(
        #     [128, 256, 512, 1024] if dataset == "simulation" else [128, 256, 512, 1000]  # 1000 = optical pool max
        # ),
        # cigarettes: n is the pre-split panel size, a tenth of it up to all 2450
        grid_fn=lambda dataset, n: (
            np.linspace(245, 2450, n, dtype=int)
            if dataset == "cigarettes"
            else np.linspace(
                128,
                1024 if dataset == "simulation" else 1000,
                n,
                dtype=int,
            )
        ),
    ),
    "m": ParamSpec(
        xlabel=r"Augmentation Folds ($m$)",
        # grid_fn=lambda dataset, n: np.array([1, 2, 4, 8, 16]),
        grid_fn=lambda dataset, n: np.arange(1, n + 1),
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
    # the three perf sweeps (src/experiments/perf.py); `wall_clock`'s key still names
    # the QueryEval field the sweeps record, and its numbers are baseline-solve
    # equivalents (perf.py:157); `feasibility` is a rate over the seed_var backends
    "wall_clock": MetricSpec("wall_clock", "cumulative runtime", "log", perf_only=True),
    "seed_var": MetricSpec("seed_var", r"stability", "linear", perf_only=True),
    "feasibility": MetricSpec("feasibility", r"feasible rate (over backends)", "linear", perf_only=True),
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
    metric: tuple[str, ...]  # the perf sweeps to run: any of `wall_clock`, `seed_var`, `feasibility`
    repeats: int = 3  # timed repeats per grid point (wall_clock), the median is kept


@dataclass(frozen=True)
class TintSpec:
    """The do-MNIST tint sweep under `query`: one image per digit, rendered at
    `sweep_samples` tints evenly spaced over `range` (0 pure blue, 1 pure red)."""

    digits: tuple[int, ...]
    sweep_samples: int = 8
    range: tuple[float, float] = (0.0, 1.0)


@dataclass(frozen=True)
class ExperimentPlan:
    """Which experiment types to run. Panel is bound to `query`. `tint` is the
    do-MNIST tint sweep, read by that orchestrator alone."""

    query: bool = False
    sweep: SweepSpec | None = None
    perf: PerfSpec | None = None
    tint: TintSpec | None = None


def _reject_unknown(got, allowed, where: str):
    unknown = sorted(set(got) - set(allowed))
    if unknown:
        raise ValueError(f"Unknown key(s) {unknown} in {where}; expected {sorted(allowed)}.")


def _check_values(values, allowed, where: str) -> tuple:
    values = tuple(values)
    _reject_unknown(values, allowed, where)
    return values


def _parse_tint(tint: Any) -> TintSpec:
    """`experiment.query.tint`: `digit` an int or a list of unique ints in 0..9,
    `sweep_samples` an int >= 2 (default 8), `range` two numbers 0 <= lo < hi <= 1
    (default [0, 1])."""
    where = "experiment.query.tint"
    if not isinstance(tint, dict):
        raise ValueError(f"{where} must be a mapping with `digit` (and `sweep_samples`, `range`); got {tint!r}.")
    _reject_unknown(tint, {"digit", "sweep_samples", "range"}, where)
    if "digit" not in tint:
        raise ValueError(f"{where} needs `digit`: an int or a list of ints in 0..9.")
    digits = tint["digit"] if isinstance(tint["digit"], list | tuple) else [tint["digit"]]
    if not digits or any(isinstance(d, bool) or not isinstance(d, int) or not 0 <= d <= 9 for d in digits):
        raise ValueError(f"{where}.digit must be an int or a non-empty list of ints in 0..9; got {tint['digit']!r}.")
    if len(set(digits)) != len(digits):
        raise ValueError(f"{where}.digit repeats a digit: {tint['digit']!r}.")
    samples = tint.get("sweep_samples", 8)
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 2:
        raise ValueError(f"{where}.sweep_samples must be an int >= 2; got {samples!r}.")
    span = tint.get("range", [0.0, 1.0])
    number = (int, float)
    valid = (
        isinstance(span, list | tuple)
        and len(span) == 2
        and all(isinstance(v, number) and not isinstance(v, bool) for v in span)
        and 0.0 <= span[0] < span[1] <= 1.0
    )
    if not valid:
        raise ValueError(f"{where}.range must be [lo, hi] with 0 <= lo < hi <= 1; got {span!r}.")
    return TintSpec(digits=tuple(int(d) for d in digits), sweep_samples=samples, range=(float(span[0]), float(span[1])))


def parse_experiment_plan(block: dict[str, Any] | None) -> ExperimentPlan:
    """Parse+validate the `experiment:` block. Unknown keys are a hard error.
    `query` is a bool, or a mapping with the one key `tint` (the do-MNIST tint
    sweep), which implies `query: true`."""
    block = dict(block or {})
    _reject_unknown(block, {"query", "sweep", "perf"}, "experiment")

    query, tint = block.get("query", False), None
    if isinstance(query, dict):
        _reject_unknown(query, {"tint"}, "experiment.query")
        tint = _parse_tint(query["tint"]) if "tint" in query else None
        query = True

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
        # the perf sweeps run on the epsilon grid and nothing else, so no `param`
        _reject_unknown(perf, {"metric", "repeats"}, "experiment.perf")
        repeats = perf.get("repeats", 3)
        if isinstance(repeats, bool) or not isinstance(repeats, int) or not 3 <= repeats <= 5:
            raise ValueError(f"experiment.perf.repeats must be an int from 3 to 5; got {repeats!r}.")
        perf = PerfSpec(
            metric=_check_values(perf.get("metric", ()), perf_metrics, "experiment.perf.metric"), repeats=repeats
        )

    return ExperimentPlan(
        query=bool(query),
        sweep=sweep,
        perf=perf,
        tint=tint,
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
    # cigarettes: one figure per treatment coordinate, the other three held at the
    # data mean (exactly 0 after FWL), plus the ray along v-hat, where every
    # homogeneous h is 0 and the symmetry alone identifies the point. The ids are
    # also the filenames (`create_query_sweep_plot`'s `fname`): the derived names
    # would be 'logp', 'logy', 'logpn' and 'logmathrmCPI'.
    "dim_p": {
        "xlabel": r"$\log p$",
        "xscale": "linear",
    },
    "dim_y": {
        "xlabel": r"$\log y$",
        "xscale": "linear",
    },
    "dim_pn": {
        "xlabel": r"$\log p_n$",
        "xscale": "linear",
    },
    "dim_cpi": {
        "xlabel": r"$\log \mathrm{CPI}$",
        "xscale": "linear",
    },
    "ray_v": {
        "xlabel": r"$c$",
        "xscale": "linear",
    },
    # the two headline figures of the neighbour-price run (SS10): beta_pn against
    # the confounding budget on the benchmarked range, and against the declared
    # real-Z leakiness budget gamma_z (the radius r_Z = s sqrt(gamma_z)) at the
    # query budget
    "beta_pn_gamma": {
        "xlabel": r"$\gamma$",
        "xscale": "linear",
    },
    "beta_pn_budget": {
        "xlabel": r"$\gamma_z$",
        "xscale": "log",
    },
    # the own-price pair, same axes
    "beta_p_gamma": {
        "xlabel": r"$\gamma$",
        "xscale": "linear",
    },
    "beta_p_budget": {
        "xlabel": r"$\gamma_z$",
        "xscale": "log",
    },
}

# do-MNIST: the tint sweep, one figure per digit (`tint_{d}`), all sharing this id
ANNOTATE_SWEEP_PLOT["tint"] = {"xlabel": r"tint", "xscale": "linear"}

validate_plot_keys("ANNOTATE_SWEEP_PLOT", ANNOTATE_SWEEP_PLOT, {"xlabel", "xscale"} | _STYLE_KEYS)

# =============================================================================
# METHOD REGISTRY
# =============================================================================

ALL_METHODS: tuple[str, ...] = (
    "ATE",
    "ERM",
    "ERM+IV",
    "DA+ERM",
    "DA+ERM+IV",
    "PI+INV",
    "PI",
    "PI+IV",
    # INV before IV, as `PI+INV` is a prefix of it: the dispatch and the TeX
    # composition stay consistent
    "PI+INV+IV",
    "DA+PI",
    "DA+PI+IV",
    "PI&DA+PI",
    "PI&DA+PI+IV",
)

# names only the do-MNIST block accepts, outside ALL_METHODS so the fallback method
# list and the linear registry of the other experiments never see them. ERM+INV is
# the prefit augmented-Lagrangian net (`InvariantGradientDescentERM`), plotted as a
# point estimate
DOMNIST_ONLY_METHODS: tuple[str, ...] = ("ERM+INV",)

# ALL_METHODS without the point-estimate IV of either kind, plus the do-MNIST-only
# ERM+INV. It DOES define the intersections -- Cor. 1 needs h_*(x) inside both
# intervals, which is a membership fact, not a claim that the two balls share a
# parameterisation.
COPSENS_METHODS: tuple[str, ...] = (
    "ATE",
    "ERM",
    "DA+ERM",
    "ERM+INV",
    "PI+INV",
    "PI",
    "DA+PI",
    "DA+PI+IV",
    "PI&DA+PI",
    "PI&DA+PI+IV",
)


def _copsens_builders(
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
    n_components,
    inv_recenter,
    calibrate_sigma,
    gamma_z_star,
):
    """The CopSens backend (do-MNIST): a latent-factor ball around a PREFIT outcome
    net, so only the centre net, the fit rows and the constraint set differ between
    the methods. `inv_recenter` picks PI+INV's centre: `off` is the X net with the
    ball on X and the pairs (X, GX), `on` the GX net with the ball on the unmixed GX
    and the pairs (GX, X), `inv` the ERM+INV net with the ball on X and the pairs
    (X, GX)."""
    common = dict(
        link=DOMNIST_CONFIG.link,
        n_components=n_components,
        n_anchors=DOMNIST_CONFIG.n_anchors,
        n_anchors_c=DOMNIST_CONFIG.n_anchors_c,
        calibrate_sigma=calibrate_sigma,
        # mu_y outside the attainable range is impossible under the SEM
        mu_clip=DOMNIST_CONFIG.attainable if DOMNIST_CONFIG.mu_clip else None,
        jax_grad=DOMNIST_CONFIG.jax_grad,
        blas_threads=DOMNIST_CONFIG.blas_threads,
        recalibrate=recalibrate,
        clipy=clipy,
        n_jobs=n_jobs,
        mean_match=mean_match,
    )
    # the standalone DA+ balls carry the step's rho; the intersections read
    # theirs off their two branches (`IntersectedCopSens.rho`)
    da_common = dict(common, rho=rho)
    inv = dict(n_constraint=DOMNIST_CONFIG.n_constraint_inv)
    iv = dict(n_constraint=DOMNIST_CONFIG.n_constraint_iv, gamma_z_star=gamma_z_star)

    def net(key):
        if outcome_models is None:
            raise ValueError(
                "copsens methods need the prefit outcome nets. "
                "`ExperimentOrchestrator.methods` names them only -- the runner "
                "must rebuild via method_factory(..., outcome_models=...) once the "
                "nets exist."
            )
        if key not in outcome_models:
            raise ValueError(
                f"copsens: no prefit {key!r} net; the replicate trains the ERM+INV net only when "
                "PI+INV runs under inv_recenter 'inv' or ERM+INV is listed (`draw_replicate(train_inv=True)`)."
            )
        return outcome_models[key]

    def pi_inv():
        if inv_recenter == "on":
            return RecentredInvCopSens(
                gamma=gamma, epsilon=epsilon, pad=False, outcome_model=net("GX"), **inv, **common
            )
        if inv_recenter not in ("off", "inv"):
            raise ValueError(f"inv_recenter must be 'off', 'on' or 'inv'; got {inv_recenter!r}.")
        # `inv`: the same class as `off`, fitted on X with the (X, GX) pairs; only
        # the centre moves, to the net trained to invariance on the DA pairs
        centre = net("INV") if inv_recenter == "inv" else net("X")
        return InvarianceConstrainedCopSens(
            gamma=gamma, epsilon=epsilon, pad=False, outcome_model=centre, **inv, **common
        )

    all_builders = {
        "ATE": lambda: None,  # the analytic target, `sem.ate_of` / `sem.h_star`
        "ERM": lambda: net("X"),  # the prefit net, not a fresh one
        "DA+ERM": lambda: net("GX"),
        "ERM+INV": lambda: net("INV"),
        "PI": lambda: CopSensPI(gamma=gamma, epsilon=epsilon, pad=False, outcome_model=net("X"), **common),
        "DA+PI": lambda: CopSensPI(gamma=gamma, epsilon=epsilon, pad=pad, outcome_model=net("GX"), **da_common),
        "PI+INV": pi_inv,
        "DA+PI+IV": lambda: IVConstrainedCopSens(
            gamma=gamma, epsilon=epsilon, epsilon_iv=epsilon_iv, pad=pad, outcome_model=net("GX"), **iv, **da_common
        ),
        # `pad` reaches the DA branch only (Cor. 1)
        "PI&DA+PI": lambda: IntersectedCopSens(
            gamma=gamma, epsilon=epsilon, pad=pad, outcome_models={"X": net("X"), "GX": net("GX")}, **common
        ),
        "PI&DA+PI+IV": lambda: IntersectedIVCopSens(
            gamma=gamma,
            epsilon=epsilon,
            epsilon_iv=epsilon_iv,
            pad=pad,
            outcome_models={"X": net("X"), "GX": net("GX")},
            **iv,
            **common,
        ),
    }
    if set(all_builders) != set(COPSENS_METHODS):
        raise ValueError("COPSENS_METHODS out of sync.")

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
        recalibrate: bool = True,
        pad: bool = False,
        pad_epsilon: float | None = None,
        clipy: bool = True,
        epsilon_iv: float | None = None,
        epsilon_iv_z: float = 0.0,
        gamma_z: float = 0.0,
        n_jobs: int = 1,
        mean_match: bool = True,
        rho: float = 1.0,
        backend: Literal["partial_r2", "copsens"] = "partial_r2",
        outcome_models: dict[str, Any] | None = None,
        n_components: int = 32,
        inv_recenter: Literal["off", "on", "inv"] = "off",
        calibrate_sigma: bool = True,
        gamma_z_star: float = 0.0,
    ) -> dict[str, Callable]:
        """
        Build only requested methods with given hyperparameters.

        `pad` is applied to DA+ methods only; the baselines PI, PI+INV, PI+IV and
        PI+INV+IV never pad. `gamma_z` is the declared real-Z budget of a non-empty
        `iv:` (SS2.6); at its default 0, which is every shipped run, the only
        instrument in play is the DA's own translation amount T. Two budgets, one
        per instrument, and they never mix: `epsilon_iv` is the radius of the
        T-as-IV constraint, `epsilon_iv_z` with `gamma_z` the radius of the
        observed instrument's. A DA+ IV method may carry an instrument mode
        (`parse_method`): bare or `DA+PI+IV(T,Z)` constrains both, `DA+PI+IV(Z)`
        the observed instrument alone, `DA+PI+IV(T)` the translation amounts
        alone. Keys are the spellings as requested.

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
            epsilon_iv: the T-as-IV budget ||E[W#|T]||, oracle `eps_iv_star`
                + EPS_TOL, and the radius of the T constraint alone. Reaches the
                IV constraint ONLY -- padding keeps the pointwise eps that
                Thm. 3.A requires.
            epsilon_iv_z: the observed instrument's measured piece: 0.0 under a
                declared radius (the Z radius is then exactly s sqrt(gamma_z)) and
                under an empty set (inert), else `eps_iv_z_star` read once per
                experiment on the experiment's base sample (the query runner: on
                its own draw) + EPS_TOL. One number for every Z constraint, non-DA
                and DA alike.
            gamma_z: leakiness budget of the observed instruments, `GAMMA_Z_DEFAULT`
                under a non-empty `iv:`; it enters the Z radius only
                (`z_bound`), never the T one
            n_jobs: query-solve workers; 1 = serial, -1 = all cores
            mean_match: solve on the mean-matched slice E_n[h(X)] = E_n[Y]
                (Lem. 2). False keeps the pre-2026-09 uncentred geometry.
            backend: which PI machinery. 'partial_r2' is the linear SOCP;
                'copsens' the do-MNIST latent-factor ball around a prefit net
                (gamma is a LATENT budget there, not the Lemma-2 gamma*).
            outcome_models: {'X': net, 'GX': net}, prefit, plus 'INV' (the
                ERM+INV net) when PI+INV runs under `inv` or ERM+INV is built.
                copsens only.
            n_components: latent dimension of the CopSens factor model. copsens only.
            inv_recenter: PI+INV's centre, 'off' (the X net), 'on' (the GX net,
                the ball on the unmixed GX) or 'inv' (the ERM+INV net, the ball on
                X). copsens only.
            calibrate_sigma: radius sigma-hat sqrt(gamma) with sigma-hat^2 the
                outcome net's own noise on the fit rows. copsens only.
            gamma_z_star: the true baseline IV budget of the T-as-IV cone
                (`copsens.iv_budget`); 0 is none. copsens only.

        Returns:
            Dictionary mapping method names to builder functions
        """
        if backend == "copsens":
            return _copsens_builders(
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
                n_components=n_components,
                inv_recenter=inv_recenter,
                calibrate_sigma=calibrate_sigma,
                gamma_z_star=gamma_z_star,
            )

        if backend != "partial_r2":
            # a hard error: an unknown backend must not fall through to the linear
            # SOCP and quietly report numbers from a model nobody asked for
            raise ValueError(f"unknown backend {backend!r}; valid: 'partial_r2', 'copsens'.")

        common = dict(
            epsilon=epsilon,
            pad_epsilon=pad_epsilon,
            recalibrate=recalibrate,
            clipy=clipy,
            n_jobs=n_jobs,
            mean_match=mean_match,
        )
        # every IV ball carries BOTH radii; which constraints it ends up with
        # follows from the instrument blocks it is FITTED with (`fit_model`), so
        # one kwarg set serves the non-DA methods, the DA+ ones and every mode
        iv_common = dict(common, epsilon_iv=epsilon_iv, epsilon_iv_z=epsilon_iv_z, gamma_z=gamma_z)
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
            # ERM with the IV moment pinned at its floor: `ERM+IV` on the observed
            # Z, `DA+ERM+IV` on the stacked Z-tilde. One stacked block is right
            # here and does NOT re-pool the two budgets of SS2.6: at gamma_z = 0
            # there are no radii, and Pi_[T,Z] r = 0 iff Pi_T r = 0 and Pi_Z r = 0,
            # so the stacked equality IS the pair of separate equalities.
            # `ERM+IV` under an empty set is a config error (`resolve_dataset_block`)
            "ERM+IV": lambda: ERMIV(fit_intercept=mean_match),
            "DA+ERM+IV": lambda: ERMIV(fit_intercept=mean_match),
            "PI+INV": lambda: InvPartialR2(gamma=gamma, pad=False, **common),
            "PI": lambda: PartialR2(gamma=gamma, pad=False, **common),
            # the baseline IV balls carry the IV budget: with an empty instrument
            # the constraint is inert and they reduce to PI and PI+INV exactly,
            # with a real Z the class raises unless `epsilon_iv` is set
            "PI+IV": lambda: IVPartialR2(gamma=gamma, pad=False, **iv_common),
            "PI+INV+IV": lambda: InvIVPartialR2(gamma=gamma, pad=False, **iv_common),
            "DA+PI": lambda: PartialR2(gamma=gamma, pad=pad, **da_common),
            "DA+PI+IV": lambda: IVPartialR2(gamma=gamma, pad=pad, **da_iv_common),
            "PI&DA+PI": lambda: IntPartialR2(gamma=gamma, pad=pad, **common),
            "PI&DA+PI+IV": lambda: IntIVPartialR2(gamma=gamma, pad=pad, **iv_common),
        }

        if set(all_builders) != set(ALL_METHODS):
            raise ValueError("ALL_METHODS out of sync.")

        # the instrument MODE is a fit-time choice (`fit_model` picks the blocks),
        # so only the intersection needs a builder per mode: it fits its own two
        # branches and has to be told what its DA branch sees. `mode=mode` binds
        # the loop variable into the lambda, which is called with no arguments
        mode_builders = {
            mode: {
                "DA+ERM+IV": all_builders["DA+ERM+IV"],
                "DA+PI+IV": all_builders["DA+PI+IV"],
                "PI&DA+PI+IV": (lambda mode=mode: IntIVPartialR2(gamma=gamma, pad=pad, instrument=mode, **iv_common)),
            }
            for mode in ("Z", "T")
        }
        for builders in mode_builders.values():
            if set(builders) != set(IV_MODE_METHODS):
                raise ValueError("IV_MODE_METHODS out of sync.")

        built = {}
        for name in method_names:
            base, mode = parse_method(name)
            if base not in all_builders:
                continue
            built[name] = all_builders[base] if mode == "T,Z" else mode_builders[mode][base]
        return built


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
        "iv",
    },
    # no `iv` here or on do_mnist: the key is rejected on those blocks, not ignored
    "optical_device": {"seed", "n_samples", "n_experiments", "sweep_samples", "methods", "augmentation"},
    "cigarettes": {
        "seed",
        "n_samples",
        "n_experiments",
        "sweep_samples",
        "methods",
        "augmentation",
        "target",
        "spec",
        "anchor",
        "da_amplitude",
        "sliver",
        "iv",
        "gamma_z",
    },
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
        "n_components",
        "mix_in",
        "target_coverage",
        "inv_recenter",
        "erm_inv_tau",
        "gamma_z_star",
        "calibrate_sigma",
        "split",
        "split_seed",
        "pop_seed",
        "exemplar_seed",
        "augmentation_amounts",
    },
}

# `normalize` is a PLOTTING switch (SS10.1), not a solver one: nothing reads it
# before `_run_sweeps`, and the pkls never move. `im-ci` is spelled with the hyphen
# in the yaml and read as `im_ci` (`resolve_dataset_block`)
TOGGLE_KEYS: set = {"recalibrate", "pad", "clipy", "n_jobs", "mean_match", "normalize", "im-ci"}

# no sensible default: the run is not reproducible / constructible without them
REQUIRED_KEYS: dict[str, set] = {
    "simulation": {"seed", "kernel_dim"},
    "optical_device": {"seed", "augmentation"},
    # `target` and `spec` decide what h_* IS, so neither has a defensible default
    "cigarettes": {"seed", "augmentation", "target", "spec"},
    # `methods` is required HERE and nowhere else: the fallback below is the whole
    # of ALL_METHODS, and the copsens backend defines only 10. Omitting it
    # would be a hard error mid-run rather than a config error up front.
    "do_mnist": {"seed", "augmentation", "gamma", "epsilon", "methods"},
}

# legal cigarette instrument names: a treatment instruments itself (included
# exogenous), an excise column is an external instrument
CIGARETTE_INSTRUMENTS: frozenset = frozenset(CIGARETTE_TREATMENTS) | {"tax_s", "tax_sn"}


def _check_instruments(name: str, block: dict[str, Any]) -> bool:
    """Validate `iv` (and `gamma_z`); True when the instrument set is non-empty.

    simulation: a non-negative int, at most `treatment_dim`; 0 or absent is no
    instrument. cigarettes: a list of column names without duplicates (a repeated
    column makes `qr` complete an arbitrary orthonormal basis and that completion
    becomes a spurious moment); [] or absent is no instrument, and the target
    then falls back to the anchor set. `gamma_z` is read only under a non-empty
    set and must then be strictly positive: a radius of exactly 0 makes the
    non-DA +IV bound 0, INFEASIBLE on every query (SS3.3). Every other block has
    no such key.
    """
    if name == "simulation":
        iv = block.get("iv", 0)
        if isinstance(iv, bool) or not isinstance(iv, int) or iv < 0:
            raise ValueError(f"config.simulation.iv must be a non-negative int (0 = no instrument); got {iv!r}.")
        if iv > block["treatment_dim"]:
            raise ValueError(f"config.simulation.iv = {iv} exceeds treatment_dim = {block['treatment_dim']}.")
        return iv > 0
    if name == "cigarettes":
        iv = block.get("iv", [])
        if not isinstance(iv, list) or not all(isinstance(column, str) for column in iv):
            raise ValueError(f"config.cigarettes.iv must be a list of column names ([] = no instrument); got {iv!r}.")
        unknown = sorted(set(iv) - CIGARETTE_INSTRUMENTS)
        if unknown:
            legal = sorted(CIGARETTE_INSTRUMENTS)
            raise ValueError(f"config.cigarettes.iv names unknown column(s) {unknown}; legal: {legal}.")
        if len(set(iv)) != len(iv):
            raise ValueError(f"config.cigarettes.iv repeats a column: {iv!r}.")
        if len(iv) > len(CIGARETTE_TREATMENTS) + 2:
            raise ValueError(f"config.cigarettes.iv lists {len(iv)} columns; at most {len(CIGARETTE_TREATMENTS) + 2}.")
        gamma_z = block.get("gamma_z", GAMMA_Z_DEFAULT)
        if isinstance(gamma_z, bool) or not isinstance(gamma_z, int | float) or not 0.0 <= gamma_z < 1.0:
            raise ValueError(f"config.cigarettes.gamma_z must be a float in [0, 1); got {gamma_z!r}.")
        if iv and gamma_z <= 0.0:
            raise ValueError(
                f"config.cigarettes.gamma_z must be strictly positive under a non-empty iv {iv!r}: a declared "
                f"radius of 0 makes the +IV bound exactly 0, INFEASIBLE on every query; got {gamma_z!r}."
            )
        return len(iv) > 0
    return False


def _check_domnist(block: dict[str, Any]) -> None:
    """The do-MNIST run knobs: every one optional (the orchestrator's constructor
    holds the default), each rejected up front rather than minutes into the nets.
    `inv_recenter` accepts YAML's bare on/off, which the loader reads as booleans,
    and `inv`. `erm_inv_tau` is on the squared scale of eps^2; above it the ERM+INV
    centre may sit outside the eps^2 ball, which is a warning, not an error."""
    prefix = "config.do_mnist"

    def number(key, low, high, integer=False, closed=True):
        value = block.get(key, ...)
        if value is ...:
            return
        kind = int if integer else (int | float)
        inside = (low <= value <= high) if closed else (low <= value < high)
        if isinstance(value, bool) or not isinstance(value, kind) or not inside:
            span = f"[{low}, {high}{']' if closed else ')'}"
            raise ValueError(f"{prefix}.{key} must be {'an int' if integer else 'a number'} in {span}; got {value!r}.")

    number("mix_in", 0.0, 1.0, closed=False)
    number("target_coverage", 0.0, 1.0)
    number("gamma_z_star", 0.0, float("inf"))
    number("n_components", 1, 10**6, integer=True)
    for key in ("split_seed", "pop_seed", "exemplar_seed", "n_pi", "n_queries"):
        number(key, 0, 2**32, integer=True)
    if "net" in block:
        from src.methods.nets import NETS

        if block["net"] not in NETS:
            raise ValueError(f"{prefix}.net must be one of {sorted(NETS)}; got {block['net']!r}.")
    if "calibrate_sigma" in block and not isinstance(block["calibrate_sigma"], bool):
        raise ValueError(f"{prefix}.calibrate_sigma must be a bool; got {block['calibrate_sigma']!r}.")
    if "inv_recenter" in block:
        recenter = {True: "on", False: "off"}.get(block["inv_recenter"], str(block["inv_recenter"]).strip().lower())
        if recenter not in ("off", "on", "inv"):
            raise ValueError(f"{prefix}.inv_recenter must be off, on or inv; got {block['inv_recenter']!r}.")
        block["inv_recenter"] = recenter
    if "erm_inv_tau" in block:
        tau = block["erm_inv_tau"]
        if isinstance(tau, bool) or not isinstance(tau, int | float) or not 0.0 < tau < float("inf"):
            raise ValueError(f"{prefix}.erm_inv_tau must be a positive number (squared scale); got {tau!r}.")
        epsilon = block.get("epsilon")
        if isinstance(epsilon, int | float) and not isinstance(epsilon, bool) and tau > float(epsilon) ** 2:
            logger.warning(
                f"{prefix}.erm_inv_tau = {tau:g} exceeds epsilon^2 = {float(epsilon) ** 2:g}: the ERM+INV centre "
                "may sit outside the eps^2 ball, and PI+INV under inv_recenter 'inv' may be infeasible."
            )
    split = block.get("split")
    if split is not None:
        parts = {"A", "B", "C"}
        if not isinstance(split, dict) or set(split) != parts:
            raise ValueError(f"{prefix}.split must be a dict with exactly the keys A, B, C; got {split!r}.")
        for part, size in split.items():
            if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                raise ValueError(f"{prefix}.split.{part} must be a positive int; got {size!r}.")
    if "pop_seed" in block and block["pop_seed"] == block["seed"] + 1:
        raise ValueError(
            f"{prefix}.pop_seed = {block['pop_seed']} equals seed + 1, the seed the selection script draws its "
            "split-C rows with; the evaluation population must not coincide with it."
        )
    amounts = block.get("augmentation_amounts")
    if amounts is not None:
        from src.data_augmentors.do_mnist import COLOR_AMOUNTS

        if not isinstance(amounts, dict) or not set(amounts) <= set(COLOR_AMOUNTS):
            raise ValueError(f"{prefix}.augmentation_amounts must map ops in {sorted(COLOR_AMOUNTS)}; got {amounts!r}.")
        for op, amount in amounts.items():
            if isinstance(amount, bool) or not isinstance(amount, int | float) or amount < 0:
                raise ValueError(f"{prefix}.augmentation_amounts.{op} must be a non-negative number; got {amount!r}.")


def resolve_dataset_block(name: str, block: dict[str, Any]) -> dict[str, Any]:
    """Validate a dataset block and fill omitted keys from `DatasetDefaults`."""
    block = dict(block)
    experiment = block.pop("experiment", None)

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

    # the toggle is bool-only: the omega factor and label key on it, and a float
    # here would solve the ball in between under the recalibrated label. The
    # solver's predict-time knob (`recalibrate=t`, the sweep) is a different thing.
    recalibrate = block.get("recalibrate", True)
    if not isinstance(recalibrate, bool):
        raise ValueError(
            f"config.{name}.recalibrate must be a bool (true = gamma/rho, false = gamma); got {recalibrate!r}."
        )
    normalize = block.get("normalize", False)
    if not isinstance(normalize, bool):
        raise ValueError(
            f"config.{name}.normalize must be a bool (divide sweep figures by the baseline); got {normalize!r}."
        )
    # the CI level, in percent; 0 / false is the raw bounds. The yaml spells it with
    # the hyphen and Python cannot take that as a keyword, so this is the one rename
    # (`im_ci` in the yaml is unknown above). bool is an int subclass: `true` is no level
    absent = "im-ci" not in block
    im_ci = block.pop("im-ci", 0)
    if absent and name != "do_mnist" and isinstance(experiment, dict) and experiment.get("sweep"):
        logger.info(f"config.{name}: im-ci absent: the sweep reads the raw bounds.")
    im_ci = 0 if im_ci is False else im_ci
    if isinstance(im_ci, bool) or not isinstance(im_ci, int | float) or not (im_ci == 0 or 0 < im_ci < 100):
        raise ValueError(f"config.{name}.im-ci must be 0/false or a percentage in (0, 100); got {im_ci!r}.")
    # `defaults:` reaches every block, do-MNIST's too: its sweep never bootstraps
    if name == "do_mnist" and im_ci:
        logger.info(f"config.do_mnist: im-ci {im_ci:g} ignored; the do-MNIST sweep reads the raw bounds.")
        im_ci = 0
    block["im_ci"] = float(im_ci)

    # dataset-specific, unlike the two guards above: these keys exist on one block
    # only, and an unknown spec would otherwise run silently at the loader's default
    if name == "cigarettes":
        for key, allowed in (
            ("target", {"iv", "plasmode"}),
            ("spec", {"s", "t1", "t2", "t3", "t4"}),
            ("anchor", {"own-tax", "own-and-neighbour-tax"}),
        ):
            value = block.get(key, ...)
            if value is not ... and value not in allowed:
                raise ValueError(f"config.cigarettes.{key} must be one of {sorted(allowed)}; got {value!r}.")
        amplitude = block.get("da_amplitude", 1.0)
        if isinstance(amplitude, bool) or not isinstance(amplitude, int | float) or not 0.0 < amplitude < 1e3:
            raise ValueError(f"config.cigarettes.da_amplitude must be a positive float; got {amplitude!r}.")
        if not isinstance(block.get("sliver", False), bool):
            raise ValueError(f"config.cigarettes.sliver must be a bool; got {block.get('sliver')!r}.")
    if name == "do_mnist":
        _check_domnist(block)

    defaults = DATASET_DEFAULTS[name]
    for key in ("n_samples", "n_experiments", "sweep_samples"):
        block.setdefault(key, getattr(defaults, key))
    if defaults.treatment_dim is not None:
        block.setdefault("treatment_dim", defaults.treatment_dim)
        dim = block["treatment_dim"]
        if isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0:
            raise ValueError(f"config.{name}.treatment_dim must be a positive int; got {dim!r}.")
    # the instrument set decides whether the observed-Z point estimate can run at
    # all, so the fallback method list omits `ERM+IV` without one; listing it by
    # hand is the error below
    has_instruments = _check_instruments(name, block)
    block.setdefault("methods", [method for method in ALL_METHODS if method != "ERM+IV" or has_instruments])
    # every entry parsed (`parse_method`): a stale method name (e.g. an old
    # underscore spelling) or a malformed mode suffix must be a config error
    # here, not silently filtered out of the run by the registry. Stored as
    # `base`, `base(Z)` or `base(T,Z)`; two entries of one (base, mode) pair
    # would collapse into one results key, so they are an error too
    methods, seen = [], {}
    for entry in block["methods"]:
        if not isinstance(entry, str):
            raise ValueError(f"config.{name}.methods must list method names; got {entry!r}.")
        try:
            base, mode = parse_method(entry)
        except ValueError as error:
            raise ValueError(f"config.{name}.methods: {error}") from None
        legal = ALL_METHODS + DOMNIST_ONLY_METHODS if name == "do_mnist" else ALL_METHODS
        if base not in legal:
            _reject_unknown([entry], legal, f"config.{name}.methods")
        if name == "do_mnist" and "(" in entry:
            raise ValueError(f"config.do_mnist.methods: {entry!r} spells an instrument mode; that backend has none.")
        if (base, mode) in seen:
            raise ValueError(
                f"config.{name}.methods lists {seen[(base, mode)]!r} and {entry!r}: the same method twice."
            )
        seen[(base, mode)] = entry
        methods.append(spelled_method(entry))
    block["methods"] = methods

    # an observed-Z point estimate with no instrument has an inert constraint and
    # is plain ERM under another name: loud and up front, never silent. PI+IV and
    # PI+INV+IV are fine under an empty set: they reduce to PI and PI+INV, as
    # DA+PI+IV(Z) reduces to DA+PI.
    for two_stage in ("ERM+IV", "DA+ERM+IV(Z)"):
        if two_stage in block["methods"] and not has_instruments:
            raise ValueError(
                f"config.{name}.methods lists {two_stage!r} but the instrument set is empty "
                f"(iv = {block.get('iv', 'absent')!r}); drop {two_stage!r} or set `iv:`."
            )

    return block
