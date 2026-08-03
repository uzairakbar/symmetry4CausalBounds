"""
Unified configuration management for experiments.
All parameters defined here - no defaults in method classes.
"""
import numpy as np
from typing import Dict, Any, Literal, Callable, Optional, Tuple, List
from dataclasses import dataclass, field

from src.methods.regression import (
    LeastSquaresClosedForm as ERM,
    TwoStageLeastSquaresIV as IV,
    GeneralizedMomentMethodIV as GMMIV,
)
from src.methods.sensitivity_models import (
    PartialR2,
    InstrumentalVariablePartialR2 as IVPartialR2,
    InvarianceConstrainedPartialR2 as InvPartialR2,
    IntersectedPartialR2 as IntPartialR2,
    IntersectedInstrumentalVariablePartialR2 as IntIVPartialR2,
)


# =============================================================================
# EXPERIMENT PARAMETERS
# =============================================================================

@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for simulation experiments."""
    gamma: float = 1.0
    epsilon: float = 2**-8
    gamma_true: Optional[float] = None      # SEM confounding; None = fully confounded
    test_fraction: float = 0.1


@dataclass(frozen=True)
class OpticalDeviceConfig:
    """Configuration for optical device experiments."""
    gamma: float = 2**-2
    epsilon: float = 2**-2
    epsilon_true: Optional[float] = None    # DA misspecification target; None = as-is
    test_fraction: float = 0.1
    dataset_index: int = 8
    ground_truth_model: Literal['linear', 'polynomial'] = 'polynomial'
# @dataclass(frozen=True)
# class OpticalDeviceConfig:
#     """Configuration for optical device experiments."""
#     gamma: float = 2**-2
#     gamma0: float = 1.0
#     delta: float = 2**-16
#     epsilon: float = 2**-1
#     test_fraction: float = 0.1
#     dataset_index: int = 2
#     ground_truth_model: Literal['linear', 'polynomial'] = 'polynomial'


# Default configurations
SIMULATION_CONFIG = SimulationConfig()
OPTICAL_CONFIG = OpticalDeviceConfig()


# =============================================================================
# DATASET DEFAULTS (root-yaml fallbacks)
# =============================================================================

@dataclass(frozen=True)
class DatasetDefaults:
    """Filled in when the root yaml omits the key."""
    n_samples: int
    n_experiments: int
    sweep_samples: int


DATASET_DEFAULTS: Dict[str, DatasetDefaults] = {
    'simulation': DatasetDefaults(n_samples=2048, n_experiments=1, sweep_samples=32),
    'optical_device': DatasetDefaults(n_samples=1000, n_experiments=8, sweep_samples=32),
}


# =============================================================================
# SWEEP PARAMETER / METRIC SPECS
# =============================================================================

@dataclass(frozen=True)
class ParamSpec:
    """Axis + policy metadata for one sweepable parameter."""
    xlabel: str
    xscale: Literal['linear', 'log'] = 'log'
    vlines: Tuple[float, ...] = ()      # reference values annotated on the x-axis
    include_ate: bool = True            # ATE is flat, useless on budget-ratio axes
    data_constant: bool = False         # False => data regenerated every step


PARAM_SPECS: Dict[str, ParamSpec] = {
    'gamma': ParamSpec(
        xlabel=r'$\gamma / \gamma^\star$', vlines=(1.0,),
        include_ate=False, data_constant=True,
    ),
    'epsilon': ParamSpec(
        xlabel=r'$\epsilon / \epsilon^\star$', vlines=(1.0,),
        include_ate=False, data_constant=True,
    ),
    'trS': ParamSpec(
        xlabel=r'$\rho \operatorname{tr}(\mathcal{S})/k$',
        xscale='linear', vlines=(1.0,),
    ),
    'n': ParamSpec(xlabel=r'$n$'),
    'm': ParamSpec(xlabel=r'Augmentation Folds ($m$)'),
}


@dataclass(frozen=True)
class MetricSpec:
    """`key` names the QueryEval field / metrics.py function."""
    key: str
    ylabel: str
    yscale: Literal['linear', 'log', 'asinh'] = 'linear'
    perf_only: bool = False


METRIC_SPECS: Dict[str, MetricSpec] = {
    'approx_error': MetricSpec(
        'approximation_error', r'average $\underline{E}_{{\bm{x}}}$', 'asinh'),
    'worst_error': MetricSpec(
        'worst_error', r'average $\overline{E}_{{\bm{x}}}$', 'asinh'),
    'width': MetricSpec('interval_width', r'average interval width'),
    'coverage': MetricSpec('coverage', r'coverage rate'),
    'wall_clock': MetricSpec('wall_clock', r'seconds per query', 'log', perf_only=True),
    'seed_var': MetricSpec('seed_var', r'SD across seeds', perf_only=True),
}


# =============================================================================
# EXPERIMENT PLAN (root-yaml `experiment:` block)
# =============================================================================

@dataclass(frozen=True)
class SweepSpec:
    param: Tuple[str, ...]
    metric: Tuple[str, ...]


@dataclass(frozen=True)
class ScatterSpec:
    param: Tuple[str, ...]
    metric: Tuple[Tuple[str, str], ...]     # (x-metric, y-metric) pairs


@dataclass(frozen=True)
class PerfSpec:
    metric: Tuple[str, ...]                 # overlay series; bar always drawn


@dataclass(frozen=True)
class ExperimentPlan:
    """Which experiment types to run. Panel is bound to `query`."""
    query: bool = False
    sweep: Optional[SweepSpec] = None
    scatter: Optional[ScatterSpec] = None
    perf: Optional[PerfSpec] = None


def _reject_unknown(got, allowed, where: str):
    unknown = sorted(set(got) - set(allowed))
    if unknown:
        raise ValueError(
            f'Unknown key(s) {unknown} in {where}; expected {sorted(allowed)}.'
        )


def _check_values(values, allowed, where: str) -> tuple:
    values = tuple(values)
    _reject_unknown(values, allowed, where)
    return values


def parse_experiment_plan(block: Optional[Dict[str, Any]]) -> ExperimentPlan:
    """Parse+validate the `experiment:` block. Unknown keys are a hard error."""
    block = dict(block or {})
    _reject_unknown(block, {'query', 'sweep', 'scatter', 'perf'}, 'experiment')

    sweep_metrics = {k for k, v in METRIC_SPECS.items() if not v.perf_only}
    perf_metrics = {k for k, v in METRIC_SPECS.items() if v.perf_only}

    sweep = block.get('sweep')
    if sweep is not None:
        _reject_unknown(sweep, {'param', 'metric'}, 'experiment.sweep')
        sweep = SweepSpec(
            param=_check_values(sweep.get('param', ()), PARAM_SPECS, 'experiment.sweep.param'),
            metric=_check_values(sweep.get('metric', ()), sweep_metrics, 'experiment.sweep.metric'),
        )

    scatter = block.get('scatter')
    if scatter is not None:
        _reject_unknown(scatter, {'param', 'metric'}, 'experiment.scatter')
        pairs = []
        for pair in scatter.get('metric', ()):
            if len(pair) != 2:
                raise ValueError(f'experiment.scatter.metric entries must be pairs, got {pair}.')
            pairs.append(_check_values(pair, sweep_metrics, 'experiment.scatter.metric'))
        scatter = ScatterSpec(
            param=_check_values(scatter.get('param', ()), PARAM_SPECS, 'experiment.scatter.param'),
            metric=tuple(pairs),
        )

    perf = block.get('perf')
    if perf is not None:
        # `param` is meaningless for perf (1-point sweep); accepted and ignored
        _reject_unknown(perf, {'param', 'metric'}, 'experiment.perf')
        perf = PerfSpec(
            metric=_check_values(perf.get('metric', ()), perf_metrics, 'experiment.perf.metric')
        )

    return ExperimentPlan(
        query=bool(block.get('query', False)),
        sweep=sweep, scatter=scatter, perf=perf,
    )


# =============================================================================
# METRIC CONFIGURATIONS
# =============================================================================

MetricName = Literal['approximation_error', 'worst_error', 'interval_width']

METRIC_CONFIGS: Dict[MetricName, Dict[str, Any]] = {
    'approximation_error': {
        'ylabel': r'average $\underline{E}_{{\bm{x}}}$',
        'yscale': 'asinh',
        'normalize': False
    },
    'worst_error': {
        'ylabel': r'average $\overline{E}_{{\bm{x}}}$',
        'normalize': False
    },
    'interval_width': {
        'ylabel': r'average interval width',
        'normalize': False
    }
}


# =============================================================================
# SWEEP CONFIGURATIONS
# =============================================================================

@dataclass(frozen=True)
class SweepConfig:
    """Configuration for parameter sweeps."""
    range_fn: Callable[[int], np.ndarray]
    xlabel: str
    xscale: Literal['linear', 'log']


SWEEP_CONFIGS: Dict[str, Dict[str, SweepConfig]] = {
    'simulation': {
        'gamma_star': SweepConfig(
            range_fn=lambda n: np.logspace(-4, 4, base=2, num=n),
            xlabel=r'$\gamma^\star$',
            xscale='log',
        ),
        'trS': SweepConfig(
            range_fn=lambda n: np.logspace(-1, 2, base=10, num=n),
            xlabel=r'$\operatorname{tr}(\mathcal{S})/k$',
            xscale='log',
        ),
        'gamma': SweepConfig(
            range_fn=lambda n: np.linspace(0.1, 1.0, num=n), #np.logspace(-3, 1, base=2, num=n),
            xlabel=r'$\gamma$',
            xscale='log',
        ),
        'folds': SweepConfig(
            range_fn=lambda n: np.array([1, 2, 4, 8, 16, 32]),
            xlabel=r'Augmentation Folds ($m$)',
            xscale='log',
        ),
    },
    'optical_device': {
        'gamma': SweepConfig(
            # range_fn=lambda n: np.logspace(-1, 1.0, base=2, num=n),
            range_fn=lambda n: np.linspace(0.1, 1.0, num=n),
            xlabel=r'$\gamma$',
            xscale='log',
        ),
        'folds': SweepConfig(
            range_fn=lambda n: np.array([1, 2, 4, 8, 16, 32]),
            xlabel=r'Augmentation Folds ($m$)',
            xscale='log',
        ),
        'gamma_star': SweepConfig(
            range_fn=lambda n: np.arange(12),      # dataset index, sorted by gamma*
            xlabel=r'$\gamma^\star$',
            xscale='linear',
        ),
        'trS': SweepConfig(
            range_fn=lambda n: np.linspace(0.01, 1.0, num=n),
            xlabel=r'$\operatorname{tr}(\mathcal{S})/k$',
            xscale='linear',
        ),
    },
}


# =============================================================================
# PLOT ANNOTATIONS
# =============================================================================

ANNOTATE_SWEEP_PLOT: Dict[str, Dict[str, Any]] = {
    'pc1': {
        'xlabel': r'$t$',
        'xscale': 'linear',
    },
    'pc2': {
        'xlabel': r'$t$',
        'xscale': 'linear',
    },
    'pc12': {
        'xlabel': r'$\theta$',
        'xscale': 'linear',
    },
}

ANNOTATE_POPULATION_PLOT: Dict[str, Dict[str, Any]] = {
    'gamma_star': {'xlabel': r'$\gamma^\star$', 'xscale': 'log', 'yscale': 'asinh'},
    'trS': {'xlabel': r'$\operatorname{tr}(\mathcal{S})/k$', 'xscale': 'linear', 'yscale': 'asinh'},
    'gamma': {'xlabel': r'$\gamma$', 'xscale': 'linear', 'yscale': 'asinh'},
    'folds': {'xlabel': r'Augmentation Folds ($m$)', 'xscale': 'log', 'yscale': 'asinh'},
}


# =============================================================================
# METHOD REGISTRY
# =============================================================================

ALL_METHODS: Tuple[str, ...] = (
    'ATE', 'ERM', 'DA+ERM', 'DA+IV',
    'PI_INV', 'PI', 'PI_IV', 'DA+PI', 'DA+PI_IV',
    'PI&DA+PI', 'PI&DA+PI_IV',
)


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
    ) -> Dict[str, Callable]:
        """
        Build only requested methods with given hyperparameters.

        `pad` is applied to DA+ methods only; baseline PI/PI_INV/PI_IV never pad.
        `gamma_z` is never set: no experiment uses instruments.

        Args:
            method_names: List of method names to build
            gamma: Confounding budget gamma (Asm. 2)
            epsilon: Invariance error epsilon (§3.1, Thm. 3.A)
            calibrate: Scale budgets by the noise level sigma (paper)
            pad: eps-pad DA+ intervals (Thm. 3.A)
            clipy: Clip intervals to the observed outcome range

        Returns:
            Dictionary mapping method names to builder functions
        """
        common = dict(epsilon=epsilon, calibrate=calibrate, clipy=clipy)

        all_builders = {
            'ATE': lambda: None,  # ATE computed analytically
            'ERM': lambda: ERM(),
            'DA+ERM': lambda: ERM(),
            'DA+IV': lambda: IV(),
            'PI_INV': lambda: InvPartialR2(gamma=gamma, pad=False, **common),
            'PI': lambda: PartialR2(gamma=gamma, pad=False, **common),
            'PI_IV': lambda: IVPartialR2(gamma=gamma, pad=False, **common),
            'DA+PI': lambda: PartialR2(gamma=gamma, pad=pad, **common),
            'DA+PI_IV': lambda: IVPartialR2(gamma=gamma, pad=pad, **common),
            'PI&DA+PI': lambda: IntPartialR2(gamma=gamma, pad=pad, **common),
            'PI&DA+PI_IV': lambda: IntIVPartialR2(gamma=gamma, pad=pad, **common),
        }

        assert set(all_builders) == set(ALL_METHODS), 'ALL_METHODS out of sync.'

        return {
            name: all_builders[name]
            for name in method_names
            if name in all_builders
        }
    
    @staticmethod
    def filter_gamma_methods(methods: Dict[str, Callable]) -> Dict[str, Callable]:
        """
        Filter out ATE for gamma sweeps (ATE is gamma-independent).
        
        Args:
            methods: Dictionary of all methods
            
        Returns:
            Dictionary with gamma-dependent methods only
        """
        return {k: v for k, v in methods.items() if 'ATE' not in k}

# =============================================================================
# ROOT-YAML DATASET BLOCK
# =============================================================================

# keys a dataset block may carry besides `experiment` and the global toggles
DATASET_KEYS: Dict[str, set] = {
    'simulation': {'seed', 'n_samples', 'n_experiments', 'sweep_samples',
                   'methods', 'augmentation', 'kernel_dim'},
    'optical_device': {'seed', 'n_samples', 'n_experiments', 'sweep_samples',
                       'methods', 'augmentation'},
}

TOGGLE_KEYS: set = {'calibrate', 'pad', 'clipy'}

# no sensible default: the run is not reproducible / constructible without them
REQUIRED_KEYS: Dict[str, set] = {
    'simulation': {'seed', 'kernel_dim'},
    'optical_device': {'seed', 'augmentation'},
}


def resolve_dataset_block(name: str, block: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a dataset block and fill omitted keys from `DatasetDefaults`."""
    block = dict(block)
    block.pop('experiment', None)

    allowed = DATASET_KEYS[name] | TOGGLE_KEYS
    _reject_unknown(block, allowed, f'config.{name}')

    missing = sorted(REQUIRED_KEYS[name] - set(block))
    if missing:
        raise ValueError(f'config.{name} is missing required key(s) {missing}.')

    defaults = DATASET_DEFAULTS[name]
    for key in ('n_samples', 'n_experiments', 'sweep_samples'):
        block.setdefault(key, getattr(defaults, key))
    block.setdefault('methods', list(ALL_METHODS))

    return block
