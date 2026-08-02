"""
Unified configuration management for experiments.
All parameters defined here - no defaults in method classes.
"""
import numpy as np
from typing import Dict, Any, Literal, Callable, Optional
from dataclasses import dataclass

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