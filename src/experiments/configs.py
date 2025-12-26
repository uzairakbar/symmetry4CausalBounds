"""
Unified configuration management for experiments.
All parameters defined here - no defaults in method classes.
"""
import numpy as np
from typing import Dict, Any, Literal, Callable
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
)


# =============================================================================
# EXPERIMENT PARAMETERS
# =============================================================================

@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for simulation experiments."""
    gamma: float = 1.0
    gamma0: float = 1.0
    delta: float = 2**-8
    epsilon: float = 2**-8
    kappa: float = 1.0
    test_fraction: float = 0.1


@dataclass(frozen=True)
class OpticalDeviceConfig:
    """Configuration for optical device experiments."""
    gamma: float = 2**2
    gamma0: float = 2**2
    delta: float = 2**-1
    epsilon: float = 2**2
    test_fraction: float = 0.1
    dataset_index: int = 9
    ground_truth_model: Literal['linear', 'polynomial'] = 'polynomial'


# Default configurations
SIMULATION_CONFIG = SimulationConfig()
OPTICAL_CONFIG = OpticalDeviceConfig()


# =============================================================================
# METRIC CONFIGURATIONS
# =============================================================================

MetricName = Literal['approximation_error', 'worst_error', 'interval_width']

METRIC_CONFIGS: Dict[MetricName, Dict[str, Any]] = {
    'approximation_error': {
        'ylabel': r'average $E_{\mathrm{approx}}^{\operatorname{do}({\bm{x}})}$',
        'normalize': False
    },
    'worst_error': {
        'ylabel': r'average $E_{\mathrm{worst}}^{\operatorname{do}({\bm{x}})}$',
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
        'kappa': SweepConfig(
            range_fn=lambda n: np.linspace(0, 1, num=n),
            xlabel=r'$\kappa$',
            xscale='linear',
        ),
        'alpha': SweepConfig(
            range_fn=lambda n: np.logspace(-1, 2, base=10, num=n),
            xlabel=r'$a$',
            xscale='log',
        ),
        'gamma': SweepConfig(
            range_fn=lambda n: np.logspace(-5, 11, base=2, num=n),
            xlabel=r'$\Gamma$',
            xscale='log',
        ),
    },
    'optical_device': {
        'gamma': SweepConfig(
            range_fn=lambda n: np.logspace(-1, 2, base=2, num=n),
            xlabel=r'$\Gamma$',
            xscale='log',
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
    'kappa': {
        'xlabel': r'$\kappa$',
        'xscale': 'linear',
        'dotted_lines': ['ERM', 'DA+ERM', 'DA+IV'],
    },
    'alpha': {
        'xlabel': r'$a$',
        'xscale': 'log',
        'dotted_lines': ['ERM', 'DA+ERM', 'DA+IV'],
    },
    'gamma': {
        'xlabel': r'$\Gamma$',
        'xscale': 'log',
        'dotted_lines': ['ERM', 'DA+ERM', 'DA+IV'],
    }
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
        gamma0: float,
        delta: float,
        epsilon: float,
    ) -> Dict[str, Callable]:
        """
        Build only requested methods with given hyperparameters.
        
        All parameters are now required - no defaults allowed.
        
        Args:
            method_names: List of method names to build
            gamma: Sensitivity parameter gamma
            gamma0: Base sensitivity parameter
            epsilon: Invariance constraint epsilon
            
        Returns:
            Dictionary mapping method names to builder functions
        """
        all_builders = {
            'ATE': lambda: None,  # ATE computed analytically
            'ERM': lambda: ERM(),
            'DA+ERM': lambda: ERM(),
            'DA+IV': lambda: IV(),
            'PI_INV': lambda: InvPartialR2(gamma=gamma, gamma0=gamma0, epsilon=epsilon),
            'PI': lambda: PartialR2(gamma=gamma, gamma0=gamma0),
            'DA+PI': lambda: PartialR2(gamma=gamma, gamma0=gamma0),
            'DA+PI_IV': lambda: IVPartialR2(gamma=gamma, gamma0=gamma0, delta=delta),
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