"""
src/experiments/configs.py - Centralized configuration
"""
import numpy as np
from src.methods.regression import LeastSquaresClosedForm as ERM
from src.methods.sensitivity_models import PartialR2, InvarianceConstrainedPartialR2 as invPartialR2


# Hyperparameters for different experiments
SIMULATION_PARAMS = {
    'gamma': 2**9,
    'gamma0': 2**9,
    'epsilon': 2**0,
    'kappa': 2**2.5,
    'test_frac': 0.1,
}

OPTICAL_PARAMS = {
    'gamma': 2**7,
    'gamma0': 2**7,
    'epsilon': 2**-3,
    'test_frac': 0.1,
    'optical_ds_idx': 9,
    'ground_truth': 'polynomial',
}

# Metric configurations for Y-axis labels and settings
METRIC_CONFIGS = {
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

# Sweep configurations (X-axis settings)
SWEEP_CONFIGS = {
    'simulation': {
        'kappa': {
            'range_fn': lambda n: np.linspace(0, 1, num=n),
            'xlabel': r'$\kappa$',
            'xscale': 'linear',
        },
        'alpha': {
            'range_fn': lambda n: np.logspace(-1, 2, base=10, num=n),
            'xlabel': r'$a$',
            'xscale': 'log',
        },
        'gamma': {
            'range_fn': lambda n: np.logspace(-5, 11, base=2, num=n),
            'xlabel': r'$\Gamma$',
            'xscale': 'log',
        },
    },
    'optical_device': {
        'gamma': {
            'range_fn': lambda n: np.logspace(-5, 11, base=2, num=n),
            'xlabel': r'$\Gamma$',
            'xscale': 'log',
        },
    },
}


class MethodRegistry:
    """Registry for building method instances"""
    
    @staticmethod
    def build_methods(method_names, gamma=None, gamma0=None, epsilon=None):
        """
        Build only requested methods with given hyperparameters.
        """
        all_builders = {
            'ATE': lambda: None,
            'ERM': lambda: ERM(),
            'DA+ERM': lambda: ERM(),
            'PI': lambda: PartialR2(gamma=gamma, gamma0=gamma0),
            'DA+PI': lambda: PartialR2(gamma=gamma, gamma0=gamma0),
            'INV+PI': lambda: invPartialR2(gamma=gamma, gamma0=gamma0, epsilon=epsilon),
        }
        
        return {
            name: all_builders[name] 
            for name in method_names 
            if name in all_builders
        }
    
    @staticmethod
    def filter_gamma_methods(methods):
        """Filter out ATE for gamma sweeps"""
        return {k: v for k, v in methods.items() if 'ATE' not in k}