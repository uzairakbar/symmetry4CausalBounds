"""
src/experiments/optical_device.py - REFACTORED with minimal duplication
"""
import numpy as np
from argparse import ArgumentParser
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

from src.data_augmentors.optical_device import OpticalDeviceDA as DA
from src.sem.optical_device import OpticalDeviceSEM as SEM
from src.experiments.base import QuerySweepRunner, ParamSweepRunner
from src.experiments.configs import MethodRegistry, OPTICAL_PARAMS, SWEEP_CONFIGS, METRIC_CONFIGS
from src.experiments.panels import PanelBuilder
from src.experiments.utils import (
    save, query_sweep_plot, param_sweep_plot,
    ANNOTATE_SWEEP_PLOT, ANNOTATE_POPULATION_PLOT, radial_sweep_pcs
)

EXPERIMENT = 'optical_device'


# ============================================================================
# Query Sweep (Visualization)
# ============================================================================

class OpticalQuerySweep(QuerySweepRunner):
    """Optical device query sweep - radial sweep in PC space"""
    
    def __init__(self, augmentation, **kwargs):
        super().__init__(**kwargs)
        self.augmentation = augmentation
        
        # Setup data
        self.sem = SEM(
            experiment=OPTICAL_PARAMS['optical_ds_idx'],
            ground_truth=OPTICAL_PARAMS['ground_truth']
        )
        self.da = DA(self.augmentation)
        self.poly = PolynomialFeatures(self.sem.poly_degree, include_bias=False)
        
        # Generate and transform data
        self.X_raw, self.y = self.sem(N=self.n_samples)
        self.GX_raw, self.G_raw = self.da(self.X_raw)
        self.X = self.poly.fit_transform(self.X_raw)
        self.GX = self.poly.fit_transform(self.GX_raw)

    def get_sweep_values(self):
        """Radial sweep using augmented geometry"""
        raw_queries = radial_sweep_pcs(self.GX_raw, self.sweep_samples)
        return self.poly.fit_transform(raw_queries)

    def setup_data(self):
        return {
            'sem': self.sem,
            'da': self.da,
            'X': self.X,
            'y': self.y,
            'GX': self.GX,
            'G': self.G_raw
        }


# ============================================================================
# Parameter Sweeps (Metrics)
# ============================================================================

class OpticalGammaSweep(ParamSweepRunner):
    """Sweep over sensitivity parameter Γ"""
    
    def __init__(self, augmentation, **kwargs):
        self.augmentation = augmentation
        super().__init__(**kwargs)
    
    def setup_sems_and_das(self):
        """Setup single SEM and DA for optical device"""
        self.sem = SEM(
            experiment=OPTICAL_PARAMS['optical_ds_idx'],
            ground_truth=OPTICAL_PARAMS['ground_truth']
        )
        self.da = DA(self.augmentation)
        self.poly = PolynomialFeatures(self.sem.poly_degree, include_bias=False)
        
        # Generate data once
        self.X_raw, self.y = self.sem(N=self.n_samples)
        self.GX_raw, self.G_raw = self.da(self.X_raw)
        self.X = self.poly.fit_transform(self.X_raw)
        self.GX = self.poly.fit_transform(self.GX_raw)
    
    def get_param_range(self):
        return SWEEP_CONFIGS['optical_device']['gamma']['range_fn'](self.sweep_samples)
    
    def get_da(self, experiment_index):
        return self.da
    
    def generate_data(self, experiment_index, param):
        """Split data for this experiment iteration"""
        X_train, X_test, y_train, y_test, GX_train, _, G_train, _ = train_test_split(
            self.X, self.y, self.GX, self.G_raw,
            test_size=OPTICAL_PARAMS['test_frac'],
            random_state=self.seed + experiment_index
        )
        estimand = X_test @ self.sem.solution
        return X_train, y_train, GX_train, G_train, X_test, estimand
    
    def get_predict_kwargs(self, param):
        return {'gamma': param, 'gamma0': OPTICAL_PARAMS['gamma0']}


# ============================================================================
# Main Entry Point
# ============================================================================

def run(seed, n_samples, sweep_samples, methods, augmentation=None,
        metric='approximation_error', sweep_mode='query', hyperparameters=None, 
        plot_panel=False, panel_only=False, n_experiments=10, **kwargs):
    """
    Main entry point for optical device experiments.
    """
    # Build methods
    active_methods = MethodRegistry.build_methods(
        methods,
        gamma=OPTICAL_PARAMS['gamma'],
        gamma0=OPTICAL_PARAMS['gamma0'],
        epsilon=OPTICAL_PARAMS['epsilon']
    )
    
    common_args = {
        'seed': seed,
        'n_samples': n_samples,
        'sweep_samples': sweep_samples,
        'hyperparameters': hyperparameters,
        'metric': metric # Pass metric to runner
    }
    
    # Get Y-axis label from metric config
    ylabel = METRIC_CONFIGS[metric]['ylabel']
    
    # ========================================================================
    # Parameter Sweep Mode
    # ========================================================================
    if sweep_mode == 'param':
        gamma_methods = MethodRegistry.filter_gamma_methods(active_methods)
        runner = OpticalGammaSweep(
            augmentation,
            n_experiments=n_experiments,
            methods=gamma_methods,
            **common_args
        )
        x, res = runner.run("Gamma Sweep")
        
        # Save with metric name in filename
        save(x, 'gamma_values', EXPERIMENT, 'pkl')
        save(res, f'gamma_{metric}', EXPERIMENT, 'pkl')
        
        # Update plot config with correct Y-label
        plot_config = ANNOTATE_POPULATION_PLOT['gamma'].copy()
        plot_config['ylabel'] = ylabel
        
        param_sweep_plot(x, res, **plot_config)
        return
    
    # ========================================================================
    # Query Sweep Mode (with optional panel)
    # ========================================================================
    
    # Create runner ONCE (deterministic seed)
    runner = OpticalQuerySweep(
        augmentation,
        seed=seed,
        n_samples=n_samples,
        n_experiments=1,  # Query sweeps are single-shot
        sweep_samples=sweep_samples,
        methods=active_methods,
        hyperparameters=hyperparameters
    )
    
    # Generate panel if requested
    if plot_panel or panel_only:
        panel_builder = PanelBuilder(
            runner,
            EXPERIMENT,
            use_augmented_geometry=True  # Use GX for optical device
        )
        panel_builder.build(sweep_samples)
        if panel_only:
            return
    
    # Standard radial sweep
    _, results = runner.run("Radial Sweep")
    angles = np.linspace(0, 2*np.pi, sweep_samples, endpoint=False)
    
    save(angles, 'treatment_values', EXPERIMENT, 'pkl')
    save(results, 'outcome_values', EXPERIMENT, 'pkl')
    query_sweep_plot(
        angles, results,
        **ANNOTATE_SWEEP_PLOT['pc12'],
        experiment=EXPERIMENT
    )


# ============================================================================
# CLI
# ============================================================================

if __name__ == '__main__':
    CLI = ArgumentParser()
    CLI.add_argument('--seed', type=int, default=42)
    CLI.add_argument('--n_samples', type=int, default=1000)
    CLI.add_argument('--sweep_samples', type=int, default=32)
    CLI.add_argument('--methods', nargs="*", type=str,
                     default=['ERM', 'DA+ERM', 'PI', 'DA+PI', 'INV+PI'])
    CLI.add_argument('--augmentation', type=str, default='all')
    CLI.add_argument('--sweep_mode', type=str,
                     choices=['query', 'param'], default='query')
    CLI.add_argument('--metric', type=str, 
                     choices=['approximation_error', 'worst_error', 'interval_width'], 
                     default='approximation_error')
    CLI.add_argument('--plot-panel', action='store_true')
    CLI.add_argument('--panel-only', action='store_true')
    args = CLI.parse_args()
    run(**vars(args))