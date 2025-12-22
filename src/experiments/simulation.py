"""
src/experiments/simulation.py - REFACTORED with minimal duplication
"""
import numpy as np
from argparse import ArgumentParser

from src.data_augmentors.simulation import NullSpaceTranslation as DA
from src.sem.simulation import LinearSimulationSEM as SEM
from src.experiments.base import QuerySweepRunner, ParamSweepRunner
from src.experiments.configs import MethodRegistry, SIMULATION_PARAMS, SWEEP_CONFIGS
from src.experiments.panels import PanelBuilder
from src.experiments.utils import (
    save, query_sweep_plot, param_sweep_plot, 
    ANNOTATE_SWEEP_PLOT, ANNOTATE_POPULATION_PLOT, radial_sweep_pcs
)

EXPERIMENT = 'simulation'


# ============================================================================
# Query Sweep (Visualization)
# ============================================================================

class SimQuerySweep(QuerySweepRunner):
    """Simulation query sweep - radial sweep in PC space"""
    
    def __init__(self, kernel_dim, **kwargs):
        super().__init__(**kwargs)
        self.kernel_dim = kernel_dim
        
        # Setup data
        self.sem = SEM()
        self.da = DA(self.sem.W_XY, kernel_dim=self.kernel_dim)
        self.X, self.y = self.sem(N=self.n_samples, kappa=SIMULATION_PARAMS['kappa'])
        self.GX, self.G = self.da(self.X)
        
        # No transformation needed for linear simulation
        self.X_raw, self.GX_raw = self.X, self.GX

    def get_sweep_values(self):
        return radial_sweep_pcs(self.X, self.sweep_samples)

    def setup_data(self):
        return {
            'sem': self.sem, 
            'da': self.da, 
            'X': self.X, 
            'y': self.y, 
            'GX': self.GX, 
            'G': self.G
        }


# ============================================================================
# Parameter Sweeps (Metrics)
# ============================================================================

class SimParamSweepBase(ParamSweepRunner):
    """Base class for simulation parameter sweeps"""
    
    def __init__(self, kernel_dim, **kwargs):
        self.kernel_dim = kernel_dim
        super().__init__(**kwargs)
    
    def setup_sems_and_das(self):
        """Initialize SEMs and DAs for all experiments"""
        self.sems = [SEM() for _ in range(self.n_experiments)]
        self.das = [DA(sem.W_XY, kernel_dim=self.kernel_dim) for sem in self.sems]
    
    def get_da(self, experiment_index):
        return self.das[experiment_index]


class KappaSweep(SimParamSweepBase):
    """Sweep over confounding strength κ"""
    
    def get_param_range(self):
        return SWEEP_CONFIGS['simulation']['kappa']['range_fn'](self.sweep_samples)
    
    def generate_data(self, experiment_index, param):
        sem, da = self.sems[experiment_index], self.das[experiment_index]
        X, y = sem(N=self.n_samples, kappa=param)
        X_test, _ = sem(
            N=int(SIMULATION_PARAMS['test_frac'] * self.n_samples), 
            intervention=True, 
            kappa=param
        )
        GX, G = da(X, gamma=1.0)
        estimand = X_test @ sem.solution
        return X, y, GX, G, X_test, estimand


class AlphaSweep(SimParamSweepBase):
    """Sweep over augmentation strength α"""
    
    def get_param_range(self):
        return SWEEP_CONFIGS['simulation']['alpha']['range_fn'](self.sweep_samples)
    
    def generate_data(self, experiment_index, param):
        sem, da = self.sems[experiment_index], self.das[experiment_index]
        X, y = sem(N=self.n_samples)
        X_test, _ = sem(
            N=int(SIMULATION_PARAMS['test_frac'] * self.n_samples), 
            intervention=True
        )
        GX, G = da(X, gamma=param)
        estimand = X_test @ sem.solution
        return X, y, GX, G, X_test, estimand


class GammaSweep(SimParamSweepBase):
    """Sweep over sensitivity parameter Γ"""
    
    def get_param_range(self):
        return SWEEP_CONFIGS['simulation']['gamma']['range_fn'](self.sweep_samples)
    
    def generate_data(self, experiment_index, param):
        sem, da = self.sems[experiment_index], self.das[experiment_index]
        X, y = sem(N=self.n_samples)
        X_test, _ = sem(
            N=int(SIMULATION_PARAMS['test_frac'] * self.n_samples), 
            intervention=True
        )
        GX, G = da(X, gamma=1.0)
        estimand = X_test @ sem.solution
        return X, y, GX, G, X_test, estimand
    
    def get_predict_kwargs(self, param):
        return {'gamma': param, 'gamma0': SIMULATION_PARAMS['gamma0']}


# ============================================================================
# Main Entry Point
# ============================================================================

def run(seed, n_samples, kernel_dim, n_experiments, sweep_samples, methods, 
        sweep_mode='query', hyperparameters=None, plot_panel=False, 
        panel_only=False, **kwargs):
    """
    Main entry point for simulation experiments.
    
    Modes:
        - query: Radial sweep visualization
        - param: Parameter sensitivity analysis (kappa, alpha, gamma)
    Flags:
        - plot_panel: Generate 4x3 panel plot
        - panel_only: Only generate panel, skip main plot
    """
    # Build methods
    active_methods = MethodRegistry.build_methods(
        methods, 
        gamma=SIMULATION_PARAMS['gamma'],
        gamma0=SIMULATION_PARAMS['gamma0'],
        epsilon=SIMULATION_PARAMS['epsilon']
    )
    
    common_args = {
        'seed': seed,
        'n_samples': n_samples,
        'kernel_dim': kernel_dim,
        'n_experiments': n_experiments,
        'sweep_samples': sweep_samples,
        'hyperparameters': hyperparameters
    }
    
    # ========================================================================
    # Parameter Sweep Mode
    # ========================================================================
    if sweep_mode == 'param':
        gamma_methods = MethodRegistry.filter_gamma_methods(active_methods)
        
        # Kappa sweep
        runner = KappaSweep(**common_args, methods=active_methods)
        x, res = runner.run("Kappa Sweep")
        save(x, 'kappa_values', EXPERIMENT, 'pkl')
        save(res, 'kappa_results', EXPERIMENT, 'pkl')
        param_sweep_plot(x, res, **ANNOTATE_POPULATION_PLOT['kappa'])
        
        # Gamma sweep
        runner = GammaSweep(**common_args, methods=gamma_methods)
        x, res = runner.run("Gamma Sweep")
        save(x, 'gamma_values', EXPERIMENT, 'pkl')
        save(res, 'gamma_results', EXPERIMENT, 'pkl')
        param_sweep_plot(x, res, **ANNOTATE_POPULATION_PLOT['gamma'])
        
        # Alpha sweep
        runner = AlphaSweep(**common_args, methods=gamma_methods)
        x, res = runner.run("Alpha Sweep")
        save(x, 'alpha_values', EXPERIMENT, 'pkl')
        save(res, 'alpha_results', EXPERIMENT, 'pkl')
        param_sweep_plot(x, res, **ANNOTATE_POPULATION_PLOT['alpha'])
        return
    
    # ========================================================================
    # Query Sweep Mode (with optional panel)
    # ========================================================================
    
    # Create runner ONCE (deterministic seed)
    runner = SimQuerySweep(
        kernel_dim=kernel_dim, 
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
            use_augmented_geometry=False  # Use X for simulation
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
    CLI.add_argument('--n_samples', type=int, default=2500)
    CLI.add_argument('--n_experiments', type=int, default=10)
    CLI.add_argument('--sweep_samples', type=int, default=10)
    CLI.add_argument('--kernel_dim', type=int, default=0)
    CLI.add_argument('--methods', nargs="*", type=str, 
                     default=['ERM', 'DA+ERM', 'PI', 'DA+PI', 'INV+PI'])
    CLI.add_argument('--sweep_mode', type=str, 
                     choices=['query', 'param'], default='query')
    CLI.add_argument('--plot-panel', action='store_true')
    CLI.add_argument('--panel-only', action='store_true')
    args = CLI.parse_args()
    run(**vars(args))