import numpy as np
from argparse import ArgumentParser
from sklearn.preprocessing import PolynomialFeatures

from src.data_augmentors.simulation import NullSpaceTranslation as DA
from src.sem.simulation import LinearSimulationSEM as SEM
from src.methods.regression import LeastSquaresClosedForm as ERM
from src.methods.sensitivity_models import PartialR2, InvarianceConstrainedPartialR2 as invPartialR2
from src.experiments.utils import (
    fit_model, query_sweep_plot, param_sweep_plot, save, ANNOTATE_SWEEP_PLOT, 
    ANNOTATE_POPULATION_PLOT, radial_sweep_pcs, sweep_along_pc
)
from src.experiments.base import QuerySweepRunner, ParamSweepRunner
from src.experiments.panels import make_panel_4x3

EXPERIMENT = 'linear_simulation'
GAMMA = 2**9
GAMMA0 = 2**9
EPSILON = 2**0
KAPPA = 2**2.5
TEST_FRAC = 0.1

class SimQuerySweep(QuerySweepRunner):
    def __init__(self, kernel_dim, **kwargs):
        super().__init__(**kwargs)
        self.kernel_dim = kernel_dim
        self.sem = SEM()
        self.da = DA(self.sem.W_XY, kernel_dim=self.kernel_dim)
        self.X, self.y = self.sem(N=self.n_samples, kappa=KAPPA)
        self.GX, self.G = self.da(self.X)

    def get_sweep_values(self):
        return radial_sweep_pcs(self.X, self.sweep_samples)

    def setup_data(self):
        return {'sem': self.sem, 'da': self.da, 'X': self.X, 'y': self.y, 'GX': self.GX, 'G': self.G}

class SimParamSweepMixin:
    def setup_runner(self):
        self.sems = [SEM() for _ in range(self.n_experiments)]
        self.das = [DA(sem.W_XY, kernel_dim=self.kernel_dim) for sem in self.sems]
    def get_da(self, experiment_index): return self.das[experiment_index]

class KappaSweep(SimParamSweepMixin, ParamSweepRunner):
    def __init__(self, kernel_dim, **kwargs):
        super().__init__(**kwargs)
        self.kernel_dim = kernel_dim
    def get_param_range(self): return np.linspace(0, 1, num=self.sweep_samples)
    def generate_data(self, experiment_index, param):
        sem, da = self.sems[experiment_index], self.das[experiment_index]
        X, y = sem(N=self.n_samples, kappa=param)
        X_test, _ = sem(N=int(TEST_FRAC * self.n_samples), intervention=True, kappa=param)
        GX, G = da(X, gamma=1.0)
        estimand = X_test @ sem.solution
        return X, y, GX, G, X_test, estimand

class AlphaSweep(SimParamSweepMixin, ParamSweepRunner):
    def __init__(self, kernel_dim, **kwargs):
        super().__init__(**kwargs)
        self.kernel_dim = kernel_dim
    def get_param_range(self): return np.logspace(-1, 2, base=10, num=self.sweep_samples)
    def generate_data(self, experiment_index, param):
        sem, da = self.sems[experiment_index], self.das[experiment_index]
        X, y = sem(N=self.n_samples)
        X_test, _ = sem(N=int(TEST_FRAC * self.n_samples), intervention=True)
        GX, G = da(X, gamma=param)
        estimand = X_test @ sem.solution
        return X, y, GX, G, X_test, estimand

class GammaSweep(SimParamSweepMixin, ParamSweepRunner):
    def __init__(self, kernel_dim, **kwargs):
        super().__init__(**kwargs)
        self.kernel_dim = kernel_dim
    def get_param_range(self): return np.logspace(-5, 11, base=2, num=self.sweep_samples)
    def generate_data(self, experiment_index, param):
        sem, da = self.sems[experiment_index], self.das[experiment_index]
        X, y = sem(N=self.n_samples)
        X_test, _ = sem(N=int(TEST_FRAC * self.n_samples), intervention=True)
        GX, G = da(X, gamma=1.0)
        estimand = X_test @ sem.solution
        return X, y, GX, G, X_test, estimand
    def get_predict_kwargs(self, param): return {'gamma': param, 'gamma0': GAMMA0}

def run_param_sweeps(args, active_methods):
    gamma_methods = {k: v for k, v in active_methods.items() if 'ATE' not in k}
    runner = KappaSweep(**args, methods=active_methods)
    x, res = runner.run("Kappa Sweep")
    save(x, 'kappa_values', EXPERIMENT, 'pkl'); save(res, 'kappa_results', EXPERIMENT, 'pkl')
    param_sweep_plot(x, res, **ANNOTATE_POPULATION_PLOT['kappa'])

    runner = GammaSweep(**args, methods=gamma_methods)
    x, res = runner.run("Gamma Sweep")
    save(x, 'gamma_values', EXPERIMENT, 'pkl'); save(res, 'gamma_results', EXPERIMENT, 'pkl')
    param_sweep_plot(x, res, **ANNOTATE_POPULATION_PLOT['gamma'])

    runner = AlphaSweep(**args, methods=gamma_methods)
    x, res = runner.run("Alpha Sweep")
    save(x, 'alpha_values', EXPERIMENT, 'pkl'); save(res, 'alpha_results', EXPERIMENT, 'pkl')
    param_sweep_plot(x, res, **ANNOTATE_POPULATION_PLOT['alpha'])

def run_panel_logic(runner, sweep_samples):
    # Use the EXISTING runner to guarantee same SEM/Data
    X0 = runner.X
    
    # 3.0 STD range
    pc1_pts, _, _, pc1_vec = sweep_along_pc(X0, 0, sweep_samples, 3.0)
    pc2_pts, _, _, pc2_vec = sweep_along_pc(X0, 1, sweep_samples, 3.0)
    rad_pts = radial_sweep_pcs(X0, sweep_samples)
    
    def run_specific(points):
        original_get = runner.get_sweep_values
        runner.get_sweep_values = lambda: points
        _, results = runner.run(desc="Panel Sweep")
        runner.get_sweep_values = original_get
        return results, results['ATE'] 

    res1, gt1 = run_specific(pc1_pts)
    res2, gt2 = run_specific(pc2_pts)
    resR, gtR = run_specific(rad_pts)
    
    mean = X0.mean(axis=0)
    hist_data = {
        'pc1': ((X0-mean)@pc1_vec, (runner.GX-mean)@pc1_vec),
        'pc2': ((X0-mean)@pc2_vec, (runner.GX-mean)@pc2_vec)
    }
    
    t1 = np.linspace( -3 * np.std((X0-mean)@pc1_vec), 3 * np.std((X0-mean)@pc1_vec), sweep_samples)
    t2 = np.linspace( -3 * np.std((X0-mean)@pc2_vec), 3 * np.std((X0-mean)@pc2_vec), sweep_samples)
    theta = np.linspace(0, 2*np.pi, sweep_samples)
    
    cols = [(res1, gt1, t1), (resR, gtR, theta), (res2, gt2, t2)]
    make_panel_4x3(EXPERIMENT, cols, hist_data)

def run(seed, n_samples, kernel_dim, n_experiments, sweep_samples, methods, 
        sweep_mode='query', hyperparameters=None, plot_panel=False, panel_only=False, **kwargs):
    
    builders = {
        'ATE': lambda: None,
        'ERM': lambda: ERM(),
        'DA+ERM': lambda: ERM(),
        'PI': lambda: PartialR2(gamma=GAMMA, gamma0=GAMMA0),
        'DA+PI': lambda: PartialR2(gamma=GAMMA, gamma0=GAMMA0),
        'INV+PI': lambda: invPartialR2(gamma=GAMMA, gamma0=GAMMA0, epsilon=EPSILON)
    }
    active_methods = {m: builders[m] for m in methods if m in builders}
    common_args = {
        'seed': seed, 'n_samples': n_samples, 'kernel_dim': kernel_dim, 
        'n_experiments': n_experiments, 'sweep_samples': sweep_samples, 
        'hyperparameters': hyperparameters
    }

    if sweep_mode == 'param':
        run_param_sweeps(common_args, active_methods)
        return

    # 1. Create Runner ONCE (Deterministic Seed)
    # Note: n_experiments=1 for query/visual sweeps
    runner = SimQuerySweep(kernel_dim=kernel_dim, seed=seed, n_samples=n_samples,
                        n_experiments=1, sweep_samples=sweep_samples,
                        methods=active_methods, hyperparameters=hyperparameters)

    # 2. Run Panel using that runner (if requested)
    if plot_panel or panel_only:
        run_panel_logic(runner, sweep_samples)
        if panel_only: return

    # 3. Standard Radial Sweep using SAME runner
    # The runner data is already set up.
    # We just call run() which calls get_sweep_values() (Radial)
    _, results = runner.run("Radial Sweep")
    
    angles = np.linspace(0, 2*np.pi, sweep_samples, endpoint=False)
    save(angles, 'treatment_values', EXPERIMENT, 'pkl')
    save(results, 'outcome_values', EXPERIMENT, 'pkl')
    query_sweep_plot(angles, results, **ANNOTATE_SWEEP_PLOT['pc12'], experiment=EXPERIMENT)

if __name__ == '__main__':
    CLI = ArgumentParser()
    CLI.add_argument('--seed', type=int, default=42)
    CLI.add_argument('--n_samples', type=int, default=2500)
    CLI.add_argument('--n_experiments', type=int, default=10)
    CLI.add_argument('--sweep_samples', type=int, default=10)
    CLI.add_argument('--kernel_dim', type=int, default=0)
    CLI.add_argument('--methods', nargs="*", type=str, default=['ERM', 'DA+ERM', 'PI', 'DA+PI', 'INV+PI'])
    CLI.add_argument('--sweep_mode', type=str, choices=['query', 'param'], default='query')
    CLI.add_argument('--plot-panel', action='store_true')
    CLI.add_argument('--panel-only', action='store_true')
    args = CLI.parse_args()
    run(**vars(args))