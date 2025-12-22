import numpy as np
from argparse import ArgumentParser
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

from src.data_augmentors.optical_device import OpticalDeviceDA as DA
from src.sem.optical_device import OpticalDeviceSEM as SEM
from src.methods.regression import LeastSquaresClosedForm as ERM
from src.methods.sensitivity_models import PartialR2, InvarianceConstrainedPartialR2 as invPartialR2
from src.experiments.utils import (
    fit_model, query_sweep_plot, param_sweep_plot, save, ANNOTATE_SWEEP_PLOT, 
    ANNOTATE_POPULATION_PLOT, radial_sweep_pcs, sweep_along_pc
)
from src.experiments.base import QuerySweepRunner, ParamSweepRunner
from src.experiments.panels import make_panel_4x3

EXPERIMENT = 'optical_device'
OPTICAL_DS_IDX = 9
GROUND_TRUTH = 'polynomial'
GAMMA = 2**7
GAMMA0 = 2**7
EPSILON = 2**-3
TEST_FRAC = 0.1

class OpticalSweepMixin:
    """Shared Data Initialization"""
    def __init__(self, augmentation, **kwargs):
        super().__init__(**kwargs)
        self.augmentation = augmentation
        self.sem = SEM(experiment=OPTICAL_DS_IDX, ground_truth=GROUND_TRUTH)
        self.da = DA(self.augmentation)
        self.poly = PolynomialFeatures(self.sem.poly_degree, include_bias=False)
        
        self.X_raw, self.y = self.sem(N=self.n_samples)
        self.GX_raw, self.G_raw = self.da(self.X_raw)
        
        self.X = self.poly.fit_transform(self.X_raw)
        self.GX = self.poly.fit_transform(self.GX_raw)

# --- VISUALIZATION ---
class OpticalQuerySweep(OpticalSweepMixin, QuerySweepRunner):
    def get_sweep_values(self):
        # Default Radial Sweep uses GX geometry (Augmented) for main plot
        # as it usually explores the invariance constraints.
        raw_queries = radial_sweep_pcs(self.GX_raw, self.sweep_samples)
        return self.poly.fit_transform(raw_queries)

    def setup_data(self):
        return {'sem': self.sem, 'da': self.da, 'X': self.X, 'y': self.y, 'GX': self.GX, 'G': self.G_raw}

# --- METRICS ---
class OpticalGammaSweep(OpticalSweepMixin, ParamSweepRunner):
    def get_param_range(self):
        return np.logspace(-5, 11, base=2, num=self.sweep_samples)
    def get_da(self, experiment_index): return self.da
    def generate_data(self, experiment_index, param):
        X_train, X_test, y_train, y_test, GX_train, _, G_train, _ = train_test_split(
            self.X, self.y, self.GX, self.G_raw, 
            test_size=TEST_FRAC, random_state=self.seed + experiment_index
        )
        estimand = X_test @ self.sem.solution
        return X_train, y_train, GX_train, G_train, X_test, estimand
    def get_predict_kwargs(self, param):
        return {'gamma': param, 'gamma0': GAMMA0}

def run_panel(seed, n_samples, sweep_samples, methods, augmentation, hyperparameters):
    runner = OpticalQuerySweep(augmentation, seed=seed, n_samples=n_samples, 
                          n_experiments=1, sweep_samples=sweep_samples, 
                          methods=methods, hyperparameters=hyperparameters)
    
    # 1. Define Geometry using AMBIENT Data (X_raw)
    # This matches the standalone script 'visualize_pca.py --basis original'
    X0 = runner.X_raw
    GX0 = runner.GX_raw
    
    # Calculate PC vectors on X0
    # sweep_along_pc returns: sweep_points, t_values, mean, pc_vector
    pc1_pts, _, mean, pc1_vec = sweep_along_pc(GX0, 0, sweep_samples, 3.0)
    pc2_pts, _, _, pc2_vec = sweep_along_pc(GX0, 1, sweep_samples, 3.0)
    
    # For radial sweep, we can also use X0
    rad_pts = radial_sweep_pcs(GX0, sweep_samples)

    # 2. Run Models on these specific geometries
    def run_specific(raw_points):
        # Transform raw points to Poly
        poly_points = runner.poly.fit_transform(raw_points)
        
        # Inject into runner
        original_get = runner.get_sweep_values
        runner.get_sweep_values = lambda: poly_points
        _, results = runner.run(desc="Panel Sweep")
        runner.get_sweep_values = original_get
        
        # ATE = Poly * Weights
        ate = poly_points @ runner.sem.solution
        return results, ate

    res1, gt1 = run_specific(pc1_pts)
    res2, gt2 = run_specific(pc2_pts)
    resR, gtR = run_specific(rad_pts)

    # 3. Construct Histogram Data
    # Project Raw Data onto the Raw PC vectors
    # IMPORTANT: Use the same 'mean' returned by sweep_along_pc (which is X0.mean)
    proj_X_pc1 = (X0 - mean) @ pc1_vec
    proj_GX_pc1 = (GX0 - mean) @ pc1_vec
    
    proj_X_pc2 = (X0 - mean) @ pc2_vec
    proj_GX_pc2 = (GX0 - mean) @ pc2_vec

    hist_data = {
        'pc1': (proj_X_pc1, proj_GX_pc1),
        'pc2': (proj_X_pc2, proj_GX_pc2)
    }

    # 4. Define Plot Grids
    # Ensure t1/t2 match the scale of the projection
    std_pc1 = np.std(proj_X_pc1)
    std_pc2 = np.std(proj_X_pc2)
    
    t1 = np.linspace(-3 * std_pc1, 3 * std_pc1, sweep_samples)
    t2 = np.linspace(-3 * std_pc2, 3 * std_pc2, sweep_samples)
    theta = np.linspace(0, 2*np.pi, sweep_samples)
    
    cols = [(res1, gt1, t1), (resR, gtR, theta), (res2, gt2, t2)]
    make_panel_4x3(EXPERIMENT, cols, hist_data)

def run(seed, n_samples, sweep_samples, methods, augmentation=None, 
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

    if sweep_mode == 'param':
        gamma_methods = {k: v for k, v in active_methods.items() if 'ATE' not in k}
        runner = OpticalGammaSweep(augmentation, seed=seed, n_samples=n_samples, 
                                   n_experiments=10, 
                                   sweep_samples=sweep_samples, 
                                   methods=gamma_methods, hyperparameters=hyperparameters)
        x, res = runner.run("Gamma Sweep")
        save(x, 'gamma_values', EXPERIMENT, 'pkl')
        save(res, 'gamma_results', EXPERIMENT, 'pkl')
        param_sweep_plot(x, res, **ANNOTATE_POPULATION_PLOT['gamma'])
        return

    if plot_panel or panel_only:
        run_panel(seed, n_samples, sweep_samples, active_methods, augmentation, hyperparameters)
        if panel_only: return

    runner = OpticalQuerySweep(augmentation, seed=seed, n_samples=n_samples, n_experiments=1,
                          sweep_samples=sweep_samples, methods=active_methods, hyperparameters=hyperparameters)
    
    # 1. Run (returns Poly transformed queries)
    _, results = runner.run("Radial Sweep")
    
    # 2. Reconstruct angles for x-axis
    angles = np.linspace(0, 2*np.pi, sweep_samples, endpoint=False)
    
    save(angles, 'treatment_values', EXPERIMENT, 'pkl')
    save(results, 'outcome_values', EXPERIMENT, 'pkl')
    query_sweep_plot(angles, results, **ANNOTATE_SWEEP_PLOT['pc12'], experiment=EXPERIMENT)

if __name__ == '__main__':
    CLI = ArgumentParser()
    CLI.add_argument('--seed', type=int, default=42)
    CLI.add_argument('--n_samples', type=int, default=1000)
    CLI.add_argument('--sweep_samples', type=int, default=32)
    CLI.add_argument('--methods', nargs="*", type=str, default=['ERM', 'DA+ERM', 'PI', 'DA+PI', 'INV+PI'])
    CLI.add_argument('--augmentation', type=str, default='all')
    CLI.add_argument('--sweep_mode', type=str, choices=['query', 'param'], default='query')
    CLI.add_argument('--plot-panel', action='store_true')
    CLI.add_argument('--panel-only', action='store_true')
    args = CLI.parse_args()
    run(**vars(args))