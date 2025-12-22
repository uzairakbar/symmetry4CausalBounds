"""
src/experiments/optical_device.py
"""
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

from src.data_augmentors.optical_device import OpticalDeviceDA as DA
from src.sem.optical_device import OpticalDeviceSEM as SEM
from src.experiments.base import QuerySweepRunner, ParamSweepRunner, ExperimentOrchestrator
from src.experiments.configs import MethodRegistry, OPTICAL_PARAMS, SWEEP_CONFIGS
from src.experiments.utils import radial_sweep_pcs

EXPERIMENT = 'optical_device'

# --- Runners ---

class OpticalQuerySweep(QuerySweepRunner):
    def __init__(self, augmentation, **kwargs):
        super().__init__(**kwargs)
        self.sem = SEM(experiment=OPTICAL_PARAMS['optical_ds_idx'], ground_truth=OPTICAL_PARAMS['ground_truth'])
        self.da = DA(augmentation)
        self.poly = PolynomialFeatures(self.sem.poly_degree, include_bias=False)
        self.X_raw, self.y = self.sem(N=self.n_samples)
        self.GX_raw, self.G_raw = self.da(self.X_raw)
        self.X = self.poly.fit_transform(self.X_raw)
        self.GX = self.poly.fit_transform(self.GX_raw)

    def get_sweep_values(self):
        return self.poly.fit_transform(radial_sweep_pcs(self.GX_raw, self.sweep_samples))
    
    def setup_data(self):
        return {'sem': self.sem, 'da': self.da, 'X': self.X, 'y': self.y, 'GX': self.GX, 'G': self.G_raw}

class OpticalGammaSweep(ParamSweepRunner):
    def __init__(self, augmentation, **kwargs):
        self.augmentation = augmentation
        super().__init__(**kwargs)

    def setup_sems_and_das(self):
        self.sem = SEM(experiment=OPTICAL_PARAMS['optical_ds_idx'], ground_truth=OPTICAL_PARAMS['ground_truth'])
        self.da = DA(self.augmentation)
        self.poly = PolynomialFeatures(self.sem.poly_degree, include_bias=False)
        self.X_raw, self.y = self.sem(N=self.n_samples)
        self.GX_raw, self.G_raw = self.da(self.X_raw)
        self.X, self.GX = self.poly.fit_transform(self.X_raw), self.poly.fit_transform(self.GX_raw)

    def get_param_range(self): return SWEEP_CONFIGS['optical_device']['gamma']['range_fn'](self.sweep_samples)
    def get_da(self, idx): return self.da
    def get_predict_kwargs(self, param): return {'gamma': param, 'gamma0': OPTICAL_PARAMS['gamma0']}
    
    def generate_data(self, idx, param):
        X_tr, X_te, y_tr, y_te, GX_tr, _, G_tr, _ = train_test_split(
            self.X, self.y, self.GX, self.G_raw,
            test_size=OPTICAL_PARAMS['test_frac'],
            random_state=self.seed + idx
        )
        return X_tr, y_tr, GX_tr, G_tr, X_te, X_te @ self.sem.solution

# --- Orchestration ---

class OpticalOrchestrator(ExperimentOrchestrator):
    def __init__(self, **kwargs):
        class OpticalRegistry(MethodRegistry):
            @staticmethod
            def build_methods(names):
                return MethodRegistry.build_methods(names, gamma=OPTICAL_PARAMS['gamma'], 
                                                    gamma0=OPTICAL_PARAMS['gamma0'], 
                                                    epsilon=OPTICAL_PARAMS['epsilon'])
        super().__init__(EXPERIMENT, OpticalRegistry(), **kwargs)

    def get_query_runner_cls(self): return OpticalQuerySweep
    def get_param_sweeps(self):
        return [(OpticalGammaSweep, 'gamma')]