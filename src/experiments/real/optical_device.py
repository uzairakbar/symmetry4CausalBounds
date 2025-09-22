import copy
import enlighten
import numpy as np
import scipy as sp
from loguru import logger
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from sklearn.preprocessing import PolynomialFeatures
from typing import Dict, Callable, Optional, List

from src.data_augmentors.real.optical_device import OpticalDeviceDA as DA

from src.sem.real.optical_device import OpticalDeviceSEM as SEM

from src.methods.abstract import pointIdentifier as Regressor
from src.methods.regression import LeastSquaresClosedForm as ERM

from src.methods.sensitivity_models import (
    MarginalSensitivityModel as partialR2
)


from src.experiments.utils import (
    save,
    set_seed,
    box_plot,
    tex_table,
    fit_model,
    sweep_plot,
    estimation_error,
    ANNOTATE_BOX_PLOT,
    ANNOTATE_SWEEP_PLOT,
    radial_sweep_pcs,
    sweep_along_pc,
)


ModelBuilder = Callable[[Optional[float]], Regressor]

MANAGER = enlighten.get_manager()
EXPERIMENT: str='optical_device'
DEFAULT_CV_SAMPLES: int=10
DEFAULT_CV_FRAC: float=0.2
DEFAULT_CV_FOLDS: int=5
DEFAULT_CV_JOBS: int=1
GROUND_TRUTH: str='polynomial'
OPTICAL_DEVICE_DATASET: int=9


class SweepExperiment:
    def __init__(
            self,
            seed: int,
            n_samples: int,
            augmentation: str,
            sweep_samples: int,
            methods: Dict[str, Callable[[Optional[float]], Regressor]],
            hyperparameters: Optional[Dict[str, Dict[str, float]]]=None
        ):
        self.seed = seed
        self.n_samples = n_samples
        self.augmentation = augmentation
        self.sweep_samples = sweep_samples
        self.methods = methods
        self.hyperparameters = hyperparameters
    
    @staticmethod
    def fit(
            method_name: str,
            method: Callable[[Optional[str]], Regressor],
            X, y, G, GX,
            param: float=10.0,
            da: Optional[DA]=None,
            hyperparameters: Optional[Dict[str, Dict[str, float]]]=None
        ) -> Regressor:
        model = method()
        
        fit_model(
            model=model,
            name=method_name,
            X=X, y=y, G=G, GX=GX,
            hyperparameters=hyperparameters,
            da=da
        )
        
        return model
    
    def generate_dataset(self, sem: SEM, da: DA, param: float=10.0):
        X, y = sem(N = self.n_samples)
        GX, G = da(X)
        return X, y, G, GX

    def query_sweep(self, X):
        queries = radial_sweep_pcs(X, self.sweep_samples)
        return queries
        # queries = sweep_along_pc(X, n_steps=self.sweep_samples, pc_index=0)
        # return queries[0]

    def compute_result(self,
               method_name: str,
               method: Callable[[Optional[str]], Regressor],
               X, y, G, GX,
               query: np.ndarray,
               da: Optional[DA]=None) -> float:
        model = self.fit(
            method_name, method, X, y, G, GX, da=da,
            hyperparameters=self.hyperparameters
        )
        bounds = model.predict(query)
        return bounds

    def run_experiment(self):
        if self.seed >= 0:
            set_seed(self.seed)

        sem = SEM(
            experiment=OPTICAL_DEVICE_DATASET,
            ground_truth=GROUND_TRUTH
        )
        da = DA(self.augmentation)
        
        features = PolynomialFeatures(
            sem.poly_degree, include_bias=False
        )
        X, y, G, GX = self.generate_dataset(sem, da)
        
        query_values = self.query_sweep(GX)
        query_values = features.fit_transform(query_values)

        X = features.fit_transform(X)
        G = features.fit_transform(G)
        GX = features.fit_transform(GX)
        
        bounds_dim = (self.sweep_samples, 1, 2)
        results = {
            name: (np.zeros(bounds_dim[:-1]) if 'PI' not in name else np.zeros(bounds_dim))
            for name in self.methods
        }
        
        experiment_name = self.__class__.__name__
        pbar_experiment = MANAGER.counter(
            total=self.sweep_samples, desc=f'{experiment_name}', unit='params'
        )
        for i, query in enumerate(query_values[:, np.newaxis, :]):

            sem_solution = sem.solution
            
            pbar_methods = MANAGER.counter(
                total=len(self.methods), desc=f'SEM {0}', unit='methods', leave=False
            )
            for method_name, method in self.methods.items():
                if method_name == 'ATE':
                    results[method_name][i][0] = query @ sem_solution
                else:
                    results[method_name][i][0] = self.compute_result(
                        method_name, method, X, y, G, GX, query, da=da
                    )

                pbar_methods.update()
            pbar_methods.close()
            pbar_experiment.update()
        pbar_experiment.close()
        return np.linspace(0, 2*np.pi, len(query_values)), results


def run(
        seed: int,
        n_samples: int,
        sweep_samples: int,
        methods: List[str],
        augmentation: Optional[List[str]]=[None],
        hyperparameters: Optional[Dict[str, Dict[str, float]]]=None
    ):
    status = MANAGER.status_bar(
        status_format=u'Optical device{fill}Sweeping {sweep}{fill}{elapsed}',
        color='bold_underline_bright_white_on_lightslategray',
        justify=enlighten.Justify.CENTER, sweep='<parameter>',
        autorefresh=True, min_delta=0.5
    )

    cv = getattr(hyperparameters, 'cv', None)
    all_methods: Dict[str, ModelBuilder] = {
        'ATE': lambda: None,
        'ERM': lambda: ERM(),
        'DA+ERM': lambda: ERM(),
        'PI': lambda: partialR2(),
        'DA+PI': lambda: partialR2(),
    }
    methods: Dict[str, ModelBuilder] = {m: all_methods[m] for m in methods}
    sweep_methods: Dict[str, ModelBuilder] = {
        m: all_methods[m] for m in methods if m in (
            'ERM', 'DA+ERM', 'ATE',
            'PI', 'DA+PI',
        )
    }
    
    # sweep over treatment queries
    status.update(sweep='treatment query')
    logger.info('Sweeping over treatment queries.')
    treatment_values, outcome_values = SweepExperiment(
        seed=seed,
        n_samples=n_samples,
        methods=sweep_methods,
        augmentation=augmentation,
        sweep_samples=sweep_samples,
        hyperparameters=hyperparameters
    ).run_experiment()
    save(
        obj=treatment_values, fname='treatment_values', experiment=EXPERIMENT, format='pkl'
    )
    save(
        obj=outcome_values, fname='outcome_values', experiment=EXPERIMENT, format='pkl'
    )
    sweep_plot(
        treatment_values, outcome_values, **ANNOTATE_SWEEP_PLOT['pc12'], experiment=EXPERIMENT
    )


if __name__ == '__main__':
    CLI = ArgumentParser(description='Optical device experiment.')
    CLI.add_argument(
        '--seed', type=int, default=42, help='Random seed for the experiment. Negative is random.'
    )
    CLI.add_argument(
        '--n_samples', type=int, default=2_500, help='Number of samples per experiment.'
    )
    CLI.add_argument('--n_experiments', type=int, default=10, help='Number of experiments.')
    CLI.add_argument(
        '--sweep_samples', type=int, default=10, help='Sweep resolution across kappa, alpha and gamma.'
    )
    CLI.add_argument(
        '--methods',
        nargs="*",
        type=str,
        default=['ERM', 'DA+ERM', 'DA+IVL-CV', 'DA+IV'],
        help='Methods to use. Specify in space-separated format -- `ERM DA+ERM DA+IVL-CV DA+IV`.'
    )
    args = CLI.parse_args()
    run(**vars(args))