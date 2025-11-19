import copy
import enlighten
import numpy as np
import scipy as sp
from loguru import logger
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from typing import Dict, Callable, Optional, List

from src.data_augmentors.simulation.linear import NullSpaceTranslation as DA

from src.sem.simulation.linear import LinearSimulationSEM as SEM

from src.methods.abstract import pointEstimator as Regressor
from src.methods.regression import LeastSquaresClosedForm as ERM
from src.methods.sensitivity_models import (
    PartialR2,
    InvarianceConstrainedPartialR2 as invPartialR2,
)

from src.experiments.utils import (
    save,
    set_seed,
    tex_table,
    fit_model,
    worst_error,
    interval_width,
    approximation_error,
    ANNOTATE_POPULATION_PLOT,
    ci_sweep_plot as sweep_plot,
)


ModelBuilder = Callable[[Optional[float]], Regressor]

MANAGER = enlighten.get_manager()
EXPERIMENT: str='linear_simulation'
DEFAULT_CV_SAMPLES: int=10
DEFAULT_CV_FRAC: float=0.2
DEFAULT_CV_FOLDS: int=5
DEFAULT_CV_JOBS: int=1
EPSILON: float=2.75
GAMMA0: float=400
GAMMA: float=400


class SweepExperiment(ABC):
    def __init__(
            self,
            seed: int,
            n_samples: int,
            kernel_dim: int,
            n_experiments: int,
            sweep_samples: int,
            methods: Dict[str, Callable[[Optional[float]], Regressor]],
            hyperparameters: Optional[Dict[str, Dict[str, float]]]=None
        ):
        self.seed = seed
        self.n_samples = n_samples
        self.kernel_dim = kernel_dim
        self.n_experiments = n_experiments
        self.sweep_samples = sweep_samples
        self.methods = methods
        self.hyperparameters = hyperparameters
    
    @staticmethod
    def fit(
            method_name: str,
            method: Callable[[Optional[str]], Regressor],
            X, y, GX,
            param: float,
            da: Optional[DA]=None,
            hyperparameters: Optional[Dict[str, Dict[str, float]]]=None
        ) -> Regressor:
        model=method()
        fit_model(
            model=model,
            name=method_name,
            X=X, y=y, GX=GX,
            hyperparameters=hyperparameters,
            da=da
        )
        return model
    
    @abstractmethod
    def predict(self, model: Regressor, X_test: np.ndarray, param: float):
        pass
    
    @abstractmethod
    def generate_dataset(self, sem: SEM, da: DA, param: float):
        pass

    @abstractmethod
    def param_sweep(self):
        pass

    def compute_result(self,
               sem_solution,
               method_name: str,
               method: Callable[[Optional[str]], Regressor],
               X, y, GX, X_test, y_test,
               param: float,
               da: Optional[DA]=None) -> float:
        
        model = self.fit(
            method_name, method, X, y, GX, param, da=da,
            hyperparameters=self.hyperparameters
        )

        estimand = X_test @ sem_solution
        estimate = self.predict(model, X_test, param)
        error = approximation_error(estimand, estimate)

        return error

    def run_experiment(self):
        if self.seed >= 0:
            set_seed(self.seed)
        param_values = self.param_sweep()

        all_sems = []
        all_augmenters = []
        for _ in range(self.n_experiments):
            sem = SEM()
            da = DA(sem.W_XY, kernel_dim=self.kernel_dim)
            all_sems.append(sem)
            all_augmenters.append(da)
        
        error_dim = (self.sweep_samples, self.n_experiments)
        results = {name: np.zeros(error_dim) for name in self.methods}
        
        experiment_name = self.__class__.__name__
        pbar_experiment = MANAGER.counter(
            total=self.sweep_samples, desc=f'{experiment_name}', unit='params'
        )
        for i, param in enumerate(param_values):

            pbar_sem = MANAGER.counter(
                total=self.n_experiments, desc=f'Param. {param:.2f}', unit='experiments', leave=False
            )
            for j, (sem, da) in enumerate(zip(all_sems, all_augmenters)):
                sem_solution = sem.solution

                X, y, GX, X_test, y_test = self.generate_dataset(sem, da, param)
                
                pbar_methods = MANAGER.counter(
                    total=len(self.methods), desc=f'SEM {j}', unit='methods', leave=False
                )
                for method_name, method in self.methods.items():
                    results[method_name][i][j] = self.compute_result(
                        sem_solution, method_name, method, X, y, GX, X_test, y_test, param, da=da
                    )

                    pbar_methods.update()
                pbar_methods.close()
                pbar_sem.update()
            pbar_sem.close()
            pbar_experiment.update()
        pbar_experiment.close()
        return param_values, results


class KappaSweep(SweepExperiment):
    def predict(self, model, X_test, param):
        return model.predict(X_test)
    
    def generate_dataset(self, sem: SEM, da: DA, param: float):
        X, y = sem(N = self.n_samples, kappa = param)
        X_test, y_test = sem(N = self.n_samples, kappa = param)
        GX, _ = da(X, gamma = 1.0)
        return X, y, GX, X_test, y_test

    def param_sweep(self):
        kappa_values = np.linspace(
            0, 1, num=self.sweep_samples
        )
        return kappa_values


class AlphaSweep(SweepExperiment):
    def predict(self, model, X_test, param):
        return model.predict(X_test)
    
    def generate_dataset(self, sem: SEM, da: DA, param: float):
        X, y = sem(N = self.n_samples)
        X_test, y_test = sem(N = self.n_samples)
        GX, _ = da(X, gamma = param)
        return X, y, GX, X_test, y_test
    
    def param_sweep(self):
        alpha_values = np.logspace(
            -1, 2, base=10, num=self.sweep_samples
        )
        return alpha_values


class GammaSweep(SweepExperiment):
    def predict(self, model, X_test, param):
        return model.predict(X_test, gamma=param, gamma0=GAMMA0)
    
    def generate_dataset(self, sem: SEM, da: DA, param: float):
        X, y = sem(N = self.n_samples)
        X_test, y_test = sem(N = self.n_samples)
        GX, _ = da(X, gamma = 1.0)
        return X, y, GX, X_test, y_test

    def param_sweep(self):
        gamma_values = np.logspace(
            0, 3, base=10, num=self.sweep_samples
        )
        return gamma_values


def run(
        seed: int,
        n_samples: int,
        kernel_dim: int,
        n_experiments: int,
        sweep_samples: int,
        methods: List[str],
        plot_panel: bool=False,
        augmentation: Optional[List[str]]=None,
        hyperparameters: Optional[Dict[str, Dict[str, float]]]=None
    ):
    status = MANAGER.status_bar(
        status_format=u'Linear simulation{fill}Sweeping {sweep}{fill}{elapsed}',
        color='bold_underline_bright_white_on_lightslategray',
        justify=enlighten.Justify.CENTER, sweep='<parameter>',
        autorefresh=True, min_delta=0.5
    )

    cv = getattr(hyperparameters, 'cv', None)
    all_methods: Dict[str, ModelBuilder] = {
        'ATE': lambda: None,
        'ERM': lambda: ERM(),
        'DA+ERM': lambda: ERM(),
        'PI': lambda: PartialR2(gamma=GAMMA, gamma0=GAMMA0),
        'DA+PI': lambda: PartialR2(gamma=GAMMA, gamma0=GAMMA0),
        'INV+PI': lambda: invPartialR2(gamma=GAMMA, gamma0=GAMMA0, epsilon=EPSILON),
    }
    methods: Dict[str, ModelBuilder] = {m: all_methods[m] for m in methods}
    sweep_methods: Dict[str, ModelBuilder] = {
        m: all_methods[m] for m in methods if 'ATE' not in m
    }
    
    # sweep over kappa parameter
    status.update(sweep='kappa')
    logger.info('Sweeping over kappa parameters.')
    kappa_values, results = KappaSweep(
        seed=seed,
        n_samples=n_samples,
        kernel_dim=kernel_dim,
        n_experiments=n_experiments,
        methods=sweep_methods,
        sweep_samples=sweep_samples,
        hyperparameters=hyperparameters
    ).run_experiment()
    save(
        obj=kappa_values, fname='kappa_values', experiment=EXPERIMENT, format='pkl'
    )
    save(
        obj=results, fname='kappa_results', experiment=EXPERIMENT, format='pkl'
    )
    sweep_plot(
        kappa_values, results, **ANNOTATE_POPULATION_PLOT['kappa']
    )

    # sweep over gamma parameter
    status.update(sweep='gamma')
    logger.info('Sweeping over gamma parameters.')
    gamma_values, results = GammaSweep(
        seed=seed,
        n_samples=n_samples,
        kernel_dim=kernel_dim,
        n_experiments=n_experiments,
        methods=sweep_methods,
        sweep_samples=sweep_samples,
        hyperparameters=hyperparameters
    ).run_experiment()
    save(
        obj=gamma_values, fname='gamma_values', experiment=EXPERIMENT, format='pkl'
    )
    save(
        obj=results, fname='gamma_results', experiment=EXPERIMENT, format='pkl'
    )
    sweep_plot(
        gamma_values, results, **ANNOTATE_POPULATION_PLOT['gamma']
    )

    # sweep over alpha parameter
    status.update(sweep='alpha')
    logger.info('Sweeping over alpha parameters.')
    alpha_values, results = AlphaSweep(
        seed=seed,
        n_samples=n_samples,
        kernel_dim=kernel_dim,
        n_experiments=n_experiments,
        methods=sweep_methods,
        sweep_samples=sweep_samples,
        hyperparameters=hyperparameters
    ).run_experiment()
    save(
        obj=alpha_values, fname='alpha_values', experiment=EXPERIMENT, format='pkl'
    )
    save(
        obj=results, fname='alpha_results', experiment=EXPERIMENT, format='pkl'
    )
    sweep_plot(
        alpha_values, results, **ANNOTATE_POPULATION_PLOT['alpha']
    )

    # # no sweep, just compare baselines with gamma=1 and kappa=1
    # status.update(sweep='N/A')
    # logger.info('Linear experiemnt with gamma=1 and kappa=1.')
    # _, results = BaselineExperiment(
    #     seed=seed,
    #     n_samples=n_samples,
    #     kernel_dim=kernel_dim,
    #     n_experiments=n_experiments,
    #     methods=methods,
    #     hyperparameters=hyperparameters
    # ).run_experiment()
    # save(
    #     obj=results, fname=EXPERIMENT, experiment=EXPERIMENT, format='pkl'
    # )
    # box_plot(
    #     results, fname=EXPERIMENT, experiment=EXPERIMENT,
    #     savefig=True, **ANNOTATE_BOX_PLOT[EXPERIMENT]
    # )
    
    # caption = fr'nCER $\pm$ one std across {n_experiments} experiments of {n_samples} samples each.',
    # table = tex_table(
    #     results, label=EXPERIMENT, caption=caption
    # )
    # save(
    #     obj=table, fname=EXPERIMENT, experiment=EXPERIMENT, format='tex'
    # )

    MANAGER.stop()


if __name__ == '__main__':
    CLI = ArgumentParser(description='Linear simulation experiment.')
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