import os
import fsspec
import numpy as np
from loguru import logger
from typing import Dict, Tuple
from numpy.typing import NDArray

from src.sem.utils import (
    select_best_degree,
    fit_ground_truth_f,
)
from src.sem.abstract import StructuralEquationModel as SEM
from src.methods.regression import LeastSquaresClosedForm as ERM


MAX_PLOYNOMIAL_DEGREE: int=3


class OpticalDeviceSEM(SEM):
    @staticmethod
    def load_dataset(directory: str='data/optical_device') -> Dict[int, NDArray]:
        
        def download_dataset(directory: str):
            fs = fsspec.filesystem(
                'github',
                org='janzing',
                repo='janzing.github.io'
            )
            fs.get(
                fs.ls("code/data_from_optical_device"),
                directory
            )

        download_dataset(directory)

        file_list = os.listdir(directory)
        file_list = [f for f in file_list if 'confounder' in f and 'random' not in f]
        
        dataset = {}
        for experiment, file_name in enumerate(file_list):
            dataset[experiment] = np.genfromtxt(
                f'{directory}/{file_name}', delimiter=' '
            )
        return dataset
    
    _DATASET: Dict[int, NDArray] = load_dataset.__func__()

    def __init__(
            self,
            experiment: int=0,
            center: bool=True,
            ground_truth: str='linear'
        ):
        experiment_data = self.get_experiment_data(experiment)

        if center:
            experiment_data -= experiment_data.mean(axis = 0)

        y = experiment_data[:, -1:]     # outcome
        XC = experiment_data[:, :-1]    # treatment and confounder
        X = XC[:, :-1]                  # treatment
        C = XC[:, -1:]                  # confounder

        best_degree = 1
        if ground_truth == 'linear':
            W_XY, _, epsilon = fit_ground_truth_f(
                X, y, C, 1
            )
        elif ground_truth == 'polynomial':
            best_degree, _ = select_best_degree(
                X, y, C, max_degree=MAX_PLOYNOMIAL_DEGREE
            )
            logger.info(
                f'Experiment {experiment} polynomial degree: {best_degree}'
            )
            W_XY, _, epsilon = fit_ground_truth_f(
                X, y, C, best_degree
            )
        else:
            raise ValueError(
                f'Ground truth {ground_truth} model not supported/implemented.'
            )
        
        self.W_XY = W_XY
        self.poly_degree = best_degree
        self.y, self.X, self.C = y, X, C

        self.varXi = np.var(epsilon * self.C)
        self.varEXiX = np.var(
            self.X @ np.linalg.pinv(self.X) @ (
                epsilon * self.C
            )
        )
        # for debugging
        # print('optical Var(xi): ', self.varXi)
        # print('optical Var(E[xi|X]): ', self.varEXiX)
    
    def sample(self, N: int=1, **kwargs) -> Tuple[NDArray, NDArray]:
        N_max, M = self.X.shape
        indices = np.arange(N_max)
        replace = N > N_max
        sampled = np.random.choice(
            indices, N, replace
        )
        return self.X[sampled], self.y[sampled]
    
    @classmethod
    def get_experiment_data(cls, n: int) -> NDArray:
        return cls._DATASET[n]
    
    @classmethod
    def num_experiments(cls) -> int:
        return len(cls._DATASET)