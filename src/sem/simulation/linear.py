import numpy as np
from typing import Tuple
from numpy.typing import NDArray

from src.sem.abstract import StructuralEquationModel as SEM


# specify default parameters
TREATMENT_DIMENSION: int=32
CONFOUNDER_DIMENSION: int=TREATMENT_DIMENSION
OUTCOME_DIMENSION: int=1
NOISE_STD: float=0.1


class LinearSimulationSEM(SEM):
    def __init__(
            self,
            treatment_dimension: int=TREATMENT_DIMENSION,
            confounder_dimension: int=CONFOUNDER_DIMENSION,
            outcome_dimension: int=OUTCOME_DIMENSION,
        ):
        self.treatment_dimension = treatment_dimension
        self.confounder_dimension = confounder_dimension
        self.outcome_dimension = outcome_dimension

        self.W_CX = np.random.randn(
            confounder_dimension,
            treatment_dimension
        )
        self.W_CY = np.random.randn(
            confounder_dimension,
            outcome_dimension
        )
        self.W_XY = np.random.randn(
            treatment_dimension, outcome_dimension
        )
        
        super(LinearSimulationSEM, self).__init__()
    
    def sample(
            self, N: int= 1, kappa: float=1.0, intervention: bool=False, **kwargs
        ) -> Tuple[NDArray, NDArray]:

        U = np.random.randn(N, self.confounder_dimension)
        N_X = np.random.randn(
            N, self.treatment_dimension
        )
        N_Y = np.random.randn(
            N, self.outcome_dimension
        )

        if intervention:
            X = N_X
        else:
            X = U @ self.W_CX + NOISE_STD * N_X
        
        Y = (
            X @ self.W_XY               # f(X)
            + kappa * U @ self.W_CY     # \xi
            + NOISE_STD * N_Y           # noise
        )

        self.varXi = np.var(Y - X @ self.W_XY)
        self.varEXiX = np.var(
            X @ np.linalg.pinv(X) @ (
                Y - X @ self.W_XY
            )
        )
        # for debugging
        # print('linear Var(xi): ', self.varXi)
        # print('linear Var(E[xi|X]): ', self.varEXiX)

        return X, Y