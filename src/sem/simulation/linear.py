import numpy as np
from typing import Tuple
from numpy.typing import NDArray

from src.sem.abstract import StructuralEquationModel as SEM


# specify default parameters
COVARIATE_DIMENSION: int=32
CONFOUNDER_DIMENSION: int=COVARIATE_DIMENSION
LABEL_DIMENSION: int=1
NOISE_STD: float=0.1
SIMULTANEITY: bool=False


class LinearSimulationSEM(SEM):
    def __init__(
            self,
            covariate_dimension: int=COVARIATE_DIMENSION,
            confounder_dimension: int=CONFOUNDER_DIMENSION,
            label_dimension: int=LABEL_DIMENSION,
            normalize: bool=SIMULTANEITY,
        ):
        self.covariate_dimension = covariate_dimension
        self.confounder_dimension = confounder_dimension
        self.label_dimension = label_dimension

        self.W_CXY = np.random.randn(
            confounder_dimension,
            covariate_dimension+label_dimension
        )
        self.W_XY = np.random.randn(
            covariate_dimension, label_dimension
        )
        self.W_YX = SIMULTANEITY * np.random.randn(
                label_dimension, covariate_dimension
        )

        if normalize:
            self.W_YX = self.W_YX / np.linalg.norm(self.W_YX)
            self.W_XY = self.W_XY / np.linalg.norm(self.W_XY)
        
        super(LinearSimulationSEM, self).__init__()
    
    def sample(
            self, N: int= 1, kappa: float=1.0, intervention: bool=False, **kwargs
        ) -> Tuple[NDArray, NDArray]:

        # check SEM solvability (unique stationary distribution):
        #   ( 1 - kappa * tau @ f ) != 0
        feedback_strength = kappa * (self.W_YX @ self.W_XY).item()
        assert not np.isclose(
            feedback_strength, 1.0
        ), f'SEM may not be solvable with kappa={kappa}. Condition κ*fᵀτ={feedback_strength:.4f} is too close to 1.'

        C = np.random.randn(N, self.confounder_dimension)
        N_XY = NOISE_STD * np.random.randn(
            N, self.covariate_dimension + self.label_dimension
        )

        # make block matrix for structural mechanism XY -> XY
        zeros_1x1 = np.zeros(
            (self.covariate_dimension, self.covariate_dimension)
        )
        zeros_2x2 = np.zeros(
            (self.label_dimension, self.label_dimension)
        )
        T = np.block([
            [zeros_1x1, self.W_XY],
            [kappa*self.W_YX, zeros_2x2],
        ])

        # solve cyclic SEM for X, Y
        exogenous = C @ self.W_CXY
        exogenous[:, -self.label_dimension:] *= kappa
        exogenous += N_XY
        I = np.eye(
            self.covariate_dimension + self.label_dimension
        )
        M = np.linalg.inv(I - T)

        XY = exogenous @ M
        X, Y = (
            XY[:, :self.covariate_dimension],
            XY[:, -self.label_dimension:],
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