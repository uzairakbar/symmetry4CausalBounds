import torch
import numpy as np
import cvxpy as cp
from loguru import logger
import torch.nn.functional as F
from joblib import Parallel, delayed

from src.methods.utils import Model, device

from src.methods.abstract import sensitivityAnalyzer as SA
from src.methods.utils import Model, device, check_feasibility
from src.methods.regression import LeastSquaresClosedForm as OLS


DEVICE: str=device()
CLOSED_FORM_SOLUTION: bool=True
LOG_FREQUENCY: int=100
MAX_BATCH: int=256
GAMMA0: float=100
EPSILON: float=1
GAMMA: float=100


class PartialR2(SA):
    def __init__(
            self,
            gamma=GAMMA,
            gamma0=GAMMA0,
            n_jobs=1,
        ):
        assert gamma >= 0.0 and gamma0 >= 0.0, \
            f'Value of {gamma} should be greater than or equal to 0.'
        self.gamma0 = gamma0
        self.n_jobs = n_jobs
        super(PartialR2, self).__init__(gamma)

    def _fit(self, X, y, **kwargs):
        # ellipsoid constraint set params
        self.h_erm = OLS().fit(X, y).solution
        self.invSigmaX = np.linalg.inv(X.T @ X)
        self.SigmaX = X.T @ X
        return self
    
    def _predict(
            self,
            X,
            gamma=None,
            gamma0=None,
        ):
        N, M = X.shape
        gamma = (
            self.gamma if gamma is None else gamma
        )
        gamma0 = (
            self.gamma0 if gamma0 is None else gamma0
        )
        radius = self._compute_radius(gamma0, gamma)

        if self.n_jobs == 1:
            # sequential
            bounds = np.zeros((N, 2))
            for i, x in enumerate(X):
                lower, upper = self._find_bounds(x, radius)
                bounds[i, 0], bounds[i, 1] = lower, upper
        else:
            # parallel
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self._find_bounds)(x, radius) for x in X
            )
            bounds = np.array(results)

        return bounds
    
    def _find_bounds(self, x, radius):
        if CLOSED_FORM_SOLUTION:
            margin = np.sqrt(radius * (x @ self.invSigmaX @ x))
            lower_bound = x @ self.h_erm - margin
            upper_bound = x @ self.h_erm + margin
            return lower_bound, upper_bound

        h = cp.Variable(self.h_erm.shape)
        constraints = ([
            cp.quad_form(
                h - self.h_erm,
                cp.psd_wrap(cp.Constant(self.SigmaX))
            ) <= radius
        ])
        cost = cp.Constant(x) @ h
        return self._optimize(cost, constraints)
    
    @staticmethod
    def _optimize(cost, constraints):
        # if not check_feasibility(constraints):
        #     logger.warning('Constraints infeasible! Skipping optimization')
        #     return float('nan'), float('nan')

        # solve minimization
        minimize = cp.Problem(cp.Minimize(cost), constraints)
        try:
            lower_bound = minimize.solve(
                solver=cp.CLARABEL,
                warm_start=True,
                verbose=False
            )
            if minimize.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                logger.warning(f'Min problem status: {minimize.status}')
                lower_bound = minimize.solve(
                    solver=cp.ECOS,
                    warm_start=True,
                    verbose=False
                )
        except Exception as e:
            logger.warning(f'CLARABEL min failed: {e}, using ECOS')
            lower_bound = minimize.solve(
                solver=cp.ECOS,
                warm_start=True,
                verbose=False
            )
        
        # solve maximization
        maximize = cp.Problem(cp.Maximize(cost), constraints)
        try:
            upper_bound = maximize.solve(
                solver=cp.CLARABEL,
                warm_start=True,
                verbose=False
            )
            if maximize.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                logger.warning(f'Max problem status: {maximize.status}')
                upper_bound = maximize.solve(
                    solver=cp.ECOS,
                    warm_start=True,
                    verbose=False
                )
        except Exception as e:
            logger.warning(f'CLARABEL max failed: {e}, using ECOS')
            upper_bound = maximize.solve(
                solver=cp.ECOS,
                warm_start=True,
                verbose=False
            )
        return lower_bound, upper_bound
    
    @staticmethod
    def _compute_radius(gamma0, gamma):
        return gamma0 * gamma


class InvarianceConstrainedPartialR2(PartialR2):
    def __init__(
            self,
            gamma=GAMMA,
            gamma0=GAMMA0,
            epsilon=EPSILON,
            n_jobs=1,
        ):
        self.epsilon = epsilon
        super(
            InvarianceConstrainedPartialR2, self
        ).__init__(gamma, gamma0, n_jobs)

    def _fit(self, X, y, GX=None, **kwargs):
        # default to vanilla partial R2 if GX is None
        if GX is None:
            GX = X
        self.X, self.GX = X, GX
        return super(
            InvarianceConstrainedPartialR2, self
        )._fit(X, y, **kwargs)
    
    def _find_bounds(self, x, radius):
        N = len(self.X)
        h = cp.Variable(self.h_erm.shape)
        constraints = ([
            cp.quad_form(
                h - self.h_erm,
                cp.psd_wrap(cp.Constant(self.SigmaX))
            ) <= radius,
            cp.norm(self.GX @ h - self.X @ h, p=2) <= N * self.epsilon
        ])
        cost = cp.Constant(x) @ h
        return self._optimize(cost, constraints)

