import torch
import numpy as np
import cvxpy as cp
from loguru import logger
import torch.nn.functional as F

from src.methods.utils import Model, device

from src.methods.utils import Model, device
from src.methods.abstract import sensitivityAnalyzer as SA
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
        ):
        assert gamma >= 0.0 and gamma0 >= 0.0, \
            f'Value of {gamma} should be greater than or equal to 0.'
        self.gamma0 = gamma0
        super(PartialR2, self).__init__(gamma)

    def _fit(self, X, y, **kwargs):
        # ellipsoid constraint set params
        self.h_erm = OLS().fit(X, y).solution
        self.metric = np.linalg.inv(X.T @ X)
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

        bounds = np.zeros((N, 2))
        for i, x in enumerate(X):
            lower, upper = self._optimize(x, radius)
            bounds[i, 0], bounds[i, 1] = lower, upper
        return bounds
    
    def _optimize(self, x, radius):
        if CLOSED_FORM_SOLUTION:
            lower_bound = (
                x @ self.h_erm
                - np.sqrt(radius * (x @ self.metric @ x))
            )
            upper_bound = (
                x @ self.h_erm
                + np.sqrt(radius * (x @ self.metric @ x))
            )
            return lower_bound, upper_bound

        h = cp.Variable(self.h_erm.shape)
        cost = cp.Constant(x) @ h
        constraints = ([
            cp.quad_form(
                h - self.h_erm,
                cp.psd_wrap(cp.Constant(np.linalg.inv(self.metric)))
            ) <= radius
        ])
        minimize = cp.Problem(
            cp.Minimize(cost),
            constraints
        )
        try:
            lower_bound = minimize.solve(solver=cp.CLARABEL)
        except:
            logger.warning(f'CLARABLE solver failed, falling back to ECOS.')
            lower_bound = minimize.solve(solver=cp.ECOS)
        
        h = cp.Variable(self.h_erm.shape)
        cost = cp.Constant(x) @ h
        constraints = ([
            cp.quad_form(
                h - self.h_erm,
                cp.psd_wrap(cp.Constant(np.linalg.inv(self.metric)))
            ) <= radius
        ])
        maximize = cp.Problem(
            cp.Maximize(cost),
            constraints
        )
        try:
            upper_bound = maximize.solve(solver=cp.CLARABEL)
        except:
            logger.warning(f'CLARABLE solver failed, falling back to ECOS.')
            upper_bound = maximize.solve(solver=cp.ECOS)

        return lower_bound, upper_bound
    
    @classmethod
    def _compute_radius(cls, gamma0, gamma):
        return gamma0 * gamma


class InvarianceConstrainedPartialR2(PartialR2):
    def __init__(
            self,
            gamma=GAMMA,
            gamma0=GAMMA0,
            epsilon=EPSILON
        ):
        self.epsilon = epsilon
        super(
            InvarianceConstrainedPartialR2, self
        ).__init__(gamma, gamma0)

    def _fit(self, X, y, GX=None, **kwargs):
        # default to vanilla partial R2 if GX is None
        if GX is None:
            GX = X
        self.X, self.GX = X, GX
        return super(
            InvarianceConstrainedPartialR2, self
        )._fit(X, y, **kwargs)
    
    def _optimize(self, x, radius):
        N = len(self.X)
        h = cp.Variable(self.h_erm.shape)
        cost = cp.Constant(x) @ h
        constraints = ([
            cp.quad_form(
                h - self.h_erm,
                cp.psd_wrap(cp.Constant(np.linalg.inv(self.metric)))
            ) <= radius,
            # self.GX @ h == self.X @ h,
            cp.norm(self.GX @ h - self.X @ h, p=2) <= N * self.epsilon
        ])
        minimize = cp.Problem(
            cp.Minimize(cost),
            constraints
        )
        try:
            lower_bound = minimize.solve(solver=cp.CLARABEL)
        except:
            logger.warning(f'CLARABLE solver failed, falling back to ECOS.')
            lower_bound = minimize.solve(solver=cp.ECOS)
        
        h = cp.Variable(self.h_erm.shape)
        cost = cp.Constant(x) @ h
        constraints = ([
            cp.quad_form(
                h - self.h_erm,
                cp.psd_wrap(cp.Constant(np.linalg.inv(self.metric)))
            ) <= radius,
            # self.GX @ h == self.X @ h,
            cp.norm(self.GX @ h - self.X @ h, p=2) <= N * self.epsilon
        ])
        maximize = cp.Problem(
            cp.Maximize(cost),
            constraints
        )
        try:
            upper_bound = maximize.solve(solver=cp.CLARABEL)
        except:
            logger.warning(f'CLARABLE solver failed, falling back to ECOS.')
            upper_bound = maximize.solve(solver=cp.ECOS)
        
        return lower_bound, upper_bound

