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
MAX_BATCH: int=256
LOG_FREQUENCY: int=100


class PartialR2(SA):
    def __init__(self, gamma = 10.0):
        assert gamma >= 0.0,\
            f'Value of {gamma} should be greater than or equal to 0.'
        self.gamma0 = gamma
        super(PartialR2, self).__init__(gamma)

    def _fit(self, X, y, **kwargs):
        # ellipsoid constraint set params
        self.h_erm = OLS().fit(X, y).solution
        self.metric = np.linalg.inv(X.T @ X)
        self.radius = (
            (self.gamma0**2) * ((self.gamma**2) - 1.0)
        )
        return self
    
    def _predict(self, X):
        N, M = X.shape
        bounds = np.zeros((N, 2))
        for i, x in enumerate(X):
            lower, upper = self._optimize(x)
            bounds[i, 0], bounds[i, 1] = lower, upper
        return bounds
    
    def _optimize(self, x):
        h = cp.Variable(self.h_erm.shape)
        cost = cp.Constant(x) @ h
        constraints = ([
            cp.quad_form(
                h - self.h_erm,
                cp.psd_wrap(cp.Constant(np.linalg.inv(self.metric)))
            ) <= self.radius
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
            ) <= self.radius
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
        
        return (
            lower_bound, upper_bound
        )


class InvarianceConstrainedPartialR2(PartialR2):
    def __init__(self, gamma = 10.0, epsilon=0.15):
        self.epsilon = epsilon
        super(
            InvarianceConstrainedPartialR2, self
        ).__init__(gamma)

    def _fit(self, X, y, GX=None, **kwargs):
        # default to vanilla partial R2 if GX is None
        if GX is None:
            GX = X
        self.X, self.GX = X, GX
        return super(
            InvarianceConstrainedPartialR2, self
        )._fit(X, y, **kwargs)
    
    def _optimize(self, x):
        N = len(self.X)
        h = cp.Variable(self.h_erm.shape)
        cost = cp.Constant(x) @ h
        constraints = ([
            cp.quad_form(
                h - self.h_erm,
                cp.psd_wrap(cp.Constant(np.linalg.inv(self.metric)))
            ) <= self.radius,
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
            ) <= self.radius,
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
        
        return (
            lower_bound, upper_bound
        )

