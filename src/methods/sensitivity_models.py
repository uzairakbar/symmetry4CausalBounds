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


class MarginalSensitivityModel(SA):
    def __init__(self, theta = 10.0):
        assert theta >= 1.0,\
            f'Value of {theta} should be greater than or equal to 1.'
        self.theta0 = theta #1.0
        super(MarginalSensitivityModel, self).__init__(theta)

    def _fit(self, X, y, **kwargs):
        # ellipsoid constraint set params
        self.h_stats = OLS().fit(X, y).solution
        self.metric = np.linalg.inv(X.T @ X)
        self.radius = (
            (self.theta0**2) * ((self.theta**2) - 1.0)
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
        Sigma_XiX = cp.Variable(self.h_stats.shape)
        cost = cp.Constant(x) @ Sigma_XiX
        constraints = ([
            cp.quad_form(
                Sigma_XiX,
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
        
        Sigma_XiX = cp.Variable(self.h_stats.shape)
        cost = cp.Constant(x) @ Sigma_XiX
        constraints = ([
            cp.quad_form(
                Sigma_XiX,
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
        
        # # closed form
        # lower_bound = - np.sqrt(self.radius * x.T @ self.metric @ x)
        # upper_bound = + np.sqrt(self.radius * x.T @ self.metric @ x)
        
        return (
            x @ self.h_stats + lower_bound,
            x @ self.h_stats + upper_bound
        )


# class ConstrainedLeastSquaresUnfaithfulIV(UIV):
#     def __init__(self, alpha = None):
#         super(ConstrainedLeastSquaresUnfaithfulIV, self).__init__(alpha)

#     def _fit(self, X, y, G=None, GX=None, **kwargs):
#         h_erm = OLS().fit(GX, y).solution

#         s1 = OLS().fit(G, GX).solution
#         GXhat = G @ s1
        
#         A = GXhat.T @ GXhat
#         b = GXhat.T @ y

#         h = cp.Variable(h_erm.shape)
#         cost = cp.norm(GX@h - y)
#         constraints = [A @ h == b]
#         prob = cp.Problem(
#             cp.Minimize(cost),
#             constraints
#         )
#         try:
#             result = prob.solve(solver=cp.CLARABEL)
#         except:
#             logger.warning(f'CLARABLE solver failed, falling back to ECOS.')
#             result = prob.solve(solver=cp.ECOS)
#         self._W = h.value
#         return self
    
#     def _predict(self, X):
#         return X @ self._W

