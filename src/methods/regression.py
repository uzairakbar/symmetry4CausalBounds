import numpy as np
import cvxpy as cp
from loguru import logger
from src.methods.abstract import pointEstimator


class LeastSquaresClosedForm(pointEstimator):
    """Closed-form least squares regression."""
    
    def _fit(self, X, y, **kwargs):
        """Fit using closed-form solution. Ignores extra kwargs."""
        self._W = np.linalg.pinv(X) @ y
        return self
    
    def _predict(self, X, **kwargs):
        """Predict outcomes. Ignores extra kwargs."""
        return X @ self._W


class LeastSquaresIterative(pointEstimator):
    def _fit(self, X, y, **kwargs):
        h0 = np.linalg.pinv(X) @ y
        h = cp.Variable(h0.shape)
        cost = cp.norm(y - X @ h)
        prob = cp.Problem(
            cp.Minimize(cost)
        )
        try:
            result = prob.solve(solver=cp.CLARABEL)
        except:
            logger.warning(f'CLARABLE solver failed, falling back to ECOS.')
            result = prob.solve(solver=cp.ECOS)
        self._W = h.value
        return self
    
    def _predict(self, X, **kwargs):
        return X @ self._W


class TwoStageLeastSquaresIV(pointEstimator):
    def __init__(self, **kwargs):
        super(TwoStageLeastSquaresIV, self).__init__(**kwargs)
    
    def _fit(self, X, y, Z, **kwargs):

        S1 = LeastSquaresClosedForm().fit(Z, X).solution
        Xhat = Z @ S1

        S2 = LeastSquaresIterative().fit(Xhat, y).solution
        self._W = S2

        return self
    
    def _predict(self, X, **kwargs):
        return X @ self._W


class GeneralizedMomentMethodIV(pointEstimator):
    def __init__(self, **kwargs):
        super(GeneralizedMomentMethodIV, self).__init__(**kwargs)
    
    def _fit(self, X, y, Z, **kwargs):
        h0 = np.linalg.pinv(X) @ y
        h = cp.Variable(h0.shape)
        Pi_Z = Z @ np.linalg.pinv(Z)
        moment_vector = cp.Constant(y) - cp.Constant(X) @ h
        cost = cp.quad_form(
            moment_vector,
            cp.psd_wrap(cp.Constant(Pi_Z))
        )
        prob = cp.Problem(
            cp.Minimize(cost)
        )
        try:
            result = prob.solve(solver=cp.CLARABEL)
        except:
            logger.warning(f'CLARABLE solver failed, falling back to ECOS.')
            result = prob.solve(solver=cp.ECOS)
        self._W = h.value
        return self
    
    def _predict(self, X, **kwargs):
        return X @ self._W