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
        return X @ self._W

