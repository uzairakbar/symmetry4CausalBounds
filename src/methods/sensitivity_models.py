import numpy as np
import cvxpy as cp
from loguru import logger
from joblib import Parallel, delayed
from typing import Literal

from src.methods.abstract import sensitivityAnalyzer as SA
from src.methods.regression import LeastSquaresClosedForm as OLS


CLOSED_FORM_SOLUTION: bool = True


def solve_with_fallback(
    problem: cp.Problem,
    objective_sense: Literal['minimize', 'maximize']
) -> float:
    """
    Solve CVXPY problem with fallback from CLARABEL to ECOS.
    
    Args:
        problem: CVXPY problem to solve
        objective_sense: 'minimize' or 'maximize'
        
    Returns:
        Optimal objective value
    """
    try:
        result = problem.solve(
            solver=cp.CLARABEL,
            warm_start=True,
            verbose=False
        )
        
        if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            logger.warning(f'{objective_sense.title()} problem status: {problem.status}')
            result = problem.solve(
                solver=cp.ECOS,
                warm_start=True,
                verbose=False
            )
    
    except Exception as e:
        logger.warning(f'CLARABEL {objective_sense} failed: {e}, using ECOS')
        result = problem.solve(
            solver=cp.ECOS,
            warm_start=True,
            verbose=False
        )
    
    return result


class PartialR2(SA):
    """Partial R² sensitivity model."""
    
    def __init__(self, gamma=None, gamma0=None, n_jobs=1):
        # Remove defaults - force explicit config
        if gamma is None or gamma0 is None:
            raise ValueError("gamma and gamma0 must be explicitly provided")
        
        assert gamma >= 0.0 and gamma0 >= 0.0, \
            f'gamma={gamma} and gamma0={gamma0} must be >= 0.'
        
        self.gamma0 = gamma0
        self.n_jobs = n_jobs
        super().__init__(gamma)

    def _fit(self, X, y, **kwargs):
        """Fit the model. Ignores GX, G, and other kwargs."""
        N = len(X)
        self.h_erm = OLS().fit(X, y).solution
        self.SigmaX = X.T @ X / N
        self.invSigmaX = np.linalg.inv(self.SigmaX)
        return self
    
    def _predict(self, X, gamma=None, gamma0=None, **kwargs):
        """
        Predict interval bounds.
        
        Args:
            X: Query points
            gamma: Override sensitivity parameter (optional)
            gamma0: Override base sensitivity (optional)
            **kwargs: Ignored
        """
        N, M = X.shape
        gamma = self.gamma if gamma is None else gamma
        gamma0 = self.gamma0 if gamma0 is None else gamma0
        radius = self._compute_radius(gamma0, gamma)

        if self.n_jobs == 1:
            bounds = np.zeros((N, 2))
            for i, x in enumerate(X):
                lower, upper = self._find_bounds(x, radius)
                bounds[i, 0], bounds[i, 1] = lower, upper
        else:
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self._find_bounds)(x, radius) for x in X
            )
            bounds = np.array(results)

        return bounds
    
    def _find_bounds(self, x, radius):
        """Find lower and upper bounds for a single query point."""
        if CLOSED_FORM_SOLUTION:
            margin = radius * np.sqrt((x @ self.invSigmaX @ x))
            lower_bound = x @ self.h_erm - margin
            upper_bound = x @ self.h_erm + margin
            return lower_bound, upper_bound

        # Optimization-based approach
        h = cp.Variable(self.h_erm.shape)
        constraints = [
            cp.quad_form(
                h - self.h_erm,
                cp.psd_wrap(cp.Constant(self.SigmaX))
            ) <= radius ** 2
        ]
        cost = cp.Constant(x) @ h
        return self._optimize(cost, constraints)
    
    @staticmethod
    def _optimize(cost, constraints):
        """Solve min/max optimization with fallback."""
        minimize = cp.Problem(cp.Minimize(cost), constraints)
        lower_bound = solve_with_fallback(minimize, 'minimize')
        
        maximize = cp.Problem(cp.Maximize(cost), constraints)
        upper_bound = solve_with_fallback(maximize, 'maximize')
        
        return lower_bound, upper_bound
    
    @staticmethod
    def _compute_radius(gamma0, gamma):
        """Compute constraint set radius."""
        return np.sqrt(gamma0 * gamma)


class InvarianceConstrainedPartialR2(PartialR2):
    """Invariance-constrained Partial R² model."""
    
    def __init__(self, gamma=None, gamma0=None, epsilon=None, n_jobs=1):
        if epsilon is None:
            raise ValueError("epsilon must be explicitly provided")
        
        self.epsilon = epsilon
        super().__init__(gamma, gamma0, n_jobs)

    def _fit(self, X, y, GX=None, **kwargs):
        """
        Fit with invariance constraints.
        
        Args:
            X: Original data
            y: Outcomes
            GX: Augmented data (required for invariance)
            **kwargs: Ignored (including G which we don't need)
        """
        # Default to vanilla partial R² if GX not provided
        if GX is None:
            GX = X
        
        self.X = X
        self.GX = GX
        
        # Call parent _fit which sets up h_erm, SigmaX, etc.
        return super()._fit(X, y, **kwargs)
    
    def _find_bounds(self, x, radius):
        """Find bounds with invariance constraint."""
        N = len(self.X)
        h = cp.Variable(self.h_erm.shape)
        
        constraints = [
            cp.quad_form(
                h - self.h_erm,
                cp.psd_wrap(cp.Constant(self.SigmaX))
            ) <= radius ** 2,
            cp.sum_squares(
                cp.Constant(self.GX - self.X) @ h
            ) <= N * self.epsilon
        ]
        
        cost = cp.Constant(x) @ h
        return self._optimize(cost, constraints)


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
            ) <= radius**2,
            cp.norm(
                cp.Constant(self.GX - self.X) @ h, p=2
            ) <= np.sqrt(N * self.epsilon)
        ])
        cost = cp.Constant(x) @ h
        return self._optimize(cost, constraints)
