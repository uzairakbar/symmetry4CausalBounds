import numpy as np
import cvxpy as cp
from loguru import logger
from joblib import Parallel, delayed
from src.methods.abstract import sensitivityAnalyzer as SA
from src.methods.regression import LeastSquaresClosedForm as OLS

# Global flag: Use closed form analytic solutions where possible (Standard PI)
CLOSED_FORM_SOLUTION: bool = True

class PartialR2(SA):
    """
    Base class implementing the Template Method pattern.
    Logic for solving and prediction is central; children only define constraints.
    """
    
    def __init__(self, gamma=None, gamma0=None, n_jobs=1):
        if gamma is None or gamma0 is None:
            raise ValueError("gamma and gamma0 must be explicitly provided")
        
        self.gamma0 = gamma0
        self.n_jobs = n_jobs
        
        # Subclasses (IV, Invariance) must set this to False
        self._supports_closed_form = True
        
        # CVX State
        self.min_problem = None
        self.max_problem = None
        self.x_param = None
        self.h_var = None
        self.radius_sq_param = None
        
        super().__init__(gamma)

    def _fit(self, X, y, **kwargs):
        """Common fit logic."""
        N = len(X)
        # Flatten solution to ensure 1D coefficient vector (M,) instead of (M, 1)
        # This fixes the "setting array element with sequence" error in closed form
        self.h_erm = OLS().fit(X, y).solution.flatten()
        
        # Precompute SigmaX for the standard Partial R2 constraint
        self.SigmaX = X.T @ X / N
        self.invSigmaX = np.linalg.inv(self.SigmaX)
        
        # Hook for child classes to compute their specific matrices
        self._precompute_matrices(X, y, **kwargs)
        
        # Setup CVX problems if we can't or won't use closed form
        if not (CLOSED_FORM_SOLUTION and self._supports_closed_form):
            self._setup_cvx_problems()
            
        return self

    def _precompute_matrices(self, X, y, **kwargs):
        """Hook for child classes to perform linear algebra before CVX setup."""
        pass

    def _get_constraints(self):
        """
        Hook for child classes to return list of constraints.
        Base implementation returns the standard PI constraint.
        """
        # Ensure matrix is treated as fixed PSD constant
        sigma_x_const = cp.psd_wrap(cp.Constant(self.SigmaX))
        h_erm_flat = self.h_erm.flatten()
        
        return [
            cp.quad_form(self.h_var - h_erm_flat, sigma_x_const) <= self.radius_sq_param
        ]

    def _setup_cvx_problems(self):
        """Sets up the optimization problem once (DPP compliant)."""
        M = len(self.h_erm.flatten())
        
        # 1. Define Parameters and Variables
        self.x_param = cp.Parameter(M)
        self.radius_sq_param = cp.Parameter(nonneg=True)
        self.h_var = cp.Variable(M)
        
        # 2. Get Constraints (Polymorphic call)
        constraints = self._get_constraints()
        
        # 3. Define Objective
        cost = self.x_param @ self.h_var
        
        # 4. Compile Problems
        self.min_problem = cp.Problem(cp.Minimize(cost), constraints)
        self.max_problem = cp.Problem(cp.Maximize(cost), constraints)
    
    def _predict(self, X, gamma=None, gamma0=None, **kwargs):
        gamma = self.gamma if gamma is None else gamma
        gamma0 = self.gamma0 if gamma0 is None else gamma0
        radius = np.sqrt(gamma0 * gamma)
        
        # VECTORIZED OPTIMIZATION for Closed Form
        # This makes PI and DA+PI practically instant
        if CLOSED_FORM_SOLUTION and self._supports_closed_form:
            # Calculate Mahalanobis distance for all X at once: sqrt(diag(X @ invSigma @ X.T))
            # Efficient way: sum( (X @ invSigma) * X, axis=1 )
            mahalanobis_sq = np.sum((X @ self.invSigmaX) * X, axis=1)
            margins = radius * np.sqrt(mahalanobis_sq)
            
            centers = X @ self.h_erm
            
            # Stack into (N, 2) array
            bounds = np.column_stack([centers - margins, centers + margins])
            return bounds

        # Standard Solver Path (for PI_INV, IV)
        if not (CLOSED_FORM_SOLUTION and self._supports_closed_form):
            self.radius_sq_param.value = radius ** 2

        if self.n_jobs == 1:
            bounds = np.zeros((len(X), 2))
            for i, x in enumerate(X):
                bounds[i] = self._solve_single(x, radius)
        else:
            bounds = np.array(Parallel(n_jobs=self.n_jobs)(
                delayed(self._solve_single)(x, radius) for x in X
            ))
        return bounds

    def _solve_single(self, x, radius):
        """Centralized solver logic with fallback."""
        # FIX 1: Only use closed form if global flag is True AND class supports it
        if CLOSED_FORM_SOLUTION and self._supports_closed_form:
            return self._solve_closed_form(x, radius)

        self.x_param.value = x.flatten()
        
        # Helper to solve one direction with robust fallback
        def solve_prob(prob):
            try:
                # Clarabel is usually faster/more accurate for QCPs
                prob.solve(solver=cp.CLARABEL, warm_start=True, verbose=False, tol_gap_abs=1e-7)
                if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                    raise RuntimeError(f"Status {prob.status}")
            except Exception:
                # Fallback to ECOS for robustness
                try:
                    prob.solve(solver=cp.ECOS, warm_start=True, verbose=False)
                except Exception:
                     # Absolute last resort (should be rare with Jitter fix)
                    return np.nan
            return prob.value

        return solve_prob(self.min_problem), solve_prob(self.max_problem)

    def _solve_closed_form(self, x, radius):
        # Base implementation for standard Partial R2
        margin = radius * np.sqrt((x @ self.invSigmaX @ x))
        center = x @ self.h_erm
        
        # FIX 2: Explicitly cast to float to avoid 1-element array assignment errors
        return float(center - margin), float(center + margin)


class InvarianceConstrainedPartialR2(PartialR2):
    
    def __init__(self, gamma=None, gamma0=None, epsilon=None, n_jobs=1):
        if epsilon is None: raise ValueError("epsilon required")
        self.epsilon = epsilon
        super().__init__(gamma, gamma0, n_jobs)
        self.Q_inv = None 
        # FIX 3: Invariance constraints have no simple closed form
        self._supports_closed_form = False

    def _precompute_matrices(self, X, y, GX=None, **kwargs):
        if GX is None: GX = X
        
        diff = GX - X
        self.Q_inv = diff.T @ diff
        self.N_samples = len(X)
        
        # FIX 4: Stabilize Matrix and Enforce Symmetry
        # High collinearity in polynomial features can make this singular.
        # Add diagonal jitter relative to the trace to ensure PSD properties.
        jitter = 1e-7 * np.trace(self.Q_inv) / len(self.Q_inv)
        self.Q_inv += jitter * np.eye(len(self.Q_inv))
        self.Q_inv = (self.Q_inv + self.Q_inv.T) / 2.0

    def _get_constraints(self):
        constraints = super()._get_constraints()
        
        # quad_form(h, Q) is equivalent to sum_squares((GX-X)@h)
        # We wrap in psd_wrap because we explicitly conditioned Q_inv to be PSD above.
        constraints.append(
            cp.quad_form(self.h_var, cp.psd_wrap(cp.Constant(self.Q_inv))) 
            <= self.N_samples * self.epsilon
        )
        return constraints


class InstrumentalVariablePartialR2(PartialR2):
    
    def __init__(self, gamma=None, gamma0=None, delta=0.0, n_jobs=1):
        self.delta = delta
        super().__init__(gamma, gamma0, n_jobs)
        self.ZtZ_inv = None
        self.ZtX = None
        self.Zty = None
        # FIX 3: IV constraints have no simple closed form
        self._supports_closed_form = False

    def _precompute_matrices(self, X, y, Z=None, **kwargs):
        if Z is None: Z = X
        
        self.N_samples = len(X)
        y_flat = y.flatten()
        
        # Precompute components in Z-space
        ZtZ = Z.T @ Z
        
        # FIX 4: Stabilize Matrix for IV (Crucial for Optical Device experiments)
        # Polynomial features create very high condition numbers.
        # Without jitter, inv(ZtZ) produces large negative eigenvalues due to noise.
        jitter = 1e-6 * np.trace(ZtZ) / len(ZtZ)
        ZtZ_stabilized = ZtZ + jitter * np.eye(len(ZtZ))
        
        # Invert and enforce symmetry
        self.ZtZ_inv = np.linalg.inv(ZtZ_stabilized)
        self.ZtZ_inv = (self.ZtZ_inv + self.ZtZ_inv.T) / 2.0
        
        self.ZtX = Z.T @ X
        self.Zty = Z.T @ y_flat

    def _get_constraints(self):
        constraints = super()._get_constraints()
        
        # IV Constraint: (y - Xh)^T P_Z (y - Xh) <= delta * N
        # Reformulated using precomputed KxK matrices
        param_vector = cp.Constant(self.Zty) - cp.Constant(self.ZtX) @ self.h_var
        
        constraints.append(
            cp.quad_form(param_vector, cp.psd_wrap(cp.Constant(self.ZtZ_inv))) 
            <= self.N_samples * self.delta
        )
        return constraints
    
