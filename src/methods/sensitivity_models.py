"""
Partial identification / sensitivity models.

Uniform signature: gamma (budget), epsilon (invariance error), pad (Thm. 3.A),
calibrate (paper's sigma-scaled budgets), clipy (clip to observed y range).
"""
import numpy as np
import cvxpy as cp
from loguru import logger
from joblib import Parallel, delayed

from src.methods.abstract import sensitivityAnalyzer as SA
from src.methods.regression import LeastSquaresClosedForm as OLS

# Global flag: use closed form analytic solutions where possible (standard PI)
CLOSED_FORM_SOLUTION: bool = False


class PartialR2(SA):
    """Bounded-confounding PI (Asm. 2). SOCP with QR compression."""

    def __init__(
        self,
        gamma=None,
        epsilon=0.0,
        pad=False,
        calibrate=False,
        clipy=True,
        n_jobs=1,
    ):
        if gamma is None:
            raise ValueError("gamma must be explicitly provided")

        self.epsilon = epsilon
        self.pad = pad
        self.calibrate = calibrate
        self.clipy = clipy
        self.n_jobs = n_jobs
        self._supports_closed_form = True

        # CVX state
        self.min_problem = None
        self.max_problem = None
        self.x_param = None
        self.h_var = None
        self.radius_param = None

        # Fit state
        self.R_constraint = None
        self.h_erm = None
        self.N_samples = 0
        self.sigma_sq = 1.0     # MMSE; sigma^2 (or sigma-tilde^2 on post-DA data)
        self.y_min = -np.inf
        self.y_max = np.inf

        super().__init__(gamma)

    # ------------------------------------------------------------------ fit

    def _fit(self, X, y, **kwargs):
        self.N_samples = len(X)
        self.h_erm = OLS().fit(X, y).solution.flatten()

        # noise level: sigma^2 = min MSE. Fit on post-DA data => sigma-tilde^2.
        residuals = y.flatten() - X @ self.h_erm
        self.sigma_sq = float(np.mean(residuals ** 2))

        # observable outcome limits (clipy)
        self.y_min, self.y_max = float(np.min(y)), float(np.max(y))

        # || X(h - h_erm) || = || R(h - h_erm) || for X = QR: solve on M x M
        _, R = np.linalg.qr(X)
        self.R_constraint = R

        if CLOSED_FORM_SOLUTION:
            SigmaX = X.T @ X / self.N_samples
            jitter = 1e-9 * np.trace(SigmaX)
            if jitter < 1e-12:
                jitter = 1e-9
            self.invSigmaX = np.linalg.inv(SigmaX + jitter * np.eye(len(SigmaX)))

        self._precompute_matrices(X, y, **kwargs)

        if not (CLOSED_FORM_SOLUTION and self._supports_closed_form):
            self._setup_cvx_problems()

        return self

    def _precompute_matrices(self, X, y, **kwargs):
        pass

    @property
    def scale(self):
        """s: sigma if calibrated (paper), else 1 (raw budgets)."""
        return float(np.sqrt(self.sigma_sq)) if self.calibrate else 1.0

    # -------------------------------------------------------------- solver

    def _get_constraints(self):
        """SOCP: || R (h - h_erm) ||_2 <= sqrt(N) * radius."""
        threshold = np.sqrt(self.N_samples) * self.radius_param
        return [
            cp.norm(cp.Constant(self.R_constraint) @ (self.h_var - cp.Constant(self.h_erm)), 2)
            <= threshold
        ]

    def _setup_cvx_problems(self):
        M = len(self.h_erm)
        self.x_param = cp.Parameter(M)
        self.radius_param = cp.Parameter(nonneg=True)
        self.h_var = cp.Variable(M)

        constraints = self._get_constraints()
        cost = self.x_param @ self.h_var

        self.min_problem = cp.Problem(cp.Minimize(cost), constraints)
        self.max_problem = cp.Problem(cp.Maximize(cost), constraints)

    def _set_solver_parameters(self, gamma):
        self.radius_param.value = self.scale * np.sqrt(gamma)

    # ------------------------------------------------------------- predict

    def _predict(self, X, gamma=None, **kwargs):
        gamma = self.gamma if gamma is None else gamma
        return self._finalize(self._raw_bounds(X, gamma))

    def _raw_bounds(self, X, gamma):
        """Unpadded, unclipped [lower, upper] per query."""
        if CLOSED_FORM_SOLUTION and self._supports_closed_form:
            radius = self.scale * np.sqrt(gamma)
            mahalanobis_sq = np.maximum(0, np.sum((X @ self.invSigmaX) * X, axis=1))
            margins = radius * np.sqrt(mahalanobis_sq)
            centers = X @ self.h_erm
            return np.column_stack([centers - margins, centers + margins])

        self._set_solver_parameters(gamma)

        if self.n_jobs == 1:
            bounds = np.zeros((len(X), 2))
            for i, x in enumerate(X):
                bounds[i] = self._solve_single(x)
        else:
            bounds = np.array(Parallel(n_jobs=self.n_jobs)(
                delayed(self._solve_single)(x) for x in X
            ))
        return bounds

    def _finalize(self, bounds):
        """eps-padding (Thm. 3.A) then clipping to observable y limits."""
        if self.pad:
            bounds = bounds + np.array([-self.epsilon, self.epsilon])
        if self.clipy:
            bounds = np.clip(bounds, self.y_min, self.y_max)
        return bounds

    def _solve_single(self, x):
        norm_x = np.linalg.norm(x)
        if norm_x < 1e-9:
            self.x_param.value = np.zeros_like(x)
            scale = 0.0
        else:
            self.x_param.value = x / norm_x
            scale = norm_x

        def solve_prob(prob):
            try:
                prob.solve(solver=cp.CLARABEL, warm_start=True, verbose=False)
            except Exception:
                pass

            if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                try:
                    prob.solve(solver=cp.ECOS, warm_start=True, verbose=False)
                except Exception:
                    return np.nan

            if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                return np.nan
            return prob.value * scale

        return solve_prob(self.min_problem), solve_prob(self.max_problem)


class InvarianceConstrainedPartialR2(PartialR2):
    """PI + explicit invariance-error constraint (§3.1): E_inv(h) <= epsilon^2."""

    def __init__(self, gamma=None, epsilon=None, **kwargs):
        if epsilon is None:
            raise ValueError("epsilon required")
        super().__init__(gamma=gamma, epsilon=epsilon, **kwargs)
        self.R_diff = None
        self._supports_closed_form = False

    def _precompute_matrices(self, X, y, GX=None, **kwargs):
        if GX is None:
            GX = X

        # || (GX - X) h ||: QR compress (N x M -> M x M), then jitter
        _, R = np.linalg.qr(GX - X)

        M = X.shape[1]
        jitter_strength = 1e-6 * np.mean(np.abs(R))
        if jitter_strength < 1e-9:
            jitter_strength = 1e-6

        self.R_diff = np.vstack([R, np.sqrt(jitter_strength) * np.eye(M)])

    def _get_constraints(self):
        constraints = super()._get_constraints()
        threshold = np.sqrt(self.N_samples * self.epsilon ** 2)

        constraints.append(
            cp.norm(cp.Constant(self.R_diff) @ self.h_var, 2) <= threshold
        )
        return constraints


class InstrumentalVariablePartialR2(PartialR2):
    """PI + leaky IV constraint (Asm. 3). Null/empty Z falls back to baseline PI."""

    def __init__(self, gamma=None, gamma_z=0.0, rho=1.0, **kwargs):
        self.gamma_z = gamma_z
        self.rho = rho
        super().__init__(gamma=gamma, **kwargs)
        self.Z_projector_R = None
        self.y_residual_base = None
        self.iv_threshold_param = None
        self._has_iv = False
        self._supports_closed_form = False

    @property
    def iv_bound(self):
        """sqrt of the budget on Var(E[Y - h|Z]); Appendix A: eps + s*sqrt(gamma_z)."""
        # calibrated: s is the *pre*-DA sigma = sqrt(sigma_sq / rho); else 1
        s = float(np.sqrt(self.sigma_sq / self.rho)) if self.calibrate else 1.0
        return self.epsilon + s * np.sqrt(self.gamma_z)

    def _precompute_matrices(self, X, y, Z=None, **kwargs):
        self._has_iv = Z is not None and np.size(Z) > 0
        if not self._has_iv:
            return

        Z = Z.reshape(len(Z), -1)
        Q_matrix, _ = np.linalg.qr(Z)

        Z_proj = Q_matrix.T @ X
        y_proj = Q_matrix.T @ y.flatten()

        M = X.shape[1]
        jitter_strength = 1e-6 * np.mean(np.abs(Z_proj))
        if jitter_strength < 1e-9:
            jitter_strength = 1e-6

        jitter_matrix = np.sqrt(jitter_strength) * np.eye(M)

        self.Z_projector_R = np.vstack([Z_proj, jitter_matrix])
        self.y_residual_base = np.concatenate([y_proj, np.zeros(M)])

    def _get_constraints(self):
        constraints = super()._get_constraints()
        if not self._has_iv:
            return constraints

        self.iv_threshold_param = cp.Parameter(nonneg=True)
        constraints.append(
            cp.norm(
                cp.Constant(self.y_residual_base) - cp.Constant(self.Z_projector_R) @ self.h_var,
                2
            ) <= self.iv_threshold_param
        )
        return constraints

    def _set_solver_parameters(self, gamma):
        super()._set_solver_parameters(gamma)
        if self._has_iv:
            self.iv_threshold_param.value = np.sqrt(self.N_samples) * self.iv_bound


class IntersectedPartialR2(PartialR2):
    """Baseline PI intersected with DA+PI (Cor. 1). Padding hits the DA branch only."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.baseline = None
        self.augmented = None

    def _branch(self, pad):
        return PartialR2(
            gamma=self.gamma, epsilon=self.epsilon, pad=pad,
            calibrate=self.calibrate, clipy=self.clipy, n_jobs=self.n_jobs,
        )

    def _fit_branches(self, X, y, GX, G):
        self.baseline = self._branch(pad=False).fit(X, y)
        self.augmented = self._branch(pad=self.pad).fit(GX, y)

    def _fit(self, X, y, GX=None, G=None, **kwargs):
        if GX is None:
            raise ValueError("GX (augmented treatment) required")

        GX = np.asarray(GX).reshape(len(GX), -1)
        self._fit_branches(X, y, GX, G)

        self.sigma_sq = self.baseline.sigma_sq
        self.y_min, self.y_max = float(np.min(y)), float(np.max(y))
        return self

    @property
    def rho(self):
        """Information-loss factor sigma-tilde^2 / sigma^2 (>= 1 by DPI)."""
        return self.augmented.sigma_sq / self.baseline.sigma_sq

    def _predict(self, X, gamma=None, **kwargs):
        lower_base, upper_base = self.baseline.predict(X, gamma=gamma, **kwargs).T
        lower_da, upper_da = self.augmented.predict(X, gamma=gamma, **kwargs).T

        lower = np.maximum(lower_base, lower_da)
        upper = np.minimum(upper_base, upper_da)

        # empty intersection: infeasible, same convention as the solver
        empty = lower > upper
        if empty.any():
            logger.warning(f'Empty intersection at {empty.sum()}/{len(empty)} queries.')
            lower = np.where(empty, np.nan, lower)
            upper = np.where(empty, np.nan, upper)

        return np.column_stack([lower, upper])


class IntersectedInstrumentalVariablePartialR2(IntersectedPartialR2):
    """Baseline PI_IV (null instrument) intersected with DA+PI_IV."""

    def __init__(self, gamma_z=0.0, **kwargs):
        self.gamma_z = gamma_z
        super().__init__(**kwargs)

    def _branch(self, pad, rho=1.0):
        return InstrumentalVariablePartialR2(
            gamma=self.gamma, gamma_z=self.gamma_z, rho=rho, epsilon=self.epsilon,
            pad=pad, calibrate=self.calibrate, clipy=self.clipy, n_jobs=self.n_jobs,
        )

    def _fit_branches(self, X, y, GX, G):
        # baseline sees no instrument => reduces to PI
        self.baseline = self._branch(pad=False).fit(X, y, Z=None)
        self.augmented = self._branch(pad=self.pad).fit(GX, y, Z=G)
        # rho known once both noise levels are: threshold is a cvx Parameter
        self.augmented.rho = self.rho
