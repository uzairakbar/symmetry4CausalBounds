"""
Partial identification / sensitivity models.

Uniform signature: gamma (budget), epsilon (invariance error), pad (Thm. 3.A),
calibrate (paper's sigma-scaled budgets), clipy (clip to observed y range).
"""

from enum import IntEnum

import cvxpy as cp
import numpy as np
from joblib import Parallel, delayed, effective_n_jobs, parallel_config
from loguru import logger
from threadpoolctl import threadpool_limits

from src.methods.abstract import sensitivityAnalyzer as SA
from src.methods.regression import LeastSquaresClosedForm as OLS

# Global flag: use closed form analytic solutions where possible (standard PI)
CLOSED_FORM_SOLUTION: bool = False


class SolveStatus(IntEnum):
    """Per-query outcome. Ordered: a pair takes the worse of its two sides."""

    OK = 0
    INFEASIBLE = 1  # solver proved the constraint set empty: data rejects the budget
    FAILURE = 2  # numerical breakdown: no answer produced


def _solve_chunk(view, chunk):
    """Worker: one model view, one chunk. Per-chunk setup runs once, not per query."""
    view._begin_chunk()
    return [view._solve_single(payload) for payload in chunk]


class BoundedSA(SA):
    """
    Contract every PI method honours: per-query status, chunk-parallel solve,
    pad-then-clip finalisation.

    Subclasses supply four hooks:
        _prepare(X, gamma) -> per-query payloads, or None if globally infeasible
        _solve_single(payload) -> (lower, upper, SolveStatus)
        _worker_view()  -> what crosses the process boundary (default: self)
        _begin_chunk()  -> per-chunk setup, run INSIDE the worker
    """

    def __init__(
        self,
        gamma=None,
        epsilon=0.0,
        pad=False,
        calibrate=False,
        clipy=True,
        n_jobs=1,
        mean_match=True,
    ):
        if gamma is None:
            raise ValueError("gamma must be explicitly provided")

        self.epsilon = epsilon
        self.pad = pad
        self.calibrate = calibrate
        self.clipy = clipy
        self.n_jobs = n_jobs
        # Lem. 2: the identified set lives on the mean-matched slice
        # H_X = {h : E[h(X)] = E[Y]}. False keeps the pre-2026-09 geometry.
        self.mean_match = mean_match
        self.query_status = None  # per-query SolveStatus, set on every predict
        self.query_diagnostics = None  # optional per-query extras, set on predict
        self.y_min = -np.inf
        self.y_max = np.inf

        super().__init__(gamma)

    # ------------------------------------------------------------- predict

    def _predict(self, X, gamma=None, epsilon=None, **kwargs):
        gamma = self.gamma if gamma is None else gamma
        if epsilon is not None:
            self.epsilon = epsilon  # constraint RHS and padding both read it
        return self._finalize(self._raw_bounds(X, gamma))

    def _raw_bounds(self, X, gamma):
        """Unpadded, unclipped [lower, upper] per query; sets `query_status`."""
        payloads = self._prepare(X, gamma)
        if payloads is None:  # the whole constraint set is empty
            self.query_status = np.full(len(X), SolveStatus.INFEASIBLE, dtype=int)
            return np.full((len(X), 2), np.nan)

        view = self._worker_view()
        if self.n_jobs == 1:
            # Never route a single query through here as a shortcut when
            # n_jobs > 1: a parent-side solve leaves an unpicklable
            # DefaultSolution on the Problem and kills every later dispatch.
            # Single BLAS thread to match the workers (inner_max_num_threads=1
            # below): numpy 2 kernels vary with thread count at the last ulp,
            # so serial and parallel would otherwise disagree (A5).
            with threadpool_limits(limits=1):
                view._begin_chunk()
                solved = [view._solve_single(p) for p in payloads]
        else:
            # Chunk, not per query: the DPP cache is per worker, so fine-grained
            # tasks re-canonicalize every query (measured ~2x slower).
            # Split INDICES, not the payload list: payloads can be ragged and
            # np.array_split would raise on them.
            n_chunks = effective_n_jobs(self.n_jobs)  # resolves -1
            chunks = [c for c in np.array_split(np.arange(len(payloads)), n_chunks) if len(c)]
            # loky sets inner threads to cpu_count//n_jobs, which a submit-script
            # OMP_NUM_THREADS overrides. Pin it.
            with parallel_config(backend="loky", n_jobs=self.n_jobs, inner_max_num_threads=1):
                out = Parallel()(delayed(_solve_chunk)(view, [payloads[i] for i in c]) for c in chunks)
            solved = [q for c in out for q in c]

        solved = np.asarray(solved, dtype=float)
        self.query_status = solved[:, 2].astype(int)
        # a subclass may append per-query diagnostics after the status; they ride
        # back from the workers with the bounds instead of being lost with the
        # worker's copy of `self` (the nets' backtrack counters do this)
        self.query_diagnostics = solved[:, 3:] if solved.shape[1] > 3 else None
        return solved[:, :2]

    def _finalize(self, bounds):
        """eps-padding (Thm. 3.A) then clipping to observable y limits."""
        if self.pad:
            bounds = bounds + np.array([-self.epsilon, self.epsilon])
        if self.clipy:
            bounds = np.clip(bounds, self.y_min, self.y_max)
        return bounds

    # ---------------------------------------------------------------- hooks

    def _worker_view(self):
        """What gets pickled to the workers. Override to strip unpicklable state."""
        return self

    def _begin_chunk(self):
        """Per-chunk setup, inside the worker. Override to amortise a JIT/cache."""

    def _prepare(self, X, gamma):
        raise NotImplementedError

    def _solve_single(self, payload):
        raise NotImplementedError


class PartialR2(BoundedSA):
    """Bounded-confounding PI (Asm. 2). SOCP with QR compression."""

    def __init__(
        self,
        gamma=None,
        epsilon=0.0,
        pad=False,
        calibrate=False,
        clipy=True,
        n_jobs=1,
        mean_match=True,
    ):
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
        # Lem. 2 slice: the design mean and the outcome mean that eliminate the
        # intercept. Zero / 0.0 under `mean_match=False`, where the ball is the
        # old uncentred second-moment one.
        self.mu_ = None
        self.y_offset_ = 0.0
        self.sigma_sq = 1.0  # MMSE; sigma^2 (or sigma-tilde^2 on post-DA data)

        super().__init__(
            gamma=gamma,
            epsilon=epsilon,
            pad=pad,
            calibrate=calibrate,
            clipy=clipy,
            n_jobs=n_jobs,
            mean_match=mean_match,
        )

    # ------------------------------------------------------------------ fit

    def _centre(self, X, y, **kwargs):
        """Move to Lem. 2's coordinates: (X - mu, y - ybar), the slice
        H_X = {h : E_n[h(X)] = E_n[Y]} with the intercept ELIMINATED rather than
        carried as a free coordinate.

        On the slice the second moment of X(h - h_erm) IS its variance, so the
        SOCP ball becomes the paper's COVARIANCE ball and `h_erm` the
        with-intercept ERM -- the centre Lem. 2 names. Bounds are returned on the
        original outcome scale by adding `y_offset_` back (Cor. 3).

        `GX` rides along with the SAME mu, so the invariance constraint
        `|| (GX - X) h ||` is untouched (the two intercepts cancel); `Z` and the
        clipping limits stay on their raw scales.
        """
        if not self.mean_match:
            self.mu_ = np.zeros(X.shape[1])
            self.y_offset_ = 0.0
            return X, y, kwargs

        self.mu_ = X.mean(axis=0)
        self.y_offset_ = float(np.mean(y))
        GX = kwargs.get("GX")
        if GX is not None:
            kwargs = {**kwargs, "GX": np.asarray(GX).reshape(len(X), -1) - self.mu_}
        return X - self.mu_, np.asarray(y) - self.y_offset_, kwargs

    def _fit(self, X, y, **kwargs):
        self.N_samples = len(X)

        # observable outcome limits (clipy) -- on the RAW outcome scale, which is
        # the scale `_finalize` clips on and the scale bounds come back in
        self.y_min, self.y_max = float(np.min(y)), float(np.max(y))

        X, y, kwargs = self._centre(X, y, **kwargs)
        self.h_erm = OLS().fit(X, y).solution.flatten()

        # noise level: sigma^2 = min MSE. Fit on post-DA data => sigma-tilde^2.
        residuals = y.flatten() - X @ self.h_erm
        self.sigma_sq = float(np.mean(residuals**2))

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
        return [cp.norm(cp.Constant(self.R_constraint) @ (self.h_var - cp.Constant(self.h_erm)), 2) <= threshold]

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

    def _raw_bounds(self, X, gamma):
        """Closed-form shortcut, else the generic chunked solve."""
        if CLOSED_FORM_SOLUTION and self._supports_closed_form:
            # Cor. 3 on the slice: h_erm(x) +- s sqrt(gamma) ||g_x||, with the
            # representer norm the Mahalanobis length of the CENTRED query
            radius = self.scale * np.sqrt(gamma)
            X = X - self.mu_
            mahalanobis_sq = np.maximum(0, np.sum((X @ self.invSigmaX) * X, axis=1))
            margins = radius * np.sqrt(mahalanobis_sq)
            centers = X @ self.h_erm + self.y_offset_
            self.query_status = np.full(len(X), SolveStatus.OK, dtype=int)
            return np.column_stack([centers - margins, centers + margins])

        return super()._raw_bounds(X, gamma)

    def _prepare(self, X, gamma):
        """One DPP parameter set for the whole batch; queries are the rows of X,
        in the same centred coordinates the ball was fitted in."""
        self._set_solver_parameters(gamma)
        return list(X - self.mu_)

    def _solve_single(self, x):
        """(lower, upper, status); bounds are NaN on any non-OK status."""
        norm_x = np.linalg.norm(x)
        if norm_x < 1e-9:
            self.x_param.value = np.zeros_like(x)
            scale = 0.0
        else:
            self.x_param.value = x / norm_x
            scale = norm_x

        def solve_prob(prob):
            status = None
            for solver in (cp.CLARABEL, cp.ECOS):
                try:
                    # warm_start=False: queries must be independent (A5). The old
                    # flag was inert -- CLARABEL ignored warm starts before
                    # cvxpy 1.6 -- but now a solve inherits state from the
                    # previous one, so chunking would change bounds at ~1e-13.
                    prob.solve(solver=solver, warm_start=False, verbose=False)
                    status = prob.status
                except Exception:
                    status = None
                if status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                    return prob.value * scale, SolveStatus.OK
                if status == cp.INFEASIBLE:
                    # a proof, not a failure; retrying costs 2-5x.
                    # INFEASIBLE_INACCURATE still gets its ECOS second opinion.
                    return np.nan, SolveStatus.INFEASIBLE

            if status in (cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE):
                return np.nan, SolveStatus.INFEASIBLE
            return np.nan, SolveStatus.FAILURE

        lower, status_lo = solve_prob(self.min_problem)
        upper, status_hi = solve_prob(self.max_problem)
        # back to the outcome scale: the eliminated intercept is ybar - mu' h
        return lower + self.y_offset_, upper + self.y_offset_, max(status_lo, status_hi)


def _trust_region_min(B, c, delta, tol=1e-12, max_iter=200):
    """min ||B u - c|| subject to ||u|| <= delta. Exact.

    The least-squares trust-region subproblem. Unconstrained solution first; if it
    is outside the ball, the optimum sits ON the boundary at the unique lambda >= 0
    solving ||u(lambda)|| = delta, and ||u(lambda)|| is strictly decreasing in
    lambda, so a bisection lands on it to machine precision.
    """
    U, s, Vt = np.linalg.svd(B, full_matrices=False)
    Utc = U.T @ c

    nonzero = s > max(s.max(initial=0.0), 1.0) * 1e-14
    coefficients = np.zeros_like(s)
    coefficients[nonzero] = Utc[nonzero] / s[nonzero]
    if np.linalg.norm(coefficients) <= delta:  # interior
        return float(np.linalg.norm(B @ (Vt.T @ coefficients) - c))

    def norm_at(lam):
        return float(np.linalg.norm(s * Utc / (s**2 + lam)))

    lo, hi = 0.0, 1.0
    while norm_at(hi) > delta:  # bracket the root
        hi *= 2.0
        if hi > 1e18:
            break
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if norm_at(mid) > delta:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol * max(hi, 1.0):
            break
    u = Vt.T @ (s * Utc / (s**2 + 0.5 * (lo + hi)))
    return float(np.linalg.norm(B @ u - c))


def constraint_floor(design, y, gamma, *, kind, GX=None, Z=None, calibrate=False, mean_match=True):
    """Lowest value the extra constraint attains on the PI ball, in BUDGET units.

    Returned SQUARED, matching `PartialR2Net._budget()`, so the smallest admissible
    budget is `sqrt(floor)` on BOTH backends even though the partial-r2 constraint
    is natively a norm. A budget below it makes every query
    INFEASIBLE -- that is what this exists to prevent.

    Closed form, not a solve: the ball is Euclidean in `R(h - h_erm)` and the
    constraint is a distance, so this is a least-squares trust-region subproblem.
    Cost is one triangular solve plus one SVD at M <= 54.

    Args:
        design: the matrix the METHOD fits on -- X for `inv`, GX for `iv`
        y: outcomes, paired with `design`
        gamma: the ball this budget will be used with
        kind: 'inv' (PI+INV) or 'iv' (DA+PI+IV)
        GX: augmented design, required for `inv`
        Z: instrument, required for `iv`
        calibrate: scale the radius by sigma-hat, as the fitted model does
        mean_match: measure the floor over the SAME ball the model solves on --
            Lem. 2's covariance ball. A floor from the other geometry would make
            the budget guard lie in both directions.

    Returns:
        floor in squared budget units
    """
    design = np.asarray(design)
    N, M = design.shape

    if mean_match:
        mu = design.mean(axis=0)
        design = design - mu
        y = np.asarray(y) - np.mean(y)
        if GX is not None:
            GX = np.asarray(GX) - mu

    h_erm = OLS().fit(design, y).solution.flatten()
    residuals = np.asarray(y).flatten() - design @ h_erm
    scale = float(np.sqrt(np.mean(residuals**2))) if calibrate else 1.0
    delta = np.sqrt(N) * scale * np.sqrt(max(float(gamma), 0.0))

    if kind == "inv":
        if GX is None:
            raise ValueError("constraint_floor(kind='inv') needs GX")
        A, b = inv_constraint_terms(design, np.asarray(GX))
    elif kind == "iv":
        if Z is None:
            raise ValueError("constraint_floor(kind='iv') needs Z")
        A, b = iv_constraint_terms(design, y, np.asarray(Z))
    else:
        raise ValueError(f"unknown constraint kind {kind!r}")

    # u = R(h - h_erm) turns the ball into ||u|| <= delta
    _, R = np.linalg.qr(design)
    B = np.linalg.lstsq(R.T, A.T, rcond=None)[0].T
    smallest = _trust_region_min(B, b - A @ h_erm, delta)
    return float((smallest / np.sqrt(N)) ** 2)


def _jittered(A, M):
    """Stack the jitter block the SOCP needs to stay strictly feasible."""
    jitter_strength = 1e-6 * np.mean(np.abs(A))
    if jitter_strength < 1e-9:
        jitter_strength = 1e-6
    return np.vstack([A, np.sqrt(jitter_strength) * np.eye(M)])


def inv_constraint_terms(X, GX):
    """(A, b) with the INV constraint as || A h - b || <= sqrt(N) eps. b = 0."""
    # || (GX - X) h ||: QR compress (N x M -> M x M), then jitter
    _, R = np.linalg.qr(GX - X)
    A = _jittered(R, X.shape[1])
    return A, np.zeros(len(A))


def iv_constraint_terms(X, y, Z):
    """(A, b) with the IV constraint as || A h - b || <= sqrt(N) eps_iv."""
    Q_matrix, _ = np.linalg.qr(Z.reshape(len(Z), -1))
    Z_proj = Q_matrix.T @ X
    y_proj = Q_matrix.T @ y.flatten()

    M = X.shape[1]
    return _jittered(Z_proj, M), np.concatenate([y_proj, np.zeros(M)])


class InvarianceConstrainedPartialR2(PartialR2):
    """PI + explicit invariance-error constraint (§3.1): E_inv(h) <= epsilon^2."""

    def __init__(self, gamma=None, epsilon=None, **kwargs):
        if epsilon is None:
            raise ValueError("epsilon required")
        super().__init__(gamma=gamma, epsilon=epsilon, **kwargs)
        self.R_diff = None
        self.eps_param = None
        self._supports_closed_form = False

    def _precompute_matrices(self, X, y, GX=None, **kwargs):
        if GX is None:
            GX = X
        self.R_diff, _ = inv_constraint_terms(X, GX)

    def _get_constraints(self):
        constraints = super()._get_constraints()

        # Parameter, not constant: epsilon is swept at predict time (robustness)
        self.eps_param = cp.Parameter(nonneg=True)
        constraints.append(
            cp.norm(cp.Constant(self.R_diff) @ self.h_var, 2) <= np.sqrt(self.N_samples) * self.eps_param
        )
        return constraints

    def _set_solver_parameters(self, gamma):
        super()._set_solver_parameters(gamma)
        self.eps_param.value = float(self.epsilon)


class InstrumentalVariablePartialR2(PartialR2):
    """PI + leaky IV constraint (Asm. 3). Null/empty Z falls back to baseline PI."""

    def __init__(self, gamma=None, gamma_z=0.0, rho=1.0, epsilon_iv=None, **kwargs):
        self.gamma_z = gamma_z
        self.rho = rho
        # epsilon_iv: the IV budget ||E[W#|Z-tilde]|| (oracle `eps_iv_star`).
        # Distinct from `epsilon`, whose only role in this class is the +/-eps
        # padding: padding validity is pointwise (Thm. 3.A Jensen step), the IV
        # budget is a projection norm. One attribute per role, one consumer each.
        self.epsilon_iv = epsilon_iv
        super().__init__(gamma=gamma, **kwargs)
        self.Z_projector_R = None
        self.y_residual_base = None
        self.iv_threshold_param = None
        self._has_iv = False
        self._supports_closed_form = False

        if gamma_z != 0.0:
            logger.warning(
                f"gamma_z={gamma_z} != 0: the IV budget is exact only at "
                "gamma_z = 0, where the cross-term vanishes. The additive "
                "s*sqrt(gamma_z) top-up is a heuristic."
            )

    @property
    def iv_bound(self):
        """sqrt of the budget on Var(E[Y - h|Z]) = ||E[W#|Z-tilde]||."""
        # calibrated: s is the *pre*-DA sigma = sqrt(sigma_sq / rho); else 1
        s = float(np.sqrt(self.sigma_sq / self.rho)) if self.calibrate else 1.0
        return self.epsilon_iv + s * np.sqrt(self.gamma_z)

    def _precompute_matrices(self, X, y, Z=None, **kwargs):
        self._has_iv = Z is not None and np.size(Z) > 0
        if not self._has_iv:
            return
        if self.epsilon_iv is None:
            raise ValueError(
                "epsilon_iv is required when an instrument is supplied; pass the oracle `eps_iv_star` (+ EPS_TOL)."
            )

        self.Z_projector_R, self.y_residual_base = iv_constraint_terms(X, y, Z)

    def _get_constraints(self):
        constraints = super()._get_constraints()
        if not self._has_iv:
            return constraints

        self.iv_threshold_param = cp.Parameter(nonneg=True)
        constraints.append(
            cp.norm(cp.Constant(self.y_residual_base) - cp.Constant(self.Z_projector_R) @ self.h_var, 2)
            <= self.iv_threshold_param
        )
        return constraints

    def _set_solver_parameters(self, gamma):
        super()._set_solver_parameters(gamma)
        if self._has_iv:
            self.iv_threshold_param.value = np.sqrt(self.N_samples) * self.iv_bound


class IntersectionMixin:
    """
    Cor. 1 at the INTERVAL level: max of the lowers, min of the uppers, worse status.

    Backend-agnostic -- it touches nothing but `self.baseline` / `self.augmented`,
    so the two balls need not share a parameterisation. Validity is a membership
    fact about h_*(x), not a geometric one.
    """

    def _predict(self, X, gamma=None, epsilon=None, **kwargs):
        if epsilon is not None:
            self.epsilon = epsilon
        branch_kwargs = dict(gamma=gamma, epsilon=epsilon, **kwargs)
        lower_base, upper_base = self.baseline.predict(X, **branch_kwargs).T
        lower_da, upper_da = self.augmented.predict(X, **branch_kwargs).T

        lower = np.maximum(lower_base, lower_da)
        upper = np.minimum(upper_base, upper_da)

        # a branch failure/infeasibility carries over to the intersection
        status = np.maximum(self.baseline.query_status, self.augmented.query_status)

        # empty intersection: infeasible, same convention as the solver
        empty = lower > upper
        if empty.any():
            logger.warning(f"Empty intersection at {empty.sum()}/{len(empty)} queries.")
            lower = np.where(empty, np.nan, lower)
            upper = np.where(empty, np.nan, upper)
            status = np.where(empty, SolveStatus.INFEASIBLE, status)

        self.query_status = status.astype(int)
        # diagnostics ride along from the branches; the WORSE of the two is the
        # honest reading for the intersection, since a starved branch starves the
        # intersection. Without this the nets' backtrack gate silently skips the
        # two intersection methods, which are the ones with the most constraints.
        parts = [self.baseline.query_diagnostics, self.augmented.query_diagnostics]
        self.query_diagnostics = None if any(p is None for p in parts) else np.maximum(*parts)
        return np.column_stack([lower, upper])


class IntersectedPartialR2(IntersectionMixin, PartialR2):
    """Baseline PI intersected with DA+PI (Cor. 1). Padding hits the DA branch only."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.baseline = None
        self.augmented = None

    def _branch(self, pad):
        return PartialR2(
            gamma=self.gamma,
            epsilon=self.epsilon,
            pad=pad,
            calibrate=self.calibrate,
            clipy=self.clipy,
            n_jobs=self.n_jobs,
            mean_match=self.mean_match,
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


class IntersectedInstrumentalVariablePartialR2(IntersectedPartialR2):
    """Baseline PI+IV (null instrument) intersected with DA+PI+IV."""

    def __init__(self, gamma_z=0.0, epsilon_iv=None, **kwargs):
        self.gamma_z = gamma_z
        self.epsilon_iv = epsilon_iv
        super().__init__(**kwargs)

    def _branch(self, pad, rho=1.0):
        return InstrumentalVariablePartialR2(
            gamma=self.gamma,
            gamma_z=self.gamma_z,
            rho=rho,
            epsilon=self.epsilon,
            epsilon_iv=self.epsilon_iv,
            pad=pad,
            calibrate=self.calibrate,
            clipy=self.clipy,
            n_jobs=self.n_jobs,
            mean_match=self.mean_match,
        )

    def _fit_branches(self, X, y, GX, G):
        # baseline sees no instrument => reduces to PI
        self.baseline = self._branch(pad=False).fit(X, y, Z=None)
        self.augmented = self._branch(pad=self.pad).fit(GX, y, Z=G)
        # rho known once both noise levels are: threshold is a cvx Parameter
        self.augmented.rho = self.rho
