"""
Partial identification / sensitivity models.

Uniform signature: gamma (budget, in the paper's sigma-scaled units), epsilon
(invariance error), pad (Thm. 3.A), recalibrate (post-DA budget gamma/rho, SS4.2),
clipy (clip to observed y range).
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


def recalibrated_gamma(gamma, rho, t) -> float:
    """The post-DA budget, SS4.2: gamma~ = gamma ((1 - t) + t / rho).

    t = 1 scales the inherited gamma down to gamma/rho, the budget the DA's
    information loss rho = sigma~^2/sigma^2 leaves on the augmented data; t = 0
    keeps gamma. Linear in between, which is what the recalibrate sweep plots.
    rho >= 1 in population (DPI); a sample rho_hat < 1 is noise and is read as 1,
    so the post-DA budget never exceeds the inherited gamma.
    """
    return float(gamma) * ((1.0 - float(t)) + float(t) / max(float(rho), 1.0))


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

    # the fitted program reads a predict-time budget: the INV cone reads epsilon,
    # the T-as-IV constraint reads `epsilon_iv`, and a plain ball only pads (the
    # perf sweep re-solves the former two and `repad`s the rest)
    solves_on_epsilon: bool = False
    # taken off the pad, never off the constraint: the sweeps set it to EPS_TOL under
    # the IM-CI (ParamSweepRunner), whose interval carries the sampling allowance the
    # tolerance used to add to the pad. 0.0 everywhere else, so every other number is
    # today's. The net backend's own intersections (`partial_r2_net._branch_kwargs`)
    # do not forward it; do-MNIST is the only caller and the config forces its
    # `im-ci` to 0, so it never leaves 0.0 there
    pad_tolerance: float = 0.0

    def __init__(
        self,
        gamma=None,
        epsilon=0.0,
        pad=False,
        pad_epsilon=None,
        recalibrate=True,
        clipy=True,
        n_jobs=1,
        mean_match=True,
        rho=1.0,
    ):
        if gamma is None:
            raise ValueError("gamma must be explicitly provided")

        self.epsilon = epsilon
        self.pad = pad
        # Thm. 3.A's epsilon and the PI+INV constraint's epsilon are DIFFERENT
        # NORMS of the same defect W = h_*(X) - h_*(X~). SS3.1 constrains
        # E_inv(h) = E|W|^2 <= eps^2 (an L2 budget); SS2.4 defines eps-approximate
        # T-invariance as sup_{x,tau} |W| <= eps, and Thm. 3.A's proof uses that
        # pointwise bound (|eta| <= eps a.s.). An L2 bound implies no pointwise
        # one, so padding by the constraint's epsilon is NOT what Thm. 3.A asks
        # for -- measured on the optical device, RMS 0.210 against a sup of 1.230.
        # `pad_epsilon` carries the padding budget separately. None keeps the old
        # behaviour (pad by `epsilon`); pass the sup-side budget to get the
        # guarantee the theorem actually states.
        self.pad_epsilon = pad_epsilon
        # SS4.2: the ball a DA+ method solves is gamma~ = gamma ((1 - t) + t / rho)
        # with t = `recalibrate` in [0, 1] and rho the information-loss factor
        # sigma~^2/sigma^2 of the DA it was fit on. Baselines keep rho = 1, so
        # t is a no-op on them. The radius is ALWAYS sigma-hat sqrt(gamma~): gamma
        # is in the paper's sigma-scaled units, on every method.
        self.rho = rho
        self.recalibrate = recalibrate
        self.clipy = clipy
        self.n_jobs = n_jobs
        # Lem. 2: the identified set lives on the mean-matched slice
        # H_X = {h : E[h(X)] = E[Y]}. False keeps the pre-2026-09 geometry.
        self.mean_match = mean_match
        self.query_status = None  # per-query SolveStatus, set on every predict
        self.query_diagnostics = None  # optional per-query extras, set on predict
        self.raw_bounds_ = None  # the last predict's unpadded bounds, for `repad`
        # one `(name, opts)` conic backend for the seed_var sweep; None is the
        # CLARABEL-then-ECOS chain
        self.backend = None
        self.y_min = -np.inf
        self.y_max = np.inf

        super().__init__(gamma)

    # ------------------------------------------------------------- budget

    @property
    def rho(self) -> float:
        """Information-loss factor sigma~^2/sigma^2 of the data this ball was fit
        on; 1 for a baseline. Intersections read it off their two branches."""
        return self._rho

    @rho.setter
    def rho(self, rho):
        rho = float(rho)
        if not np.isfinite(rho) or rho <= 0.0:
            raise ValueError(f"rho must be a positive finite float; got {rho!r}")
        # DPI: rho >= 1 in population; a sample value below 1 is read as 1, so the
        # recalibrated budget never exceeds the inherited gamma (`recalibrated_gamma`)
        self._rho = max(rho, 1.0)

    @property
    def recalibrate(self) -> float:
        """t in [0, 1]: 0 keeps the inherited gamma, 1 solves at gamma/rho."""
        return self._recalibrate

    @recalibrate.setter
    def recalibrate(self, t):
        t = float(t)
        if not np.isfinite(t) or not 0.0 <= t <= 1.0:
            raise ValueError(f"recalibrate must lie in [0, 1]; got {t!r}")
        self._recalibrate = t

    def budget(self, gamma) -> float:
        """The budget actually solved at: `recalibrated_gamma(gamma, rho, t)`."""
        return recalibrated_gamma(gamma, self.rho, self.recalibrate)

    # ------------------------------------------------------------- predict

    def _predict(self, X, gamma=None, epsilon=None, recalibrate=None, **kwargs):
        gamma = self.gamma if gamma is None else gamma
        if epsilon is not None:
            self.epsilon = epsilon  # the CONSTRAINT RHS; `pad_epsilon` is separate
        if recalibrate is not None:
            self.recalibrate = recalibrate  # swept at predict time, like gamma
        self.raw_bounds_ = self._raw_bounds(X, gamma)
        return self._finalize(self.raw_bounds_)

    def repad(self, epsilon):
        """The last predict's raw bounds finalised at another padding epsilon, no
        solve; what the perf sweep times for a method whose program does not read
        epsilon."""
        self.epsilon = epsilon
        return self._finalize(self.raw_bounds_)

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

    @property
    def pad_amount(self) -> float:
        """Thm. 3.A's epsilon: `pad_epsilon` when supplied, else the constraint's
        own (L2) epsilon -- see `__init__` for why those are not the same thing --
        less `pad_tolerance` (0.0 unless a sweep runs under the IM-CI)."""
        return max(float(self.epsilon if self.pad_epsilon is None else self.pad_epsilon) - self.pad_tolerance, 0.0)

    def _finalize(self, bounds):
        """eps-padding (Thm. 3.A) then clipping to observable y limits."""
        if self.pad:
            bounds = bounds + np.array([-self.pad_amount, self.pad_amount])
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
        pad_epsilon=None,
        recalibrate=True,
        clipy=True,
        n_jobs=1,
        mean_match=True,
        rho=1.0,
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
            pad_epsilon=pad_epsilon,
            recalibrate=recalibrate,
            clipy=clipy,
            n_jobs=n_jobs,
            mean_match=mean_match,
            rho=rho,
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

        `GX` and `X_pre` ride along with the SAME mu, so the invariance constraint
        `|| (GX - X) h ||` and the augmentation shift `X - X_pre` are untouched
        (the two intercepts cancel); `Z`, `T` and the clipping limits stay on
        their raw scales.
        """
        if not self.mean_match:
            self.mu_ = np.zeros(X.shape[1])
            self.y_offset_ = 0.0
            return X, y, kwargs

        self.mu_ = X.mean(axis=0)
        self.y_offset_ = float(np.mean(y))
        for key in ("GX", "X_pre"):
            value = kwargs.get(key)
            if value is not None:
                kwargs = {**kwargs, key: np.asarray(value).reshape(len(X), -1) - self.mu_}
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
        """s = sigma-hat: the MMSE of the data this ball was fit on (paper's units)."""
        return float(np.sqrt(self.sigma_sq))

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
        self.radius_param.value = self.scale * np.sqrt(self.budget(gamma))

    # ------------------------------------------------------------- predict

    def _raw_bounds(self, X, gamma):
        """Closed-form shortcut, else the generic chunked solve."""
        if CLOSED_FORM_SOLUTION and self._supports_closed_form:
            # Cor. 3 on the slice: h_erm(x) +- s sqrt(gamma~) ||g_x||, with the
            # representer norm the Mahalanobis length of the CENTRED query
            radius = self.scale * np.sqrt(self.budget(gamma))
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
            for solver, opts in self._solver_chain():
                try:
                    # warm_start=False: queries must be independent (A5). The old
                    # flag was inert -- CLARABEL ignored warm starts before
                    # cvxpy 1.6 -- but now a solve inherits state from the
                    # previous one, so chunking would change bounds at ~1e-13.
                    prob.solve(solver=solver, warm_start=False, verbose=False, **opts)
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

    def _solver_chain(self):
        """The pinned `(name, opts)` backend alone, else CLARABEL then ECOS."""
        if self.backend is not None:
            return (tuple(self.backend),)
        return ((cp.CLARABEL, {}), (cp.ECOS, {}))


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


def constraint_floor(design, y, gamma, *, kind, GX=None, Z=None, mean_match=True, rho=1.0, recalibrate=True):
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
        kind: 'inv' (the invariance cone) or 'iv' (ONE instrument constraint)
        GX: augmented design, required for `inv`
        Z: the instrument block of the ONE constraint being measured, required
            for `iv`. The T budget's floor is measured with the translation amounts
            alone, the Z radius is never floor-reported (SS2.6), and a PAIR's feasibility
            is not a floor at all: it is the smallest Z moment on the ball cap the
            T constraint.
        mean_match: measure the floor over the SAME ball the model solves on --
            Lem. 2's covariance ball. A floor from the other geometry would make
            the floor report lie in both directions.
        rho, recalibrate: the DA ball's factor and toggle, so the floor is over
            the recalibrated budget the fitted model actually solves at
            (`BoundedSA.budget`). Leave both at their defaults for a baseline.

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
    scale = float(np.sqrt(np.mean(residuals**2)))
    delta = np.sqrt(N) * scale * np.sqrt(max(recalibrated_gamma(gamma, rho, recalibrate), 0.0))

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


def _instrument_columns(Z, n):
    """`Z` as the (n, m) float block a constraint takes; None or empty is (n, 0),
    which contributes no constraint at all.

    A local twin of `model_fitting.instrument_columns` on purpose: `src.methods`
    does not import from `src.experiments`, and three lines are cheaper than that
    dependency.
    """
    if Z is None:
        return np.zeros((n, 0))
    Z = np.asarray(Z, dtype=float)
    return Z.reshape(n, -1) if Z.size else Z.reshape(n, 0)


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

    solves_on_epsilon = True  # `eps_param` is the constraint's RHS

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
    """PI + leaky IV constraints (Asm. 3), ONE PER INSTRUMENT.

    Two instrument blocks reach `fit` by their own keyword: `T`, the DA
    translation amounts of SS4.3, and `Z`, the observed instrument. Every block
    that is present gets its own SOC constraint at its own radius -- `t_bound`
    from `epsilon_iv`, `z_bound` from `epsilon_iv_z` and `gamma_z` -- so one
    instrument's budget is never spent by the other. A missing block is (n, 0)
    and contributes nothing; with both missing this is baseline PI exactly.

    Each constraint is valid on its own by the contraction step of Thm. 3.B's
    proof (E[U + xi | .] is conserved, T independent of (xi, U, Z)). The pair is
    NOT in general inside the single constraint on Z-tilde = (T, Z) the paper
    writes: that one admits every h whose joint moment norm is under
    sqrt(r_T^2 + r_Z^2), which on the T side is looser than r_T. The two can also
    be jointly infeasible where each alone is feasible, and no floor predicts
    where: the binding quantity is the smallest Z moment on the ball intersected
    with the T constraint. That reads INFEASIBLE per query, as any empty
    constraint set does, and nothing is inflated to hide it.
    """

    def __init__(self, gamma=None, gamma_z=0.0, epsilon_iv=None, epsilon_iv_z=0.0, **kwargs):
        self.gamma_z = gamma_z
        # epsilon_iv: the T-as-IV budget ||E[W#|T]|| (oracle `eps_iv_star`), the
        # radius of the T constraint and of nothing else. Distinct from
        # `epsilon`, whose only role in this class is the +/-eps padding: padding
        # validity is pointwise (Thm. 3.A Jensen step), an IV budget is a
        # projection norm. One attribute per role, one consumer each.
        self.epsilon_iv = epsilon_iv
        # epsilon_iv_z: the observed instrument's measured piece, beside the
        # declared radius s sqrt(gamma_z) it is combined with (SS2.6)
        self.epsilon_iv_z = epsilon_iv_z
        super().__init__(gamma=gamma, **kwargs)
        self.T_projector_R = None
        self.t_residual_base = None
        self.Z_projector_R = None
        self.z_residual_base = None
        self.t_threshold_param = None
        self.z_threshold_param = None
        self._has_t = False
        self._has_z = False
        # how far the augmentation can move a DECLARED Z moment; set at fit
        self._z_allowance = 0.0
        self._budget_logged = False
        self._supports_closed_form = False

    @property
    def _has_iv(self) -> bool:
        """Any instrument at all; False leaves the Lem. 2 ball on its own."""
        return self._has_t or self._has_z

    @property
    def solves_on_epsilon(self) -> bool:
        """The T threshold is read at predict time (the epsilon sweep moves the T
        budget with the ratio), so a model with a T constraint re-solves per grid
        point; one without it only pads. `PI+INV` and `PI+INV+IV` keep the class
        attribute, which shadows this."""
        return self._has_t

    @property
    def t_bound(self) -> float | None:
        """Radius of the T constraint: the T-as-IV budget, alone. None when the
        model was built without one (it then carries no T block either)."""
        return None if self.epsilon_iv is None else float(self.epsilon_iv)

    @property
    def z_bound(self) -> float:
        """Radius of the Z constraint: the measured piece `epsilon_iv_z` and the
        declared one in root sum square, s = sigma-hat of the PRE-DA data.

        The declared piece is `s sqrt(gamma_z) + _z_allowance`. The first term is
        the leak the config declares, a statement about the instrument's moment on
        the ORIGINAL design; the second pays for the Z moment the augmentation adds
        to the element this design's program must admit (`_declared_allowance`).
        Without that term the DA branch's constraint excludes h#, the element it
        has to admit, however large gamma_z is -- measured, and it is what emptied
        the decoupled pair on the plasmode sweep. `_z_allowance` is 0 for a model
        fitted with no `X_pre` (every non-DA method) and at gamma_z = 0, so the
        radius is then exactly what it was.

        Exactly s sqrt(gamma_z) on a declared non-DA method, and exactly
        `epsilon_iv_z` at gamma_z = 0.
        """
        s_sq = self.sigma_sq / self.rho
        # the allowance rides with a DECLARED leak and disappears with it: the F2
        # figure sweeps `gamma_z` on a fitted model down to 0, and there the radius
        # must be exactly `epsilon_iv_z` again, not the fit-time allowance
        allowance = self._z_allowance if self.gamma_z else 0.0
        declared = np.sqrt(s_sq * self.gamma_z) + allowance
        return float(np.sqrt(self.epsilon_iv_z**2 + declared**2))

    def _precompute_matrices(self, X, y, Z=None, T=None, X_pre=None, **kwargs):
        T, Z = _instrument_columns(T, len(X)), _instrument_columns(Z, len(X))
        self._has_t, self._has_z = T.shape[1] > 0, Z.shape[1] > 0
        if self._has_t and self.epsilon_iv is None:
            raise ValueError("epsilon_iv is required with T-as-IV; pass the oracle `eps_iv_star` (+ EPS_TOL).")
        if self._has_t:
            self.T_projector_R, self.t_residual_base = iv_constraint_terms(X, y, T)
        if self._has_z:
            self.Z_projector_R, self.z_residual_base = iv_constraint_terms(X, y, Z)
            self._z_allowance = self._declared_allowance(X, Z, X_pre)
        if self._has_z and self.z_bound == 0.0:
            # the usual cause is a caller handing the translation amounts as `Z`:
            # T-as-IV goes in the `T` block, and the Z block's radius is 0 until
            # `epsilon_iv_z` or `gamma_z` says otherwise
            logger.warning(
                "IV: an observed instrument with a zero radius (epsilon_iv_z 0, gamma_z 0) "
                "forces its moment to hold exactly; every query will read INFEASIBLE."
            )

    def _declared_allowance(self, X, Z, X_pre) -> float:
        """How much of the DECLARED Z radius the augmentation itself can consume.

        The element this program must admit is not the target but h#, the target
        plus what this design can absorb of the invariance signal w, and h# carries
        w's residual Z moment || Q_Z' (I - P_X) w || / sqrt(N). Two facts bound it:
        || w || <= sqrt(N) epsilon by the definition of the invariance budget, and w
        lies in the span of what the augmentation moved, X - X_pre. So the worst
        case over THAT subspace, after the projection the residual already carries,
        is `|| Q_Z' (I - P_X) Q_D ||_2 * epsilon`. Bounding it over every direction
        of norm epsilon instead (dropping Q_D) is also valid and measurably useless:
        it takes the radius far enough that the constraint stops binding at all.

        Without this term the DA branch's constraint excludes h# however large
        gamma_z is, which is what a pooled radius used to hide (SS9.6 of the
        round-14 plan). `epsilon` is the one this model was FITTED at, so the radius
        does not move with the swept ratio (R2). Zero without a pre-augmentation
        design, without a declared leak, or when the augmentation moved nothing.
        """
        if X_pre is None or not self.gamma_z:
            return 0.0
        X = np.asarray(X, dtype=float)
        difference = X - np.asarray(X_pre, dtype=float).reshape(len(X), -1)
        if not np.any(difference):
            return 0.0
        # an orthonormal basis of the SPAN of the shift, truncated: a plain QR of a
        # rank deficient shift completes the basis arbitrarily, and those made-up
        # directions would inflate kappa
        left, singular, _ = np.linalg.svd(difference, full_matrices=False)
        keep = singular > max(float(singular[0]), 1.0) * 1e-12
        if not keep.any():
            return 0.0
        Q_Z, _ = np.linalg.qr(Z)
        Q_X, _ = np.linalg.qr(X)
        Q_D = left[:, keep]
        residual_basis = Q_D - Q_X @ (Q_X.T @ Q_D)
        kappa = float(np.linalg.svd(Q_Z.T @ residual_basis, compute_uv=False)[0])
        return kappa * float(self.epsilon)

    def _get_constraints(self):
        constraints = super()._get_constraints()
        # one SOC per non-empty block, T first: the order the cvx problem carries
        # is the ball, then the instrument blocks, then an INV cone on top
        if self._has_t:
            self.t_threshold_param = cp.Parameter(nonneg=True)
            constraints.append(
                cp.norm(cp.Constant(self.t_residual_base) - cp.Constant(self.T_projector_R) @ self.h_var, 2)
                <= self.t_threshold_param
            )
        if self._has_z:
            self.z_threshold_param = cp.Parameter(nonneg=True)
            constraints.append(
                cp.norm(cp.Constant(self.z_residual_base) - cp.Constant(self.Z_projector_R) @ self.h_var, 2)
                <= self.z_threshold_param
            )
        return constraints

    def _set_solver_parameters(self, gamma):
        super()._set_solver_parameters(gamma)
        if self._has_t:
            self.t_threshold_param.value = np.sqrt(self.N_samples) * self.t_bound
        if self._has_z:
            self.z_threshold_param.value = np.sqrt(self.N_samples) * self.z_bound
            if self.gamma_z != 0.0 and not self._budget_logged:
                # once per fitted model, at the first solve: rho is final here (an
                # intersection sets its DA branch's after fit), so s is too
                self._budget_logged = True
                logger.info(
                    f"IV: gamma_z={self.gamma_z:g}, s={np.sqrt(self.sigma_sq / self.rho):.6g}, "
                    f"DA-side allowance {self._z_allowance:.6g}; r_Z={self.z_bound:.6g}, "
                    f"r_T={'none' if self.t_bound is None else format(self.t_bound, '.6g')}. "
                    "Two radii, one per instrument, never pooled."
                )

    def _predict(self, X, epsilon_iv=None, **kwargs):
        """`epsilon_iv` is the T budget at predict time; it reaches the T threshold
        and nothing else, and a model without a T constraint ignores it. The program
        is unchanged, so this re-solves without re-canonicalising."""
        if epsilon_iv is not None and self._has_t:
            self.epsilon_iv = float(epsilon_iv)
        return super()._predict(X, **kwargs)


class InvarianceConstrainedInstrumentalVariablePartialR2(InstrumentalVariablePartialR2):
    """PI + INV + IV on the ORIGINAL design: the Lem. 2 ball, the invariance cone
    of SS3.1 (`epsilon` on ||(GX - X) h||) and the leaky IV constraint of Asm. 3
    on the observed instrument at `z_bound`. It is a non-DA method, so it carries
    no T-as-IV moment: that condition holds on the augmented design, not on X.
    An empty Z reduces it to PI+INV exactly."""

    solves_on_epsilon = True

    def __init__(self, gamma=None, epsilon=None, **kwargs):
        if epsilon is None:
            raise ValueError("epsilon required")
        super().__init__(gamma=gamma, epsilon=epsilon, **kwargs)
        self.R_diff = None
        self.eps_param = None

    def _precompute_matrices(self, X, y, GX=None, Z=None, T=None, **kwargs):
        super()._precompute_matrices(X, y, Z=Z, T=T, **kwargs)
        self.R_diff, _ = inv_constraint_terms(X, X if GX is None else GX)

    def _get_constraints(self):
        constraints = super()._get_constraints()
        self.eps_param = cp.Parameter(nonneg=True)
        constraints.append(
            cp.norm(cp.Constant(self.R_diff) @ self.h_var, 2) <= np.sqrt(self.N_samples) * self.eps_param
        )
        return constraints

    def _set_solver_parameters(self, gamma):
        super()._set_solver_parameters(gamma)
        self.eps_param.value = float(self.epsilon)


class IntersectionMixin:
    """
    Cor. 1 at the INTERVAL level: max of the lowers, min of the uppers, worse status.

    Backend-agnostic -- it touches nothing but `self.baseline` / `self.augmented`,
    so the two balls need not share a parameterisation. Validity is a membership
    fact about h_*(x), not a geometric one.
    """

    @property
    def solves_on_epsilon(self) -> bool:
        """The intersection re-solves when either branch does."""
        return bool(self.baseline.solves_on_epsilon or self.augmented.solves_on_epsilon)

    def _predict(self, X, gamma=None, epsilon=None, recalibrate=None, **kwargs):
        if epsilon is not None:
            self.epsilon = epsilon
        if recalibrate is not None:
            self.recalibrate = recalibrate
        # both branches see t and the T budget; the baseline has no T constraint,
        # so only the DA branch acts on the latter
        branch_kwargs = dict(gamma=gamma, epsilon=epsilon, recalibrate=recalibrate, **kwargs)
        return self._combine(self.baseline.predict(X, **branch_kwargs), self.augmented.predict(X, **branch_kwargs))

    def repad(self, epsilon):
        """Both branches re-finalised at `epsilon` (the DA branch pads, the baseline
        never does) and intersected again, no solve."""
        self.epsilon = epsilon
        return self._combine(self.baseline.repad(epsilon), self.augmented.repad(epsilon))

    def _combine(self, base, da):
        lower_base, upper_base = base.T
        lower_da, upper_da = da.T

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
        # rho = 1 at construction: the DA branch's factor is only known once
        # both branches are fitted (`_fit_branches` sets it)
        branch = PartialR2(
            gamma=self.gamma,
            epsilon=self.epsilon,
            pad=pad,
            recalibrate=self.recalibrate,
            rho=1.0,
            clipy=self.clipy,
            n_jobs=self.n_jobs,
            mean_match=self.mean_match,
        )
        branch.pad_tolerance = self.pad_tolerance  # the sweeps' IM-CI setting reaches the DA branch
        return branch

    def _fit_branches(self, X, y, GX, G, Z=None):
        self.baseline = self._branch(pad=False).fit(X, y)
        self.augmented = self._branch(pad=self.pad).fit(GX, y)
        # rho known once both noise levels are; the DA branch solves at gamma~
        self.augmented.rho = self.rho

    def _fit(self, X, y, GX=None, G=None, Z=None, **kwargs):
        if GX is None:
            raise ValueError("GX (augmented treatment) required")

        GX = np.asarray(GX).reshape(len(GX), -1)
        self._fit_branches(X, y, GX, G, Z)

        self.sigma_sq = self.baseline.sigma_sq
        self.y_min, self.y_max = float(np.min(y)), float(np.max(y))
        return self

    # the base setter stays (a bare @property would drop it and `__init__`'s
    # assignment would raise); the getter reads the branches once they exist
    @BoundedSA.rho.getter
    def rho(self):
        """Information-loss factor sigma-tilde^2 / sigma^2 (>= 1 by DPI), read off
        the two fitted branches; the constructor's value until then."""
        baseline, augmented = getattr(self, "baseline", None), getattr(self, "augmented", None)
        if baseline is None or augmented is None:
            return self._rho
        return augmented.sigma_sq / baseline.sigma_sq


class IntersectedInstrumentalVariablePartialR2(IntersectedPartialR2):
    """Baseline PI+IV on the observed Z intersected with DA+PI+IV on the
    augmented design (Cor. 1).

    Each branch carries one constraint per instrument block it is handed. The
    baseline only ever sees Z: no T-as-IV moment holds on the un-augmented
    design. `instrument` says what the DA branch sees -- "T,Z" both (the
    default), "Z" the observed instrument alone, "T" the translation amounts
    alone. Both radii travel to both branches and each uses the ones its blocks
    ask for: `epsilon_iv` is r_T, `epsilon_iv_z` with `gamma_z` is r_Z (0.0 under
    a declared budget makes it exactly s sqrt(gamma_z)). An empty Z makes the
    baseline plain PI and the DA branch T-only, which is today's run."""

    def __init__(self, gamma_z=0.0, epsilon_iv=None, epsilon_iv_z=0.0, instrument="T,Z", **kwargs):
        if instrument not in ("T,Z", "Z", "T"):
            raise ValueError(f"instrument must be 'T,Z', 'Z' or 'T'; got {instrument!r}")
        self.gamma_z = gamma_z
        self.epsilon_iv = epsilon_iv
        self.epsilon_iv_z = epsilon_iv_z
        self.instrument = instrument
        super().__init__(**kwargs)

    def _branch(self, pad):
        # rho = 1 at construction: the DA branch's factor is only known once both
        # branches are fitted (`_fit_branches` sets it)
        branch = InstrumentalVariablePartialR2(
            gamma=self.gamma,
            gamma_z=self.gamma_z,
            epsilon=self.epsilon,
            epsilon_iv=self.epsilon_iv,
            epsilon_iv_z=self.epsilon_iv_z,
            pad=pad,
            recalibrate=self.recalibrate,
            rho=1.0,
            clipy=self.clipy,
            n_jobs=self.n_jobs,
            mean_match=self.mean_match,
        )
        branch.pad_tolerance = self.pad_tolerance
        return branch

    def _fit_branches(self, X, y, GX, G, Z=None):
        # empty is spelled (n, 0), and a branch handed it carries no constraint
        Z = _instrument_columns(Z, len(X))
        T = _instrument_columns(G, len(X))
        self.baseline = self._branch(pad=False).fit(X, y, Z=Z)
        # `X_pre` is the design before augmentation: the DA branch's declared Z
        # radius needs it to carry the declaration across (D20)
        self.augmented = self._branch(pad=self.pad).fit(
            GX,
            y,
            Z=Z if self.instrument in ("T,Z", "Z") else None,
            T=T if self.instrument in ("T,Z", "T") else None,
            X_pre=X,
        )
        # rho known once both noise levels are: the ball and both IV thresholds
        # are cvx Parameters, set at predict
        self.augmented.rho = self.rho
