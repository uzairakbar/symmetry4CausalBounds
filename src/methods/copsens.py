"""
CopSens partial identification (Zheng / D'Amour / Franks, JMLR 26(36) 2025), cast
into `BoundedSA`: the do-MNIST backend.

A latent-factor sensitivity model. Where `PartialR2` puts a ball around h_erm in
FUNCTION space and maximises a linear objective (an SOCP), CopSens puts a ball
around the origin in an m-dimensional LATENT sensitivity space and maximises a
non-convex one (SLSQP, multi-start). The two gammas are different
parameterisations and are NOT comparable: each is selected against population
coverage before the two are compared.

    U | x ~ N(mu_u(x), Sigma)                       factor model on the treatments
    h_g(x) = E_{x' ~ anchors}[ F( a + (mu_u(x') - mu_u(x)).g ) ]
    a      = g_inv(mu_y(x), s),  s = radius ||v||,  g = radius Sigma^{-1/2} v
    bounds = [min, max] h_g(x)  over  ||v|| <= 1,  i.e. g' Sigma g <= gamma sigma^2

`a` de-convolves the posterior smearing, so h_0(x) == mu_y(x) exactly. The outcome
model mu_y is a PREFIT net handed in by the runner: every PI variant of one run
shares it, so the ERM and the PI centre are the same function.

Every number here follows the mixin implementation this backend was ported from,
given identical fit arrays, nets and seeds: the factor model, the anchor draw
(before the constraint rows, off one `rng_`), sigma-hat^2 = E[mu(1 - mu)] on the
fit rows, the radius sqrt(gamma sigma-hat^2), the start set, the SLSQP options, the
acceptance test, the floor solve and both budgets. Four things differ by design:

  1. the constraint and the floor are evaluated at the OBJECTIVE'S radius, so a
     predict(gamma=...) override moves the whole program, not half of it;
  2. a query with no accepted start on either side is NaN on both sides with status
     `FAILURE` (the `BoundedSA` convention), where the mixin kept the other side;
  3. the failed-start fraction per query rides back as `query_diagnostics`, one
     column, instead of a global solve report;
  4. the floor is cached on (radius, n_starts): it depends on the ball and the
     fitted latent model, never on the budget, and a gamma grid re-solves it once
     per point instead of once per predict.

`sigma_model` (default None, the centre's own net) lets a ball borrow sigma-hat^2
from another prefit net, read on the observed rows: the do-MNIST PI+INV takes the
ERM's under the X-ball centres (off, inv), so its radius is PI's.

`recalibrate` and `rho` reach the radius through `BoundedSA.budget`; `mean_match`
is accepted for the uniform signature and does nothing (the latent ball has no
mean-matched slice); `pad` and `clipy` are `BoundedSA._finalize`'s.
"""

import contextlib
import copy

import cvxpy as cp
import numpy as np
from loguru import logger
from scipy.optimize import NonlinearConstraint, minimize
from scipy.stats import norm
from sklearn.decomposition import FactorAnalysis
from threadpoolctl import threadpool_limits

from src.methods.sensitivity_models import BoundedSA, IntersectionMixin, SolveStatus

EPS = 1e-9
CLIP = 1e-6
FTOL = 1e-8
MAXITER = 200
N_EXTRA_STARTS = 3
FLOOR_STARTS = 6
FEAS_TOL = 1e-6
# fit kwargs that describe the design, never the outcome model's training
_DESIGN_KWARGS = ("GX", "G", "Z", "T", "X_pre")


# ------------------------------------------------------------------ latent model


class LatentTreatmentModel:
    """Factor model on the treatments -> posterior U|x ~ N(mu_u(x), Sigma)."""

    def __init__(self, n_components=32, random_state=0):
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X):
        m = min(self.n_components, X.shape[1] - 1)
        fa = FactorAnalysis(n_components=m, random_state=self.random_state).fit(X)
        W, psi = fa.components_.T, np.maximum(fa.noise_variance_, EPS)
        self.mean_ = fa.mean_
        self.W_, self.psi_ = W, psi

        WtPsi = W.T / psi  # (m, d)
        self.Sigma_ = np.linalg.inv(np.eye(m) + WtPsi @ W)
        self.Sigma_ = (self.Sigma_ + self.Sigma_.T) / 2
        self.proj_ = WtPsi.T @ self.Sigma_  # (d, m)

        w, V = np.linalg.eigh(self.Sigma_)
        w = np.maximum(w, EPS)
        self.half_ = V @ np.diag(np.sqrt(w)) @ V.T  # Sigma^{1/2}
        self.halfinv_ = V @ np.diag(1 / np.sqrt(w)) @ V.T  # Sigma^{-1/2}
        return self

    def transform(self, X):
        return (X - self.mean_) @ self.proj_


# ------------------------------------------------------------------------- links


class Probit:
    """g(a, s) = Phi(a / sqrt(1 + s^2)). Closed form: no quadrature, no bisection."""

    @staticmethod
    def g_inv(mu, s):
        return np.sqrt(1.0 + s**2) * norm.ppf(np.clip(mu, CLIP, 1 - CLIP))

    @staticmethod
    def h(base, s, axis):
        return norm.cdf(base / np.sqrt(1.0 + s**2)).mean(axis=axis)


class Gaussian:
    """Identity link. Linear in g, so every program on it is convex."""

    @staticmethod
    def g_inv(mu, s):
        return mu

    @staticmethod
    def h(base, s, axis):
        return base.mean(axis=axis)


LINKS = {"probit": Probit, "gaussian": Gaussian}


def iv_budget(epsilon, gamma_z_star=0.0, sigma2=None, rho=None, calibrate_sigma=False) -> float:
    """Absolute leaky-IV budget from App. A, Thm. 3.B.

    sigma~ sqrt(gz~) >= eps + sigma sqrt(gz*)  =>  t_abs := sigma~^2 gz~ = (eps + s sqrt(gz*))^2.
    s = sigma = sqrt(sigma~^2 / rho) when calibrated (sigma2 IS sigma~^2: the model is
    fit on the post-DA data), else 1. gz* = 0 (the baseline carries no IV constraint)
    gives eps^2.
    """
    gz = max(float(gamma_z_star), 0.0)
    if gz == 0.0:
        return float(epsilon) ** 2  # s drops out; rho/sigma2 not needed
    if calibrate_sigma:
        if sigma2 is None or rho is None:
            raise ValueError("calibrated iv_budget needs sigma2 and rho when gamma_z_star > 0")
        s = float(np.sqrt(sigma2 / max(rho, 1e-12)))
    else:
        s = 1.0
    return float((float(epsilon) + s * np.sqrt(gz)) ** 2)


# --------------------------------------------------------------- solver plumbing


class _Cached:
    """value_and_grad behind a 1-entry cache: scipy asks for `fun` and `jac`
    separately at the same x, and the reverse-mode pass is not cheap."""

    def __init__(self, value_and_grad):
        self.vg, self._key, self._val, self._grad = value_and_grad, None, None, None

    def _eval(self, v):
        v = np.asarray(v, dtype=float)
        key = v.tobytes()
        if key != self._key:
            val, grad = self.vg(v)
            self._key, self._val, self._grad = key, float(val), np.asarray(grad, float)
        return self._val, self._grad

    def val(self, v):
        return self._eval(v)[0]

    def grad(self, v):
        return self._eval(v)[1]


def _build_terms(model, radius):
    """(objective vg, constraint vg) from JAX, or (None, None) to stay on finite
    differences. Built inside the worker, so one XLA compile serves a whole chunk."""
    if not model.jax_grad:
        return None, None
    try:
        from src.methods import copsens_jax

        return copsens_jax.build_terms(model, radius)
    except Exception as exc:  # jax missing, jit failure: degrade to finite differences
        logger.warning(f"JAX unavailable ({type(exc).__name__}: {exc}); falling back to finite differences")
        return None, None


def _unit_ball():
    """||v||^2 <= 1, with its trivial analytic jacobian."""
    return NonlinearConstraint(lambda v: float(v @ v), -np.inf, 1.0, jac=lambda v: 2.0 * np.asarray(v, dtype=float))


def _slsqp(fun, v0, jac, constraints):
    return minimize(
        fun, v0, method="SLSQP", jac=jac, constraints=constraints, options={"maxiter": MAXITER, "ftol": FTOL}
    )


def socp_bounds(mu_y, w_x, half, radius, soc_terms, backend=None):
    """min/max of mu_y + g'w over ||half g|| <= radius and each ||A g - b|| <= t.

    One DPP parameter carries the objective, so the program is canonicalised once
    per call. Returns `(bounds (n, 2), status (n,))`; a side the chain cannot solve
    is NaN and the query takes the worse of its two statuses.
    """
    m = half.shape[0]
    g = cp.Variable(m)
    constraints = [cp.norm(cp.Constant(half) @ g, 2) <= radius]
    for A, b, t in soc_terms:
        constraints.append(cp.norm(cp.Constant(A) @ g - cp.Constant(b), 2) <= t)
    objective = cp.Parameter(m)
    problems = (
        cp.Problem(cp.Minimize(objective @ g), constraints),
        cp.Problem(cp.Maximize(objective @ g), constraints),
    )
    chain = (tuple(backend),) if backend is not None else ((cp.CLARABEL, {}), (cp.ECOS, {}))

    def solve(problem):
        status = None
        for solver, opts in chain:
            try:
                # warm_start=False: queries must be independent of their order
                problem.solve(solver=solver, warm_start=False, verbose=False, **opts)
                status = problem.status
            except Exception:
                status = None
            if status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                return float(problem.value), SolveStatus.OK
            if status == cp.INFEASIBLE:
                return np.nan, SolveStatus.INFEASIBLE
        if status in (cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE):
            return np.nan, SolveStatus.INFEASIBLE
        return np.nan, SolveStatus.FAILURE

    bounds = np.empty((len(w_x), 2))
    status = np.empty(len(w_x), dtype=int)
    for i, w in enumerate(w_x):
        objective.value = np.asarray(w, dtype=float)
        (lo, s_lo), (hi, s_hi) = (solve(p) for p in problems)
        bounds[i] = (mu_y[i] + lo, mu_y[i] + hi)
        status[i] = max(s_lo, s_hi)
    return bounds, status


# ---------------------------------------------------------------------- estimator


class CopSensPI(BoundedSA):
    """Bounds on E[Y|do(x)] over the covariance ball {g : g' Sigma g <= gamma sigma^2}.

    Invariants: `predict(X, gamma=0)` equals mu_y on both sides; the bounds nest in
    gamma; the fitted state never depends on gamma (`_prepare` is its only reader).
    """

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
        link="probit",
        n_components=32,
        n_anchors=128,
        n_anchors_c=48,
        calibrate_sigma=True,
        outcome_model=None,
        sigma_model=None,
        mu_clip=None,
        jax_grad=True,
        seed=0,
        blas_threads=None,
    ):
        if link not in LINKS:
            raise ValueError(f"unknown link {link!r}; accepted: {sorted(LINKS)}")
        self.link_name = link
        self.n_components = n_components
        self.n_anchors = n_anchors
        self.n_anchors_c = n_anchors_c
        # radius sigma-hat sqrt(gamma) with sigma-hat^2 = E[mu(1 - mu)] on the fit
        # rows (probit) or the residual variance (gaussian); False solves raw budgets
        self.calibrate_sigma = calibrate_sigma
        self.outcome_model = outcome_model
        # sigma_model: a PREFIT net that sets sigma-hat^2 in place of the centre net,
        # read on the observed rows (`_sigma_rows`). None keeps the centre's own. The
        # do-MNIST PI+INV passes the ERM under off and inv, so its radius is PI's
        # (Prop. 3 and Cor. 1 take sigma from the ERM); the post-DA ball under on
        # keeps the DA+ERM's sigma-tilde
        self.sigma_model = sigma_model
        # mu_clip: (lo, hi) the observational law can actually occupy. Outside it the
        # probit bounds CONTRACT to nothing (a fixed latent shift moves Phi by ~0
        # once mu saturates), so a confident ERM gets a vacuously narrow interval.
        self.mu_clip = mu_clip
        self.jax_grad = jax_grad
        self.seed = seed
        # a cap on the BLAS threads of the fit (the factor analysis); None leaves the
        # pool alone. On a loaded many-core node the full pool oversubscribes
        self.blas_threads = blas_threads
        self._ctx = None
        self._kernels = None
        self._floor_cache = {}
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
        if mean_match:
            logger.info(f"{type(self).__name__}: mean_match is inert on the latent ball (no mean-matched slice)")

    # ------------------------------------------------------------------- fit

    def _fit(self, X, y, **kwargs):
        cap = (
            threadpool_limits(int(self.blas_threads), user_api="blas")
            if self.blas_threads is not None
            else contextlib.nullcontext()
        )
        with cap:
            return self._fit_model(X, y, **kwargs)

    def _fit_model(self, X, y, **kwargs):
        self.rng_ = np.random.default_rng(self.seed)
        self.link_ = LINKS[self.link_name]

        self.latent_ = LatentTreatmentModel(self.n_components, self.seed).fit(X)
        self.mu_tr_ = self.latent_.transform(X)

        model = self.outcome_model
        model = model() if callable(model) else model
        if model is None:
            raise ValueError(
                f"{type(self).__name__} needs an outcome model: pass the prefit net that owns "
                "mu_y. Refitting one per PI variant would train on the PI rows and break "
                "the ERM/PI matching."
            )
        if not getattr(model, "prefit_", False):
            model = model.fit(X, y, **{k: v for k, v in kwargs.items() if k not in _DESIGN_KWARGS})
        self.outcome_ = model

        self.mu_ = self._mu(X)
        yf = np.asarray(y, dtype=float).ravel()
        mu_sigma = self.mu_ if self.sigma_model is None else self._sigma_mu(X, **kwargs)
        self.sigma2_ = (
            float(np.var(yf - mu_sigma)) if self.link_name == "gaussian" else float(np.mean(mu_sigma * (1 - mu_sigma)))
        )

        # anchors FIRST, constraint rows SECOND, off the same generator
        idx = self.rng_.choice(len(X), size=min(self.n_anchors, len(X)), replace=False)
        self.anchors_ = self.mu_tr_[idx]  # (J, m)
        self.anchors_c_ = self.anchors_[: self.n_anchors_c]  # coarse set, for the constraint

        self.y_min, self.y_max = float(np.min(yf)), float(np.max(yf))
        self._floor_cache = {}  # keyed on (radius, n_starts); a refit invalidates it
        self._precompute(X, y, **kwargs)
        return self

    def _precompute(self, X, y, **kwargs):
        """Constrained variants build their constraint sample here."""

    def _mu(self, X, model=None):
        """Observational mean of `model` (the centre net when None), clipped to the
        attainable range if one was given."""
        model = self.outcome_ if model is None else model
        mu = np.asarray(model.predict_mean(X)).ravel()
        return np.clip(mu, *self.mu_clip) if self.mu_clip else mu

    def _sigma_rows(self, X, **kwargs):
        """The observed rows sigma-hat^2 is read on: the fit rows here."""
        return X

    def _sigma_mu(self, X, **kwargs):
        """The sigma model's clipped mean on `_sigma_rows`. It must be prefit: the
        radius has to come from the same net as the ball it borrows."""
        model = self.sigma_model
        model = model() if callable(model) else model
        if not getattr(model, "prefit_", False):
            raise ValueError(f"{type(self).__name__}: sigma_model must be a prefit net")
        return self._mu(self._sigma_rows(X, **kwargs), model)

    # -------------------------------------------------------------- geometry

    @property
    def scale(self) -> float:
        """sigma-hat under `calibrate_sigma`, else 1: the factor the ball radius
        carries beside sqrt(gamma)."""
        return float(np.sqrt(self.sigma2_)) if self.calibrate_sigma else 1.0

    def _radius(self, gamma) -> float:
        """Ball radius in whitened coordinates, g = radius Sigma^{-1/2} v: the mixin's
        sqrt(gamma sigma-hat^2) written as one square root so the last ulp agrees,
        with gamma first put through `budget` (recalibration, rho)."""
        c = self.sigma2_ if self.calibrate_sigma else 1.0
        return float(np.sqrt(max(self.budget(gamma) * c, 0.0)))

    def _h(self, mu_q, Gam, s, mu_y, anchors=None):
        """E[Y|do(x); g] for every (query, candidate) -> (n_q, n_cand)."""
        anchors = self.anchors_ if anchors is None else anchors
        a = self.link_.g_inv(mu_y, s)  # (n_q,)
        base = a[:, None, None] + (anchors @ Gam.T)[None, :, :] - (mu_q @ Gam.T)[:, None, :]
        return self.link_.h(base, s, axis=1)

    def _h_v(self, v, mu_q_i, mu_y_i, radius):
        """Scalar h_g(x_i); the NLP hot path when JAX is unavailable."""
        v = np.asarray(v, dtype=float)
        g = radius * (v @ self.latent_.halfinv_)
        s = radius * float(np.sqrt(v @ v))
        a = float(np.ravel(self.link_.g_inv(np.atleast_1d(mu_y_i), s))[0])
        d = a + self.anchors_ @ g - float(mu_q_i @ g)  # (J,)
        return float(self.link_.h(d, s, axis=0))

    def _con_v(self, v, radius):
        """Scalar constraint value at the whitened point v on the ball of `radius`."""
        v = np.asarray(v, dtype=float)
        c = self._constraint(np.atleast_2d(radius * (v @ self.latent_.halfinv_)), radius * float(np.linalg.norm(v)))
        return np.inf if c is None else float(np.ravel(c)[0])

    def _constraint(self, Gam, s):
        """Constraint value per candidate; None if unconstrained."""
        return None

    def _budget(self):
        """RHS of the extra constraint; None if unconstrained."""
        return None

    def _con_at_zero(self) -> float:
        """Constraint value at gamma = 0, i.e. at the ERM itself."""
        m = self.latent_.Sigma_.shape[0]
        return float(np.ravel(self._constraint(np.zeros((1, m)), 0.0))[0])

    def constraint_floor(self, radius, n_starts=FLOOR_STARTS) -> float:
        """min over the ball of the extra constraint: the lower limit any budget must
        clear. Solved with the SAME backend as the bounds: it GATES their NaN
        return, so a different optimiser here invites disagreement. 0.0 when there
        is no constraint. Cached on (radius, n_starts)."""
        m = self.latent_.Sigma_.shape[0]
        if self._constraint(np.zeros((1, m)), 0.0) is None:
            return 0.0
        key = (float(radius), int(n_starts))
        cached = getattr(self, "_floor_cache", None)
        if cached is None:
            cached = self._floor_cache = {}
        if key in cached:
            return cached[key]

        _, con_vg = _build_terms(self, radius)
        cache = _Cached(con_vg) if con_vg is not None else None
        fun = cache.val if cache is not None else (lambda v: self._con_v(v, radius))
        jac = cache.grad if cache is not None else None

        rng = np.random.default_rng(self.seed + 1)
        best = fun(np.zeros(m))
        for _ in range(n_starts):
            v0 = rng.standard_normal(m)
            v0 /= max(np.linalg.norm(v0), EPS)
            result = _slsqp(fun, v0 * 0.5, jac, [_unit_ball()])
            if result.success and result.x @ result.x <= 1 + FEAS_TOL:
                best = min(best, float(result.fun))
        cached[key] = best
        return best

    def _starts(self, mu_q, n_extra=N_EXTRA_STARTS):
        """Multi-start points: +-the linearisation direction, plus `n_extra` random
        unit directions shared by every query of one predict call.

        NOT the origin. s = radius ||v|| is non-differentiable there (d||v||/dv is
        0/0), so a solve seeded at the origin fails every time; it never passed the
        feasibility check, so dropping it is exactly output-neutral.
        """
        rng = np.random.default_rng(self.seed)
        m = self.latent_.Sigma_.shape[0]
        L = (mu_q - self.anchors_.mean(axis=0)) @ self.latent_.halfinv_
        L = L / np.maximum(np.linalg.norm(L, axis=1, keepdims=True), EPS)
        R = rng.standard_normal((n_extra, m))
        R /= np.maximum(np.linalg.norm(R, axis=1, keepdims=True), EPS)
        return [[0.99 * L[i], -0.99 * L[i]] + [0.9 * r for r in R] for i in range(len(mu_q))]

    def _soc_terms(self):
        """Second-order-cone terms `(A, b, t)` of the extra constraint on the
        gaussian link, each ||A g - b|| <= t; empty when unconstrained."""
        return []

    def _floor_gate(self, radius, budget) -> bool:
        """True when the budget clears the floor; logs either way."""
        floor = self.constraint_floor(radius)
        logger.info(f"{type(self).__name__}: constraint floor {floor:.4g} vs budget {budget:.4g}")
        if floor > budget:
            logger.error(
                f"{type(self).__name__}: infeasible, budget {budget:.4g} is below the "
                f"attainable floor {floor:.4g}. All queries INFEASIBLE."
            )
            return False
        return True

    # --------------------------------------------------------- BoundedSA hooks

    def _raw_bounds(self, X, gamma):
        """The gaussian link is linear in g: unconstrained, min/max over the ellipsoid
        is the dual norm (closed form); constrained, one SOCP per query. Probit
        goes through the chunked multi-start NLP."""
        if self.link_name != "gaussian":
            return super()._raw_bounds(X, gamma)

        radius = self._radius(gamma)
        mu_q = self.latent_.transform(X)
        mu_y = self._mu(X)
        budget = self._budget()
        self.query_diagnostics = None
        if budget is None:
            margin = radius * np.linalg.norm((mu_q - self.anchors_.mean(axis=0)) @ self.latent_.halfinv_, axis=1)
            self.query_status = np.full(len(X), SolveStatus.OK, dtype=int)
            return np.column_stack([mu_y - margin, mu_y + margin])
        if not self._floor_gate(radius, budget):
            self.query_status = np.full(len(X), SolveStatus.INFEASIBLE, dtype=int)
            return np.full((len(X), 2), np.nan)
        w = self.anchors_.mean(axis=0) - mu_q  # h = mu_y + g' w
        bounds, self.query_status = socp_bounds(mu_y, w, self.latent_.half_, radius, self._soc_terms(), self.backend)
        failed = self.query_status == SolveStatus.FAILURE
        if failed.any():
            logger.warning(f"{type(self).__name__}: SOCP failed at {failed.sum()}/{len(failed)} queries")
        return bounds

    def _prepare(self, X, gamma):
        radius = self._radius(gamma)
        budget = self._budget()
        self.query_diagnostics = None
        if budget is not None and not self._floor_gate(radius, budget):
            return None

        self._ctx = dict(radius=radius, has_con=budget is not None, budget=(np.inf if budget is None else budget))
        self._kernels = None  # the radius moved, so the JIT must be rebuilt

        mu_q = self.latent_.transform(X)
        mu_y = self._mu(X)  # precomputed: workers never call the net
        return list(zip(mu_q, mu_y, self._starts(mu_q), strict=False))

    def _worker_view(self):
        """Picklable snapshot. Drops exactly what the per-query solve never touches
        and what would not survive a process boundary: `outcome_` (a torch net,
        often on the GPU), `outcome_model` (often a lambda), `mu_tr_` (n_pi x m,
        already summarised by `anchors_`) and the jitted kernels. Shallow, so
        `self` keeps all of them."""
        view = copy.copy(self)
        view.outcome_ = view.outcome_model = view.mu_tr_ = None
        view._kernels = None
        return view

    def _begin_chunk(self):
        """Build the JAX kernels ONCE per chunk. Per query would mean one XLA
        compile per query, which costs more than the solve it replaces."""
        if self._kernels is not None:
            return
        radius, has_con = self._ctx["radius"], self._ctx["has_con"]
        objective, con_vg = _build_terms(self, radius)

        cache = _Cached(con_vg) if con_vg is not None else None
        con_val = cache.val if cache is not None else (lambda v: self._con_v(v, radius))

        constraints = [_unit_ball()]
        if has_con:
            constraints.append(
                NonlinearConstraint(
                    con_val, -np.inf, self._ctx["budget"], **({"jac": cache.grad} if cache is not None else {})
                )
            )
        self._kernels = (objective, con_val, constraints)

    def _solve_single(self, payload):
        """(lower, upper, status, failed_fraction) for one query, by multi-start
        SLSQP on both signs. The fraction counts the starts (over both sides) that
        SLSQP rejected or that landed outside the feasible set.

        Do NOT warm-start from a neighbouring query's optimum: the problem is
        non-convex (that is why there is a multi-start), so seeding from a nearby
        solution biases toward a local optimum and yields NARROWER, i.e. silently
        invalid, bounds.
        """
        mu_q_i, mu_y_i, starts = payload
        objective, con_val, constraints = self._kernels
        radius, has_con, budget = (self._ctx["radius"], self._ctx["has_con"], self._ctx["budget"])

        cache = _Cached(lambda v: objective(v, mu_q_i, mu_y_i)) if objective is not None else None

        def h_at(v):
            return cache.val(v) if cache is not None else self._h_v(v, mu_q_i, mu_y_i, radius)

        best = [np.inf, -np.inf]
        got = [False, False]
        n_solves = n_failed = 0
        for sign, side in ((+1.0, 0), (-1.0, 1)):
            fun = lambda v, s=sign: s * h_at(v)  # noqa: E731
            jac = (lambda v, s=sign: s * cache.grad(v)) if cache is not None else None
            for v0 in starts:
                n_solves += 1
                try:
                    result = _slsqp(fun, v0, jac, constraints)
                except Exception:  # noqa: S112 - a failed start is routine in a multi-start; the survivors cover
                    n_failed += 1
                    continue
                feasible = (result.x @ result.x <= 1.0 + FEAS_TOL) and (
                    not has_con or con_val(result.x) <= budget * (1 + FEAS_TOL)
                )
                if not (bool(result.success) and feasible):
                    n_failed += 1
                    continue
                value = h_at(result.x)
                best[side] = min(best[side], value) if side == 0 else max(best[side], value)
                got[side] = True

        failed_fraction = n_failed / max(n_solves, 1)
        if not (got[0] and got[1]):
            return np.nan, np.nan, SolveStatus.FAILURE, failed_fraction
        return best[0], best[1], SolveStatus.OK, failed_fraction


# --------------------------------------------------------- constrained variants


class InvarianceConstrainedCopSens(CopSensPI):
    """CopSens + explicit invariance-error constraint: E|h(X) - h(GX)|^2 <= eps^2.

    `epsilon` is read at predict time (`BoundedSA._predict(epsilon=...)`), so the
    budget follows a swept epsilon and the program re-solves.
    """

    solves_on_epsilon = True

    def __init__(self, gamma=None, epsilon=None, n_constraint=192, **kwargs):
        if epsilon is None:
            raise ValueError("epsilon required")
        self.n_constraint = n_constraint
        super().__init__(gamma=gamma, epsilon=epsilon, **kwargs)

    def _precompute(self, X, y, GX=None, **kwargs):
        GX = X if GX is None else np.asarray(GX).reshape(len(X), -1)
        idx = self.rng_.choice(len(X), size=min(self.n_constraint, len(X)), replace=False)
        self.cX_ = self.latent_.transform(X[idx])
        self.cGX_ = self.latent_.transform(GX[idx])
        self.cmuX_ = self._mu(X[idx])
        self.cmuGX_ = self._mu(GX[idx])

    def _constraint(self, Gam, s):
        d = self._h(self.cX_, Gam, s, self.cmuX_, self.anchors_c_) - self._h(
            self.cGX_, Gam, s, self.cmuGX_, self.anchors_c_
        )
        return (d**2).mean(axis=0)

    def _budget(self):
        return float(self.epsilon) ** 2

    def _soc_terms(self):
        n = len(self.cX_)
        D = self.cX_ - self.cGX_  # (n, m)
        b = self.cmuX_ - self.cmuGX_  # (n,)
        return [(-D, -b, np.sqrt(n) * float(self.epsilon))]


class RecentredInvCopSens(InvarianceConstrainedCopSens):
    """PI+INV fitted against the POST-DA measure: the ball on GX around the GX
    net, the invariance pairs (GX, X).

    E_inv is symmetric in (X, GX), so swapping them leaves the CONSTRAINT
    untouched; only the ambient ball moves, from H_pi to H_pi~, whose centre may
    already sit near the invariant slice when the X-centred one cannot reach it.
    """

    def _fit(self, X, y, GX=None, **kwargs):
        if GX is None:
            raise ValueError("GX (augmented treatment) required")
        GX = np.asarray(GX).reshape(len(GX), -1)
        return super()._fit(GX, y, GX=X, **kwargs)

    def _sigma_rows(self, X, GX=None, **kwargs):
        """The fit rows are the augmented ones here; the observed X rides as `GX`."""
        return X if GX is None else np.asarray(GX).reshape(len(X), -1)


class IVConstrainedCopSens(CopSensPI):
    """CopSens + leaky-IV constraint: Var(E[y - h(X) | Z]) <= t_abs, with t_abs the
    absolute budget of App. A, Thm. 3.B, (eps_iv + s sqrt(gamma_z*))^2.

    The instrument is `T` (the DA's own parameters, what the runner hands a DA+
    method) when given, else `Z`; neither is refused. At gamma_z* = 0, no baseline
    IV constraint, t_abs is eps_iv^2 and the calibration factor cancels; the budget
    is still computed as `delta = t_abs / sigma-hat^2; budget = delta sigma-hat^2`,
    the mixin's two steps, so the last ulp agrees.
    """

    solves_on_epsilon = True

    def __init__(self, gamma=None, epsilon_iv=None, gamma_z_star=0.0, n_constraint=384, **kwargs):
        if epsilon_iv is None:
            raise ValueError("epsilon_iv required")
        self.epsilon_iv = epsilon_iv
        self.gamma_z_star = gamma_z_star
        self.n_constraint = n_constraint
        super().__init__(gamma=gamma, **kwargs)

    @staticmethod
    def _present(Z):
        return Z is not None and np.asarray(Z).size > 0

    def _precompute(self, X, y, Z=None, T=None, X_pre=None, **kwargs):
        instrument = T if self._present(T) else (Z if self._present(Z) else None)
        if instrument is None:
            raise ValueError(
                f"{type(self).__name__} needs an instrument (T, the DA parameters, or Z). "
                "Instrumenting on X itself is a live constraint, not a no-op; for a "
                "baseline branch use a plain CopSensPI."
            )
        idx = self.rng_.choice(len(X), size=min(self.n_constraint, len(X)), replace=False)
        self.cX_ = self.latent_.transform(X[idx])
        self.cmuX_ = self._mu(X[idx])
        self.cy_ = np.asarray(y, dtype=float).ravel()[idx]
        Zc = np.column_stack([np.ones(len(idx)), np.asarray(instrument)[idx].reshape(len(idx), -1)])
        self.Qz_, _ = np.linalg.qr(Zc)  # (n, k) orthonormal
        c = self.sigma2_ if self.calibrate_sigma else 1.0
        t_abs = self._t_abs()
        logger.info(
            f"{type(self).__name__}: gamma_z={t_abs / max(c, EPS):.4g} t_abs={t_abs:.4g} "
            f"(eps_iv={float(self.epsilon_iv):.4g} gz*={float(self.gamma_z_star):.4g} "
            f"rho={self.rho:.4g} sigma_tilde^2={c:.4g})"
        )

    def _t_abs(self) -> float:
        return iv_budget(self.epsilon_iv, self.gamma_z_star, self.sigma2_, self.rho, self.calibrate_sigma)

    def _constraint(self, Gam, s):
        residual = self.cy_[:, None] - self._h(self.cX_, Gam, s, self.cmuX_, self.anchors_c_)
        fitted = self.Qz_ @ (self.Qz_.T @ residual)
        return (fitted**2).mean(axis=0) - fitted.mean(axis=0) ** 2

    def _budget(self):
        c = self.sigma2_ if self.calibrate_sigma else 1.0
        delta = float(self._t_abs() / max(c, EPS))
        return delta * c

    def _soc_terms(self):
        n = len(self.cX_)
        w = self.anchors_.mean(axis=0) - self.cX_  # h = mu_y + g' w
        r0 = self.cy_ - self.cmuX_  # residual at g = 0
        C = np.eye(n) - np.ones((n, n)) / n  # centring
        A = self.Qz_ @ (self.Qz_.T @ (C @ w))
        b = -self.Qz_ @ (self.Qz_.T @ (C @ r0))
        # radius^2 = n * _budget(), which carries the calibration factor
        return [(A, b, np.sqrt(n * self._budget()))]

    def _predict(self, X, epsilon_iv=None, **kwargs):
        """`epsilon_iv` is the T budget at predict time; it reaches the IV budget and
        nothing else."""
        if epsilon_iv is not None:
            self.epsilon_iv = float(epsilon_iv)
        return super()._predict(X, **kwargs)


# ------------------------------------------------------------------ intersections


class IntersectedCopSens(IntersectionMixin, CopSensPI):
    """Baseline PI intersected with DA+PI (Cor. 1). Padding hits the DA branch only.

    `_fit` is overridden entirely, so none of CopSensPI's own fit state (latent_,
    anchors_, outcome_) is ever built here: each branch owns its own, exactly as
    `IntersectedPartialR2` relates to `PartialR2`.

    Cor. 1 needs h_*(x) to lie in BOTH intervals, a membership fact; it does not
    need the two balls to share a parameterisation. They do not here: each branch
    fits its own factor model, so different Sigma, different sigma-hat^2, different
    net. Validity survives that; Prop. 4 sharpness does not, so this is the
    conservative inclusion of Remark 1, not a sharpness claim.
    """

    def __init__(self, outcome_models=None, **kwargs):
        if outcome_models is None or not {"X", "GX"} <= set(outcome_models):
            raise ValueError(
                f'{type(self).__name__} needs the prefit outcome nets {{"X": .., "GX": ..}}: '
                "the baseline branch fits on X with the X net, the DA branch on GX "
                f"with the GX net. Got {sorted(outcome_models or {})}."
            )
        self.outcome_models = outcome_models
        # outcome_model=None is safe: our _fit never reaches CopSensPI._fit's check
        super().__init__(outcome_model=None, **kwargs)
        self.baseline = self.augmented = None

    def _branch_kwargs(self, key, pad):
        """Everything a sibling CopSensPI needs. One place, so a new knob on
        CopSensPI cannot silently stop reaching the branches. rho = 1 at
        construction: the DA branch's factor is only known once both branches are
        fitted (`_fit_branches` sets it)."""
        return dict(
            gamma=self.gamma,
            epsilon=self.epsilon,
            pad=pad,
            pad_epsilon=self.pad_epsilon,
            recalibrate=self.recalibrate,
            clipy=self.clipy,
            n_jobs=self.n_jobs,
            mean_match=self.mean_match,
            rho=1.0,
            link=self.link_name,
            n_components=self.n_components,
            n_anchors=self.n_anchors,
            n_anchors_c=self.n_anchors_c,
            calibrate_sigma=self.calibrate_sigma,
            sigma_model=None,  # each branch keeps its own net's sigma-hat
            mu_clip=self.mu_clip,
            jax_grad=self.jax_grad,
            seed=self.seed,
            blas_threads=self.blas_threads,
            outcome_model=self.outcome_models[key],
        )

    def _branch(self, key, pad, cls=CopSensPI, **extra):
        branch = cls(**extra, **self._branch_kwargs(key, pad))
        if pad:
            branch.pad_tolerance = self.pad_tolerance  # the sweeps' CI setting reaches the DA branch
        return branch

    def _fit_branches(self, X, y, GX, G, Z=None):
        self.baseline = self._branch("X", pad=False).fit(X, y)
        self.augmented = self._branch("GX", pad=self.pad).fit(GX, y)
        # rho known once both noise levels are; the DA branch solves at gamma~
        self.augmented.rho = self.rho

    def _fit(self, X, y, GX=None, G=None, Z=None, **kwargs):
        if GX is None:
            raise ValueError("GX (augmented treatment) required")
        GX = np.asarray(GX).reshape(len(GX), -1)
        self._fit_branches(X, y, GX, G, Z)

        self.sigma2_ = self.baseline.sigma2_
        yf = np.asarray(y, dtype=float).ravel()
        self.y_min, self.y_max = float(np.min(yf)), float(np.max(yf))
        logger.info(
            f"{type(self).__name__}: sigma2 ratio (GX/X) {self.rho:.4f}, read as the DA "
            "branch's rho; NOT the paper's rho (see `rho`)"
        )
        return self

    # the base setter stays (a bare @property would drop it and `__init__`'s
    # assignment would raise); the getter reads the branches once they exist
    @BoundedSA.rho.getter
    def rho(self):
        """sigma-hat^2(GX) / sigma-hat^2(X) off the two fitted branches, the
        constructor's value until then. NOT the paper's rho: CopSens's sigma-hat^2 is
        E[mu(1 - mu)] on the latent index scale, not a min-MSE, so it does not
        inherit the DPI bound rho >= 1 (the setter on the DA branch clamps it)."""
        baseline, augmented = getattr(self, "baseline", None), getattr(self, "augmented", None)
        if baseline is None or augmented is None:
            return self._rho
        return augmented.sigma2_ / baseline.sigma2_


class IntersectedIVCopSens(IntersectedCopSens):
    """Baseline PI (no instrument) intersected with DA+PI+IV on the DA parameters."""

    def __init__(self, epsilon_iv=None, gamma_z_star=0.0, n_constraint=384, **kwargs):
        if epsilon_iv is None:
            raise ValueError("epsilon_iv required")
        self.epsilon_iv = epsilon_iv
        self.gamma_z_star = gamma_z_star
        self.n_constraint = n_constraint
        super().__init__(**kwargs)

    def _fit_branches(self, X, y, GX, G, Z=None):
        # the baseline carries no instrument, so it is a PLAIN CopSensPI: handing
        # IVConstrainedCopSens no instrument raises rather than dropping the constraint
        self.baseline = self._branch("X", pad=False).fit(X, y)
        self.augmented = self._branch(
            "GX",
            pad=self.pad,
            cls=IVConstrainedCopSens,
            epsilon_iv=self.epsilon_iv,
            gamma_z_star=self.gamma_z_star,
            n_constraint=self.n_constraint,
        ).fit(GX, y, T=G, Z=Z)
        self.augmented.rho = self.rho
