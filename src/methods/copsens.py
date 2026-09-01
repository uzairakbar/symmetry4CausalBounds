"""
CopSens partial identification (Zheng / D'Amour / Franks, JMLR 26(36) 2025).

A latent-factor sensitivity model, and the alternative this repo's `PartialR2`
is measured against. Where PartialR2 puts a ball around h_erm in FUNCTION space
and maximises a linear objective (an SOCP), CopSens puts a ball around the origin
in a 32-d LATENT sensitivity space and maximises a non-convex one (SLSQP,
multi-start). The two gammas are different parameterisations and are NOT
comparable -- select each against population coverage before comparing them.

    U | x ~ N(mu_u(x), Sigma)                       factor model on the treatments
    h_g(x) = E_{x' ~ anchors}[ F( a + (mu_u(x') - mu_u(x)).g ) ]
    a      = g_inv(mu_y(x), s),  s = radius ||v||,  g = radius Sigma^{-1/2} v
    bounds = [min, max] h_g(x)  over  ||v|| <= 1,  i.e. g' Sigma g <= gamma sigma^2

`a` de-convolves the posterior smearing, so h_0(x) == mu_y(x) exactly.
"""

import copy

import numpy as np
from loguru import logger
from scipy.optimize import NonlinearConstraint, minimize
from scipy.stats import norm
from sklearn.decomposition import FactorAnalysis

from src.methods.sensitivity_models import BoundedSA, IntersectionMixin, SolveStatus

EPS = 1e-9
CLIP = 1e-6
FTOL = 1e-8
MAXITER = 200
N_EXTRA_STARTS = 3
FLOOR_STARTS = 6


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
    """g(a, s) = Phi(a / sqrt(1+s^2)). Closed form -- no quadrature, no bisection."""

    @staticmethod
    def g_inv(mu, s):
        return np.sqrt(1.0 + s**2) * norm.ppf(np.clip(mu, CLIP, 1 - CLIP))

    @staticmethod
    def h(base, s, axis):
        return norm.cdf(base / np.sqrt(1.0 + s**2)).mean(axis=axis)


class Gaussian:
    """Identity link. The right geometry when the confounding is additive in
    probability, as do-MNIST's is; kept as a backend option (see `link`)."""

    @staticmethod
    def g_inv(mu, s):
        return mu

    @staticmethod
    def h(base, s, axis):
        return base.mean(axis=axis)


LINKS = {"probit": Probit, "gaussian": Gaussian}


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
    except Exception as exc:  # jax missing, jit failure -> degrade, don't die
        logger.warning(f"JAX unavailable ({type(exc).__name__}: {exc}); falling back to finite differences")
        return None, None


def _unit_ball():
    """||v||^2 <= 1, with its trivial analytic jacobian."""
    return NonlinearConstraint(lambda v: float(v @ v), -np.inf, 1.0, jac=lambda v: 2.0 * np.asarray(v, dtype=float))


# ---------------------------------------------------------------------- estimator


class CopSensPI(BoundedSA):
    """Bounds on E[Y|do(x)] over the covariance ball {g : g' Sigma g <= gamma sigma^2}."""

    def __init__(
        self,
        gamma=None,
        epsilon=0.0,
        pad=False,
        calibrate=False,
        clipy=True,
        n_jobs=1,
        n_components=32,
        link="probit",
        n_anchors=256,
        n_anchors_c=48,
        outcome_model=None,
        mu_clip=None,
        jax_grad=True,
        seed=0,
    ):
        if link not in LINKS:
            raise ValueError(f"unknown link {link!r}; valid: {sorted(LINKS)}")
        self.n_components = n_components
        self.link_name = link
        self.n_anchors = n_anchors
        self.n_anchors_c = n_anchors_c
        self.outcome_model = outcome_model
        # mu_clip: (lo, hi) the observational law can actually occupy. Outside it the
        # probit bounds CONTRACT to nothing -- a fixed latent shift moves Phi by ~0
        # once mu saturates -- so a confident ERM gets a vacuously narrow interval.
        self.mu_clip = mu_clip
        self.jax_grad = jax_grad
        self.seed = seed
        self._ctx = None
        self._kernels = None
        super().__init__(gamma=gamma, epsilon=epsilon, pad=pad, calibrate=calibrate, clipy=clipy, n_jobs=n_jobs)

    # ------------------------------------------------------------------- fit

    def _fit(self, X, y, **kwargs):
        self.rng_ = np.random.default_rng(self.seed)
        self.link_ = LINKS[self.link_name]

        self.latent_ = LatentTreatmentModel(self.n_components, self.seed).fit(X)
        self.mu_tr_ = self.latent_.transform(X)

        model = self.outcome_model
        model = model() if callable(model) else model
        if model is None:
            raise ValueError(
                "CopSensPI needs an outcome model: pass the prefit net that owns "
                "mu_y. Refitting one per PI variant would train on the n_pi subset "
                "and break the ERM/PI matching."
            )
        if not getattr(model, "prefit_", False):
            model = model.fit(X, y, **kwargs)
        self.outcome_ = model

        mu_hat = self._mu(X)
        self.sigma2_ = (
            float(np.var(np.asarray(y).ravel() - mu_hat))
            if self.link_name == "gaussian"
            else float(np.mean(mu_hat * (1 - mu_hat)))
        )

        idx = self.rng_.choice(len(X), size=min(self.n_anchors, len(X)), replace=False)
        self.anchors_ = self.mu_tr_[idx]  # (J, m)
        self.anchors_c_ = self.anchors_[: self.n_anchors_c]  # coarse set, for feasibility

        # clipy clips to what the DATA says, not to [0, 1]
        self.y_min, self.y_max = float(np.min(y)), float(np.max(y))

        self._floor_cache = {}  # keyed on radius; a refit invalidates it
        self._precompute(X, y, **kwargs)
        return self

    def _precompute(self, X, y, **kwargs):
        """Constrained variants build their constraint sample here."""

    def _mu(self, X):
        """Observational mean, clipped to the attainable range if one was given."""
        mu = np.asarray(self.outcome_.predict_mean(X)).ravel()
        return np.clip(mu, *self.mu_clip) if self.mu_clip else mu

    # -------------------------------------------------------------- geometry

    @property
    def scale(self):
        """s: sigma-hat if calibrated (paper), else 1 (raw budgets). Same role as
        PartialR2.scale, but sigma-hat^2 = E[mu(1-mu)] on the latent index scale."""
        return float(np.sqrt(self.sigma2_)) if self.calibrate else 1.0

    def _radius(self, gamma):
        """Ball radius in whitened coordinates: g = radius * Sigma^{-1/2} v."""
        return float(np.sqrt(max(gamma, 0.0))) * self.scale

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
        c = self._constraint(
            np.atleast_2d(radius * (np.asarray(v, float) @ self.latent_.halfinv_)), radius * float(np.linalg.norm(v))
        )
        return np.inf if c is None else float(np.ravel(c)[0])

    def _constraint(self, Gam, s):
        """Constraint value per candidate; None if unconstrained."""
        return None

    def _budget(self):
        """RHS of the extra constraint; None if unconstrained."""
        return None

    def constraint_floor(self, radius, n_starts=FLOOR_STARTS):
        """min over the ball of the extra constraint: the lower limit any budget must
        clear. Solved with the SAME backend as the bounds -- it GATES their NaN
        return, so a different optimiser here invites disagreement.

        Cached on radius: it depends on the ball and the fitted latent model, never
        on the budget, but `_prepare` asks for it on EVERY predict. Budget sweeps
        (scripts/select_domnist_budgets.py) would otherwise pay 6 SLSQP solves a probe.
        """
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

        m = self.latent_.Sigma_.shape[0]
        rng = np.random.default_rng(self.seed + 1)
        best = fun(np.zeros(m))
        for _ in range(n_starts):
            v0 = rng.standard_normal(m)
            v0 /= max(np.linalg.norm(v0), EPS)
            result = minimize(
                fun,
                v0 * 0.5,
                method="SLSQP",
                jac=jac,
                constraints=[_unit_ball()],
                options={"maxiter": MAXITER, "ftol": FTOL},
            )
            if result.success and result.x @ result.x <= 1 + 1e-6:
                best = min(best, float(result.fun))
        cached[key] = best
        return best

    def _starts(self, mu_q, n_extra=N_EXTRA_STARTS):
        """Multi-start points: +-the linearisation direction, plus random.

        NOT the origin. s = radius*||v|| is non-differentiable there (d||v||/dv is
        0/0), so a solve seeded at the origin fails 100% of the time -- that one
        start WAS the entire 16.7% failure rate SOURCE reported. Dropping it is
        exactly output-neutral: it never passed the feasibility check.
        """
        rng = np.random.default_rng(self.seed)
        m = self.latent_.Sigma_.shape[0]
        L = (mu_q - self.anchors_.mean(axis=0)) @ self.latent_.halfinv_
        L = L / np.maximum(np.linalg.norm(L, axis=1, keepdims=True), EPS)
        R = rng.standard_normal((n_extra, m))
        R /= np.maximum(np.linalg.norm(R, axis=1, keepdims=True), EPS)
        return [[0.99 * L[i], -0.99 * L[i]] + [0.9 * r for r in R] for i in range(len(mu_q))]

    # --------------------------------------------------------- BoundedSA hooks

    def _raw_bounds(self, X, gamma):
        """Unconstrained gaussian is linear in g, so min/max over the ellipsoid is
        the dual norm -- exact, and no solver of any kind."""
        if self.link_name == "gaussian" and self._budget() is None:
            radius = self._radius(gamma)
            mu_q = self.latent_.transform(X)
            margin = radius * np.linalg.norm((mu_q - self.anchors_.mean(axis=0)) @ self.latent_.halfinv_, axis=1)
            mu_y = self._mu(X)
            self.query_status = np.full(len(X), SolveStatus.OK, dtype=int)
            return np.column_stack([mu_y - margin, mu_y + margin])

        return super()._raw_bounds(X, gamma)

    def _prepare(self, X, gamma):
        radius = self._radius(gamma)
        budget = self._budget()

        if budget is not None:
            floor = self.constraint_floor(radius)
            logger.info(f"{type(self).__name__}: constraint floor {floor:.4g} vs budget {budget:.4g}")
            if floor > budget:
                logger.error(
                    f"{type(self).__name__}: infeasible -- budget {budget:.4g} is "
                    f"below the attainable floor {floor:.4g}. All queries INFEASIBLE."
                )
                return None

        self._ctx = dict(radius=radius, has_con=budget is not None, budget=(np.inf if budget is None else budget))
        self._kernels = None  # radius moved => the JIT must be rebuilt

        mu_q = self.latent_.transform(X)
        mu_y = self._mu(X)  # precomputed: workers never call the net
        return list(zip(mu_q, mu_y, self._starts(mu_q)))

    def _worker_view(self):
        """Picklable snapshot. Drops exactly what the per-query solve never touches
        and what would not survive a process boundary: `outcome_` (a torch net,
        often on the GPU), `outcome_model` (often a lambda), and `mu_tr_` (n_pi x m,
        already summarised by `anchors_`). Shallow, so `self` keeps all three."""
        view = copy.copy(self)
        view.outcome_ = view.outcome_model = view.mu_tr_ = None
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
        """(lower, upper, status) for one query, by multi-start SLSQP on both signs.

        Do NOT warm-start from a neighbouring query's optimum: the problem is
        non-convex (that is why there is a multi-start), so seeding from a nearby
        solution biases toward a local optimum and yields NARROWER -- i.e. silently
        invalid -- bounds.
        """
        mu_q_i, mu_y_i, starts = payload
        objective, con_val, constraints = self._kernels
        radius, has_con, budget = (self._ctx["radius"], self._ctx["has_con"], self._ctx["budget"])

        cache = _Cached(lambda v: objective(v, mu_q_i, mu_y_i)) if objective is not None else None

        def h_at(v):
            return cache.val(v) if cache is not None else self._h_v(v, mu_q_i, mu_y_i, radius)

        best = [np.inf, -np.inf]
        got = [False, False]
        for sign, side in ((+1.0, 0), (-1.0, 1)):
            fun = lambda v, s=sign: s * h_at(v)  # noqa: E731
            jac = (lambda v, s=sign: s * cache.grad(v)) if cache is not None else None
            for v0 in starts:
                try:
                    result = minimize(
                        fun,
                        v0,
                        method="SLSQP",
                        jac=jac,
                        constraints=constraints,
                        options={"maxiter": MAXITER, "ftol": FTOL},
                    )
                except Exception:
                    continue
                feasible = (result.x @ result.x <= 1.0 + 1e-6) and (
                    not has_con or con_val(result.x) <= budget * (1 + 1e-6)
                )
                if not (bool(result.success) and feasible):
                    continue
                value = h_at(result.x)
                best[side] = min(best[side], value) if side == 0 else max(best[side], value)
                got[side] = True

        if not (got[0] and got[1]):
            return np.nan, np.nan, SolveStatus.FAILURE
        return best[0], best[1], SolveStatus.OK


# --------------------------------------------------------- constrained variants


class InvarianceConstrainedCopSens(CopSensPI):
    """CopSens + explicit invariance-error constraint: E|h(X) - h(GX)|^2 <= eps^2."""

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
        return self.epsilon**2


class RecentredInvCopSens(InvarianceConstrainedCopSens):
    """PI_INV, fitted against the POST-DA measure.

    From an X-centred ball, PI_INV is empty at any reasonable eps: a fixed net plus
    a 32-d latent shift cannot reach the invariant slice from a centre whose own
    invariance defect IS the confounding (measured floor 0.0357 vs eps^2 0.01).
    E_inv is symmetric in (X, GX), so swapping them leaves the CONSTRAINT untouched;
    only the ambient ball moves, H_pi -> H_pi~, whose centre already sits near that
    slice (floor 0.0078 < 0.01).
    """

    def _fit(self, X, y, GX=None, **kwargs):
        if GX is None:
            raise ValueError("GX (augmented treatment) required")
        GX = np.asarray(GX).reshape(len(GX), -1)
        return super()._fit(GX, y, GX=X, **kwargs)


class IVConstrainedCopSens(CopSensPI):
    """CopSens + leaky-IV constraint: Var(E[y - h(X) | Z]) <= epsilon_iv^2.

    At gamma_z* = 0 -- no baseline IV constraint, which is how every experiment here
    runs -- Thm. 3.B's absolute budget (eps + s sqrt(gamma_z*))^2 collapses to
    eps_iv^2, and the calibrate factor cancels out of it entirely.
    """

    def __init__(self, gamma=None, epsilon_iv=None, n_constraint=384, **kwargs):
        if epsilon_iv is None:
            raise ValueError("epsilon_iv required")
        self.epsilon_iv = epsilon_iv
        self.n_constraint = n_constraint
        super().__init__(gamma=gamma, **kwargs)

    def _precompute(self, X, y, Z=None, **kwargs):
        if Z is None:
            # NOT the partial-r2 semantics. IVPartialR2 treats Z=None as "no
            # instrument" and drops the constraint entirely; the old default here
            # was `Z = X`, i.e. instrument on the treatment, which keeps a live and
            # possibly infeasible constraint. Refuse rather than silently differ.
            raise ValueError(
                "IVConstrainedCopSens needs an instrument. Z=None would instrument "
                "on X, which is a live constraint, not the no-op that the "
                "partial-r2 backend gives you. For a baseline branch use a plain "
                "CopSensPI."
            )
        idx = self.rng_.choice(len(X), size=min(self.n_constraint, len(X)), replace=False)
        self.cX_ = self.latent_.transform(X[idx])
        self.cmuX_ = self._mu(X[idx])
        self.cy_ = np.asarray(y).ravel()[idx]
        Zc = np.column_stack([np.ones(len(idx)), np.asarray(Z)[idx].reshape(len(idx), -1)])
        self.Qz_, _ = np.linalg.qr(Zc)  # (n, k) orthonormal

    def _constraint(self, Gam, s):
        residual = self.cy_[:, None] - self._h(self.cX_, Gam, s, self.cmuX_, self.anchors_c_)
        fitted = self.Qz_ @ (self.Qz_.T @ residual)
        return (fitted**2).mean(axis=0) - fitted.mean(axis=0) ** 2

    def _budget(self):
        return self.epsilon_iv**2


# ------------------------------------------------------------------ intersections


class IntersectedCopSens(IntersectionMixin, CopSensPI):
    """
    Baseline PI intersected with DA+PI (Cor. 1). Padding hits the DA branch only.

    `_fit` is overridden entirely, so none of CopSensPI's own fit state (latent_,
    anchors_, outcome_) is ever built here -- each branch owns its own. Exactly how
    IntersectedPartialR2 relates to PartialR2.

    Cor. 1 needs h_*(x) to lie in BOTH intervals, which is a membership fact; it does
    not need the two balls to share a parameterisation. They do not here: each branch
    fits its own FactorAnalysis, so different Sigma, different sigma-hat^2, different
    net. Validity survives that; Prop. 4 sharpness does not, so this is the
    conservative inclusion of Remark 1, not a sharpness claim.

    COST: the branches are re-fits, so a run carrying PI, DA+PI and both
    intersections pays for the same FactorAnalysis three times over (~30 s each at
    n_pi = 60k) and solves every query twice per intersection. That redundancy is
    inherited from IntersectedPartialR2 deliberately -- sharing fitted branches
    across methods would couple their lifetimes to the runner's build order.
    """

    def __init__(self, outcome_models=None, **kwargs):
        if outcome_models is None or not {"X", "GX"} <= set(outcome_models):
            raise ValueError(
                'IntersectedCopSens needs the prefit outcome nets {"X": .., "GX": ..}: '
                "the baseline branch fits on X with the X net, the DA branch on GX "
                f"with the GX net. Got {sorted(outcome_models or {})}."
            )
        self.outcome_models = outcome_models
        # outcome_model=None is safe: our _fit never reaches CopSensPI._fit's check
        super().__init__(outcome_model=None, **kwargs)
        self.baseline = self.augmented = None

    def _branch_kwargs(self, key, pad):
        """Everything a sibling CopSensPI needs. One place, so a new knob on
        CopSensPI cannot silently stop reaching the branches."""
        return dict(
            gamma=self.gamma,
            epsilon=self.epsilon,
            pad=pad,
            calibrate=self.calibrate,
            clipy=self.clipy,
            n_jobs=self.n_jobs,
            n_components=self.n_components,
            link=self.link_name,
            n_anchors=self.n_anchors,
            n_anchors_c=self.n_anchors_c,
            mu_clip=self.mu_clip,
            jax_grad=self.jax_grad,
            seed=self.seed,
            outcome_model=self.outcome_models[key],
        )

    def _fit_branches(self, X, y, GX, G):
        self.baseline = CopSensPI(**self._branch_kwargs("X", pad=False)).fit(X, y)
        self.augmented = CopSensPI(**self._branch_kwargs("GX", pad=self.pad)).fit(GX, y)

    def _fit(self, X, y, GX=None, G=None, **kwargs):
        if GX is None:
            raise ValueError("GX (augmented treatment) required")
        GX = np.asarray(GX).reshape(len(GX), -1)
        self._fit_branches(X, y, GX, G)

        self.sigma2_ = self.baseline.sigma2_
        self.y_min, self.y_max = float(np.min(y)), float(np.max(y))
        logger.info(
            f"{type(self).__name__}: sigma2 ratio (GX/X) "
            f"{self.sigma2_ratio:.4f} -- the DA ball is "
            f"{np.sqrt(self.sigma2_ratio):.4f}x the baseline radius"
        )
        return self

    @property
    def sigma2_ratio(self):
        """sigma-hat^2(GX) / sigma-hat^2(X): the SQUARED ratio of the two ball radii
        under `calibrate`. NOT the paper's rho -- CopSens's sigma-hat^2 is
        E[mu(1-mu)] on the latent index scale, not a min-MSE, so it does not inherit
        the DPI bound rho >= 1."""
        return self.augmented.sigma2_ / self.baseline.sigma2_


class IntersectedIVCopSens(IntersectedCopSens):
    """Baseline PI (null instrument) intersected with DA+PI_IV."""

    def __init__(self, epsilon_iv=None, n_constraint=384, **kwargs):
        if epsilon_iv is None:
            raise ValueError("epsilon_iv required")
        self.epsilon_iv, self.n_constraint = epsilon_iv, n_constraint
        super().__init__(**kwargs)

    def _fit_branches(self, X, y, GX, G):
        # The baseline carries no instrument, so it is a PLAIN CopSensPI, built
        # explicitly. Do NOT reach for IVConstrainedCopSens(...).fit(Z=None) for
        # symmetry with the partial-r2 backend: there Z=None drops the constraint,
        # here it would instrument on X. That path now raises.
        self.baseline = CopSensPI(**self._branch_kwargs("X", pad=False)).fit(X, y)
        self.augmented = IVConstrainedCopSens(
            epsilon_iv=self.epsilon_iv, n_constraint=self.n_constraint, **self._branch_kwargs("GX", pad=self.pad)
        ).fit(GX, y, Z=G)
