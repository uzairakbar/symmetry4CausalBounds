"""
Partial identification over a last-`l`-layer refit of the prefit ERM net (App. D).

The finite-n program (P2), instantiated with H_theta = {head_theta o phi}: phi is
the FROZEN trunk of the prefit net (activations at the unfreeze boundary) and theta
the weights of its last `unfrozen_layers` layers. Per query x:

    min / max  h_theta(x)                     h_theta = link(index_theta(phi(x)))
    s.t.  E_r2(h)  = mean (h(x_i) - mu_hat(x_i))^2        <= sigma_hat^2 gamma
          R_iv(h)  = || Qz' (y - h(X)) ||^2 / n           <= eps_iv^2
          E_inv(h) = mean (h(x_i) - h(G x_i))^2           <= eps^2
          | mean h(x_i) - ybar |                          <= tau            (Lem. 2)

Same function-space ball around the ERM as `PartialR2`, but the class is nonlinear
in theta, so per-query multi-start SLSQP (JAX gradients) replaces the SOCP. The
index (pre-link output) is what the solver optimises: link is monotone, so
min/max h = link(min/max index), and the index never saturates the gradients.

phi is precomputed ONCE (torch, on the device) for the n_pi constraint rows and
the queries; the whole NLP then lives on cached features in JAX on the CPU.
"""

import copy

import numpy as np
from loguru import logger
from scipy.optimize import NonlinearConstraint, minimize
from scipy.special import expit, ndtr
from scipy.stats import norm

from src.methods.sensitivity_models import BoundedSA, IntersectionMixin, SolveStatus

EPS = 1e-9
FTOL = 1e-8
FEAS_TOL = 1e-6
MAXITER = 300
# per-query polish budget for UNCONSTRAINED programs. Profiled at n_pi=6k: one
# 300-iteration SLSQP is ~0.5 s and the 12-solve multi-start ~4.6 s/query, while
# on programs with no (or an inert) extra constraint the backtracked directed
# candidates already carry the bounds (widths move < 1e-3 with the full budget) --
# there each side gets ONE polish solve from its best candidate, capped here.
# Extra-constrained programs are NOT covered by that measurement and keep the
# full multi-start polish (see _solve_single).
POLISH_MAXITER = 80
# Lem. 2's slice E[h(X)] = E[Y] is nonlinear in theta here (the link), and it is
# only known to the sampling error of the level, so it enters as a BAND rather
# than an equality:
#
#     tau = MEAN_BAND_SE * sigma_hat * sqrt(1 + gamma) / sqrt(n)
#
# sqrt(Var(U + xi) / n) is the standard error of the level, and Lem. 2's own proof
# bounds Var(E[U|X]) <= sigma^2 gamma (the (**)-(::) chain of SS.F.9), so
# sigma_hat^2 (1 + gamma) is the sensitivity model's OWN bound on Var(U + xi):
# the band is "MEAN_BAND_SE standard errors" for ANY dataset and budget, not just
# at do-MNIST's small gamma*. h_* misses the empirical slice by the same
# quantity, so at 2 SE it stays band-feasible w.p. ~95% per fit.
#
# An EQUALITY would also break the solver: `_to_feasible` backtracks along the
# SEGMENT to the anchor, and the segment between two points of a curved slice
# leaves it -- every start would collapse onto the anchor and the bounds would
# silently NARROW.
MEAN_BAND_SE = 2.0
# The band is a live constraint at production tau, so a mean-matched PI/DA+PI is
# an extra-constrained program and takes the full multi-start polish (a single
# polish under a live constraint is what silently narrowed PI+INV once). Flip
# only on full-scale evidence: `a27 --polish-compare` prints the width difference.
SINGLE_POLISH_WITH_BAND = False
# The slab anchor is solved against a slightly TIGHTER band than tau. SLSQP meets
# the SQUARED constraint m^2 <= tau^2 to its own tolerance, and that slop reappears
# in |m| divided by 2 tau -- for a thin slab it lands the anchor a hair OUTSIDE the
# band, `_to_feasible` then backtracks every start onto it, rejects them all, and
# the whole query reports FAILURE (seen at `a27 --micro --band-se 0.146`, where
# |m| 0.003288 vs tau 0.003287 cost 1 query in 4 on five of six methods). Anchoring
# in the interior costs a 1% narrower slab in the floor it reports and nothing at
# production tau, where theta_c is inside the band and this path never runs.
BAND_ANCHOR_MARGIN = 0.01
N_EXTRA_STARTS = 3
JITTER = 0.05
FLOOR_STARTS = 6
# above this head size SLSQP's dense O(d^2) workspace is unaffordable (l=2 is ~74k
# params -> ~130 GB); switch to the augmented-Lagrangian path
SLSQP_DIM_MAX = 4096
AL_ROUNDS = 10
AL_INNER_MAXITER = 300
# sigmoid(t) ~ Phi(t / 1.702), so a trained sigmoid head lands on the probit index
# scale by dividing its last layer by 1.702. The recentre fit absorbs the residual.
LOGISTIC_TO_PROBIT = 1.702

LINKS = {"probit": ndtr, "logistic": expit}
LINK_SLOPES = {"probit": norm.pdf, "logistic": lambda a: expit(a) * (1.0 - expit(a))}


# ------------------------------------------------------------------ head splitting


def split_head(net, unfrozen_layers):
    """Split a prefit `GradientDescentERM` into (trunk, head shapes, theta0).

    The head is the last `unfrozen_layers` parameterised layers, terminal link
    stripped -- the refit supplies its own. Hidden ReLUs between unfrozen layers
    stay part of the head function.

    Args:
        net: prefit GradientDescentERM whose `.f` is an nn.Sequential
        unfrozen_layers: how many trailing parameterised layers become theta

    Returns:
        (trunk, shapes, theta0): frozen nn.Sequential up to the boundary, the
        (out, in) shape of each head Linear, and the packed flat weights.

    Raises:
        ValueError: `unfrozen_layers` out of range, or the tail is not an MLP.
    """
    from torch import nn

    modules = list(net.f)
    param_at = [i for i, m in enumerate(modules) if list(m.parameters(recurse=False))]
    if not 1 <= unfrozen_layers <= len(param_at):
        raise ValueError(f"unfrozen_layers={unfrozen_layers}; net has {len(param_at)} parameterised layers")

    boundary = param_at[-unfrozen_layers]
    for m in modules[boundary:]:
        # anything else cannot be packed into (W, b) chains
        if not isinstance(m, (nn.Linear, nn.ReLU, nn.Sigmoid)):
            raise ValueError(f"unfrozen tail must be Linear/ReLU/link, got {type(m).__name__}")

    head = [m for m in modules[boundary:] if isinstance(m, nn.Linear)]

    def _flat(tensor):
        return tensor.detach().cpu().double().numpy().ravel()

    shapes = [tuple(m.weight.shape) for m in head]
    theta0 = np.concatenate([np.concatenate([_flat(m.weight), _flat(m.bias)]) for m in head])
    trunk = nn.Sequential(*modules[:boundary]).eval()
    return trunk, shapes, theta0


def head_index(theta, phi, shapes):
    """Index (pre-link head output) on (n, d) features -> (n,). Numpy mirror of the
    JAX kernel in `partial_r2_net_jax`; the feasibility checks run on THIS one, so
    the two cannot drift without a status change (gate a27)."""
    a = phi
    offset = 0
    # runaway solver iterates land here through the feasibility checks; their
    # overflow to inf is exactly what makes them infeasible, not worth a warning
    with np.errstate(over="ignore"):
        for i, (dout, din) in enumerate(shapes):
            W = theta[offset : offset + dout * din].reshape(dout, din)
            offset += dout * din
            b = theta[offset : offset + dout]
            offset += dout
            a = a @ W.T + b
            if i < len(shapes) - 1:
                a = np.maximum(a, 0.0)
    return a[:, 0]


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


def _minimize_slsqp(fun_vg, x0, cons, maxiter=MAXITER):
    """One SLSQP solve of min fun s.t. c <= B for cons = [(vg, B), ...]."""
    cache = _Cached(fun_vg)
    constraints = []
    for con_vg, budget in cons:
        c = _Cached(con_vg)
        constraints.append(NonlinearConstraint(c.val, -np.inf, budget, jac=c.grad))
    result = minimize(
        cache.val,
        x0,
        method="SLSQP",
        jac=cache.grad,
        constraints=constraints,
        options={"maxiter": maxiter, "ftol": FTOL},
    )
    # status 9 = out of iterations while still making progress. The point is a
    # feasible inner approximation exactly like a converged local optimum (the
    # caller's feasibility check gates validity), so keep it -- rejecting it is
    # what turned whole queries into FAILURE. Real breakdowns (singular matrix,
    # incompatible constraints, linesearch failure) stay rejected.
    return result.x, bool(result.success or result.status == 9)


def _minimize_al(fun_vg, x0, cons, maxiter=MAXITER):
    """Augmented-Lagrangian fallback for heads past `SLSQP_DIM_MAX`: L-BFGS-B inner
    solves on the AL merit, multiplier update per round, penalty grown while the
    violation resists. Acceptance still rests on the caller's feasibility check.
    `maxiter` caps each inner L-BFGS-B run."""
    lam = np.zeros(len(cons))
    mu = 10.0
    x = np.asarray(x0, dtype=float)

    def violations(z):
        return np.array([float(con_vg(z)[0]) - budget for con_vg, budget in cons])

    previous = np.inf
    for _ in range(AL_ROUNDS):

        def merit(z, lam=lam, mu=mu):
            value, grad = fun_vg(z)
            value, grad = float(value), np.asarray(grad, dtype=float)
            for j, (con_vg, budget) in enumerate(cons):
                c, cg = con_vg(z)
                t = lam[j] + mu * (float(c) - budget)
                if t > 0.0:
                    value += (t * t - lam[j] ** 2) / (2.0 * mu)
                    grad = grad + t * np.asarray(cg, dtype=float)
            return value, grad

        result = minimize(merit, x, method="L-BFGS-B", jac=True, options={"maxiter": min(maxiter, AL_INNER_MAXITER)})
        x = result.x
        viol = violations(x)
        worst = float(np.max(viol, initial=0.0))
        if worst <= FTOL:
            return x, True
        lam = np.maximum(0.0, lam + mu * viol)
        if worst > 0.5 * previous:  # not shrinking fast enough: lean on the penalty
            mu *= 4.0
        previous = worst
    return x, float(np.max(violations(x), initial=0.0)) <= FTOL


def _cg(matvec, b, iters=50, tol=1e-10):
    """Conjugate gradients on a PSD matvec. A partial solve is exactly what the
    directed starts want: early iterations already favour the cheap directions."""
    x = np.zeros_like(b)
    r = b.copy()
    p = r.copy()
    rs = float(r @ r)
    for _ in range(iters):
        Ap = matvec(p)
        alpha = rs / max(float(p @ Ap), 1e-300)
        x += alpha * p
        r -= alpha * Ap
        rs_new = float(r @ r)
        if rs_new < tol * float(b @ b):
            break
        p = r + (rs_new / rs) * p
        rs = rs_new
    return x


# ---------------------------------------------------------------------- estimator


class PartialR2Net(BoundedSA):
    """Bounds on E[Y|do(x)] over the (P2) ball in a last-`l`-layer refit class."""

    def __init__(
        self,
        gamma=None,
        epsilon=0.0,
        pad=False,
        calibrate=False,
        clipy=True,
        n_jobs=1,
        mean_match=True,
        link="probit",
        unfrozen_layers=1,
        outcome_model=None,
        seed=0,
    ):
        if link not in LINKS:
            raise ValueError(f"unknown link {link!r}; valid: {sorted(LINKS)}")
        self.link_name = link
        self.unfrozen_layers = unfrozen_layers
        self.outcome_model = outcome_model
        self.seed = seed
        self._ctx = None
        self._tau = None  # Lem. 2 band half-width; per-gamma, set in _prepare
        self._terms = None  # jitted (objective, r2, extra, gram, band); fit-state only
        self._kernels = None  # (objective, cons, cons_np) at the CURRENT budgets
        super().__init__(
            gamma=gamma,
            epsilon=epsilon,
            pad=pad,
            calibrate=calibrate,
            clipy=clipy,
            n_jobs=n_jobs,
            mean_match=mean_match,
        )

    # ------------------------------------------------------------------- fit

    def _fit(self, X, y, **kwargs):
        self.rng_ = np.random.default_rng(self.seed)

        model = self.outcome_model
        model = model() if callable(model) else model
        if model is None:
            raise ValueError(
                "PartialR2Net needs an outcome model: pass the prefit net that owns "
                "mu_hat and the frozen trunk. Refitting one per PI variant would "
                "train on the n_pi subset and break the ERM/PI matching."
            )
        if not getattr(model, "prefit_", False):
            model = model.fit(X, y, **kwargs)
        self.outcome_ = model

        self.trunk_, self.head_shapes_, theta0 = split_head(model, self.unfrozen_layers)
        if self.link_name == "probit":
            # only the LAST layer rescales: it alone sets the index scale
            dout, din = self.head_shapes_[-1]
            theta0[-(dout * din + dout) :] /= LOGISTIC_TO_PROBIT

        y = np.asarray(y).ravel().astype(float)
        self.y_ = y
        self.phi_ = self._features(X)
        self.mu_ = np.asarray(model.predict_mean(X)).ravel().astype(float)
        self.sigma2_ = float(np.mean((y - self.mu_) ** 2))
        self.y_min, self.y_max = float(np.min(y)), float(np.max(y))
        # Lem. 2's slice is E_n[h(X)] = E_n[Y]; `ybar_` is that right-hand side,
        # read by the jitted band term as well as by the numpy mirror
        self.ybar_ = float(np.mean(y))

        self._precompute(X, y, **kwargs)
        self.theta_c_, self.r2_floor_ = self._recentre(theta0)
        self._fit_directions()
        self._floor_cache = {}  # keyed on gamma; a refit invalidates it
        self._band_cache = {}  # (anchor, E_r2 floor) on the slab, keyed on gamma
        return self

    def _fit_directions(self):
        """Pseudo-inverse of the slope-weighted feature Gram, for the DIRECTED
        starts: linearising E_r2 at theta_c gives d' S d with
        S = mean(link'(a_i)^2 phi~_i phi~_i'), so d = t S^+ phi~_q is the cheapest
        excursion toward moving the query. Exact geometry for a single-Linear head
        (the index is linear in theta there); skipped for deeper heads."""
        if len(self.head_shapes_) > 1:
            self._S_pinv = None
            return
        slope = LINK_SLOPES[self.link_name](head_index(self.theta_c_, self.phi_, self.head_shapes_))
        design = np.column_stack([self.phi_, np.ones(len(self.phi_))])
        weighted = slope[:, None] * design
        self._S_pinv = np.linalg.pinv(weighted.T @ weighted / len(weighted))
        # grad of the LEVEL at theta_c: the directed starts are projected onto its
        # tangent so they travel along the slab instead of straight out of it
        self._band_grad = (slope[:, None] * design).mean(axis=0)

    def _precompute(self, X, y, **kwargs):
        """Constrained variants cache their constraint features here."""

    def _features(self, X):
        """phi(x): frozen-trunk activations, chunked on the device -> (n, d)."""
        import torch

        X = np.asarray(X).reshape(len(X), -1)
        dev = next(self.outcome_.f.parameters()).device
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 8192):
                xb = torch.tensor(X[i : i + 8192], dtype=torch.float, device=dev)
                out.append(self.trunk_(xb).cpu().double().numpy())
        return np.concatenate(out)

    def _recentre(self, theta0):
        """theta_c = argmin E_r2 from the rescaled head: the ball centre in theta
        space, and the one guaranteed-feasible start. Also swallows the ~0.01
        sigmoid-vs-probit link mismatch of the raw rescale."""
        r2_vg = self._get_terms()[1]
        before = float(r2_vg(theta0)[0])
        cache = _Cached(r2_vg)
        result = minimize(cache.val, theta0, method="L-BFGS-B", jac=cache.grad, options={"maxiter": 500})
        floor = float(result.fun)
        logger.info(f"{type(self).__name__}: theta_init E_r2 {before:.4g} -> {floor:.4g} (head recentred on mu_hat)")
        return np.asarray(result.x, dtype=float), floor

    # -------------------------------------------------------------- numpy mirror

    def _link_np(self, a):
        return LINKS[self.link_name](a)

    def r2_value(self, theta):
        """E_r2(theta), numpy. The feasibility gate for every accepted solve."""
        h = self._link_np(head_index(theta, self.phi_, self.head_shapes_))
        return float(np.mean((h - self.mu_) ** 2))

    def mean_value(self, theta):
        """m(theta) = mean_n h_theta(x_i) - ybar: the SIGNED level defect, numpy
        mirror of the jitted band term (which squares it). Lem. 2 asks for 0; the
        band asks for |m| <= tau."""
        h = self._link_np(head_index(theta, self.phi_, self.head_shapes_))
        return float(np.mean(h) - self.ybar_)

    def _extra_value(self, theta):
        """Extra constraint value; None if unconstrained."""
        return None

    def _budget(self):
        """RHS of the extra constraint; None if unconstrained."""
        return None

    # -------------------------------------------------------------- geometry

    @property
    def scale(self):
        """s: sigma-hat if calibrated (paper), else 1 (raw budgets). Identical role
        to PartialR2.scale; sigma-hat^2 is the ERM net's MSE on the n_pi rows."""
        return float(np.sqrt(self.sigma2_)) if self.calibrate else 1.0

    def _r2_budget(self, gamma):
        return self.scale**2 * max(float(gamma), 0.0)

    def _band_tau(self, gamma):
        """Half-width of Lem. 2's slab at this budget; None when mean matching is
        off. See MEAN_BAND_SE for the derivation of the sqrt(1 + gamma) factor."""
        if not self.mean_match:
            return None
        variance = self.sigma2_ * (1.0 + max(float(gamma), 0.0))
        return float(MEAN_BAND_SE * np.sqrt(variance / len(self.y_)))

    def _band_cons(self):
        """The band as ONE inequality m^2 <= tau^2, in the (value_and_grad,
        budget) protocol both solver paths already take."""
        if self._tau is None:
            return []
        band_vg = self._get_terms()[4]
        return [(band_vg, self._tau**2)]

    def _band_anchor(self, gamma):
        """(anchor, E_r2 floor) for this budget.

        theta_c is the ball centre and the natural anchor, but nothing makes the
        prefit net's level land inside the slab, so when it does not the anchor
        becomes the slab point of least E_r2 -- and THAT E_r2 is the floor the
        infeasibility test must use, since no candidate can do better.

        The anchor is the one point every start may fall back to, so it is held to
        the REAL band however it was solved for; when no start can exhibit such a
        point the floor comes back infinite and the caller reports every query
        INFEASIBLE, which is the honest reading of "no feasible point was found"
        (and far better than anchoring outside the band, which reports FAILURE).
        """
        if self._tau is None:
            return self.theta_c_, self.r2_floor_

        key = float(gamma)
        if key in self._band_cache:
            return self._band_cache[key]

        defect = self.mean_value(self.theta_c_)
        if abs(defect) <= self._tau:
            self._band_cache[key] = (self.theta_c_, self.r2_floor_)
            return self._band_cache[key]

        r2_vg, band_vg = self._get_terms()[1], self._get_terms()[4]
        cons = [(band_vg, ((1.0 - BAND_ANCHOR_MARGIN) * self._tau) ** 2)]
        rng = np.random.default_rng(self.seed + 2)
        jitter = JITTER * float(np.sqrt(np.mean(self.theta_c_**2)))
        best, at = np.inf, None
        for i in range(FLOOR_STARTS):
            theta0 = self.theta_c_ if i == 0 else self.theta_c_ + jitter * rng.standard_normal(self.theta_c_.size)
            x, ok = self._minimize(r2_vg, theta0, cons)
            if not ok or abs(self.mean_value(x)) > self._tau * (1 + FEAS_TOL) + EPS:
                continue
            value = self.r2_value(x)
            if value < best:
                best, at = value, x

        if at is None:
            logger.error(
                f"{type(self).__name__}: theta_c level defect {defect:+.4g} exceeds tau "
                f"{self._tau:.4g} and no band-feasible anchor was found in {FLOOR_STARTS} starts"
            )
            self._band_cache[key] = (self.theta_c_, np.inf)
            return self._band_cache[key]

        logger.info(
            f"{type(self).__name__}: theta_c level defect {defect:+.4g} exceeds tau "
            f"{self._tau:.4g}; anchored on the slab at E_r2 {best:.4g}"
        )
        self._band_cache[key] = (at, best)
        return self._band_cache[key]

    def _get_terms(self):
        if self._terms is None:
            from src.methods import partial_r2_net_jax

            self._terms = partial_r2_net_jax.build_terms(self)
        return self._terms

    def _minimize(self, fun_vg, x0, cons, maxiter=MAXITER):
        if len(self.theta_c_) <= SLSQP_DIM_MAX:
            return _minimize_slsqp(fun_vg, x0, cons, maxiter=maxiter)
        return _minimize_al(fun_vg, x0, cons, maxiter=maxiter)

    def _feasible(self, theta, b_r2, budget):
        """Acceptance runs on the NUMPY mirror, so a JAX/numpy drift surfaces as a
        status change rather than staying invisible."""
        if self.r2_value(theta) > b_r2 * (1 + FEAS_TOL) + EPS:
            return False
        if self._tau is not None and abs(self.mean_value(theta)) > self._tau * (1 + FEAS_TOL) + EPS:
            return False
        if budget is None:
            return True
        return self._extra_value(theta) <= budget * (1 + FEAS_TOL) + EPS

    def constraint_floor(self, gamma, n_starts=FLOOR_STARTS):
        """min of the extra constraint over the E_r2 ball: the lower limit any
        budget must clear. Solved with the SAME backend as the bounds -- it GATES
        their all-INFEASIBLE return. Cached on gamma: `_prepare` asks on every
        predict and budget sweeps would otherwise pay `n_starts` solves a probe."""
        return self._floor_point(gamma, n_starts)[0]

    def _floor_point(self, gamma, n_starts=FLOOR_STARTS):
        """(floor, argmin): the argmin doubles as the feasible ANCHOR the query
        solves start from and backtrack to -- theta_c itself can sit outside the
        extra constraint."""
        key = (float(gamma), int(n_starts))
        cached = getattr(self, "_floor_cache", None)
        if cached is None:
            cached = self._floor_cache = {}
        if key in cached:
            return cached[key]

        _, r2_vg, extra_vg, _, _ = self._get_terms()
        # the floor is over ball AND slab: a point outside the slab is not a
        # candidate, so a floor measured without the band would under-report and
        # let an infeasible budget through
        cons = [(r2_vg, self._r2_budget(gamma))] + self._band_cons()
        anchor, _ = self._band_anchor(gamma)
        rng = np.random.default_rng(self.seed + 1)
        jitter = JITTER * float(np.sqrt(np.mean(self.theta_c_**2)))
        best, at = self._extra_value(anchor), anchor
        for i in range(n_starts):
            theta0 = anchor if i == 0 else anchor + jitter * rng.standard_normal(anchor.size)
            x, ok = self._minimize(extra_vg, theta0, cons)
            if ok and self._feasible(x, self._r2_budget(gamma), None):
                value = self._extra_value(x)
                if value < best:
                    best, at = value, x
        cached[key] = (best, at)
        return best, at

    # --------------------------------------------------------- BoundedSA hooks

    def _prepare(self, X, gamma):
        b_r2 = self._r2_budget(gamma)
        # tau BEFORE the anchor: the slab is what the anchor is anchored in, and
        # both depend on the gamma actually in force (predict may override the fit
        # one), which is why neither is computed at fit time
        self._tau = self._band_tau(gamma)
        anchor, r2_floor = self._band_anchor(gamma)
        if r2_floor > b_r2 * (1 + FEAS_TOL):
            logger.error(
                f"{type(self).__name__}: infeasible -- the class cannot reach mu_hat within "
                f"the ball (and Lem. 2's slab, where it applies): E_r2 floor {r2_floor:.4g} "
                f"> budget {b_r2:.4g}. "
                "All queries INFEASIBLE."
            )
            return None

        budget = self._budget()
        if budget is not None:
            floor, at = self._floor_point(gamma)
            logger.info(f"{type(self).__name__}: constraint floor {floor:.4g} vs budget {budget:.4g}")
            if floor > budget:
                logger.error(
                    f"{type(self).__name__}: infeasible -- budget {budget:.4g} is "
                    f"below the attainable floor {floor:.4g}. All queries INFEASIBLE."
                )
                return None
            if self._extra_value(anchor) > budget:
                anchor = at  # theta_c sits outside the extra budget; anchor on the slice

        self._ctx = dict(b_r2=b_r2, budget=budget, anchor=anchor, tau=self._tau)
        self._kernels = None  # budgets moved; the jitted terms themselves survive

        phi_q = self._features(X)  # precomputed: workers never touch the net
        seeds = self.rng_.integers(0, 2**31, size=len(X))
        return list(zip(phi_q, seeds, strict=True))

    def _starts_for(self, phi_q, seed):
        """Multi-start points: the anchor, +-the linearisation direction sized to
        ~81% of the budget, plus random jitters.

        The directed pair is load-bearing, not an optimisation: pushing the index
        toward the far side saturates the link on the constraint rows, the
        Jacobian vanishes and SLSQP's LSQ subproblem goes singular -- from the
        anchor alone the far side fails outright. Starting near the boundary on
        each side skips the flat zone. Deterministic per query (A5); built lazily
        from the seed so l=2 payloads stay small.
        """
        anchor = self._ctx["anchor"]
        starts = [anchor]
        if self._S_pinv is not None:
            direction = self._S_pinv @ np.append(phi_q, 1.0)
            if self._tau is not None:
                # travel ALONG the slab: subtract the S-orthogonal component of
                # the level gradient, so the directed pair moves the query
                # without moving the mean. `quad` below still equals d'Sd after
                # this (the cross terms cancel) -- do not "fix" that line.
                lift = self._S_pinv @ self._band_grad
                denominator = float(self._band_grad @ lift)
                if denominator > EPS:
                    direction = direction - (float(self._band_grad @ direction) / denominator) * lift
            quad = float(direction @ np.append(phi_q, 1.0))  # = d'Sd for psd S
            step = 0.9 * np.sqrt(self._ctx["b_r2"] / max(quad, EPS))
            starts += [anchor + step * direction, anchor - step * direction]
        rng = np.random.default_rng(seed)
        jitter = JITTER * float(np.sqrt(np.mean(self.theta_c_**2)))
        return starts + [anchor + jitter * rng.standard_normal(anchor.size) for _ in range(N_EXTRA_STARTS)]

    def _directed_deep(self, cache, gram, b_r2):
        """Directed pair for deep heads, where the dense S^+ is unaffordable.

        d solves (S + lam I) d = grad index(anchor): with more parameters than
        constraint rows S is rank-deficient, so the regularised CG solution leans
        into the near-null directions -- the moves that shift the query at almost
        no constraint cost, which is exactly where the identified set is widest.
        The step is then sized on the TRUE E_r2 (the linearisation undershoots
        once the link saturates).
        """
        anchor = self._ctx["anchor"]
        gradient = cache.grad(anchor)
        norm = float(np.linalg.norm(gradient))
        if norm < EPS:
            return []
        direction = _cg(lambda v: np.asarray(gram(anchor, v), dtype=float) + 1e-10 * v, gradient / norm)

        # double until the ball is left, then bisect back inside ~0.9 of it
        quad = float(direction @ np.asarray(gram(anchor, direction), dtype=float))
        step = 0.5 * np.sqrt(b_r2 / max(quad, EPS))
        for _ in range(30):
            if self.r2_value(anchor + step * direction) > 0.9 * b_r2 or step > 1e12:
                break
            step *= 2.0
        lo, hi = 0.0, step
        for _ in range(15):
            mid = 0.5 * (lo + hi)
            if self.r2_value(anchor + mid * direction) <= 0.9 * b_r2:
                lo = mid
            else:
                hi = mid
        return [anchor + lo * direction, anchor - lo * direction]

    def _to_feasible(self, x, b_r2, budget):
        """A final iterate can sit a hair outside tolerance (iteration cap). Pull
        it back along the segment to the strictly feasible anchor -- the excursion
        survives almost whole -- rather than discarding the whole solve. None when
        even the anchor fails (then the query is honestly FAILURE)."""
        if self._feasible(x, b_r2, budget):
            return x
        anchor = self._ctx["anchor"]
        if not self._feasible(anchor, b_r2, budget):
            return None
        lo, hi = 0.0, 1.0
        for _ in range(25):
            mid = 0.5 * (lo + hi)
            if self._feasible(anchor + mid * (x - anchor), b_r2, budget):
                lo = mid
            else:
                hi = mid
        return anchor + lo * (x - anchor) if lo > 0.0 else anchor

    def _worker_view(self):
        """Picklable snapshot: drop the torch net and the jitted terms; the solve
        runs entirely on the cached numpy features."""
        view = copy.copy(self)
        view.outcome_ = view.outcome_model = view.trunk_ = None
        view._terms = view._kernels = None
        return view

    def _begin_chunk(self):
        """Bind the jitted terms to the CURRENT budgets, once per chunk."""
        if self._kernels is not None:
            return
        objective, r2_vg, extra_vg, gram, _ = self._get_terms()
        cons = [(r2_vg, self._ctx["b_r2"])] + self._band_cons()
        if extra_vg is not None:
            cons.append((extra_vg, self._ctx["budget"]))
        self._kernels = (objective, cons, gram)

    def _solve_single(self, payload):
        """(lower, upper, status) for one query: multi-start solve on both signs of
        the INDEX, mapped through the link at the end.

        Do NOT warm-start from a neighbouring query's optimum: the constraint set
        is non-convex (that is why there is a multi-start), and seeding from a
        nearby solution biases toward its local optimum -- silently NARROWER, i.e.
        invalid, bounds.
        """
        phi_q, seed = payload
        objective, cons, gram = self._kernels
        b_r2, budget = self._ctx["b_r2"], self._ctx["budget"]

        cache = _Cached(lambda theta: objective(theta, phi_q))
        starts = self._starts_for(phi_q, seed)
        if self._S_pinv is None:  # deep head: directed pair via the Gram matvec
            starts[1:1] = self._directed_deep(cache, gram, b_r2)

        # the backtracked starts are candidates in their own right: the directed
        # pair already sits near the boundary on each side and carries most of the
        # bound (profiled: the full multi-start moves the widths by < 1e-3)
        candidates = []
        anchored = 0
        for theta0 in starts:
            x = self._to_feasible(theta0, b_r2, budget)
            if x is not None:
                # a start that backtracks all the way to the anchor carried no
                # excursion: counted, so a band too thin to travel in is visible
                # as a number rather than as quietly narrower bounds
                anchored += int(np.array_equal(x, self._ctx["anchor"]))
                candidates.append((cache.val(x), x))

        best = [np.inf, -np.inf]
        got = [False, False]
        for sign, side in ((+1.0, 0), (-1.0, 1)):
            values = [value for value, _ in candidates]

            def fun_vg(theta, s=sign):
                return s * cache.val(theta), s * cache.grad(theta)

            if candidates:
                # Unconstrained/inert programs: ONE polish per side from the
                # leading candidate -- solving from every start cost ~4.6 s/query
                # and moved THOSE widths by < 1e-3. Extra-constrained programs
                # keep the full multi-start (every RAW start, full budget): their
                # optima need not sit along the r2-sized directed line, and the
                # single-polish shortcut measurably NARROWED PI+INV (0.91 ->
                # 0.80 at the a27 fixture) -- a validity-side loss, not a
                # tolerable speed trade.
                # The band is a LIVE extra constraint at production tau, so a
                # mean-matched program is an extra-constrained one and takes the
                # full multi-start too -- the single polish is only ever safe
                # where nothing but the ball binds.
                if budget is None and (self._tau is None or SINGLE_POLISH_WITH_BAND):
                    pick = min if side == 0 else max
                    polish_starts = [pick(candidates, key=lambda c: c[0])[1]]
                    maxiter = POLISH_MAXITER
                else:
                    polish_starts = starts
                    maxiter = MAXITER
                for theta0 in polish_starts:
                    try:
                        x, _ = self._minimize(fun_vg, theta0, cons, maxiter=maxiter)
                        # any backtracked iterate is a feasible inner point,
                        # wherever the solver stopped -- runaways just backtrack
                        x = self._to_feasible(x, b_r2, budget)
                        if x is not None:
                            values.append(cache.val(x))
                    except Exception as exc:  # a lost polish costs width, not validity
                        logger.debug(f"{type(self).__name__}: polish failed ({exc}); candidate bound kept")

            if values:
                best[side] = min(values) if side == 0 else max(values)
                got[side] = True

        fraction = anchored / max(len(starts), 1)
        if not (got[0] and got[1]):
            return np.nan, np.nan, SolveStatus.FAILURE, fraction
        return float(self._link_np(best[0])), float(self._link_np(best[1])), SolveStatus.OK, fraction


# --------------------------------------------------------- constrained variants


class RecentredInvPartialR2Net(PartialR2Net):
    """PI+INV, fitted against the POST-DA measure.

    Recentred for the obvious reason: from an X-centred ball the invariant slice
    sits an invariance-defect away -- and that defect IS the
    confounding -- so PI+INV is empty at any reasonable eps. E_inv is symmetric in
    (X, GX), so swapping them leaves the CONSTRAINT untouched; only the ball moves
    to the post-DA centre, which already sits near the slice.

    Realizability at unfrozen_layers=1 is empirical: a shallow refit could contain
    no near-invariant function, in which case all-INFEASIBLE is the honest answer,
    not a bug. Measured at l=1, gamma*, n_pi=6k the DA-trunk head DOES reach the
    slice (floor 0.004 < eps^2 0.01) -- read the floor/budget pair logged per
    predict before touching the budget either way.
    """

    def __init__(self, gamma=None, epsilon=None, **kwargs):
        if epsilon is None:
            raise ValueError("epsilon required")
        super().__init__(gamma=gamma, epsilon=epsilon, **kwargs)

    def _fit(self, X, y, GX=None, **kwargs):
        if GX is None:
            raise ValueError("GX (augmented treatment) required")
        GX = np.asarray(GX).reshape(len(GX), -1)
        return super()._fit(GX, y, GX=X, **kwargs)

    def _precompute(self, X, y, GX=None, **kwargs):
        # the ball features `phi_` already cover the X side of every pair
        GX = X if GX is None else np.asarray(GX).reshape(len(X), -1)
        self.phi_gx_ = self._features(GX)

    def _extra_value(self, theta):
        h = self._link_np(head_index(theta, self.phi_, self.head_shapes_))
        h_g = self._link_np(head_index(theta, self.phi_gx_, self.head_shapes_))
        return float(np.mean((h - h_g) ** 2))

    def _budget(self):
        return self.epsilon**2


class IVConstrainedPartialR2Net(PartialR2Net):
    """PI + leaky-IV constraint (Asm. 3): || Qz'(y - h(X)) ||^2 / n <= eps_iv^2,
    the GMM form with phi_z(z) = z and W = Sigma_z^+ -- the pseudo-inverse weight
    collapses to the same QR geometry as `iv_constraint_terms`."""

    def __init__(self, gamma=None, epsilon_iv=None, **kwargs):
        if epsilon_iv is None:
            raise ValueError("epsilon_iv required")
        self.epsilon_iv = epsilon_iv
        super().__init__(gamma=gamma, **kwargs)

    def _precompute(self, X, y, Z=None, **kwargs):
        if Z is None:
            # a baseline branch wants a plain PartialR2Net, not a silently
            # self-instrumented constraint
            raise ValueError("IVConstrainedPartialR2Net needs an instrument; use PartialR2Net for a null-IV baseline.")
        self.Qz_, _ = np.linalg.qr(np.asarray(Z, dtype=float).reshape(len(Z), -1))

    def _extra_value(self, theta):
        residual = self.y_ - self._link_np(head_index(theta, self.phi_, self.head_shapes_))
        projected = self.Qz_.T @ residual
        return float(projected @ projected / len(self.y_))

    def _budget(self):
        return self.epsilon_iv**2


# ------------------------------------------------------------------ intersections


class IntersectedPartialR2Net(IntersectionMixin, PartialR2Net):
    """Baseline PI intersected with DA+PI (Cor. 1). Padding hits the DA branch only.

    `_fit` is overridden entirely -- each branch owns its own trunk, features and
    centre. Cor. 1 needs h_*(x) inside BOTH intervals, a membership fact, so the
    branches need not share a parameterisation.
    """

    def __init__(self, outcome_models=None, **kwargs):
        if outcome_models is None or not {"X", "GX"} <= set(outcome_models):
            raise ValueError(
                'IntersectedPartialR2Net needs the prefit outcome nets {"X": .., "GX": ..}: '
                "the baseline branch fits on X with the X net, the DA branch on GX "
                f"with the GX net. Got {sorted(outcome_models or {})}."
            )
        self.outcome_models = outcome_models
        # outcome_model=None is safe: our _fit never reaches PartialR2Net._fit's check
        super().__init__(outcome_model=None, **kwargs)
        self.baseline = self.augmented = None

    def _branch_kwargs(self, key, pad):
        """Everything a sibling PartialR2Net needs. One place, so a new knob cannot
        silently stop reaching the branches."""
        return dict(
            gamma=self.gamma,
            epsilon=self.epsilon,
            pad=pad,
            calibrate=self.calibrate,
            clipy=self.clipy,
            n_jobs=self.n_jobs,
            mean_match=self.mean_match,
            link=self.link_name,
            unfrozen_layers=self.unfrozen_layers,
            seed=self.seed,
            outcome_model=self.outcome_models[key],
        )

    def _fit_branches(self, X, y, GX, G):
        self.baseline = PartialR2Net(**self._branch_kwargs("X", pad=False)).fit(X, y)
        self.augmented = PartialR2Net(**self._branch_kwargs("GX", pad=self.pad)).fit(GX, y)

    def _fit(self, X, y, GX=None, G=None, **kwargs):
        if GX is None:
            raise ValueError("GX (augmented treatment) required")
        GX = np.asarray(GX).reshape(len(GX), -1)
        self._fit_branches(X, y, GX, G)

        self.sigma2_ = self.baseline.sigma2_
        self.y_min, self.y_max = float(np.min(y)), float(np.max(y))
        logger.info(f"{type(self).__name__}: rho (GX/X noise ratio) {self.rho:.4f}")
        return self

    @property
    def rho(self):
        """Information-loss factor sigma-tilde^2 / sigma^2 on the nets' MSE."""
        return self.augmented.sigma2_ / self.baseline.sigma2_


class IntersectedIVPartialR2Net(IntersectedPartialR2Net):
    """Baseline PI (null instrument) intersected with DA+PI+IV."""

    def __init__(self, epsilon_iv=None, **kwargs):
        if epsilon_iv is None:
            raise ValueError("epsilon_iv required")
        self.epsilon_iv = epsilon_iv
        super().__init__(**kwargs)

    def _fit_branches(self, X, y, GX, G):
        # the baseline carries no instrument, so it is a PLAIN PartialR2Net
        self.baseline = PartialR2Net(**self._branch_kwargs("X", pad=False)).fit(X, y)
        branch_kwargs = self._branch_kwargs("GX", pad=self.pad)
        self.augmented = IVConstrainedPartialR2Net(epsilon_iv=self.epsilon_iv, **branch_kwargs).fit(GX, y, Z=G)
