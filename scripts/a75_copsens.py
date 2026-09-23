"""A75: the copsens backend on a synthetic factor SEM.

The mixin's `tests/test_copsens.py` and `tests/test_copsens_speed.py` as one gate,
on the mixin's `FactorSEM` (copied below as the fixture, so nothing new enters
`src/sem`). Every leg is about the ANSWER: the numerics of `src/methods/copsens.py`
against closed forms, against each other, and against finite differences. Legs:

  (i)    construction: unknown link, missing outcome model, missing epsilon and a
         missing instrument raise; `mean_match` is inert.
  (ii)   gamma 0 collapses to the ERM; the ERM sits inside ordered bounds in [0, 1].
  (iii)  gaussian equals the gcalibrate closed form; probit equals the bcalibrate
         identity; the latent model recovers the factors up to rotation.
  (iv)   coverage of the true ATE at a large budget.
  (v)    INV and IV: gamma 0 collapses, widths monotone in gamma, nested in PI,
         under SLSQP (probit) and the SOCP (gaussian); the budgets follow a
         predict-time epsilon / epsilon_iv.
  (vi)   `iv_budget` at gz* 0 and > 0; SOCP radius^2 = n budget; E_inv symmetric
         in (X, GX); RecentredInv puts the ball on GX.
  (vii)  JAX == numpy in value (1e-10) and against finite differences in gradient
         (1e-4 relative); the floor agrees (1e-4 relative); the bounds agree within
         5e-3 of the width; jax_grad is an exact no-op on the gaussian SOCP.
  (viii) n_jobs 4 == n_jobs 1 bit for bit on the probit NLP (PI and PI+INV).
  (ix)   `recalibrate` and `rho` reach the radius through `budget`; two predicts
         at one gamma agree bit for bit; an infeasible budget reads INFEASIBLE on
         every query; `query_diagnostics` is one column in [0, 1].
  (x)    the intersections: max of lowers, min of uppers, the worse status; the IV
         intersection instruments its DA branch on T.

    uv run python scripts/a75_copsens.py [--only LEG]
"""

import argparse
import os
import sys
import time

import numpy as np
from loguru import logger
from scipy.optimize import approx_fprime
from scipy.stats import norm
from sklearn.linear_model import LinearRegression, LogisticRegression

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from src.methods.copsens import (  # noqa: E402
    CopSensPI,
    IntersectedCopSens,
    IntersectedIVCopSens,
    InvarianceConstrainedCopSens,
    IVConstrainedCopSens,
    LatentTreatmentModel,
    RecentredInvCopSens,
    iv_budget,
)
from src.methods.sensitivity_models import SolveStatus  # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAIL.append(name)


# ------------------------------------------------------------------- fixture


class FactorSEM:
    """U ~ N(0, I_m); X = W U + eps; E[Y | x, u] = F(beta'x + g'u). The mixin's
    `src/sem/copsens_sim.py::FactorSEM`, without the abstract base."""

    def __init__(self, treatment_dimension=32, latent_dimension=3, noise_std=1.0, confounding=2.0, link="probit"):
        self.dx, self.m, self.noise_std, self.link = treatment_dimension, latent_dimension, noise_std, link
        W = np.random.randn(self.dx, self.m)
        self.W = W / np.linalg.norm(W, axis=0, keepdims=True) * np.sqrt(self.dx) * noise_std
        beta = np.random.randn(self.dx, 1)
        self.W_XY = beta / np.linalg.norm(beta)
        gc = np.random.randn(self.m)
        self.g_star = confounding * gc / np.linalg.norm(gc)
        z, w = np.polynomial.hermite_e.hermegauss(41)
        self.z, self.w = z, w / w.sum()

    def _F(self, a):
        return a if self.link == "gaussian" else norm.cdf(a)

    def ate(self, X):
        """E[Y | do(x)] = E_U[F(beta'x + g'U)], U ~ N(0, I): scalar Gaussian smoothing."""
        a = (X.reshape(len(X), -1) @ self.W_XY).ravel()
        s = float(np.linalg.norm(self.g_star))
        if self.link == "gaussian":
            return a[:, None]
        return sum(wk * self._F(a + s * zk) for zk, wk in zip(self.z, self.w, strict=True))[:, None]

    def sample(self, N):
        U = np.random.randn(N, self.m)
        X = U @ self.W.T + self.noise_std * np.random.randn(N, self.dx)
        p = self._F((X @ self.W_XY).ravel() + U @ self.g_star)
        Y = (np.random.rand(N) < p).astype(float) if self.link != "gaussian" else p + np.random.randn(N)
        return X, Y[:, None]


class SkOutcome:
    """A prefit outcome model on the fixture: logistic for probit, OLS for gaussian."""

    def __init__(self, link):
        self.link = link
        self.prefit_ = False

    def fit(self, X, y, **kwargs):
        y = np.asarray(y, dtype=float).ravel()
        if self.link == "gaussian":
            self.m_ = LinearRegression().fit(X, y)
        else:
            self.m_ = LogisticRegression(max_iter=2000, C=1e4).fit(X, (y > 0.5).astype(int))
        self.prefit_ = True
        return self

    def predict_mean(self, X):
        return self.m_.predict(X) if self.link == "gaussian" else self.m_.predict_proba(X)[:, 1]


_CACHE = {}


def fixture(link, n=600, seed=0, d=12):
    """(sem, X, y, GX, Z, prefit outcome model). Directional DA: an isotropic one
    leaves the INV constraint inert."""
    key = (link, n, seed, d)
    if key not in _CACHE:
        np.random.seed(seed)
        sem = FactorSEM(treatment_dimension=d, latent_dimension=3, link=link)
        X, y = sem.sample(n)
        u = np.zeros(X.shape[1])
        u[0] = 1.0
        GX = X + 1.5 * u
        Z = np.random.default_rng(seed).normal(0, 1, (len(X), 2))
        _CACHE[key] = (sem, X, y, GX, Z, SkOutcome(link).fit(X, y))
    return _CACHE[key]


def kw(link, om, **extra):
    """Ctor knobs shared by every model of one leg: no recalibration, no padding,
    no clipping, so the bounds are the raw program's."""
    return dict(link=link, n_components=3, outcome_model=om, recalibrate=False, pad=False, clipy=False) | extra


def feasible_eps(model, slack=1.5):
    """An epsilon whose budget clears the constraint at gamma 0 with `slack`."""
    return float(np.sqrt(slack * model._con_at_zero()))


def models(link, jax_grad=True, n_jobs=1, eps_slack=1.5):
    """PI+INV and DA+PI+IV on the fixture, at budgets that clear their floors."""
    _, X, y, GX, Z, om = fixture(link)
    inv = InvarianceConstrainedCopSens(gamma=0.25, epsilon=1.0, **kw(link, om, jax_grad=jax_grad, n_jobs=n_jobs))
    inv.fit(X, y, GX=GX)
    inv.epsilon = feasible_eps(inv, eps_slack)
    iv = IVConstrainedCopSens(gamma=0.25, epsilon_iv=1.0, **kw(link, om, jax_grad=jax_grad, n_jobs=n_jobs))
    iv.fit(X, y, Z=Z)
    iv.epsilon_iv = feasible_eps(iv, eps_slack)
    return [("PI+INV", inv), ("DA+PI+IV", iv)]


def ikw(link, **extra):
    """`kw` for an intersection, which names its nets `outcome_models`."""
    return {k: v for k, v in kw(link, None, **extra).items() if k != "outcome_model"}


def width(bounds):
    return float(np.nanmean(bounds[:, 1] - bounds[:, 0]))


# ---------------------------------------------------------------------- legs


def leg_i():
    _, X, y, GX, Z, om = fixture("probit")
    for name, thunk in (
        ("unknown link raises", lambda: CopSensPI(gamma=0.1, link="logit", outcome_model=om)),
        ("missing outcome model raises at fit", lambda: CopSensPI(gamma=0.1, n_components=3).fit(X, y)),
        ("PI+INV without epsilon raises", lambda: InvarianceConstrainedCopSens(gamma=0.1, **kw("probit", om))),
        ("IV without epsilon_iv raises", lambda: IVConstrainedCopSens(gamma=0.1, **kw("probit", om))),
        (
            "IV without an instrument raises",
            lambda: IVConstrainedCopSens(gamma=0.1, epsilon_iv=0.1, **kw("probit", om)).fit(X, y),
        ),
        (
            "IV with an EMPTY instrument raises",
            lambda: IVConstrainedCopSens(gamma=0.1, epsilon_iv=0.1, **kw("probit", om)).fit(
                X, y, T=np.zeros((len(X), 0)), Z=None
            ),
        ),
        (
            "intersection without both nets raises",
            lambda: IntersectedCopSens(gamma=0.1, outcome_models={"X": om}, **ikw("probit")),
        ),
    ):
        try:
            thunk()
            check(f"(i) {name}", False, "no raise")
        except ValueError as error:
            check(f"(i) {name}", True, f"ValueError: {str(error)[:50]}")
    Q = X[:6]
    a = CopSensPI(gamma=0.25, mean_match=True, **kw("probit", om)).fit(X, y).predict(Q)
    b = CopSensPI(gamma=0.25, mean_match=False, **kw("probit", om)).fit(X, y).predict(Q)
    check("(i) mean_match is inert", np.array_equal(a, b), f"max|diff| {np.abs(a - b).max():.1e}")


def leg_ii():
    for link in ("probit", "gaussian"):
        _, X, y, _, _, om = fixture(link)
        Q = X[:32]
        model = CopSensPI(gamma=0.5, **kw(link, om)).fit(X, y)
        mu = om.predict_mean(Q)
        b0 = model.predict(Q, gamma=0.0)
        err = max(np.abs(b0[:, 0] - mu).max(), np.abs(b0[:, 1] - mu).max())
        check(f"(ii) {link}: gamma 0 collapses to the ERM", err < 1e-8, f"max|b - mu| {err:.1e}")
        b = model.predict(Q)
        inside = np.all(b[:, 0] <= mu + 1e-9) and np.all(mu <= b[:, 1] + 1e-9)
        ordered = np.all(b[:, 1] >= b[:, 0])
        check(f"(ii) {link}: ERM inside ordered bounds", inside and ordered)
        if link == "probit":
            check("(ii) probit: bounds in [0, 1]", np.all(b >= -1e-9) and np.all(b <= 1 + 1e-9))
            check("(ii) probit: every query OK", np.all(model.query_status == SolveStatus.OK))


def leg_iii():
    _, X, y, _, _, om = fixture("gaussian")
    model = CopSensPI(gamma=0.4, calibrate_sigma=True, **kw("gaussian", om)).fit(X, y)
    Q = X[:40]
    b = model.predict(Q)
    mu_y = om.predict_mean(Q)
    shift = model.latent_.transform(Q) - model.anchors_.mean(axis=0)
    rad = np.sqrt(0.4 * model.sigma2_) * np.linalg.norm(shift @ model.latent_.halfinv_, axis=1)
    err = max(np.abs(b[:, 0] - (mu_y - rad)).max(), np.abs(b[:, 1] - (mu_y + rad)).max())
    check("(iii) gaussian == gcalibrate closed form", err < 1e-8, f"max err {err:.1e}")

    _, X, y, _, _, om = fixture("probit")
    model = CopSensPI(gamma=0.3, **kw("probit", om)).fit(X, y)
    Q = X[:24]
    mu_y, mu_u = om.predict_mean(Q), model.latent_.transform(Q)
    v = np.random.default_rng(0).standard_normal(3)
    v /= np.linalg.norm(v)
    s = np.sqrt(0.3)
    g = s * (v @ model.latent_.halfinv_)
    ours = model._h(mu_u, g[None, :], s, mu_y).ravel()
    delta = (model.anchors_ @ g)[None, :] - (mu_u @ g)[:, None]
    ref = norm.cdf(
        (norm.ppf(np.clip(mu_y, 1e-6, 1 - 1e-6))[:, None] * np.sqrt(1 + s**2) + delta) / np.sqrt(1 + s**2)
    ).mean(axis=1)
    err = np.abs(ours - ref).max()
    check("(iii) probit == bcalibrate identity", err < 1e-10, f"max err {err:.1e}")

    np.random.seed(0)
    sem = FactorSEM(treatment_dimension=32, latent_dimension=3)
    U = np.random.randn(4000, 3)
    Xf = U @ sem.W.T + sem.noise_std * np.random.randn(4000, 32)
    Uh = LatentTreatmentModel(3).fit(Xf).transform(Xf)
    cc = np.linalg.svd(np.linalg.qr(U)[0].T @ np.linalg.qr(Uh)[0], compute_uv=False)
    check("(iii) latent model recovers the factors", cc.min() > 0.9, f"min canonical corr {cc.min():.3f}")


def leg_iv():
    sem, X, y, _, _, om = fixture("probit", n=3000)
    model = CopSensPI(gamma=4.0, **kw("probit", om)).fit(X, y)
    Q = X[:100]
    b = model.predict(Q)
    t = sem.ate(Q).ravel()
    cov = float(np.mean((b[:, 0] <= t) & (t <= b[:, 1])))
    check("(iv) coverage of the true ATE at gamma 4", cov > 0.8, f"coverage {cov:.3f}")


def leg_v():
    for link in ("gaussian", "probit"):
        _, X, y, _, _, om = fixture(link)
        Q = X[:6]
        base = CopSensPI(gamma=0.25, **kw(link, om)).fit(X, y)
        for name, model in models(link):
            tag = f"(v) {link}/{name}"
            b0 = model.predict(Q, gamma=0.0)
            check(
                f"{tag}: finite and collapsed at gamma 0",
                np.all(np.isfinite(b0)) and np.nanmax(b0[:, 1] - b0[:, 0]) < 1e-6,
            )
            w = [width(model.predict(Q, gamma=g)) for g in (0.05, 0.25, 1.0)]
            check(f"{tag}: width monotone in gamma", w[0] < w[1] < w[2], f"{w[0]:.4f} < {w[1]:.4f} < {w[2]:.4f}")
            bp, bc = base.predict(Q, gamma=0.25), model.predict(Q, gamma=0.25)
            viol = float(np.nanmax(np.maximum(bp[:, 0] - bc[:, 0], bc[:, 1] - bp[:, 1])))
            check(f"{tag}: nested in PI", viol < 1e-6, f"violation {viol:.1e}")
            check(f"{tag}: every query OK", np.all(model.query_status == SolveStatus.OK))
        inv, iv = (m for _, m in models(link))
        inv.predict(Q, epsilon=0.123)
        check(f"(v) {link}: PI+INV budget follows predict-time epsilon", inv._budget() == 0.123**2)
        iv.predict(Q, epsilon_iv=0.321)
        check(f"(v) {link}: IV budget follows predict-time epsilon_iv", abs(iv._budget() - 0.321**2) < 1e-15)
        check(f"(v) {link}: both constrained models solve on epsilon", inv.solves_on_epsilon and iv.solves_on_epsilon)


def leg_vi():
    eps, s2, rho = 0.1, 0.13, 1.25
    ok = abs(iv_budget(eps) - eps**2) < 1e-15 and abs(iv_budget(eps, 0.0, calibrate_sigma=True) - eps**2) < 1e-15
    check("(vi) iv_budget at gz* 0 is eps^2", ok)
    check("(vi) iv_budget at gz* > 0, uncalibrated", abs(iv_budget(eps, 0.04) - (eps + 0.2) ** 2) < 1e-12)
    s = np.sqrt(s2 / rho)
    got = iv_budget(eps, 0.04, sigma2=s2, rho=rho, calibrate_sigma=True)
    check("(vi) iv_budget at gz* > 0, calibrated", abs(got - (eps + s * 0.2) ** 2) < 1e-12, f"{got:.4g}")
    try:
        iv_budget(eps, 0.04, calibrate_sigma=True)
        check("(vi) calibrated iv_budget without rho raises", False)
    except ValueError:
        check("(vi) calibrated iv_budget without rho raises", True)

    _, X, y, GX, Z, om = fixture("gaussian", n=400)
    for cal in (False, True):
        model = IVConstrainedCopSens(gamma=0.25, epsilon_iv=0.1, calibrate_sigma=cal, **kw("gaussian", om)).fit(
            X, y, Z=Z
        )
        ((_, _, t),) = model._soc_terms()
        n = len(model.cX_)
        check(
            f"(vi) SOCP radius^2 = n budget (calibrate_sigma={cal})",
            abs(t**2 - n * model._budget()) < 1e-9 * max(1.0, n * model._budget()),
            f"{t**2 / n:.6g} == {model._budget():.6g}",
        )

    a = InvarianceConstrainedCopSens(gamma=0.25, epsilon=0.2, **kw("gaussian", om)).fit(X, y, GX=GX)
    b = InvarianceConstrainedCopSens(gamma=0.25, epsilon=0.2, **kw("gaussian", om)).fit(X, y, GX=GX)
    b.cX_, b.cGX_, b.cmuX_, b.cmuGX_ = a.cGX_, a.cX_, a.cmuGX_, a.cmuX_
    g = np.random.default_rng(1).normal(0, 0.1, (5, 3))
    d = np.abs(a._constraint(g, 0.3) - b._constraint(g, 0.3)).max()
    check("(vi) E_inv symmetric in (X, GX)", d < 1e-12, f"max|diff| {d:.1e}")

    _, X, y, GX, _, om = fixture("probit")
    om_gx = SkOutcome("probit").fit(GX, y)
    rec = RecentredInvCopSens(gamma=0.25, epsilon=0.2, **kw("probit", om_gx)).fit(X, y, GX=GX)
    on_gx = CopSensPI(gamma=0.25, **kw("probit", om_gx)).fit(GX, y)
    same_ball = rec.sigma2_ == on_gx.sigma2_ and np.array_equal(rec.latent_.mean_, on_gx.latent_.mean_)
    check("(vi) RecentredInv puts the ball on GX", same_ball and np.array_equal(rec.anchors_, on_gx.anchors_))
    # replay the fit's generator: anchors first, constraint rows second
    rng = np.random.default_rng(rec.seed)
    rng.choice(len(X), size=rec.n_anchors, replace=False)
    idx = rng.choice(len(X), size=rec.n_constraint, replace=False)
    pairs = np.array_equal(rec.cX_, rec.latent_.transform(GX[idx])) and np.array_equal(
        rec.cGX_, rec.latent_.transform(X[idx])
    )
    check("(vi) RecentredInv pairs are (GX, X)", pairs and np.array_equal(rec.cmuGX_, rec._mu(X[idx])))


def leg_vii():
    from src.methods import copsens_jax

    rng = np.random.default_rng(0)
    for link in ("probit", "gaussian"):
        _, X, _, _, _, _ = fixture(link)
        for name, model in models(link):
            Q = X[:4]
            mu_q, mu_y, radius = model.latent_.transform(Q), model._mu(Q), model._radius(0.25)
            obj, con = copsens_jax.build_terms(model, radius)
            dv = dc = gv = gc = 0.0
            x64 = True
            for _ in range(6):
                v = rng.standard_normal(3) * 0.4
                hv, hg = obj(v, mu_q[0], mu_y[0])
                x64 &= np.asarray(hg).dtype == np.float64
                dv = max(dv, abs(float(hv) - model._h_v(v, mu_q[0], mu_y[0], radius)))
                fd = approx_fprime(v, lambda w, m=model, q=mu_q[0], y=mu_y[0], r=radius: m._h_v(w, q, y, r), 1e-7)
                gv = max(gv, np.abs(np.asarray(hg) - fd).max() / max(np.abs(fd).max(), 1e-12))
                cv, cg = con(v)
                dc = max(dc, abs(float(cv) - model._con_v(v, radius)))
                fdc = approx_fprime(v, lambda w, m=model, r=radius: m._con_v(w, r), 1e-7)
                gc = max(gc, np.abs(np.asarray(cg) - fdc).max() / max(np.abs(fdc).max(), 1e-12))
            check(
                f"(vii) {link}/{name}: JAX == numpy in value",
                dv < 1e-10 and dc < 1e-10 and x64,
                f"obj {dv:.1e} con {dc:.1e}",
            )
            check(f"(vii) {link}/{name}: JAX gradient == FD", gv < 1e-4 and gc < 1e-4, f"rel obj {gv:.1e} con {gc:.1e}")

    def bounds(jax_grad, n_q):
        out = {}
        for name, model in models("probit", jax_grad=jax_grad):
            t = time.time()
            out[name] = (
                model.predict(X[:n_q], gamma=0.25),
                time.time() - t,
                model.constraint_floor(model._radius(0.25)),
            )
        return out

    _, X, _, _, _, _ = fixture("probit")
    ref, got = bounds(False, 25), bounds(True, 25)
    for name in ref:
        a, b = ref[name][0], got[name][0]
        w, d = width(a), float(np.nanmax(np.abs(a - b)))
        check(
            f"(vii) probit/{name}: bounds JAX == FD within 5e-3 width", d < 5e-3 * w, f"max|diff| {d:.1e} width {w:.4f}"
        )
        fa, fb = ref[name][2], got[name][2]
        check(
            f"(vii) probit/{name}: floor JAX == FD",
            abs(fa - fb) <= 1e-4 * max(abs(fa), 1e-12) + 1e-9,
            f"fd {fa:.6g} jax {fb:.6g}",
        )
    slow, fast = ref["PI+INV"][1], got["PI+INV"][1]
    check("(vii) JAX faster than FD on PI+INV", fast < slow, f"fd {slow:.2f}s jax {fast:.2f}s")

    _, X, y, GX, _, om = fixture("gaussian")
    Q = X[:10]
    a = InvarianceConstrainedCopSens(gamma=0.25, epsilon=0.3, jax_grad=False, **kw("gaussian", om)).fit(X, y, GX=GX)
    b = InvarianceConstrainedCopSens(gamma=0.25, epsilon=0.3, jax_grad=True, **kw("gaussian", om)).fit(X, y, GX=GX)
    d = float(np.nanmax(np.abs(a.predict(Q) - b.predict(Q))))
    check("(vii) gaussian SOCP: jax_grad is a no-op", d == 0.0, f"max|diff| {d:.1e}")


def leg_viii():
    _, X, y, _, _, om = fixture("probit")
    Q = X[:12]
    one = CopSensPI(gamma=0.25, n_jobs=1, **kw("probit", om)).fit(X, y).predict(Q)
    four = CopSensPI(gamma=0.25, n_jobs=4, **kw("probit", om)).fit(X, y).predict(Q)
    check(
        "(viii) PI: n_jobs 4 == n_jobs 1 bit for bit",
        np.array_equal(one, four),
        f"max|diff| {np.abs(one - four).max():.1e}",
    )
    (_, inv1), _ = models("probit", n_jobs=1)
    (_, inv4), _ = models("probit", n_jobs=4)
    a, b = inv1.predict(Q), inv4.predict(Q)
    check(
        "(viii) PI+INV: n_jobs 4 == n_jobs 1 bit for bit", np.array_equal(a, b), f"max|diff| {np.abs(a - b).max():.1e}"
    )
    check(
        "(viii) diagnostics ride back from the workers",
        inv4.query_diagnostics is not None and np.array_equal(inv1.query_diagnostics, inv4.query_diagnostics),
    )


def leg_ix():
    _, X, y, GX, _, om = fixture("gaussian")
    Q = X[:8]
    plain = CopSensPI(gamma=0.4, **kw("gaussian", om, rho=1.0, recalibrate=0.0)).fit(X, y)
    half = CopSensPI(gamma=0.2, **kw("gaussian", om, rho=1.0, recalibrate=0.0)).fit(X, y)
    rho2 = CopSensPI(gamma=0.4, **kw("gaussian", om, rho=2.0, recalibrate=1.0)).fit(X, y)
    check("(ix) rho 2, recalibrate 1 solves at gamma/2", np.allclose(rho2.predict(Q), half.predict(Q), atol=1e-12))
    rho2.predict(Q, recalibrate=0.0)
    check("(ix) recalibrate 0 keeps gamma", np.allclose(rho2.predict(Q, recalibrate=0.0), plain.predict(Q), atol=1e-12))
    check("(ix) _radius is sqrt(budget sigma2)", abs(rho2._radius(0.4) - np.sqrt(0.4 * rho2.sigma2_)) < 1e-15)

    _, X, y, GX, _, om = fixture("probit")
    Q = X[:6]
    model = CopSensPI(gamma=0.25, **kw("probit", om)).fit(X, y)
    a = model.predict(Q)
    model.predict(Q, gamma=1.0)
    b = model.predict(Q)
    check("(ix) fitted state independent of gamma: repeat predict bit for bit", np.array_equal(a, b))
    diag = model.query_diagnostics
    check(
        "(ix) query_diagnostics is one column in [0, 1]",
        diag is not None and diag.shape == (len(Q), 1) and np.all((diag >= 0) & (diag <= 1)),
        f"failed-start fractions {np.ravel(diag)}",
    )
    inv = InvarianceConstrainedCopSens(gamma=0.25, epsilon=1e-6, **kw("probit", om)).fit(X, y, GX=GX)
    b = inv.predict(Q)
    check(
        "(ix) budget below the floor: every query INFEASIBLE and NaN",
        np.all(inv.query_status == SolveStatus.INFEASIBLE) and np.all(np.isnan(b)) and inv.query_diagnostics is None,
    )
    floor = inv.constraint_floor(inv._radius(0.25))
    check("(ix) floor is cached on the radius", inv._floor_cache[(inv._radius(0.25), 6)] == floor and floor > 1e-12)
    # a shift DA makes the gaussian INV constraint one linear equation, solvable to
    # a floor of 0; a scaling DA leaves a positive floor for the gate to sit under
    _, Xg, yg, _, _, omg = fixture("gaussian")
    ginv = InvarianceConstrainedCopSens(gamma=0.25, epsilon=1e-6, **kw("gaussian", omg)).fit(Xg, yg, GX=1.3 * Xg)
    b = ginv.predict(Xg[:6])
    check(
        "(ix) gaussian below the floor: every query INFEASIBLE",
        np.all(ginv.query_status == SolveStatus.INFEASIBLE) and np.all(np.isnan(b)),
        f"floor {ginv.constraint_floor(ginv._radius(0.25)):.2e}",
    )


def leg_x():
    for link in ("probit", "gaussian"):
        _, X, y, GX, Z, om = fixture(link)
        om_gx = SkOutcome(link).fit(GX, y)
        Q = X[:6]
        nets = {"X": om, "GX": om_gx}
        both = IntersectedCopSens(gamma=0.25, outcome_models=nets, **ikw(link)).fit(X, y, GX=GX, G=Z)
        b = both.predict(Q)
        lo = np.maximum(both.baseline.predict(Q)[:, 0], both.augmented.predict(Q)[:, 0])
        hi = np.minimum(both.baseline.predict(Q)[:, 1], both.augmented.predict(Q)[:, 1])
        empty = lo > hi
        ok = (
            np.allclose(b[~empty, 0], lo[~empty])
            and np.allclose(b[~empty, 1], hi[~empty])
            and np.all(np.isnan(b[empty]))
        )
        check(f"(x) {link}: PI&DA+PI is max-lower / min-upper", ok, f"{empty.sum()} empty")
        check(
            f"(x) {link}: branches are plain CopSensPI on their own nets",
            type(both.baseline) is CopSensPI and both.augmented.outcome_ is om_gx,
        )
        ratio = both.augmented.sigma2_ / both.baseline.sigma2_
        check(
            f"(x) {link}: rho reads the sigma2 ratio; DA branch clamps it",
            both.rho == ratio and both.augmented.rho == max(ratio, 1.0),
        )

        ivx = IntersectedIVCopSens(gamma=0.25, epsilon_iv=1.0, outcome_models=nets, **ikw(link)).fit(X, y, GX=GX, G=Z)
        ivx.augmented.epsilon_iv = feasible_eps(ivx.augmented)
        b = ivx.predict(Q)
        aug = ivx.augmented
        rng = np.random.default_rng(aug.seed)
        rng.choice(len(X), size=aug.n_anchors, replace=False)
        idx = rng.choice(len(X), size=aug.n_constraint, replace=False)
        Qz, _ = np.linalg.qr(np.column_stack([np.ones(len(idx)), Z[idx]]))
        check(
            f"(x) {link}: PI&DA+PI+IV instruments its DA branch on T",
            isinstance(aug, IVConstrainedCopSens) and np.array_equal(aug.Qz_, Qz),
        )
        check(
            f"(x) {link}: intersection status is the worse of the branches",
            np.array_equal(ivx.query_status, np.maximum(ivx.baseline.query_status, aug.query_status)),
        )
        check(f"(x) {link}: intersection solves on epsilon", ivx.solves_on_epsilon and not both.solves_on_epsilon)
        check(
            f"(x) {link}: intersection nested in PI",
            np.all(b[:, 0] >= both.baseline.predict(Q)[:, 0] - 1e-9)
            and np.all(b[:, 1] <= both.baseline.predict(Q)[:, 1] + 1e-9),
        )


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    os.chdir(REPO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=None)
    args = parser.parse_args()
    legs = [
        ("i", leg_i),
        ("ii", leg_ii),
        ("iii", leg_iii),
        ("iv", leg_iv),
        ("v", leg_v),
        ("vi", leg_vi),
        ("vii", leg_vii),
        ("viii", leg_viii),
        ("ix", leg_ix),
        ("x", leg_x),
    ]
    if args.only:
        legs = [(tag, leg) for tag, leg in legs if tag.lower() == args.only.lower()]
        if not legs:
            sys.exit(f"unknown leg {args.only!r}")
    t0 = time.time()
    for tag, leg in legs:
        try:
            leg()
        except Exception as error:  # a raise is a FAIL line, and the later legs still report
            check(f"({tag}) ran without raising", False, f"{type(error).__name__}: {error}")
    print(f"  {time.time() - t0:.0f}s")
    if not FAIL:
        print("A75 PASS")
    else:
        print(f"A75 FAIL: {FAIL}")
        sys.exit(1)
