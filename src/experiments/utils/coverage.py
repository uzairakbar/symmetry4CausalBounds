"""Gamma selection by POPULATION coverage: the helpers behind
`scripts/select_domnist_gamma.py`.

gamma* = the SMALLEST gamma whose empirical coverage of h* on the selection population
reaches the target, by bisection on log gamma for ANY link/constraint (`bisect_gamma`).
Valid because the sensitivity set at gamma1 < gamma2 is the image of the same unit ball
under a smaller scale, so the exact bounds nest and coverage of a fixed target is
non-decreasing in gamma; the fitted PI is gamma-independent, so only `predict` runs
per step. NaN rows (failed solves) count as uncovered, so the result errs wide.

For the unconstrained gaussian link the same number is closed form, kept ONLY as a
cross-check (`required_gamma`): the bound is mu_y +- sqrt(gamma c) r(x), so

    gamma_i = (|mu_y(x_i) - h*(x_i)| / r(x_i))^2 / c,   r(x) = ||(mu_u|x - abar) Sigma^-1/2||

and gamma* is the ceil(target n)-th order statistic of gamma_i.

Every helper takes any object with `predict(X, gamma=...)` returning (N, 2) bounds.
"""

import numpy as np
from loguru import logger

GAMMA_GRID = (1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0)


def coverage_curve(pi, X, target, grid=GAMMA_GRID, chunk: int = 4096) -> dict:
    """Empirical coverage/width over a gamma grid, chunked so N=60k never materialises
    one big block in the solver."""
    X, h = np.asarray(X), np.asarray(target).ravel()
    out = {}
    for g in grid:
        cov = wid = n = 0
        for i in range(0, len(X), chunk):
            b = np.asarray(pi.predict(X[i : i + chunk], gamma=g), dtype=float)
            hh = h[i : i + chunk]
            cov += float(np.nansum((b[:, 0] <= hh) & (hh <= b[:, 1])))
            wid += float(np.nansum(b[:, 1] - b[:, 0]))
            n += len(hh)
        out[g] = dict(coverage=cov / n, mean_width=wid / n)
    return out


def coverage_at(pi, X, h, gamma, chunk: int = 4096) -> tuple[float, float]:
    """(coverage, mean width) at one gamma; one predict pass over X."""
    r = coverage_curve(pi, X, h, grid=(gamma,), chunk=chunk)[gamma]
    return r["coverage"], r["mean_width"]


def bisect_gamma(
    pi, X, h, target: float = 0.95, lo: float = 1e-4, hi: float = 10.0, tol: float = 0.05, max_iter: int = 20
) -> dict:
    """Smallest gamma with coverage(h) >= target, by bisection on log gamma.

    Checks the bracket first (if cov(hi) < target returns hi with a warning, if
    cov(lo) >= target returns lo), then halves log(hi/lo) until hi/lo <= 1+tol.
    Coverage is monotone in gamma up to solver noise, so the bracket is never released
    and NO monotonicity assert is made on real traces. Returns the conservative end.
    """
    trace: dict[float, tuple[float, float]] = {}

    def cov(g):
        trace[g] = coverage_at(pi, X, h, g)
        logger.info(f"  gamma={g:.5g}: coverage {trace[g][0]:.4f} width {trace[g][1]:.4f}")
        return trace[g][0]

    lo, hi = float(lo), float(hi)
    if not 0.0 < lo < hi:
        raise ValueError(f"need 0 < gamma_lo < gamma_hi, got {lo:g}, {hi:g}")
    if cov(hi) < target:
        logger.warning(f"coverage {trace[hi][0]:.4f} < {target} even at gamma_hi={hi:g}; returning gamma_hi")
    elif cov(lo) >= target:
        hi = lo
    else:
        for _ in range(max_iter):
            if hi / lo <= 1.0 + tol:
                break
            mid = float(np.sqrt(lo * hi))
            if cov(mid) >= target:
                hi = mid
            else:
                lo = mid
        if hi / lo > 1.0 + tol:
            logger.warning(
                f"max_iter={max_iter} exhausted at hi/lo={hi / lo:.4g} > 1+tol; returning the conservative end {hi:.5g}"
            )
    return {
        "gamma": hi,
        "coverage": trace[hi][0],
        "width": trace[hi][1],
        "n_eval": len(trace),
        "trace": [(g, c, w) for g, (c, w) in sorted(trace.items())],
    }


def grid_curves(pi, X, h, gammas) -> dict[str, np.ndarray]:
    """coverage/width on a COMMON gamma grid, for the plot only."""
    cw = np.array([coverage_at(pi, X, h, g) for g in gammas])
    return {"coverage": cw[:, 0], "width": cw[:, 1]}


def _required(r, d, c):
    """Per-sample smallest covering gamma from the closed-form primitives."""
    with np.errstate(divide="ignore", over="ignore"):
        g = (d / r) ** 2 / c
    return np.where(r > 0, g, np.where(d <= 0, 0.0, np.inf))


def required_gamma(pi, X, target, pad: float = 0.0) -> np.ndarray:
    """Per-sample smallest gamma whose interval covers the target. Unconstrained
    gaussian only: reads `latent_`, `anchors_`, `sigma2_`, `calibrate_sigma`, `_mu`
    and `_budget` off a fitted CopSensPI."""
    if getattr(pi, "link", None) != "gaussian" or pi._budget() is not None:
        raise ValueError("closed form is for the unconstrained gaussian link; use coverage_curve for the rest")
    X = np.asarray(X)
    mu_q = pi.latent_.transform(X)
    r = np.linalg.norm((mu_q - pi.anchors_.mean(axis=0)) @ pi.latent_.halfinv_, axis=1)
    c = float(max(pi.sigma2_ if pi.calibrate_sigma else 1.0, 1e-300))
    d = np.maximum(np.abs(np.asarray(pi._mu(X)).ravel() - np.asarray(target).ravel()) - pad, 0.0)
    return _required(r, d, c)
