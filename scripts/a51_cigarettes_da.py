"""A51: the cigarette DA translates along v and nothing else, and its two knobs do
what Prop. 2 and the robustness sweep need.

`ScaleTranslation` is `NullSpaceTranslation` handed a basis of null(v), so the
translation direction is v-hat itself. Legs:

  (i)   direction: `augment` moves X only along v -- the component off span(v) is
        machine zero -- it moves it at all, and `param_dimension` is 1. Catches: a
        basis built from the wrong side of v (the DA then translates inside null(v),
        where h_* is NOT invariant, or not at all). Misses: the amplitude, which is
        the orchestrator's to set.
  (ii)  exactness: against an exactly homogeneous h_*, eps* at `strength = 0` is
        zero to machine precision, so nothing on this path is padded. Catches: a
        non-zero default strength, a direction that leaks out of ker(h_*).
  (iii) the knob: eps* is monotone and LINEAR in `strength`, and the repo's own
        bisection (`recalibrated_da_epsilon`) lands on a target of 0.5 to 1e-6.
        This is what the epsilon sweep rides. Catches: a clamped or saturating
        knob, which turns the robustness axis flat.
  (iv)  Prop. 2: over a 2^-3..2^3 amplitude grid tr(S)/k falls monotonically to the
        analytic floor (k-1)/k and rho tr(S)/k stays under 1, and every point
        matches 1 - (1/k) a/(1+a) with a = s^2 v'Sigma^-1 v computed from the
        CONFIGURED v. The pin on v is what makes the leg falsifiable: the formula
        holds for ANY unit direction with its own a, so an unpinned version would
        pass under any translation. tr(S)/k is pooled over draws, because the
        formula is a population statement and one draw of c carries 2% of
        resampling noise in its own variance.
  (v)   `W_perp` rows are unit norm and the augmentation is still "translate". The
        parent normalises its basis in the Frobenius norm, so an inherited W_perp
        puts a 1/sqrt(k-1) factor on `strength`; harmless (the knob is bisected)
        but wrong as documentation.

    python scripts/a51_cigarettes_da.py
"""

import os
import sys

import numpy as np
from loguru import logger

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from src.data_augmentors.cigarettes import ScaleTranslation  # noqa: E402
from src.experiments.utils.metrics import rho_hat, trace_S_over_k  # noqa: E402
from src.oracle import epsilon_star, recalibrated_da_epsilon  # noqa: E402
from src.sem.cigarettes import V, build_design, load_panel, restricted_fit  # noqa: E402

SPEC = "t3"
SEED = 0
AMPLITUDE_GRID = 2.0 ** np.linspace(-3, 3, 7)
TRS_ENDS = (0.9882, 0.7518)  # MEASURED, plan SS4; single-draw values
TRS_TOL = 2e-3
LAW_TOL = 1e-3
TRS_DRAWS = 32
EPSILON_TARGET = 0.5
STRENGTH_GRID = (0.25, 0.5, 1.0, 2.0)
FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAIL.append(name)


class Homogeneous:
    """The target alone: h_*(x) = x'b_r with v'b_r = 0. The SEM that carries it is
    a52's business; eps* needs nothing but `f`."""

    def __init__(self, b):
        self.W_XY = np.asarray(b, dtype=float).reshape(-1, 1)

    def f(self, X):
        return X @ self.W_XY


def setup():
    design = build_design(load_panel(), spec=SPEC)
    b_r, _ = restricted_fit(design)
    amplitude = float(np.std(design.X @ (V / np.linalg.norm(V))))
    np.random.seed(SEED)
    return design, Homogeneous(b_r), ScaleTranslation(V, std=amplitude), amplitude


def leg_i(design, da):
    print("(i) the augmentation moves X along v and nowhere else")
    check("(i) param_dimension == 1", da.param_dimension == 1, f"{da.param_dimension}")
    np.random.seed(SEED)
    GX, _ = da(design.X, scale=1.0)
    delta = GX - design.X
    along = delta @ da.direction
    residual = float(np.abs(delta - np.outer(along, da.direction)).max())
    check("(i) no component off span(v)", residual < 1e-12, f"max {residual:.2e}")
    check("(i) the DA actually moves X", float(np.std(along)) > 0.0, f"sd(c) {np.std(along):.4f}")


def leg_ii(sem, da, X):
    print("(ii) exactly invariant at strength 0")
    # read BEFORE it is set: a non-zero default is exactly what this catches
    check("(ii) built with strength 0", da.strength == 0.0, f"{da.strength}")
    da.strength = 0.0
    eps = epsilon_star(sem, da, X=X)
    check("(ii) eps* == 0", eps < 1e-12, f"{eps:.3e}")


def leg_iii(sem, da, X):
    print("(iii) the strength knob is monotone, linear and reachable")
    achieved = []
    for strength in STRENGTH_GRID:
        da.strength = strength
        np.random.seed(SEED + 1)
        achieved.append(epsilon_star(sem, da, X=X))
    print("      strength " + " ".join(f"{s:8.3f}" for s in STRENGTH_GRID))
    print("      eps*     " + " ".join(f"{e:8.4f}" for e in achieved))
    check("(iii) monotone in strength", bool(np.all(np.diff(achieved) > 0)), f"{np.round(achieved, 4)}")
    slope = np.array(achieved) / np.array(STRENGTH_GRID)
    spread = float(np.max(slope) / np.min(slope) - 1.0)
    check("(iii) linear to 1% over [0, 2]", spread < 0.01, f"slope spread {spread:.2e}")
    da.strength = 0.0
    try:
        landed = recalibrated_da_epsilon(sem, da, EPSILON_TARGET, X=X)
    except ValueError as error:
        landed = float("nan")
        print(f"      the bisection refused the target: {error}")
    check(
        f"(iii) bisection reaches eps* = {EPSILON_TARGET}",
        abs(landed - EPSILON_TARGET) < 1e-6,
        f"{landed:.8f} at strength {da.strength:.6g}",
    )
    da.strength = 0.0


def leg_iv(design, da, amplitude):
    print("(iv) Prop. 2 over the amplitude grid, against the analytic law")
    da.strength = 0.0
    direction = V / np.linalg.norm(V)
    precision = float(direction @ np.linalg.inv(design.Sigma) @ direction)
    print(f"      {'knob':>7s} {'tr(S)/k':>9s} {'analytic':>9s} {'gap':>10s} {'rho':>8s} {'rho tr(S)/k':>12s}")
    measured, analytic, expansion = [], [], []
    for knob in AMPLITUDE_GRID:
        pooled = []
        for draw in range(TRS_DRAWS):
            np.random.seed(SEED + 100 + draw)
            GX, _ = da(design.X, scale=float(knob))
            pooled.append(trace_S_over_k(design.X, GX))
        np.random.seed(SEED + 100)
        GX, _ = da(design.X, scale=float(knob))
        rho = rho_hat(design.X, GX, design.y, intercept=True)
        a = (knob * amplitude) ** 2 * precision
        law = 1.0 - (1.0 / design.k) * a / (1.0 + a)
        measured.append(float(np.mean(pooled)))
        analytic.append(law)
        expansion.append(rho * measured[-1])
        print(
            f"      {knob:7.3f} {measured[-1]:9.5f} {law:9.5f} {measured[-1] - law:+10.2e} "
            f"{rho:8.4f} {expansion[-1]:12.5f}"
        )
    check("(iv) tr(S)/k is monotone decreasing", bool(np.all(np.diff(measured) < 0)), f"{np.round(measured, 4)}")
    for end, want in zip((measured[0], measured[-1]), TRS_ENDS, strict=True):
        check(f"(iv) tr(S)/k end {want}", abs(end - want) < TRS_TOL, f"{end:.4f}")
    floor = (design.k - 1) / design.k
    check("(iv) saturates at (k-1)/k", measured[-1] > floor, f"{measured[-1]:.5f} vs {floor}")
    gap = float(np.max(np.abs(np.array(measured) - np.array(analytic))))
    check("(iv) matches the analytic law", gap < LAW_TOL, f"max gap {gap:.2e}, a from the configured v")
    check("(iv) rho tr(S)/k <= 1 everywhere", max(expansion) <= 1.0, f"max {max(expansion):.5f}")


def leg_v(da):
    print("(v) the perpendicular channel is unit-normed")
    norms = np.linalg.norm(da.W_perp, axis=1)
    check("(v) W_perp rows are unit norm", bool(np.allclose(norms, 1.0, atol=1e-12)), f"{np.round(norms, 4)}")
    check("(v) W_perp spans null(v)", float(np.abs(da.W_perp @ V).max()) < 1e-12, f"{np.abs(da.W_perp @ V).max():.2e}")
    check("(v) augmentation name", da.augmentation == "translate", f"{da.augmentation!r}")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    os.chdir(REPO)
    design, sem, da, amplitude = setup()
    print(f"{SPEC}: sd(X . v-hat) = {amplitude:.6f}, the DA amplitude at da_amplitude = 1")
    leg_i(design, da)
    leg_ii(sem, da, design.X)
    leg_iii(sem, da, design.X)
    leg_iv(design, da, amplitude)
    leg_v(da)
    if FAIL:
        print(f"A51 FAIL: {FAIL}")
        sys.exit(1)
    print("A51 PASS")
