"""A69: the moment-constrained point estimate, ERM+IV / DA+ERM+IV.

SS6 retires the 2SLS point estimates `IV` and `DA+IV` and replaces them with the ERM
that minimises `||yc - Xc h||` subject to the IV moment pinned at its attainable
floor, `A h == P b` with `A = Q' Xc`, `b = Q' yc` and `Q` a rank-revealing basis of
the centred instrument. That is (P2) at `gamma_z = 0`: an affine feasible set, always
non-empty, no tolerance constant. Legs:

  (i)    moment orthogonality at three data scales. The normalised residual
         `||Zc' r|| / (||Zc||_F ||yc||)` is under 1e-6 at scale 1e-3, 1 and 1e3.
         Catches: the repo's `_jittered` ridge stacked onto (A, b), and a slack ball
         `m* + 1e-6` in place of the equality -- MEASURED 3.4e-06 and 6.5e-06, both
         at scale 1e-3 ALONE, which is why the sweep and not the threshold is the
         leg. NOT a break: leaving Z uncentred while (X, y) are centred. A centred
         residual is mean-zero, so `Z' r == Zc' r` identically and the leg reads
         2.3e-16 either way; the convention still matters to the ESTIMATE, which
         (iii) and (vi) pin. Misses: WHICH feasible point was returned -- that
         is (ii).
  (ii)   ERM optimality within the moment set: 200 normalised directions of ker(A),
         stepped both ways at 1e-3 ||h||, none beats the fit. Catches: a solve that
         returns any feasible point, which is exactly what 2SLS does on an
         under-identified instrument.
  (iii)  degeneracy to 2SLS where d_z == d_h == 5: the moment system is square and
         the two estimators agree to 1e-8 relative, both backends pinned.
  (iv)   over-identified feasibility: d_z = d_h + 4, the exact-IV set is empty, the
         fit is finite and the residual sits ON the floor to 1e-9. The break is
         demonstrated rather than asserted: the same program on `A h == P b` is
         `optimal` and on `A h == b`, the equality with `P b` dropped, INFEASIBLE.
  (v)    an empty instrument is plain ERM BIT FOR BIT: the early branch and
         LeastSquaresClosedForm are both `pinv`, so 0.0 and not a tolerance. The
         conic path agrees only to 6.5e-16, so `== 0.0` is what makes a missing
         branch detectable. Everything is read off the object that was BUILT, never
         off the return of `.fit`: a branch written as
         `return LeastSquaresClosedForm(...).fit(...)` hands back a populated
         stranger while leaving `_W` / `_mu` / `_offset` unset on the object every
         live caller keeps, and reading the return would miss it.
  (vi)   `mean_match`: the fitted line matches the outcome mean to 1e-8 relative.
  (vii)  `perf.set_backend` reaches the class: a pin on a built CONIC_FITS model
         changes `.backend`. Catches the silent one -- a missing isinstance arm
         leaves every backend reporting the same numbers.
  (viii) style: every point estimate is dashed and every interval solid, with and
         without a real Z (the ERMs are point estimates whatever their instrument;
         a real Z shows in the label, never in the line). FAILED before SS6 on
         `DA+IV(Z)`, which was drawn both ways.
  (ix)   hue on literals and the label pooling: the three mode spellings share one
         label with and without a real Z, and nothing else does under one.
         DA+ERM+IV(Z) on deep 3 (red) is the load-bearing one -- a (Z) spelling
         aliased onto its green base would draw a red method green.
  (x)    the rank-revealing basis, the ONLY leg that guards it. Two assertions run
         against the CLASS: its fit on a duplicated instrument column must stay
         ERM-optimal over the TRUE kernel of the rank cut, and must equal the fit
         without that column to 1e-9. Under a plain QR the fabricated direction is
         forced to zero by the equality, one degree of freedom is lost and the
         coefficient misses by O(1). Three further checks are FIXTURE SANITY on this
         script's own basis copy and pass under that mutation, so they are labelled
         as such. Legs (i) and (ii) both PASS against a plain QR too, so without
         this leg the requirement ships unguarded.
  (xi)   the legend key: in one column with a real Z six handles draw five rows --
         erm, erm+iv, erm~, erm~+iv, erm~+iv~ on hues 0, 0, 3, 3, 2, the two green
         mode spellings pooled onto one and the five labels pairwise distinct; in a
         merged grid (ERM and DA+ERM from a null-Z column, ERM+IV, DA+ERM+IV(Z) and
         DA+ERM+IV(T,Z) from a Z column) three rows, erm+iv, erm~+iv, erm~+iv~.
         Catches the render fold not happening (six rows), a label aliased onto
         another, which collapses the set to four or three, and a merge that keeps a
         null-Z column's own entries.

Usage:
    MPLBACKEND=Agg python scripts/a69_erm_iv.py [--only LEG]
"""

import argparse
import os
import sys

import cvxpy as cp
import numpy as np
from loguru import logger

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402

import src.experiments.perf as perf  # noqa: E402
from src.aggregate import _legend  # noqa: E402
from src.experiments.configs import ALL_METHODS, DOMNIST_ONLY_METHODS, EPS_TOL, MethodRegistry  # noqa: E402
from src.experiments.utils.constants import (  # noqa: E402
    DEEP_HEX,
    DEEP_INDEX,
    IV_MODE_METHODS,
    IV_MODES,
    PARTIAL_IDENTIFICATION_STYLE,
    POINT_ESTIMATE_STYLE,
    alpha,
    hue,
    is_interval,
    is_point_estimate,
    line_style,
    method_style,
    resolve,
)
from src.experiments.utils.constants import label as method_label  # noqa: E402
from src.methods.regression import (  # noqa: E402
    LeastSquaresClosedForm,
    MomentConstrainedLeastSquares,
    TwoStageLeastSquaresIV,
)

GAMMA = 0.25
TOGGLES = dict(recalibrate=True, clipy=False, mean_match=True, n_jobs=1)
FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAIL.append(name)


def fixture(n, d_h, d_z, seed=0):
    """A confounded linear design with a valid instrument. `d_z < d_h` is the
    under-identified regime the method exists for; `d_z > d_h` over-determines the
    moment system and empties the exact-IV set."""
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, d_z))
    U = rng.normal(size=n)
    X = Z @ rng.normal(size=(d_z, d_h)) + U[:, None] + rng.normal(size=(n, d_h))
    y = X @ rng.normal(size=d_h) + 2.0 * U + rng.normal(size=n)
    return X, y, Z


def centred(X, y, Z, fit_intercept=True):
    """The (Xc, yc, Zc) the estimator solves on, under its own convention."""
    if not fit_intercept:
        return np.asarray(X, float), np.asarray(y, float), np.asarray(Z, float)
    X, y, Z = np.asarray(X, float), np.asarray(y, float), np.asarray(Z, float)
    return X - X.mean(axis=0), y - float(np.mean(y)), Z - Z.mean(axis=0)


def basis(Zc):
    """The rank-revealing basis of the instrument span, the repo's own rank cut."""
    left, singular, _ = np.linalg.svd(Zc, full_matrices=False)
    keep = singular > max(float(singular[0]), 1.0) * 1e-12 if singular.size else singular.astype(bool)
    return left[:, keep]


def moment_residual(h, Xc, yc, Zc):
    """`||Zc' r|| / (||Zc||_F ||yc||)`: normalised, so the threshold cannot loosen
    with the data scale. An absolute one would."""
    r = yc - Xc @ h
    return float(np.linalg.norm(Zc.T @ r)) / (np.linalg.norm(Zc, "fro") * np.linalg.norm(yc))


def leg_i():
    print("(i) the moment is zeroed at three data scales")
    # the scale sweep IS the leg: a jitter ridge or a slack ball leaves the residual
    # under 1e-6 at scale 1 and 1e3 and only breaks out at 1e-3
    X, y, Z = fixture(128, 8, 3)
    for scale in (1e-3, 1.0, 1e3):
        model = MomentConstrainedLeastSquares(fit_intercept=True).fit(X=X * scale, y=y * scale, Z=Z)
        Xc, yc, Zc = centred(X * scale, y * scale, Z)
        got = moment_residual(model._W, Xc, yc, Zc)
        check(f"(i) scale {scale:g}: normalised moment residual under 1e-6", got < 1e-6, f"{got:.2e}")


def leg_ii():
    print("(ii) the fit is the ERM optimum WITHIN the moment set, not just feasible")
    X, y, Z = fixture(128, 8, 3)
    Xc, yc, Zc = centred(X, y, Z)
    A = basis(Zc).T @ Xc
    kernel = np.linalg.svd(A)[2][np.linalg.matrix_rank(A) :].T
    check("(ii) the fixture is under-identified, so ker(A) is non-trivial", kernel.shape[1] > 0, f"{kernel.shape}")
    # the new estimator, and 2SLS standing in as the broken one: any feasible point
    for label, h in (
        ("DA+ERM+IV", MomentConstrainedLeastSquares(fit_intercept=True).fit(X=X, y=y, Z=Z)._W),
        ("a 2SLS fit", TwoStageLeastSquaresIV(fit_intercept=True).fit(X=X, y=y, Z=Z)._W),
    ):
        objective = float(np.linalg.norm(yc - Xc @ h))
        rng = np.random.default_rng(0)
        worst = 1.0
        for _ in range(200):
            step = kernel @ rng.normal(size=kernel.shape[1])
            step /= np.linalg.norm(step)
            for t in (1e-3 * np.linalg.norm(h), -1e-3 * np.linalg.norm(h)):
                worst = min(worst, float(np.linalg.norm(yc - Xc @ (h + t * step))) / objective)
        if label == "a 2SLS fit":
            # the control: the leg is only a leg if it separates the two
            check("(ii) and 2SLS on the same set is NOT the optimum", worst < 1.0 - 1e-9, f"{worst:.9f}")
        else:
            check("(ii) no feasible step beats the fit", worst >= 1.0 - 1e-9, f"{worst:.9f}")


def leg_iii():
    print("(iii) exact identification degenerates to 2SLS")
    # BACKENDS[0] is CLARABEL's OWN default tolerance triple, so the pin is a no-op
    # on CLARABEL and buys cross-backend reproducibility, not accuracy. What 1e-8
    # actually has to survive is the fixture's conditioning: measured 1e-14 here,
    # 8.7e-10 at cond(X) ~ 1e6, so one order of headroom on a badly scaled design
    for fit_intercept in (False, True):
        for seed in (0, 1, 2):
            X, y, Z = fixture(400, 5, 5, seed=seed)
            new = MomentConstrainedLeastSquares(fit_intercept=fit_intercept)
            old = TwoStageLeastSquaresIV(fit_intercept=fit_intercept)
            for model in (new, old):
                perf.set_backend(model, perf.BACKENDS[0])
            new.fit(X=X, y=y, Z=Z)
            old.fit(X=X, y=y, Z=Z)
            scale = max(1.0, float(np.abs(old._W).max()))
            gap = float(np.abs(new._W - old._W).max()) / scale
            check(
                f"(iii) fit_intercept={fit_intercept}, seed {seed}: equals 2SLS to 1e-8 relative",
                gap < 1e-8,
                f"{gap:.2e}",
            )


def leg_iv():
    print("(iv) over-identification is feasible at the floor, and says so")
    d_h = 4
    X, y, Z = fixture(400, d_h, d_h + 4)
    model = MomentConstrainedLeastSquares(fit_intercept=True).fit(X=X, y=y, Z=Z)
    Xc, yc, Zc = centred(X, y, Z)
    Q = basis(Zc)
    A, b = Q.T @ Xc, Q.T @ yc
    m_star = float(np.linalg.norm(b - A @ (np.linalg.pinv(A) @ b)))
    empty_set = m_star > 1e-9 * max(float(np.linalg.norm(b)), 1.0)
    check("(iv) the exact-IV set really is empty here", empty_set, f"{m_star:.3e}")
    check("(iv) the coefficients are finite", bool(np.all(np.isfinite(model._W))))
    check("(iv) and it predicts finite", bool(np.all(np.isfinite(np.asarray(model.predict(X), dtype=float)))))
    gap = abs(float(np.linalg.norm(b - A @ model._W)) - m_star)
    check("(iv) the residual sits ON the floor to 1e-9", gap < 1e-9, f"{gap:.2e}")
    # the same program at the floor, and the one it becomes if `P b` is dropped and
    # the moment is asked to vanish outright. The first is optimal, the second is
    # INFEASIBLE, which is the break this leg exists for
    for label, rhs, want in (("A h == P b", A @ (np.linalg.pinv(A) @ b), True), ("A h == b", b, False)):
        h = cp.Variable(X.shape[1])
        prob = cp.Problem(cp.Minimize(cp.norm(yc - Xc @ h)), [A @ h == rhs])
        try:
            prob.solve(solver=cp.CLARABEL)
            status = prob.status
        except cp.SolverError as error:
            status = f"SolverError: {error}"
        check(f"(iv) the conic solve on `{label}` is optimal: {want}", (status == "optimal") is want, f"{status}")


def leg_v():
    print("(v) an empty instrument is plain ERM bit for bit")
    X, y, _ = fixture(256, 6, 3)
    empty = np.zeros((len(X), 0))
    for fit_intercept in (False, True):
        for scale in (1e-3, 1.0, 1e3):
            tag = f"(v) fit_intercept={fit_intercept}, scale {scale:g}"
            # EVERYTHING below reads the object that was BUILT, never the return of
            # `.fit`. `pointEstimator.fit` hands back whatever `_fit` returns, so a
            # branch written as `return LeastSquaresClosedForm(...).fit(...)` would
            # return a populated foreign estimator while leaving _W / _mu / _offset
            # unset on the object every live caller keeps (perf.py, model_fitting)
            model = MomentConstrainedLeastSquares(fit_intercept=fit_intercept)
            model.fit(X=X * scale, y=y * scale, Z=empty)
            erm = LeastSquaresClosedForm(fit_intercept=fit_intercept)
            erm.fit(X * scale, y * scale)
            try:
                predicted = np.asarray(model.predict(X * scale), dtype=float)
                coefficients = np.asarray(model._W, dtype=float)
                raised = None
            except AttributeError as error:  # a delegated branch leaves the object bare
                predicted, coefficients, raised = None, None, error
            check(f"{tag}: the BUILT object predicts", raised is None, f"{raised!r}" if raised else "")
            if raised is not None:
                continue
            gap = float(np.abs(coefficients - erm._W).max())
            # EXACTLY 0.0: both paths are pinv. The conic path agrees only to
            # 6.5e-16, so a 1e-12 threshold here would pass with the branch deleted
            check(f"{tag}: equals ERM to exactly 0.0", gap == 0.0, f"{gap:.3e}")
            gap = float(np.abs(predicted - np.asarray(erm.predict(X * scale), dtype=float)).max())
            check(f"{tag}: and predicts as ERM", gap == 0.0, f"{gap:.3e}")


def leg_vi():
    print("(vi) mean_match: the fitted line carries the outcome mean")
    X, y, Z = fixture(256, 6, 3)
    model = MomentConstrainedLeastSquares(fit_intercept=True).fit(X=X, y=y, Z=Z)
    gap = abs(float(np.sum(model.predict(X)) - np.sum(y))) / abs(float(np.sum(y)))
    check("(vi) sum predict(X) equals sum y to 1e-8 relative", gap < 1e-8, f"{gap:.2e}")
    # and it does NOT hold without the intercept, which is what makes (vi) a leg
    plain = MomentConstrainedLeastSquares(fit_intercept=False).fit(X=X, y=y, Z=Z)
    gap = abs(float(np.sum(plain.predict(X)) - np.sum(y))) / abs(float(np.sum(y)))
    check("(vi) and fit_intercept=False does not match the mean", gap > 1e-6, f"{gap:.2e}")


def leg_vii():
    print("(vii) the backend pin reaches the new class")
    check(
        "(vii) CONIC_FITS is the two new names (bookkeeping, transcribes perf.py)",
        set(perf.CONIC_FITS) == {"ERM+IV", "DA+ERM+IV"},
        f"{perf.CONIC_FITS}",
    )
    built = MethodRegistry.build_methods(
        list(perf.CONIC_FITS), gamma=GAMMA, epsilon=EPS_TOL, epsilon_iv=EPS_TOL, **TOGGLES
    )
    for name in perf.CONIC_FITS:
        model = built[name]()
        check(f"(vii) {name} builds with backend unset", model.backend is None, f"{model.backend!r}")
        perf.set_backend(model, perf.BACKENDS[0])
        check(f"(vii) set_backend pins {name}", model.backend == perf.BACKENDS[0], f"{model.backend!r}")


def leg_viii():
    print("(viii) the style channel: point estimates dashed, intervals solid")
    names = [
        spelling
        for base in ALL_METHODS + DOMNIST_ONLY_METHODS
        for spelling in ([base] + [f"{base}({m})" for m in IV_MODES] if base in IV_MODE_METHODS else [base])
    ]
    # a point estimate drawn like an interval's edge, or the reverse, reads as the
    # other kind. This FAILED before SS6: `DA+IV(Z)` was drawn both ways
    wrong = [
        (n, hz)
        for n in names
        for hz in (True, False)
        if line_style(n, hz) != (POINT_ESTIMATE_STYLE if is_point_estimate(n) else PARTIAL_IDENTIFICATION_STYLE)
    ]
    check("(viii) every point estimate dashed, every interval solid, with or without a Z", not wrong, f"{wrong}")
    solid = [n for n in names if is_interval(n) and resolve(n, True).side.iv_set == frozenset("Z")]
    check(
        "(viii) the four real-Z interval members are solid too",
        len(solid) == 4 and all(line_style(n, True) == PARTIAL_IDENTIFICATION_STYLE for n in solid),
        f"{solid}",
    )


def leg_ix():
    print("(ix) hue on literals, labels pooled on the mode spellings alone")
    # literals, not a comparison with DA+ERM's hue: the comparison form passes for
    # the wrong reason if the family's own hue is ever edited
    for name, want in (("ERM+IV", 0), ("DA+ERM+IV", 2), ("DA+ERM+IV(T)", 2), ("DA+ERM+IV(T,Z)", 2)):
        got = DEEP_INDEX[hue(name, True)]
        check(f"(ix) {name} is deep {want}", got == want, f"{got!r}")
    # the load-bearing one: a (Z) spelling aliased onto its green base would draw a
    # red method green
    got = DEEP_INDEX[hue("DA+ERM+IV(Z)", True)]
    check("(ix) DA+ERM+IV(Z) is deep 3", got == 3, f"{got!r}")
    for name in ("ERM+IV", "DA+ERM+IV", "DA+ERM+IV(Z)", "DA+ERM+IV(T)", "DA+ERM+IV(T,Z)"):
        check(f"(ix) {name} is solid (alpha 1)", alpha(name, True) == 1.0, f"{alpha(name, True)!r}")
    for has_z in (True, False):
        pooled = (
            method_label("DA+ERM+IV", has_z)
            == method_label("DA+ERM+IV(T)", has_z)
            == method_label("DA+ERM+IV(T,Z)", has_z)
        )
        check(f"(ix) the three mode spellings share one label, has_z {has_z}", pooled)
    # the first, second and last pairs share a hue, an alpha and a dash, so the
    # label is the ONLY thing keeping them apart in the legend -- ERM+IV against
    # ERM is blue, solid alpha, point-estimate dash on both sides
    for a, b in (
        ("ERM+IV", "ERM"),
        ("DA+ERM+IV(Z)", "DA+ERM"),
        ("DA+ERM+IV(Z)", "DA+ERM+IV"),
        ("DA+ERM+IV", "DA+ERM"),
        ("ATE", "DA+ERM"),
    ):
        check(f"(ix) with a Z, {a} is labelled apart from {b}", method_label(a, True) != method_label(b, True))


def leg_x():
    print("(x) the instrument basis is rank-revealing")
    X, y, Z = fixture(400, 6, 3)
    Z_dup = np.column_stack([Z, 2.0 * Z[:, :1]])
    _, _, Zc_dup = centred(X, y, Z_dup)
    rank = int(np.linalg.matrix_rank(Z_dup))
    check("(x) the fixture is rank deficient and under-identified after the cut", rank == 3 < X.shape[1], f"{rank}")
    # FIXTURE SANITY, not the leg: these four run against this script's own
    # `basis()` copy, so they establish that the fixture is the hazard 6.1.2 names
    # and that a plain QR would fabricate a fourth direction. They cannot see the
    # class, and they all PASS under the plain-QR mutation. The two assertions
    # below are the ones that guard `regression.py`
    Q = basis(Zc_dup)
    check(f"(x) fixture: an SVD basis keeps {rank} columns, not {Z_dup.shape[1]}", Q.shape[1] == rank, f"{Q.shape[1]}")
    projector = Q @ Q.T - Zc_dup @ np.linalg.pinv(Zc_dup)
    check("(x) fixture: and it IS the projector onto col(Zc)", float(np.abs(projector).max()) < 1e-9)
    plain = np.linalg.qr(Zc_dup)[0]
    check("(x) fixture: a plain QR would not be", plain.shape[1] != rank, f"{plain.shape[1]}")

    # (a), at the FIT level: the class's own feasible set must be the TRUE one, so
    # its answer has to be ERM-optimal over ker(Q_svd' Xc), which is 3-dimensional
    # here. Under a plain QR the class solves on a 2-dimensional subset of that, so
    # the fabricated direction strictly improves the objective and this fires
    dup_model = MomentConstrainedLeastSquares(fit_intercept=True)
    dup_model.fit(X=X, y=y, Z=Z_dup)
    Xc, yc, _ = centred(X, y, Z_dup)
    A = Q.T @ Xc
    kernel = np.linalg.svd(A)[2][np.linalg.matrix_rank(A) :].T
    check(
        f"(x) the true kernel is {X.shape[1] - rank}-dimensional, so a fabricated direction costs one",
        kernel.shape[1] == X.shape[1] - rank,
        f"{kernel.shape[1]}",
    )
    h = np.asarray(dup_model._W, dtype=float)
    objective = float(np.linalg.norm(yc - Xc @ h))
    rng = np.random.default_rng(0)
    worst = 1.0
    for _ in range(200):
        step = kernel @ rng.normal(size=kernel.shape[1])
        step /= np.linalg.norm(step)
        for t in (1e-3 * np.linalg.norm(h), -1e-3 * np.linalg.norm(h)):
            worst = min(worst, float(np.linalg.norm(yc - Xc @ (h + t * step))) / objective)
    check("(x) the fit is ERM-optimal over the TRUE kernel of the rank cut", worst >= 1.0 - 1e-9, f"{worst:.9f}")

    # (b): and the duplicate column must change nothing at all
    plain_fit = MomentConstrainedLeastSquares(fit_intercept=True)
    plain_fit.fit(X=X, y=y, Z=Z)
    scale = max(1.0, float(np.abs(plain_fit._W).max()))
    gap = float(np.abs(h - np.asarray(plain_fit._W, dtype=float)).max()) / scale
    check("(x) the duplicate column changes nothing to 1e-9 relative", gap < 1e-9, f"{gap:.2e}")


def drawn_rows(columns, merged):
    """(labels, colours, handle count) of `_legend` over [(names, has_z)] columns,
    each name styled as `_metric_grid` styles it."""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    handles = {}
    for names, has_z in columns:
        for name in names:
            style = method_style(name, has_z, merged=merged)
            (line,) = ax.plot([0, 1], [0, 1], color=style.colour, linestyle=style.linestyle)
            handles.setdefault(style.signature, (line, style, name))
    legend = _legend(fig, handles)
    texts = [text.get_text() for text in legend.get_texts()]
    colours = [to_rgb(handle.get_color()) for handle in legend.legend_handles]
    plt.close(fig)
    return texts, colours, sum(len(names) for names, _ in columns)


def leg_xi():
    print("(xi) the legend pools the mode spellings and nothing else")
    names = ["ERM", "ERM+IV", "DA+ERM", "DA+ERM+IV(Z)", "DA+ERM+IV(T)", "DA+ERM+IV(T,Z)"]
    texts, colours, n = drawn_rows([(names, True)], merged=False)
    # six handles, five rows: `_legend` folds entries that render as the same pixels,
    # which is the ONE pooling SS6.4 asks for. A label aliased onto another collapses
    # the set further -- to four with the (Z) label aliased, to three under a blanket alias
    entries = set(zip(texts, colours, strict=True))
    check("(xi) six handles draw five legend rows", len(texts) == 5, f"{len(texts)} of {n} handles")
    check("(xi) five distinct (label, hue) legend entries", len(entries) == 5, f"{len(entries)}")
    rows = ("ERM", "ERM+IV", "DA+ERM", "DA+ERM+IV(Z)", "DA+ERM+IV")
    want = [method_label(name, True) for name in rows]
    check("(xi) the rows are erm, erm+iv, erm~, erm~+iv, erm~+iv~ in that order", texts == want, f"{texts}")
    hues = [DEEP_INDEX[hue(name, True)] for name in rows]
    deep = [to_rgb(DEEP_HEX[name]) for name in ("blue", "blue", "red", "red", "green")]
    check("(xi) their hues are 0, 0, 3, 3, 2", hues == [0, 0, 3, 3, 2] and colours == deep, f"{hues}")
    green = {text for text, colour in entries if colour == to_rgb(DEEP_HEX["green"])}
    check("(xi) the (T) and (T,Z) spellings share the one green entry", len(green) == 1, f"{len(green)}")
    check("(xi) the five labels are pairwise distinct", len({text for text, _ in entries}) == 5, f"{len(set(texts))}")

    # a merged grid: the null-Z column's ERM and DA+ERM draw as their Z counterparts
    null, with_z = ["ERM", "DA+ERM"], ["ERM+IV", "DA+ERM+IV(Z)", "DA+ERM+IV(T,Z)"]
    texts, _, _ = drawn_rows([(null, False), (with_z, True)], merged=True)
    want = [method_label(name, True) for name in with_z]
    check("(xi) merged: three rows erm+iv, erm~+iv, erm~+iv~", texts == want, f"{texts}")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    os.chdir(REPO)
    os.environ.setdefault("MPLBACKEND", "Agg")
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
        ("xi", leg_xi),
    ]
    if args.only:
        legs = [(tag, leg) for tag, leg in legs if tag.lower() == args.only.lower()]
        if not legs:
            sys.exit(f"unknown leg {args.only!r}")
    for tag, leg in legs:
        try:
            leg()
        except Exception as error:  # a raise is a FAIL line, and the later legs still report
            check(f"({tag}) ran without raising", False, f"{type(error).__name__}: {error}")
    if not FAIL:
        print("A69 PASS")
    else:
        print(f"A69 FAIL: {FAIL}")
        sys.exit(1)
