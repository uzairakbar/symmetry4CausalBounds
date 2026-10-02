"""A57: the IV rows are solved right, and an empty block is exact.

A DA+ IV ball takes the translation amounts as `T` and the observed instrument as
`Z`; a fit with one block carries that block's SOC row, a fit with both the rows of
`IV_LAYOUT` (the joint row on span(T, Z)), each at
sqrt(s^2 (1 + gamma~) gamma_n(d; g / (1 + gamma~))) with g the declared `gamma_z`
(non-DA) or App. D's (eps / sigma~ + sqrt(gamma_z / rho))^2 (DA); the intersection
hands its baseline branch the observed Z alone. Everything is measured on the
cigarette design (t3, own-tax, sigma-normalised, n = 2450), four coefficient
queries, serial solves. Legs:

  (D)   the digest leg (scripts/digest_leg.py), as a56: one query panel and one
        gamma sweep step per dataset at the shipped configuration against the
        recorded reference, no tolerance. Catches: the `(n, 0)` spelling of an
        empty Z not being exactly `None`, a T row that is not G elementwise.
        Misses: anything at gamma_z != 0, which (iv) and (vii) cover.
  (i)   `Z=None` and `Z=zeros((n, 0))` reproduce PartialR2 to EXACTLY 0.0.
        Catches: an empty instrument that is not inert (a jitter block or a stray
        row reaching the problem), a block reading an (n, 0) array as present.
  (ii)  PI+INV+IV with an empty Z reproduces PI+INV to 0.0. Catches: the INV cone
        of the IV class differing from InvarianceConstrainedPartialR2's.
  (iii) a non-DA Z row at a zero leak (gamma_z 0) returns INFEASIBLE on every
        query with NaN bounds, never a silent [0, 0]: the jitter block makes
        ||Q'(y - Xh)||^2 + j||h||^2 <= 0 unsatisfiable. Catches: the jitter block
        dropped, a solver failure read as infeasible.
  (iv)  the phase-b set [tax_s, y, cpi] at gamma 0.25, gamma_z 2^-8, raw: the
        beta_pn intervals of PI+IV, PI+INV+IV, DA+PI+IV (T and Z, so the joint
        row) and the intersection's two branches, RECORDED to 1e-6, every solve
        OK (PI+IV and PI+INV+IV read the old r_Z = 0.0625 pins exactly: s = 1
        on this panel). Catches: any change to the IV or INV arithmetic, a Z that is not the
        FWL'd column, a branch handed the wrong block. Misses: a solver bump under
        1e-6.
  (v)   the rows on the fitted attributes: PI+IV carries ("z",) of width m, a T
        only DA fit ("t",) of width p, a T and Z DA fit the joint ("tz",) of
        width p + m (`IV_LAYOUT`); the intersection's baseline ("z",) and its DA
        branch ("tz",); each row's moments are Q'y for its own Q, bit for bit,
        and its intercept column Q'1; with Z=None the baseline carries no row and
        the DA branch exactly G's ("t",), array_equal to a zeros((n, 0)) fit and
        to a direct `T=G` fit. Catches: Z reaching the DA branch only, a row
        built on the wrong block, `None` reaching column_stack.
  (vi)  the radii: raw (alpha 0) a non-DA Z row is s sqrt(gamma_z) and a DA row
        eps + sigma~ sqrt(gamma_z / rho) (so a DA row with gamma_z 0 is eps),
        rho-aware; padded (alpha 0.05) every row is the formula at level
        alpha / 3 (B = 1 row) on n_eff, d the row's width, and above its raw
        value; a DA row moves with a predict-time epsilon; a fitted model logs
        one INFO line naming gamma_z and the row radii at its first solve, not
        at fit. Catches: the non-DA leak on a DA row, a level not split, the
        old allowance arithmetic, a line printed before rho is final.
  (vii) solver equivalence: PI+IV, DA+PI+IV and the intersection (raw and
        padded) solved through the class equal an independent cvxpy program
        written here from the raw arrays (centred design, the Lem. 2 ball, the
        jittered IV rows at the class's radii) to 1e-6 relative to the width,
        and the CLARABEL and ECOS backends agree to the same tolerance; padded,
        the written-out program carries the mean row's delta in the objective,
        the ball and every IV residual.
        Catches: a constraint the class builds differently from the program it
        states, a backend-dependent answer.

    MPLBACKEND=Agg python scripts/a57_iv_solvers.py [--seed 0] [--reference JSON] [--skip-digest]

`--skip-digest` drops leg (D), about five minutes; it exists for the break-it runs
of the other legs and is never used on the committed state.
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

import digest_leg  # noqa: E402

import src.methods.sensitivity_models as solvers  # noqa: E402
from src.data_augmentors.cigarettes import ScaleTranslation  # noqa: E402
from src.experiments.cigarettes import absorbed_rate  # noqa: E402
from src.experiments.configs import EPS_TOL  # noqa: E402
from src.methods.sensitivity_models import (  # noqa: E402
    GAMMA_N_ROWS,
    InstrumentalVariablePartialR2,
    IntersectedInstrumentalVariablePartialR2,
    InvarianceConstrainedInstrumentalVariablePartialR2,
    InvarianceConstrainedPartialR2,
    PartialR2,
    SolveStatus,
)
from src.methods.sensitivity_models import finite_sample_budget as fsb  # noqa: E402
from src.sem.cigarettes import TREATMENTS, CigaretteSEM, V, build_design  # noqa: E402

GAMMA = 0.25
GAMMA_Z = 2**-8
ALPHA = 0.05
COMMON = dict(clipy=False, mean_match=True, n_jobs=1, recalibrate=True, pad=False)
PHASE_B = ("tax_s", "y", "cpi")
# raw beta_pn, raw log units, one seed-0 DA draw, gamma 0.25, gamma_z 2^-8, eps
# EPS_TOL, raw program: RECORDED on the gamma_n rows (the non-DA Z row at
# s sqrt(gamma_z), the DA joint row at eps + sigma~ sqrt(gamma_z / rho))
RECORDED = {
    "PI+IV": (0.358594946, 1.658163776),
    "PI+INV+IV": (0.941185206, 1.658163802),
    "DA+PI+IV": (0.725170893, 1.668974822),
    "PI&DA+PI+IV baseline": (0.358594946, 1.658163776),
    "PI&DA+PI+IV DA branch": (0.725170893, 1.668974822),
}
INTERVAL_TOL = 1e-6
SOLVER_TOL = 1e-6
FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAIL.append(name)


def design_and_draw(seed):
    """The t3 / own-tax design, the phase-b instrument matrix and one DA draw."""
    design = build_design(CigaretteSEM.panel(), spec="t3", anchor="own-tax")
    # every ball charged the FWL controls, as the orchestrator builds them: s = 1
    COMMON["absorbed_rate"] = absorbed_rate(design)
    columns = {name: design.X[:, j] for j, name in enumerate(TREATMENTS)}
    Z = np.column_stack([design.Z[:, 0] if name == "tax_s" else columns[name] for name in PHASE_B])
    np.random.seed(seed)
    GX, G = ScaleTranslation(V, std=float(np.std(design.X @ (V / np.linalg.norm(V)))))(design.X)
    return design, Z, GX, G.reshape(len(G), -1)


def bounds(model, k=4, **kwargs):
    return np.asarray(model.predict(np.eye(k), gamma=GAMMA, **kwargs), dtype=float)


def fitted(model, **arrays):
    """`model.fit(**arrays)` and None, or None and the exception, so a raise is a
    FAIL line in the log and not a traceback that hides the later legs."""
    try:
        return model.fit(**arrays), None
    except Exception as error:  # reported by the leg
        return None, error


def models_at(design, Z, GX, G, alpha=0.0, gamma_z=GAMMA_Z):
    """PI+IV, PI+INV+IV, DA+PI+IV on both blocks and the intersection, fitted."""
    X, y = design.X, design.y
    kw = dict(gamma=GAMMA, epsilon=EPS_TOL, gamma_z=gamma_z, gamma_n_alpha=alpha, **COMMON)
    pi_iv = InstrumentalVariablePartialR2(**kw).fit(X=X, y=y, Z=Z)
    pi_inv_iv = InvarianceConstrainedInstrumentalVariablePartialR2(**kw).fit(X=X, y=y, GX=GX, Z=Z)
    da_pi_iv = InstrumentalVariablePartialR2(**kw).fit(X=GX, y=y, T=G, Z=Z, X_pre=X)
    da_pi_iv.rho = da_pi_iv.sigma_sq / PartialR2(gamma=GAMMA, **COMMON).fit(X, y).sigma_sq
    intersection = IntersectedInstrumentalVariablePartialR2(**kw).fit(X=X, y=y, GX=GX, G=G, Z=Z)
    return {
        "PI+IV": pi_iv,
        "PI+INV+IV": pi_inv_iv,
        "DA+PI+IV": da_pi_iv,
        "PI&DA+PI+IV baseline": intersection.baseline,
        "PI&DA+PI+IV DA branch": intersection.augmented,
        "PI&DA+PI+IV": intersection,
    }


def leg_d(reference):
    print("(D) the shipped configuration does not move: query panel and gamma sweep step, three datasets")
    for label, ok, detail in digest_leg.compare(REPO, reference):
        check(f"(D) {label}", ok, detail)


def leg_i(design):
    print("(i) an empty instrument is exactly PartialR2")
    X, y = design.X, design.y
    reference = bounds(PartialR2(gamma=GAMMA, **COMMON).fit(X, y))
    for tag, Z in (("None", None), ("zeros((n, 0))", np.zeros((design.n, 0)))):
        model, error = fitted(InstrumentalVariablePartialR2(gamma=GAMMA, **COMMON), X=X, y=y, Z=Z)
        check(f"(i) Z={tag}: fits", error is None, repr(error) if error else "")
        if error is not None:
            continue
        gap = float(np.abs(bounds(model) - reference).max())
        check(f"(i) Z={tag}: max |PI+IV - PI| == 0.0", gap == 0.0, f"{gap!r}")
        check(f"(i) Z={tag}: no instrument read, no row", not model._has_iv and model.rows == ())


def leg_ii(design, GX):
    print("(ii) PI+INV+IV with an empty instrument is exactly PI+INV")
    X, y = design.X, design.y
    reference = bounds(InvarianceConstrainedPartialR2(gamma=GAMMA, epsilon=EPS_TOL, **COMMON).fit(X, y, GX=GX))
    for tag, Z in (("None", None), ("zeros((n, 0))", np.zeros((design.n, 0)))):
        model = InvarianceConstrainedInstrumentalVariablePartialR2(gamma=GAMMA, epsilon=EPS_TOL, **COMMON)
        model, error = fitted(model, X=X, y=y, GX=GX, Z=Z)
        check(f"(ii) Z={tag}: fits", error is None, repr(error) if error else "")
        if error is not None:
            continue
        gap = float(np.abs(bounds(model) - reference).max())
        check(f"(ii) Z={tag}: max |PI+INV+IV - PI+INV| == 0.0", gap == 0.0, f"{gap!r}")


def leg_iii(design, Z):
    print("(iii) a non-DA Z row at a zero leak is INFEASIBLE, not a silent [0, 0]")
    model = InstrumentalVariablePartialR2(gamma=GAMMA, gamma_z=0.0, **COMMON)
    model, error = fitted(model, X=design.X, y=design.y, Z=Z)
    check("(iii) fits at a zero leak", error is None, repr(error) if error else "")
    if error is not None:
        return
    check("(iii) the Z radius is 0", model.iv_radius("z") == 0.0, repr(model.iv_radius("z")))
    out = bounds(model)
    status = np.asarray(model.query_status)
    check("(iii) every query INFEASIBLE", np.all(status == SolveStatus.INFEASIBLE), f"status {status.tolist()}")
    check("(iii) every bound NaN", np.all(np.isnan(out)), f"{out.tolist()}")


def leg_iv(design, Z, GX, G):
    print("(iv) the phase-b set at gamma 0.25, gamma_z 2^-8, raw: the RECORDED beta_pn intervals")
    scale = design.sigma
    models = models_at(design, Z, GX, G)
    for name, want in RECORDED.items():
        model = models[name]
        out = bounds(model)
        status = np.asarray(model.query_status)
        got = scale * out[2]
        gap = float(np.abs(got - np.asarray(want)).max())
        check(f"(iv) {name}: every solve OK", np.all(status == SolveStatus.OK), f"status {status.tolist()}")
        check(
            f"(iv) {name}: beta_pn [{want[0]:.6f}, {want[1]:.6f}] to 1e-6",
            gap < INTERVAL_TOL,
            f"got [{got[0]:.9f}, {got[1]:.9f}], gap {gap:.2e}",
        )


def _moments(block, y):
    Q = np.linalg.qr(block)[0]
    return Q.T @ (np.asarray(y).ravel() - np.mean(y)), Q.T @ np.ones(len(block))


def leg_v(design, Z, GX, G):
    print("(v) the rows on the fitted attributes")
    X, y, M = design.X, design.y, design.k
    m, p = Z.shape[1], G.shape[1]
    models = models_at(design, Z, GX, G)
    pi_iv, da, base, aug = (
        models["PI+IV"],
        models["DA+PI+IV"],
        models["PI&DA+PI+IV baseline"],
        models["PI&DA+PI+IV DA branch"],
    )
    tz = np.column_stack([G, Z])
    for name, model, row, block in (
        ("PI+IV", pi_iv, "z", Z),
        ("DA+PI+IV", da, "tz", tz),
        ("the intersection's baseline", base, "z", Z),
        ("the intersection's DA branch", aug, "tz", tz),
    ):
        A, b, intercept, d = model.iv_terms_[row]
        moments, ones = _moments(block, y)
        width = block.shape[1]
        check(
            f"(v) {name}: rows ({row!r},) of width {width}",
            model.rows == (row,) and d == width and A.shape == (width + M, M),
            f"{model.rows} {A.shape}",
        )
        check(f"(v) {name}: moments Q'y bit for bit", np.array_equal(b[:width], moments) and not b[width:].any())
        check(f"(v) {name}: intercept column Q'1", np.allclose(intercept[:width], ones, atol=1e-12))
    t_only = InstrumentalVariablePartialR2(gamma=GAMMA, epsilon=EPS_TOL, **COMMON).fit(X=GX, y=y, T=G, X_pre=X)
    check(f"(v) a T-only DA fit: rows ('t',) of width {p}", t_only.rows == ("t",) and t_only.iv_terms_["t"][3] == p)
    check(f"(v) the IV_LAYOUT of a T and Z fit is the joint row, width {p} + {m}", solvers.IV_LAYOUT == ("tz",))

    common = dict(gamma=GAMMA, epsilon=EPS_TOL, gamma_z=GAMMA_Z, **COMMON)
    empty = IntersectedInstrumentalVariablePartialR2(**common).fit(X=X, y=y, GX=GX, G=G, Z=None)
    zeros = IntersectedInstrumentalVariablePartialR2(**common).fit(X=X, y=y, GX=GX, G=G, Z=np.zeros((design.n, 0)))
    direct = InstrumentalVariablePartialR2(**common).fit(X=GX, y=y, T=G, X_pre=X)
    check("(v) Z=None: the baseline carries no row", not empty.baseline._has_iv and empty.baseline.rows == ())
    check("(v) Z=None: the DA branch carries exactly G's ('t',) row", empty.augmented.rows == ("t",))
    for tag, other in (("zeros((n, 0))", zeros.augmented), ("a direct T=G fit", direct)):
        same = all(
            np.array_equal(left, right)
            for left, right in zip(empty.augmented.iv_terms_["t"][:3], other.iv_terms_["t"][:3], strict=True)
        )
        check(f"(v) Z=None DA arrays array_equal to {tag}", same)
    check("(v) Z=None and zeros((n, 0)) baselines agree", not zeros.baseline._has_iv)


def leg_vi(design, Z, GX, G):
    print("(vi) the radii")
    X, y = design.X, design.y
    raw = models_at(design, Z, GX, G)
    pi_iv, da = raw["PI+IV"], raw["DA+PI+IV"]
    want = float(np.sqrt(pi_iv.sigma_sq * GAMMA_Z))
    got = pi_iv.iv_radius("z")
    check("(vi) raw non-DA Z row: s sqrt(gamma_z)", abs(got - want) <= 1e-12 * want, f"{got:.9f} vs {want:.9f}")
    want = EPS_TOL + da.scale * np.sqrt(GAMMA_Z / da.rho)
    got = da.iv_radius("tz")
    check("(vi) raw DA joint row: eps + sigma~ sqrt(gamma_z / rho)", abs(got - want) <= 1e-12 * want, f"{got:.9f}")
    no_z = InstrumentalVariablePartialR2(gamma=GAMMA, epsilon=EPS_TOL, gamma_z=0.0, **COMMON)
    no_z.fit(X=GX, y=y, T=G, X_pre=X)
    got = no_z.iv_radius("t")
    check("(vi) raw DA T row at gamma_z 0 is eps", abs(got - EPS_TOL) <= 1e-12, f"{got!r}")
    rho_before = da.iv_radius("tz")
    da.rho = 4.0 * da.rho
    want = EPS_TOL + da.scale * np.sqrt(GAMMA_Z / da.rho)
    check("(vi) the DA row is rho-aware", abs(da.iv_radius("tz") - want) <= 1e-12 * want and want < rho_before)

    padded = models_at(design, Z, GX, G, alpha=ALPHA)
    for name in ("PI+IV", "PI+INV+IV", "DA+PI+IV", "PI&DA+PI+IV baseline", "PI&DA+PI+IV DA branch"):
        model = padded[name]
        (row,) = model.rows
        d = model.iv_terms_[row][3]
        budget = model.budget(GAMMA)
        g = GAMMA_Z if not model._da_fit else (EPS_TOL / model.scale + np.sqrt(GAMMA_Z / model.rho)) ** 2
        level = ALPHA / (GAMMA_N_ROWS * 1)
        want = float(np.sqrt(model.sigma_sq * (1 + budget) * fsb(d, g / (1 + budget), model.n_eff, level)))
        got = model.iv_radius(row)
        population = float(np.sqrt(model.sigma_sq * g))
        check(
            f"(vi) padded {name}: the row formula at alpha / 3 on n_eff {model.n_eff}, d {d}, above raw",
            abs(got - want) <= 1e-12 * want and got > population,
            f"{got:.6f} (raw {population:.6f})",
        )
    da = padded["DA+PI+IV"]
    before = da.iv_radius("tz")
    bounds(da, epsilon=2 * EPS_TOL)
    check("(vi) a DA row moves with a predict-time epsilon", da.iv_radius("tz") > before, f"{before:.6f}")

    # the line fires at the first solve, after rho is final, once per fitted model
    records = []
    marker = "IV: gamma_z="
    sink = logger.add(lambda message: records.append(message.record), level="DEBUG")
    try:
        logged = InstrumentalVariablePartialR2(gamma=GAMMA, epsilon=EPS_TOL, gamma_z=GAMMA_Z, **COMMON)
        logged.fit(X=X, y=y, Z=Z)
        at_fit = sum(marker in r["message"] for r in records)
        bounds(logged)
        bounds(logged)
    finally:
        logger.remove(sink)
    lines = [r for r in records if marker in r["message"]]
    check("(vi) nothing logged at fit, rho is not final there", at_fit == 0, f"{at_fit} lines")
    check("(vi) a fitted IV model logs its rows once over two solves", len(lines) == 1, f"{len(lines)} lines")
    message = lines[0]["message"] if lines else ""
    check("(vi) at INFO, not WARNING", bool(lines) and all(r["level"].name == "INFO" for r in lines), message)
    printed = float(message.split("r_Z=")[1].split(" ")[0].rstrip(".")) if "r_Z=" in message else np.nan
    check(
        "(vi) the line names gamma_z and the Z row's radius",
        f"gamma_z={GAMMA_Z:g}" in message and abs(printed - logged.iv_radius("z")) < 1e-5,
        message,
    )


def reference_bounds(model, X, y, blocks, queries):
    """The program written out from the raw arrays: max / min x_c'h + ybar over the
    Lem. 2 ball || X_c (h - h_erm) || <= sqrt(N) r and, per IV row,
    || (Q'(y_c - X_c h), sqrt(j) h) || <= sqrt(N) r_row, j = 1e-6 mean |Q'X_c|,
    at the class's radii (whose formula (vi) pins). Padded, the mean row's delta
    joins the objective, the ball (|| (X_c (h - h_erm), sqrt(N) delta) ||) and every
    IV residual (less delta Q'1), with |delta| <= the mean radius."""
    N, M = X.shape
    mu, ybar = X.mean(axis=0), float(np.mean(y))
    Xc, yc = X - mu, np.asarray(y).ravel() - ybar
    h_erm = np.linalg.lstsq(Xc, yc, rcond=None)[0]
    h, delta = cp.Variable(M), cp.Variable()
    padded = model.gamma_n_alpha > 0.0
    offset = Xc @ (h - h_erm)
    ball = cp.hstack([offset, np.sqrt(N) * cp.reshape(delta, (1,), order="F")]) if padded else offset
    constraints = [cp.norm(ball, 2) <= np.sqrt(N) * model.ball_radius(GAMMA)]
    constraints += [cp.abs(delta) <= model.mean_radius(GAMMA)] if padded else [delta == 0]
    for row in model.rows:
        Q = np.linalg.qr(blocks[row])[0]
        A = Q.T @ Xc
        jitter = 1e-6 * np.mean(np.abs(A))
        jitter = 1e-6 if jitter < 1e-9 else jitter
        residual = cp.hstack([Q.T @ yc - A @ h - (Q.T @ np.ones(N)) * delta, -np.sqrt(jitter) * h])
        constraints.append(cp.norm(residual, 2) <= np.sqrt(N) * model.iv_radius(row, GAMMA))
    out = []
    for x in queries - mu:
        lo = cp.Problem(cp.Minimize(x @ h + delta), constraints)
        lo.solve(solver=cp.CLARABEL)
        hi = cp.Problem(cp.Maximize(x @ h + delta), constraints)
        hi.solve(solver=cp.CLARABEL)
        out.append((lo.value + ybar, hi.value + ybar))
    return np.asarray(out)


def leg_vii(design, Z, GX, G):
    print("(vii) solver equivalence: the class against the program written out, and across backends")
    X, y = design.X, design.y
    queries = np.eye(design.k)
    for alpha in (0.0, ALPHA):
        models = models_at(design, Z, GX, G, alpha=alpha)
        for name, design_x, blocks in (
            ("PI+IV", X, {"z": Z}),
            ("DA+PI+IV", GX, {"tz": np.column_stack([G, Z])}),
            ("PI&DA+PI+IV baseline", X, {"z": Z}),
            ("PI&DA+PI+IV DA branch", GX, {"tz": np.column_stack([G, Z])}),
        ):
            model = models[name]
            got = bounds(model)
            want = reference_bounds(model, design_x, y, blocks, queries)
            width = float(np.max(want[:, 1] - want[:, 0]))
            gap = float(np.max(np.abs(got - want))) / width
            check(f"(vii) alpha {alpha:g} {name}: class == the program written out", gap < SOLVER_TOL, f"{gap:.1e}")
            per_backend = []
            for backend in ((cp.CLARABEL, {}), (cp.ECOS, {})):
                model.backend = backend
                per_backend.append(bounds(model))
            model.backend = None
            gap = float(np.max(np.abs(per_backend[0] - per_backend[1]))) / width
            check(f"(vii) alpha {alpha:g} {name}: CLARABEL == ECOS", gap < SOLVER_TOL, f"{gap:.1e}")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    os.chdir(REPO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reference", default=digest_leg.DEFAULT_REFERENCE)
    parser.add_argument("--skip-digest", action="store_true")
    args = parser.parse_args()
    print(f"tree: {solvers.__file__}")
    design, Z, GX, G = design_and_draw(args.seed)
    for tag, leg in (
        ("(i)", lambda: leg_i(design)),
        ("(ii)", lambda: leg_ii(design, GX)),
        ("(iii)", lambda: leg_iii(design, Z)),
        ("(iv)", lambda: leg_iv(design, Z, GX, G)),
        ("(v)", lambda: leg_v(design, Z, GX, G)),
        ("(vi)", lambda: leg_vi(design, Z, GX, G)),
        ("(vii)", lambda: leg_vii(design, Z, GX, G)),
    ):
        try:
            leg()
        except Exception as error:  # a raise is a FAIL line, and the later legs still report
            check(f"{tag} ran without raising", False, f"{type(error).__name__}: {error}")
    if not args.skip_digest:
        leg_d(args.reference)
    else:
        print("(D) SKIPPED by --skip-digest: a break-it run, not the committed state")
    if not FAIL:
        print("A57 PASS")
    else:
        print(f"A57 FAIL: {FAIL}")
        sys.exit(1)
