"""A57: the solvers see a real instrument: empty is exact, the budget is root sum square.

refactor2 touches sensitivity_models.py only: the intersection hands its baseline
branch the real Z and its DA branch Z-tilde = (T, Z), and `iv_bound` becomes
sqrt(epsilon_iv^2 + s^2 gamma_z). Everything is measured on the cigarette design
(t3, own-tax, sigma-normalised, n = 2450), four coefficient queries, serial solves.
Legs:

  (D)   the digest leg (scripts/digest_leg.py), as a56: one query panel and one
        gamma sweep step per dataset at the shipped configuration against the
        reference recorded on 14509db, no tolerance. Catches: the `(n, 0)` spelling
        of an empty Z not being exactly today's `None`, a Z-tilde that is not G
        elementwise, an `iv_bound` that moved at gamma_z = 0. Misses: anything at
        gamma_z != 0, which is why (vi) exists.
  (i)   `Z=None` and `Z=zeros((n, 0))` reproduce PartialR2 to EXACTLY 0.0
        (MEASURED p7). Catches: an empty instrument that is not inert (a jitter
        block or a stray constraint reaching the problem), `_has_iv` reading an
        (n, 0) array as present. Misses: an empty Z on a DA design, which (v)
        covers through the intersection.
  (ii)  PI+INV+IV with an empty Z reproduces PI+INV to 0.0. Catches: the INV cone
        of the new class differing from InvarianceConstrainedPartialR2's (a wrong
        GX, a missing centring, `eps_param` never set). Misses: the IV cone, (iv).
  (iii) `iv_bound = 0` (epsilon_iv 0, gamma_z 0) returns INFEASIBLE on every query
        with NaN bounds, never a silent [0, 0] (MEASURED p3): the jitter block makes
        ||Q'(y - Xh)||^2 + j||h||^2 <= 0 unsatisfiable. Catches: the jitter block
        dropped (three moments in four unknowns are then exactly solvable inside
        the gamma = 0.25 ball and the status flips to OK), a solver failure read
        as infeasible. Misses: a positive bound the ball cannot reach (a61-ii).
  (iv)  the phase-b set [tax_s, y, cpi] at gamma 0.25, IV bound 0.0625 reproduces
        the beta_pn intervals measured at 14509db with the plan's p8 prototype
        (logs/batchA/a57_reference_probe_14509db.log) to 1e-6: PI+IV [0.374940,
        1.644600], PI+INV+IV [0.951673, 1.644600], DA+PI+IV on Z-tilde [1.149780,
        1.648402], raw log units, SS4.1's [0.375, 1.645], [0.952, 1.645],
        [1.150, 1.648]; every solve OK. The two IV rows are read off the standalone
        classes AND off the intersection's two branches, the latter at epsilon_iv 0
        and gamma_z 2^-8, where both branch bounds read 0.0625 on this sigma-
        normalised panel (s = 1). Catches: any change to the IV or INV arithmetic,
        a Z that is not the FWL'd column (p7: +7 moves beta_pn by 0.004), a branch
        handed the wrong instrument. Misses: a solver bump under 1e-6, which is
        why the tolerance is not 1e-9.
  (v)   the intersection's baseline branch is fitted with Z and its DA branch with
        Z-tilde = column_stack([G, Z]), read off the fitted attributes: the rows
        of `Z_projector_R` above M count the instrument (m, then p + m) and
        `y_residual_base` is Q'y for the right Q, bit for bit, so [Z, G] or a
        dropped G fails. With Z=None the baseline reads no instrument, the DA
        branch reads exactly G, and its arrays are array_equal to a fit with
        zeros((n, 0)) and to today's direct `Z=G` fit. Catches: Z passed to the
        DA branch only, Z-tilde in the wrong order, `None` reaching column_stack.
        Misses: the numbers, (iv).
  (vi)  `iv_bound` is the root sum square: bit-equal to `epsilon_iv` at gamma_z 0
        for several budgets (sqrt(e^2) is exact in binary arithmetic, so leg (D)
        cannot move), 0.069877 to 1e-9 at epsilon_iv 2^-5, gamma_z 2^-8, s 1,
        strictly below the plain sum 0.09375, and rho-aware (s is the pre-DA
        sigma). Then at fit: PI+IV at epsilon_iv 0, gamma_z 2^-8 reproduces the
        epsilon_iv 0.0625, gamma_z 0 interval to 1e-6 (p10 E), and a gamma_z
        model logs one INFO line, not WARNING, naming gamma_z, epsilon_iv, s and
        the joint bound, at its first solve (rho is final there) and not at fit.
        Catches: the plain sum put back (the DA bound would read 0.09375 and (D)
        would not notice, since it runs at gamma_z 0), s read post-DA, the old
        warning, a line printed before an intersection's DA branch knows its rho.
        Misses: the 0.7% overlap of span(T) and span(Z) the plan states (SS2.6),
        which a60 records with G and Z held apart.
  (vii) the intersection budgets per branch (SS2.6, the coordinator's ruling): the
        baseline branch gets epsilon_iv 0.0, so at epsilon_iv 2^-5, gamma_z 2^-8,
        s = 1 its bound is exactly r_Z = 0.0625 while the DA branch's is the joint
        0.069877, both to 1e-6; the intersection's own epsilon_iv is untouched;
        the DA branch's rho is set after fit; under an empty Z the baseline has no
        IV constraint at all (so leg (D) cannot move) and the DA branch keeps the
        joint bound on T. Catches: both branches handed the same budget (the
        baseline would read 0.069877), the T-as-IV term leaking into the baseline.
        Misses: the numbers under a real Z at those two budgets (a60/a61).

    MPLBACKEND=Agg python scripts/a57_iv_solvers.py [--seed 0] [--reference JSON] [--skip-digest]

`--skip-digest` drops leg (D), about five minutes; it exists for the break-it runs
of the other legs and is never used on the committed state.
"""

import argparse
import os
import sys

import numpy as np
from loguru import logger

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import digest_leg  # noqa: E402

import src.methods.sensitivity_models as solvers  # noqa: E402
from src.data_augmentors.cigarettes import ScaleTranslation  # noqa: E402
from src.experiments.configs import EPS_TOL  # noqa: E402
from src.methods.sensitivity_models import (  # noqa: E402
    InstrumentalVariablePartialR2,
    IntersectedInstrumentalVariablePartialR2,
    InvarianceConstrainedInstrumentalVariablePartialR2,
    InvarianceConstrainedPartialR2,
    PartialR2,
    SolveStatus,
)
from src.sem.cigarettes import TREATMENTS, CigaretteSEM, V, build_design  # noqa: E402

GAMMA = 0.25
IV_BOUND = 0.0625
GAMMA_Z = 2**-8
COMMON = dict(clipy=False, mean_match=True, n_jobs=1, recalibrate=True, pad=False)
PHASE_B = ("tax_s", "y", "cpi")
# raw beta_pn at 14509db with the p8 prototype, one seed-0 DA draw (the probe log
# beside this batch's report); the plan's SS4.1 digits in the labels
REFERENCE = {
    "PI+IV": (0.37494044773078467, 1.6446003962347882),
    "PI+INV+IV": (0.9516730404805537, 1.6446004352912316),
    "DA+PI+IV": (1.1497795786532348, 1.6484022876967956),
}
PLAN_DIGITS = {"PI+IV": "[0.375, 1.645]", "PI+INV+IV": "[0.952, 1.645]", "DA+PI+IV": "[1.150, 1.648]"}
INTERVAL_TOL = 1e-6
RSS_BOUND = 0.069877  # sqrt(0.03125^2 + 0.0625^2)
PLAIN_SUM = 0.09375
BUDGETS = (0.03125, 0.0625, 0.1, 0.017749, 0.3, 1e-3, 0.7071067811865476)
FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAIL.append(name)


def design_and_draw(seed):
    """The t3 / own-tax design, the phase-b instrument matrix and one DA draw."""
    design = build_design(CigaretteSEM.panel(), spec="t3", anchor="own-tax")
    columns = {name: design.X[:, j] for j, name in enumerate(TREATMENTS)}
    Z = np.column_stack([design.Z[:, 0] if name == "tax_s" else columns[name] for name in PHASE_B])
    np.random.seed(seed)
    GX, G = ScaleTranslation(V, std=float(np.std(design.X @ (V / np.linalg.norm(V)))))(design.X)
    return design, Z, GX, G


def bounds(model, k=4):
    return np.asarray(model.predict(np.eye(k), gamma=GAMMA), dtype=float)


def fitted(model, **arrays):
    """`model.fit(**arrays)` and None, or None and the exception, so a raise is a
    FAIL line in the log and not a traceback that hides the later legs."""
    try:
        return model.fit(**arrays), None
    except Exception as error:  # reported by the leg
        return None, error


def leg_d(reference):
    print("(D) the shipped configuration does not move: query panel and gamma sweep step, three datasets")
    for label, ok, detail in digest_leg.compare(REPO, reference):
        check(f"(D) {label}", ok, detail)


def leg_i(design):
    print("(i) an empty instrument is exactly PartialR2")
    X, y = design.X, design.y
    reference = bounds(PartialR2(gamma=GAMMA, **COMMON).fit(X, y))
    for tag, Z in (("None", None), ("zeros((n, 0))", np.zeros((design.n, 0)))):
        model, error = fitted(InstrumentalVariablePartialR2(gamma=GAMMA, epsilon_iv=None, **COMMON), X=X, y=y, Z=Z)
        check(f"(i) Z={tag}: fits", error is None, repr(error) if error else "")
        if error is not None:
            continue
        gap = float(np.abs(bounds(model) - reference).max())
        check(f"(i) Z={tag}: max |PI+IV - PI| == 0.0", gap == 0.0, f"{gap!r}")
        check(f"(i) Z={tag}: no instrument read", not model._has_iv)


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
    print("(iii) a zero IV bound is INFEASIBLE, not a silent [0, 0]")
    model = InstrumentalVariablePartialR2(gamma=GAMMA, epsilon_iv=0.0, gamma_z=0.0, **COMMON)
    model, error = fitted(model, X=design.X, y=design.y, Z=Z)
    check("(iii) fits at a zero bound", error is None, repr(error) if error else "")
    if error is not None:
        return
    check("(iii) iv_bound is 0", model.iv_bound == 0.0, repr(model.iv_bound))
    out = bounds(model)
    status = np.asarray(model.query_status)
    check("(iii) every query INFEASIBLE", np.all(status == SolveStatus.INFEASIBLE), f"status {status.tolist()}")
    check("(iii) every bound NaN", np.all(np.isnan(out)), f"{out.tolist()}")


def leg_iv(design, Z, GX, G):
    print("(iv) the phase-b set reproduces the 14509db intervals at gamma 0.25, IV bound 0.0625")
    X, y, scale = design.X, design.y, design.sigma
    pi_iv = InstrumentalVariablePartialR2(gamma=GAMMA, epsilon_iv=IV_BOUND, **COMMON).fit(X=X, y=y, Z=Z)
    pi_inv_iv = InvarianceConstrainedInstrumentalVariablePartialR2(
        gamma=GAMMA, epsilon=EPS_TOL, epsilon_iv=IV_BOUND, **COMMON
    )
    pi_inv_iv.fit(X=X, y=y, GX=GX, Z=Z)
    da_pi_iv = InstrumentalVariablePartialR2(gamma=GAMMA, epsilon_iv=IV_BOUND, **COMMON)
    da_pi_iv.fit(X=GX, y=y, Z=np.column_stack([G, Z]))
    da_pi_iv.rho = da_pi_iv.sigma_sq / PartialR2(gamma=GAMMA, **COMMON).fit(X, y).sigma_sq
    # the intersection budgets per branch (SS2.6): the baseline at r_Z = s sqrt(gamma_z)
    # with epsilon_iv 0, the DA branch at the root sum square. With epsilon_iv 0 and
    # gamma_z 2^-8 both read 0.0625 on this sigma-normalised panel (s = 1), the
    # setting the reference rows were measured at
    intersection = IntersectedInstrumentalVariablePartialR2(
        gamma=GAMMA, epsilon=EPS_TOL, epsilon_iv=0.0, gamma_z=GAMMA_Z, **COMMON
    )
    intersection.fit(X=X, y=y, GX=GX, G=G, Z=Z)
    models = {
        "PI+IV": pi_iv,
        "PI+INV+IV": pi_inv_iv,
        "DA+PI+IV": da_pi_iv,
        "PI&DA+PI+IV baseline at r_Z (PI+IV)": intersection.baseline,
        "PI&DA+PI+IV DA branch at the joint bound (DA+PI+IV)": intersection.augmented,
    }
    for name, model in models.items():
        key = name.split("(")[-1].rstrip(")") if "(" in name else name
        out = bounds(model)
        status = np.asarray(model.query_status)
        got = scale * out[2]
        want = np.asarray(REFERENCE[key])
        gap = float(np.abs(got - want).max())
        check(f"(iv) {name}: every solve OK", np.all(status == SolveStatus.OK), f"status {status.tolist()}")
        check(
            f"(iv) {name}: beta_pn {PLAN_DIGITS[key]} to 1e-6",
            gap < INTERVAL_TOL,
            f"got [{got[0]:.9f}, {got[1]:.9f}] vs [{want[0]:.9f}, {want[1]:.9f}], gap {gap:.2e}",
        )


def _expected_moments(Z, y):
    Q = np.linalg.qr(Z)[0]
    return Q.T @ (np.asarray(y).ravel() - np.mean(y))


def leg_v(design, Z, GX, G):
    print("(v) the intersection fits its baseline on Z and its DA branch on Z-tilde = (T, Z)")
    X, y, M = design.X, design.y, design.k
    z = Z[:, :1]
    m, p = z.shape[1], G.reshape(len(G), -1).shape[1]
    model = IntersectedInstrumentalVariablePartialR2(gamma=GAMMA, epsilon=EPS_TOL, epsilon_iv=IV_BOUND, **COMMON)
    model.fit(X=X, y=y, GX=GX, G=G, Z=z)
    base, aug = model.baseline, model.augmented
    check("(v) baseline read the instrument", base._has_iv)
    check(f"(v) baseline instrument width {m}", base.Z_projector_R.shape == (m + M, M), f"{base.Z_projector_R.shape}")
    check(
        "(v) baseline moments are Q_Z' y, bit for bit",
        np.array_equal(base.y_residual_base[:m], _expected_moments(z, y)),
    )
    check("(v) DA branch read the instrument", aug._has_iv)
    check(
        f"(v) DA branch instrument width {p + m}",
        aug.Z_projector_R.shape == (p + m + M, M),
        f"{aug.Z_projector_R.shape}",
    )
    tilde = np.column_stack([G, z])
    check(
        "(v) DA branch moments are Q_[G,Z]' y, bit for bit",
        np.array_equal(aug.y_residual_base[: p + m], _expected_moments(tilde, y)),
    )
    swapped = np.array_equal(aug.y_residual_base[: p + m], _expected_moments(np.column_stack([z, G]), y))
    check("(v) and not Q_[Z,G]' y (the order is part of the contract)", not swapped)
    check(
        "(v) DA branch design is GX, centred",
        np.array_equal(aug.Z_projector_R[: p + m], np.linalg.qr(tilde)[0].T @ (GX - GX.mean(axis=0))),
    )

    empty = IntersectedInstrumentalVariablePartialR2(gamma=GAMMA, epsilon=EPS_TOL, epsilon_iv=IV_BOUND, **COMMON)
    empty.fit(X=X, y=y, GX=GX, G=G, Z=None)
    zeros = IntersectedInstrumentalVariablePartialR2(gamma=GAMMA, epsilon=EPS_TOL, epsilon_iv=IV_BOUND, **COMMON)
    zeros.fit(X=X, y=y, GX=GX, G=G, Z=np.zeros((design.n, 0)))
    direct = InstrumentalVariablePartialR2(gamma=GAMMA, epsilon_iv=IV_BOUND, **COMMON).fit(X=GX, y=y, Z=G)
    check("(v) Z=None: baseline reads no instrument", not empty.baseline._has_iv)
    check(
        "(v) Z=None: DA branch reads exactly G",
        empty.augmented._has_iv and empty.augmented.Z_projector_R.shape == (p + M, M),
    )
    for tag, other in (("zeros((n, 0))", zeros.augmented), ("today's direct Z=G fit", direct)):
        same = np.array_equal(empty.augmented.Z_projector_R, other.Z_projector_R) and np.array_equal(
            empty.augmented.y_residual_base, other.y_residual_base
        )
        check(f"(v) Z=None DA arrays array_equal to {tag}", same)
    check("(v) Z=None and zeros((n, 0)) baselines agree", not zeros.baseline._has_iv)


def leg_vi(design, Z):
    print("(vi) iv_bound is the root sum square, exact at gamma_z = 0")
    X, y = design.X, design.y

    def bound(epsilon_iv, gamma_z, sigma_sq=1.0, rho=1.0):
        model = InstrumentalVariablePartialR2(gamma=GAMMA, epsilon_iv=epsilon_iv, gamma_z=gamma_z, rho=rho, **COMMON)
        model.sigma_sq = sigma_sq
        return model.iv_bound

    for epsilon_iv in BUDGETS:
        check(
            f"(vi) gamma_z 0: iv_bound == epsilon_iv {epsilon_iv!r} bit for bit", bound(epsilon_iv, 0.0) == epsilon_iv
        )
    joint = bound(EPS_TOL, GAMMA_Z)
    check(
        "(vi) sqrt(0.03125^2 + 0.0625^2) = 0.069877 to 1e-9",
        abs(joint - np.hypot(EPS_TOL, np.sqrt(GAMMA_Z))) < 1e-9 and abs(joint - RSS_BOUND) < 1e-6,
        f"{joint:.9f}",
    )
    check("(vi) strictly below the plain sum 0.09375", joint < PLAIN_SUM, f"{joint:.6f} < {PLAIN_SUM}")
    check(
        "(vi) non-DA at epsilon_iv 0 reads exactly s sqrt(gamma_z)",
        bound(0.0, GAMMA_Z) == np.sqrt(GAMMA_Z),
        f"{bound(0.0, GAMMA_Z)!r}",
    )
    scaled = bound(EPS_TOL, GAMMA_Z, sigma_sq=4.0, rho=4.0)
    check(
        "(vi) s is the pre-DA sigma sqrt(sigma_sq / rho)",
        abs(scaled - joint) < 1e-12,
        f"{scaled:.9f} at sigma_sq 4, rho 4",
    )

    declared = InstrumentalVariablePartialR2(gamma=GAMMA, epsilon_iv=0.0, gamma_z=GAMMA_Z, **COMMON).fit(X=X, y=y, Z=Z)
    explicit = InstrumentalVariablePartialR2(gamma=GAMMA, epsilon_iv=IV_BOUND, gamma_z=0.0, **COMMON).fit(X=X, y=y, Z=Z)
    gap = float(np.abs(bounds(declared) - bounds(explicit)).max())
    check(
        "(vi) declared gamma_z 2^-8 at epsilon_iv 0 == explicit bound 0.0625, to 1e-6",
        gap < INTERVAL_TOL,
        f"gap {gap:.2e}",
    )

    # the line fires at the first solve, after rho is final, once per fitted model
    records = []
    sink = logger.add(lambda message: records.append(message.record), level="DEBUG")
    try:
        logged = InstrumentalVariablePartialR2(gamma=GAMMA, epsilon_iv=EPS_TOL, gamma_z=GAMMA_Z, **COMMON)
        logged.fit(X=X, y=y, Z=Z)
        at_fit = sum("IV budget" in r["message"] for r in records)
        bounds(logged)
        bounds(logged)
    finally:
        logger.remove(sink)
    lines = [r for r in records if "IV budget" in r["message"]]
    check("(vi) nothing logged at fit, rho is not final there", at_fit == 0, f"{at_fit} lines")
    check("(vi) a gamma_z model logs the joint bound once over two solves", len(lines) == 1, f"{len(lines)} lines")
    message = lines[0]["message"] if lines else ""
    check("(vi) at INFO, not WARNING", bool(lines) and all(r["level"].name == "INFO" for r in lines), message)
    printed = float(message.split("joint bound")[1].split("= ")[1].split(";")[0]) if lines else np.nan
    check("(vi) the line names the joint bound", abs(printed - joint) < 1e-6, f"printed {printed!r} vs {joint:.9f}")
    named = f"gamma_z={GAMMA_Z:g}" in message and f"epsilon_iv={EPS_TOL:g}" in message and "s=" in message
    check("(vi) the line names gamma_z, epsilon_iv and s", named, message)


def leg_vii(design, Z, GX, G):
    print("(vii) the intersection budgets per branch: baseline r_Z, DA branch the joint bound")
    X, y = design.X, design.y
    model = IntersectedInstrumentalVariablePartialR2(
        gamma=GAMMA, epsilon=EPS_TOL, epsilon_iv=EPS_TOL, gamma_z=GAMMA_Z, **COMMON
    )
    model.fit(X=X, y=y, GX=GX, G=G, Z=Z)
    base, aug = model.baseline, model.augmented
    check("(vii) the intersection keeps its own epsilon_iv 2^-5", model.epsilon_iv == EPS_TOL)
    check("(vii) baseline branch epsilon_iv is 0.0", base.epsilon_iv == 0.0, repr(base.epsilon_iv))
    check("(vii) DA branch epsilon_iv is 2^-5", aug.epsilon_iv == EPS_TOL, repr(aug.epsilon_iv))
    check("(vii) both branches carry gamma_z 2^-8", base.gamma_z == GAMMA_Z and aug.gamma_z == GAMMA_Z)
    check(
        "(vii) baseline bound is r_Z = 0.0625 to 1e-6 (s = 1)",
        abs(base.iv_bound - IV_BOUND) < 1e-6,
        f"{base.iv_bound:.9f}",
    )
    check(
        "(vii) DA branch bound is 0.069877 to 1e-6 (s = 1)", abs(aug.iv_bound - RSS_BOUND) < 1e-6, f"{aug.iv_bound:.9f}"
    )
    check("(vii) DA branch rho set after fit", aug.rho == model.rho and aug.rho > 1.0, f"{aug.rho:.6f}")
    empty = IntersectedInstrumentalVariablePartialR2(
        gamma=GAMMA, epsilon=EPS_TOL, epsilon_iv=EPS_TOL, gamma_z=GAMMA_Z, **COMMON
    )
    empty.fit(X=X, y=y, GX=GX, G=G, Z=None)
    check("(vii) empty Z: baseline has no IV constraint at all", not empty.baseline._has_iv)
    check(
        "(vii) empty Z: DA branch keeps the joint bound on T",
        empty.augmented._has_iv and abs(empty.augmented.iv_bound - RSS_BOUND) < 1e-6,
    )


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
    leg_i(design)
    leg_ii(design, GX)
    leg_iii(design, Z)
    leg_iv(design, Z, GX, G)
    leg_v(design, Z, GX, G)
    leg_vi(design, Z)
    leg_vii(design, Z, GX, G)
    if not args.skip_digest:
        leg_d(args.reference)
    else:
        print("(D) SKIPPED by --skip-digest: a break-it run, not the committed state")
    if not FAIL:
        print("A57 PASS")
    else:
        print(f"A57 FAIL: {FAIL}")
        sys.exit(1)
