"""A57: the solvers carry ONE CONSTRAINT PER INSTRUMENT, and an empty block is exact.

refactor10 touches sensitivity_models.py only: a DA+ IV ball takes the translation
amounts as `T` and the observed instrument as `Z`, each block getting its own SOC
constraint at its own radius (`t_bound` from `epsilon_iv`, `z_bound` from
`epsilon_iv_z` and `gamma_z`), and the intersection hands its baseline branch the
observed Z alone. Nothing is pooled. Everything is measured on the cigarette design
(t3, own-tax, sigma-normalised, n = 2450), four coefficient queries, serial solves.
Legs:

  (D)   the digest leg (scripts/digest_leg.py), as a56: one query panel and one
        gamma sweep step per dataset at the shipped configuration against the
        reference recorded on 14509db, no tolerance. Catches: the `(n, 0)` spelling
        of an empty Z not being exactly today's `None`, a T block that is not G
        elementwise, a T radius that moved at gamma_z = 0. Misses: anything at
        gamma_z != 0, which is why (vi) exists.
  (i)   `Z=None` and `Z=zeros((n, 0))` reproduce PartialR2 to EXACTLY 0.0
        (MEASURED p7). Catches: an empty instrument that is not inert (a jitter
        block or a stray constraint reaching the problem), a block reading an
        (n, 0) array as present. Misses: an empty Z on a DA design, which (v)
        covers through the intersection.
  (ii)  PI+INV+IV with an empty Z reproduces PI+INV to 0.0. Catches: the INV cone
        of the new class differing from InvarianceConstrainedPartialR2's (a wrong
        GX, a missing centring, `eps_param` never set). Misses: the IV cone, (iv).
  (iii) a zero Z radius (epsilon_iv_z 0, gamma_z 0) returns INFEASIBLE on every
        query with NaN bounds, never a silent [0, 0] (MEASURED p3): the jitter block
        makes ||Q'(y - Xh)||^2 + j||h||^2 <= 0 unsatisfiable, and the class warns.
        Catches: the jitter block dropped (three moments in four unknowns are then
        exactly solvable inside the gamma = 0.25 ball and the status flips to OK), a
        solver failure read as infeasible. Misses: a positive radius the ball cannot
        reach (a61-ii).
  (iv)  the phase-b set [tax_s, y, cpi] at gamma 0.25. The two rows whose PROGRAM
        is unchanged by the decoupling -- PI+IV and PI+INV+IV, which only ever
        carried a Z constraint -- still reproduce the beta_pn intervals measured at
        14509db (logs/batchA/a57_reference_probe_14509db.log) to 1e-6 at
        r_Z = 0.0625: [0.374940, 1.644600] and [0.951673, 1.644600], raw log units.
        The DA rows changed program: `DA+PI+IV` now carries T at r_T = 0.0625 AND Z
        at r_Z = 0.0625 instead of one pooled constraint on the stacked matrix at
        0.0625, so its 14509db row [1.149780, 1.648402] is SUPERSEDED and the leg
        pins the round-14 values RECORDED here instead. The standalone row and the
        intersection's DA branch are pinned APART: the branch is fitted through the
        class, so its declared radius also carries the D20 allowance and it reads
        wider. Catches: any change to the
        IV or INV arithmetic, a Z that is not the FWL'd column (p7: +7 moves
        beta_pn by 0.004), a branch handed the wrong block. Misses: a solver bump
        under 1e-6, which is why the tolerance is not 1e-9.
  (v)   the intersection fits its baseline on Z alone and its DA branch on BOTH
        blocks, read off the fitted attributes: `Z_projector_R` counts the observed
        instrument (m rows above M) on both branches, `T_projector_R` counts the
        translation amounts (p rows) on the DA branch and is absent on the
        baseline, and each residual base is Q'y for its OWN Q, bit for bit, so a
        stacked QR fails. With Z=None the baseline reads nothing at all, the DA
        branch reads exactly G in its T block, and its arrays are array_equal to a
        fit with zeros((n, 0)) and to a direct `T=G` fit. Catches: Z reaching the
        DA branch only, the two blocks pooled into one QR, `None` reaching
        column_stack. Misses: the numbers, (iv).
  (vi)  the two radii. `t_bound` is `epsilon_iv` bit for bit at EVERY gamma_z (the
        leak budget never touches the T radius), and `z_bound` is
        sqrt(epsilon_iv_z^2 + (s sqrt(gamma_z))^2): 0.069877 to 1e-9 at
        epsilon_iv_z 2^-5, gamma_z 2^-8, s 1, strictly below the plain sum 0.09375,
        and rho-aware (s is the pre-DA sigma). Then at fit: a declared radius at
        epsilon_iv_z 0, gamma_z 2^-8 reproduces the explicit 0.0625, gamma_z 0
        interval to 1e-6 (p10 E), and a gamma_z model logs one INFO line, not
        WARNING, naming gamma_z, s, the allowance and both radii, at its first solve
        (rho is final there) and not at fit. Catches: gamma_z leaking into the T
        radius, the plain sum put back, s read post-DA, the old warning, a line
        printed before an intersection's DA branch knows its rho. Misses: the
        DA-side allowance on a fitted pair, which a67-(i) pins.
  (vii) the intersection budgets per branch: BOTH radii travel to both branches and
        each uses the ones its blocks ask for. The baseline carries no T block, so
        at epsilon_iv 2^-5, epsilon_iv_z 0, gamma_z 2^-8, s = 1 its Z radius is
        exactly r_Z = 0.0625 and `_has_t` is False, while the DA branch reads
        r_T = 2^-5 and the same r_Z at its own s; the oracle case, epsilon_iv_z 0.05
        and gamma_z 0: both branches read a Z radius of 0.05 and the DA branch's T
        radius is still its own epsilon_iv; the intersection's own epsilon_iv is
        untouched; the DA branch's rho is set after fit; under an empty Z the
        baseline has no constraint at all (so leg (D) cannot move) and the DA branch
        keeps its T constraint at epsilon_iv. Catches: the T block reaching the
        baseline, `epsilon_iv` spent on a Z constraint, `epsilon_iv_z` ignored (the
        oracle baseline would read 0 and be INFEASIBLE). Misses: the numbers under a
        real Z (a60/a61).

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
}
PLAN_DIGITS = {"PI+IV": "[0.375, 1.645]", "PI+INV+IV": "[0.952, 1.645]"}
# the DA rows changed PROGRAM in round 14: one pooled constraint on the stacked
# matrix at 0.0625 became a T constraint at r_T = 0.0625 beside a Z constraint at
# r_Z = 0.0625, which admits more. RECORDED here, superseding [1.149780, 1.648402]
# the standalone row is at r_Z = 0.0625 with no `X_pre`; the intersection's DA
# branch is fitted through the class, so its declared radius also carries the D20
# allowance and it is a WIDER interval, which is why the two are pinned apart
RECORDED = {
    "DA+PI+IV": (0.8513706565948552, 1.6487486320559676),
    "DA+PI+IV branch": (0.6966065657601866, 1.6547022161437242),
}
INTERVAL_TOL = 1e-6
RSS_BOUND = 0.069877  # sqrt(0.03125^2 + 0.0625^2)
ORACLE_Z = 0.05  # the oracle case's epsilon_iv_z in (vii)
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
    print("(iii) a zero Z radius is INFEASIBLE, not a silent [0, 0]")
    model = InstrumentalVariablePartialR2(gamma=GAMMA, epsilon_iv=EPS_TOL, epsilon_iv_z=0.0, gamma_z=0.0, **COMMON)
    model, error = fitted(model, X=design.X, y=design.y, Z=Z)
    check("(iii) fits at a zero radius", error is None, repr(error) if error else "")
    if error is not None:
        return
    check("(iii) z_bound is 0", model.z_bound == 0.0, repr(model.z_bound))
    out = bounds(model)
    status = np.asarray(model.query_status)
    check("(iii) every query INFEASIBLE", np.all(status == SolveStatus.INFEASIBLE), f"status {status.tolist()}")
    check("(iii) every bound NaN", np.all(np.isnan(out)), f"{out.tolist()}")


def leg_iv(design, Z, GX, G):
    print("(iv) the phase-b set reproduces the 14509db intervals at gamma 0.25, IV bound 0.0625")
    X, y, scale = design.X, design.y, design.sigma
    # the non-DA rows carry the observed instrument alone: `epsilon_iv_z` is its
    # radius now, `epsilon_iv` is the T one and is inert without a T block
    pi_iv = InstrumentalVariablePartialR2(gamma=GAMMA, epsilon_iv=EPS_TOL, epsilon_iv_z=IV_BOUND, **COMMON).fit(
        X=X, y=y, Z=Z
    )
    pi_inv_iv = InvarianceConstrainedInstrumentalVariablePartialR2(
        gamma=GAMMA, epsilon=EPS_TOL, epsilon_iv=EPS_TOL, epsilon_iv_z=IV_BOUND, **COMMON
    )
    pi_inv_iv.fit(X=X, y=y, GX=GX, Z=Z)
    # the DA row carries BOTH blocks, each at 0.0625: a different program from the
    # 14509db pooled one, hence RECORDED rather than REFERENCE
    da_pi_iv = InstrumentalVariablePartialR2(gamma=GAMMA, epsilon_iv=IV_BOUND, epsilon_iv_z=IV_BOUND, **COMMON)
    da_pi_iv.fit(X=GX, y=y, T=G, Z=Z)
    da_pi_iv.rho = da_pi_iv.sigma_sq / PartialR2(gamma=GAMMA, **COMMON).fit(X, y).sigma_sq
    # both radii travel to both branches and each uses the ones its blocks ask for:
    # the baseline sees Z alone at r_Z = s sqrt(gamma_z) = 0.0625 (s = 1 on this
    # sigma-normalised panel), the DA branch sees T at 0.0625 and Z at the same r_Z
    intersection = IntersectedInstrumentalVariablePartialR2(
        gamma=GAMMA, epsilon=EPS_TOL, epsilon_iv=IV_BOUND, gamma_z=GAMMA_Z, **COMMON
    )
    intersection.fit(X=X, y=y, GX=GX, G=G, Z=Z)
    models = {
        "PI+IV": pi_iv,
        "PI+INV+IV": pi_inv_iv,
        "DA+PI+IV": da_pi_iv,
        "PI&DA+PI+IV baseline at r_Z (PI+IV)": intersection.baseline,
        "PI&DA+PI+IV DA branch on both blocks (DA+PI+IV branch)": intersection.augmented,
    }
    for name, model in models.items():
        key = name.split("(")[-1].rstrip(")") if "(" in name else name
        out = bounds(model)
        status = np.asarray(model.query_status)
        got = scale * out[2]
        pinned, source = (REFERENCE, "14509db") if key in REFERENCE else (RECORDED, "round 14")
        want = np.asarray(pinned[key])
        gap = float(np.abs(got - want).max())
        digits = PLAN_DIGITS.get(key, f"[{want[0]:.3f}, {want[1]:.3f}]")
        check(f"(iv) {name}: every solve OK", np.all(status == SolveStatus.OK), f"status {status.tolist()}")
        check(
            f"(iv) {name}: beta_pn {digits} to 1e-6 ({source})",
            gap < INTERVAL_TOL,
            f"got [{got[0]:.9f}, {got[1]:.9f}] vs [{want[0]:.9f}, {want[1]:.9f}], gap {gap:.2e}",
        )


def _expected_moments(Z, y):
    Q = np.linalg.qr(Z)[0]
    return Q.T @ (np.asarray(y).ravel() - np.mean(y))


def leg_v(design, Z, GX, G):
    print("(v) the intersection fits its baseline on Z alone and its DA branch on both blocks")
    X, y, M = design.X, design.y, design.k
    z = Z[:, :1]
    m, p = z.shape[1], G.reshape(len(G), -1).shape[1]
    common = dict(gamma=GAMMA, epsilon=EPS_TOL, epsilon_iv=IV_BOUND, epsilon_iv_z=IV_BOUND, **COMMON)
    model = IntersectedInstrumentalVariablePartialR2(**common)
    model.fit(X=X, y=y, GX=GX, G=G, Z=z)
    base, aug = model.baseline, model.augmented
    check("(v) baseline read the observed instrument and no T", base._has_z and not base._has_t)
    check(f"(v) baseline Z width {m}", base.Z_projector_R.shape == (m + M, M), f"{base.Z_projector_R.shape}")
    check(
        "(v) baseline moments are Q_Z' y, bit for bit",
        np.array_equal(base.z_residual_base[:m], _expected_moments(z, y)),
    )
    check("(v) DA branch read both blocks", aug._has_t and aug._has_z)
    check(f"(v) DA branch T width {p}", aug.T_projector_R.shape == (p + M, M), f"{aug.T_projector_R.shape}")
    check(f"(v) DA branch Z width {m}", aug.Z_projector_R.shape == (m + M, M), f"{aug.Z_projector_R.shape}")
    check(
        "(v) DA branch T moments are Q_G' y, bit for bit",
        np.array_equal(aug.t_residual_base[:p], _expected_moments(G, y)),
    )
    check(
        "(v) DA branch Z moments are Q_Z' y, bit for bit",
        np.array_equal(aug.z_residual_base[:m], _expected_moments(z, y)),
    )
    stacked = _expected_moments(np.column_stack([G, z]), y)
    pooled = np.array_equal(aug.t_residual_base[:p], stacked[:p]) and np.array_equal(
        aug.z_residual_base[:m], stacked[p : p + m]
    )
    check("(v) the two blocks are separate: the DA branch's T rows are NOT the stacked QR", not pooled)
    check(
        "(v) DA branch design is GX, centred",
        np.array_equal(aug.T_projector_R[:p], np.linalg.qr(G.reshape(len(G), -1))[0].T @ (GX - GX.mean(axis=0))),
    )

    empty = IntersectedInstrumentalVariablePartialR2(**common)
    empty.fit(X=X, y=y, GX=GX, G=G, Z=None)
    zeros = IntersectedInstrumentalVariablePartialR2(**common)
    zeros.fit(X=X, y=y, GX=GX, G=G, Z=np.zeros((design.n, 0)))
    direct = InstrumentalVariablePartialR2(gamma=GAMMA, epsilon_iv=IV_BOUND, **COMMON).fit(X=GX, y=y, T=G)
    check("(v) Z=None: baseline reads no instrument", not empty.baseline._has_iv)
    check(
        "(v) Z=None: DA branch reads exactly G in its T block and no Z",
        empty.augmented._has_t and not empty.augmented._has_z and empty.augmented.T_projector_R.shape == (p + M, M),
    )
    for tag, other in (("zeros((n, 0))", zeros.augmented), ("a direct T=G fit", direct)):
        same = np.array_equal(empty.augmented.T_projector_R, other.T_projector_R) and np.array_equal(
            empty.augmented.t_residual_base, other.t_residual_base
        )
        check(f"(v) Z=None DA arrays array_equal to {tag}", same)
    check("(v) Z=None and zeros((n, 0)) baselines agree", not zeros.baseline._has_iv)


def leg_vi(design, Z):
    print("(vi) the two radii: gamma_z never touches the T one")
    X, y = design.X, design.y

    def radii(epsilon_iv, gamma_z, epsilon_iv_z=0.0, sigma_sq=1.0, rho=1.0):
        model = InstrumentalVariablePartialR2(
            gamma=GAMMA, epsilon_iv=epsilon_iv, epsilon_iv_z=epsilon_iv_z, gamma_z=gamma_z, rho=rho, **COMMON
        )
        model.sigma_sq = sigma_sq
        return model.t_bound, model.z_bound

    for epsilon_iv in BUDGETS:
        for gamma_z in (0.0, GAMMA_Z, 0.25):
            t_radius, _ = radii(epsilon_iv, gamma_z)
            check(
                f"(vi) gamma_z {gamma_z!r}: t_bound == epsilon_iv {epsilon_iv!r} bit for bit",
                t_radius == epsilon_iv,
            )
    joint = radii(EPS_TOL, GAMMA_Z, epsilon_iv_z=EPS_TOL)[1]
    check(
        "(vi) z_bound sqrt(0.03125^2 + 0.0625^2) = 0.069877 to 1e-9",
        abs(joint - np.hypot(EPS_TOL, np.sqrt(GAMMA_Z))) < 1e-9 and abs(joint - RSS_BOUND) < 1e-6,
        f"{joint:.9f}",
    )
    check("(vi) strictly below the plain sum 0.09375", joint < PLAIN_SUM, f"{joint:.6f} < {PLAIN_SUM}")
    check(
        "(vi) at epsilon_iv_z 0 the Z radius is exactly s sqrt(gamma_z)",
        radii(EPS_TOL, GAMMA_Z)[1] == np.sqrt(GAMMA_Z),
        f"{radii(EPS_TOL, GAMMA_Z)[1]!r}",
    )
    check(
        "(vi) at gamma_z 0 the Z radius is exactly epsilon_iv_z",
        radii(EPS_TOL, 0.0, epsilon_iv_z=IV_BOUND)[1] == IV_BOUND,
        f"{radii(EPS_TOL, 0.0, epsilon_iv_z=IV_BOUND)[1]!r}",
    )
    scaled = radii(EPS_TOL, GAMMA_Z, epsilon_iv_z=EPS_TOL, sigma_sq=4.0, rho=4.0)[1]
    check(
        "(vi) s is the pre-DA sigma sqrt(sigma_sq / rho)",
        abs(scaled - joint) < 1e-12,
        f"{scaled:.9f} at sigma_sq 4, rho 4",
    )

    declared = InstrumentalVariablePartialR2(
        gamma=GAMMA, epsilon_iv=EPS_TOL, epsilon_iv_z=0.0, gamma_z=GAMMA_Z, **COMMON
    ).fit(X=X, y=y, Z=Z)
    explicit = InstrumentalVariablePartialR2(
        gamma=GAMMA, epsilon_iv=EPS_TOL, epsilon_iv_z=IV_BOUND, gamma_z=0.0, **COMMON
    ).fit(X=X, y=y, Z=Z)
    gap = float(np.abs(bounds(declared) - bounds(explicit)).max())
    check(
        "(vi) declared gamma_z 2^-8 at epsilon_iv 0 == explicit bound 0.0625, to 1e-6",
        gap < INTERVAL_TOL,
        f"gap {gap:.2e}",
    )

    # the line fires at the first solve, after rho is final, once per fitted model
    records = []
    marker = "IV: gamma_z="
    sink = logger.add(lambda message: records.append(message.record), level="DEBUG")
    try:
        logged = InstrumentalVariablePartialR2(
            gamma=GAMMA, epsilon_iv=EPS_TOL, epsilon_iv_z=EPS_TOL, gamma_z=GAMMA_Z, **COMMON
        )
        logged.fit(X=X, y=y, Z=Z)
        at_fit = sum(marker in r["message"] for r in records)
        bounds(logged)
        bounds(logged)
    finally:
        logger.remove(sink)
    lines = [r for r in records if marker in r["message"]]
    check("(vi) nothing logged at fit, rho is not final there", at_fit == 0, f"{at_fit} lines")
    check("(vi) a gamma_z model logs the two radii once over two solves", len(lines) == 1, f"{len(lines)} lines")
    message = lines[0]["message"] if lines else ""
    check("(vi) at INFO, not WARNING", bool(lines) and all(r["level"].name == "INFO" for r in lines), message)
    printed = float(message.split("r_Z=")[1].split(",")[0]) if lines else np.nan
    check("(vi) the line names r_Z", abs(printed - joint) < 1e-6, f"printed {printed!r} vs {joint:.9f}")
    named = (
        f"gamma_z={GAMMA_Z:g}" in message
        and "s=" in message
        and "allowance" in message
        and f"r_T={EPS_TOL:.6g}" in message
    )
    check("(vi) the line names gamma_z, s, the allowance and both radii", named, message)


def leg_vii(design, Z, GX, G):
    print("(vii) the intersection budgets per branch: both radii travel, each block uses its own")
    X, y = design.X, design.y
    model = IntersectedInstrumentalVariablePartialR2(
        gamma=GAMMA, epsilon=EPS_TOL, epsilon_iv=EPS_TOL, gamma_z=GAMMA_Z, **COMMON
    )
    model.fit(X=X, y=y, GX=GX, G=G, Z=Z)
    base, aug = model.baseline, model.augmented
    check("(vii) the intersection keeps its own epsilon_iv 2^-5", model.epsilon_iv == EPS_TOL)
    check("(vii) baseline branch carries no T block", not base._has_t, repr(base._has_t))
    check("(vii) DA branch T radius is 2^-5", aug.t_bound == EPS_TOL, repr(aug.t_bound))
    check("(vii) both branches carry gamma_z 2^-8", base.gamma_z == GAMMA_Z and aug.gamma_z == GAMMA_Z)
    check(
        "(vii) baseline Z radius is r_Z = 0.0625 to 1e-6 (s = 1)",
        abs(base.z_bound - IV_BOUND) < 1e-6,
        f"{base.z_bound:.9f}",
    )
    check(
        "(vii) DA branch Z radius is r_Z at its own s, to 1e-6",
        abs(aug.z_bound - float(np.sqrt(aug.sigma_sq / aug.rho * GAMMA_Z)) - aug._z_allowance) < 1e-6,
        f"{aug.z_bound:.9f}",
    )
    check("(vii) DA branch rho set after fit", aug.rho == model.rho and aug.rho > 1.0, f"{aug.rho:.6f}")
    # the oracle case: no declared radius, so both Z constraints read the measured
    # piece and the DA branch's T radius is still its own epsilon_iv
    oracle_case = IntersectedInstrumentalVariablePartialR2(
        gamma=GAMMA, epsilon=EPS_TOL, epsilon_iv=EPS_TOL, epsilon_iv_z=ORACLE_Z, gamma_z=0.0, **COMMON
    )
    oracle_case.fit(X=X, y=y, GX=GX, G=G, Z=Z)
    base, aug = oracle_case.baseline, oracle_case.augmented
    check("(vii) oracle case: baseline Z radius is epsilon_iv_z 0.05", base.z_bound == ORACLE_Z, f"{base.z_bound!r}")
    check("(vii) oracle case: DA branch Z radius is the same 0.05", aug.z_bound == ORACLE_Z, f"{aug.z_bound!r}")
    check(
        "(vii) oracle case: epsilon_iv_z never reaches a T radius (both read 2^-5)",
        aug.t_bound == EPS_TOL and base.t_bound == EPS_TOL,
        f"{aug.t_bound!r}",
    )
    bounds(oracle_case)
    check(
        "(vii) oracle case: both branches solve OK on every coefficient query",
        np.all(np.asarray(base.query_status) == SolveStatus.OK)
        and np.all(np.asarray(aug.query_status) == SolveStatus.OK),
    )
    empty = IntersectedInstrumentalVariablePartialR2(
        gamma=GAMMA, epsilon=EPS_TOL, epsilon_iv=EPS_TOL, gamma_z=GAMMA_Z, **COMMON
    )
    empty.fit(X=X, y=y, GX=GX, G=G, Z=None)
    check("(vii) empty Z: baseline has no IV constraint at all", not empty.baseline._has_iv)
    check(
        "(vii) empty Z: the DA branch keeps its T constraint at epsilon_iv",
        empty.augmented._has_t and not empty.augmented._has_z and empty.augmented.t_bound == EPS_TOL,
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
