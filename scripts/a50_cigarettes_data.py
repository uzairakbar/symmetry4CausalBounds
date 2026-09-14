"""A50: the cigarette panel is the panel the plan measured, and the FWL design is
in the paper's units.

The loader is numpy and the stdlib `csv` module; every number below is read off
`src/sem/cigarettes.py` and compared against the planning probes. Legs:

  (i)   provenance: the four raw sources hash to the recorded digests and the
        built panel hashes to its own. Catches: a source silently re-fetched or
        edited, a loader that drops or reorders a row. Misses: a change upstream
        AND in the pin at once.
  (ii)  shape: 2450 rows, 49 states, 50 years, balanced, nothing missing, and the
        minimum state excise is $0.02 so the logs need no floor. Catches: AK and
        HI back in (no land neighbour, so p_n is undefined there), a broken join.
        Misses: whether the VALUES are right; that is (iii) and (iv).
  (iii) nominal: the loader's `p` IS the CDC "Average Cost per pack" column, and
        the raw share of Var(log X) along v = (1,1,1,1) is 0.9422 against an
        isotropic 0.25. Catches: a deflation anywhere in the loader -- dividing by
        the CPI would impose homogeneity by construction and is the one mistake
        that makes the whole experiment vacuous. Misses: a deflation applied to y
        alone, which (iv)'s ladder would move.
  (iv)  the trend ladder reproduces: rho_max, the classical iid statistic and the
        state- and year-clustered Walds at all five specs. Catches: a trend that
        is not t = (year - 1994)/25, a control matrix without the state dummies,
        a different neighbour aggregator. Misses: nothing about the instrument.
  (v)   units: the FWL residuals are exactly mean zero (the state dummies span the
        intercept, so `mean_match`'s mu is zero) and sigma-hat^2 of the design is
        1, which is what makes `EPS_TOL` -- a module constant in outcome units --
        mean the same here as on the other datasets. Catches: a missing
        normalisation, controls without the dummies.
  (vi)  the anchor and the restricted target at t3: first-stage F under both
        clusterings, b_r in raw log units, v'b_r = 0 to machine precision, and
        gamma* = 0.1922. Catches: the federal tax in Z (7 year-common changes,
        collinear with any rich time control), an unrestricted fit passed off as
        the target. Misses: whether the SEM wires this b_r through; that is a52.
  (vii) no pandas under `src/`. It is not a dependency of this project (it reaches
        the venv as a transitive of seaborn), so importing it from `src/` would
        make a transitive load-bearing.

    python scripts/a50_cigarettes_data.py
"""

import csv
import hashlib
import os
import sys

import numpy as np
from loguru import logger

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from src.experiments.utils.metrics import sigma_sq_hat  # noqa: E402
from src.sem.cigarettes import (  # noqa: E402
    ANCHORS,
    PANEL_FILE,
    SPECS,
    TREATMENTS,
    V,
    build_design,
    data_directory,
    first_stage_f,
    homogeneity_f,
    homogeneity_wald,
    load_panel,
    restricted_fit,
    rho_max,
)

# sha256 prefixes: the four sources of plan SS2.1, and the panel this loader builds
# from them. The panel digest is of THIS serialisation (shortest-repr floats), not
# of the planning probe's pandas `to_csv`; the numbers the panel is FOR are pinned
# in (ii) to (vi), which no serialisation can move.
RAW_DIGESTS = {
    "taxburden.csv": "400370089bdee037",
    "SAINC1__ALL_AREAS_1929_2025.csv": "a47d3f8894bfd578",
    "CPIAUCSL.csv": "f8ecddf53a9a9a74",
    "neighbors.csv": "8e4b2a1581f1ff58",
}
PANEL_DIGEST = "dead08a23428602f"
ROWS, STATES, YEARS = 2450, 49, 50
MIN_EXCISE = 0.02
RAW_SHARE_V = 0.9422
# spec -> (rho_max, W_iid classical, W_state, W_year); MEASURED, plan SS0.4
LADDER = {
    "s": (1.0121, 29.0, 4.9, 8.0),
    "t1": (1.1135, 272.0, 86.8, 53.4),
    "t2": (1.0156, 37.3, 7.8, 8.4),
    "t3": (1.0020, 4.8, 2.4, 1.6),
    "t4": (1.0025, 6.0, 2.9, 2.9),
}
SPEC = "t3"
FIRST_STAGE = {"state": 67.6, "year": 324.4}
B_R = np.array([-1.992, 0.507, 1.237, 0.249])
GAMMA_STAR = 0.1922
FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAIL.append(name)


def digest(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()[:16]


def leg_i(directory):
    print("(i) the raw sources and the built panel hash to the recorded digests")
    for name, want in RAW_DIGESTS.items():
        path = os.path.join(directory, "raw", name)
        got = digest(path) if os.path.isfile(path) else "MISSING"
        check(f"(i) {name}", got == want, f"{got} vs {want}")
    got = digest(os.path.join(directory, PANEL_FILE))
    check("(i) panel.csv", got == PANEL_DIGEST, f"{got} vs {PANEL_DIGEST}")


def leg_ii(panel):
    print("(ii) shape, balance and the excise floor")
    states, years = np.unique(panel["st"]), np.unique(panel["year"])
    counts = np.array([int((panel["st"] == state).sum()) for state in states])
    check("(ii) rows", len(panel["st"]) == ROWS, f"{len(panel['st'])}")
    check("(ii) states", len(states) == STATES, f"{len(states)}")
    check("(ii) years", len(years) == YEARS, f"{years.min()}-{years.max()}")
    check("(ii) balanced", bool(np.all(counts == YEARS)), f"min {counts.min()} max {counts.max()}")
    check("(ii) AK and HI dropped", not ({"AK", "HI"} & set(states)), "no land neighbour, so p_n is undefined")
    missing = {name: int(np.isnan(panel[name]).sum()) for name in panel if panel[name].dtype.kind == "f"}
    check("(ii) no NaN", not any(missing.values()), f"{ {k: v for k, v in missing.items() if v} }")
    excise = float(panel["tax_s"].min())
    check("(ii) minimum state excise", np.isclose(excise, MIN_EXCISE), f"${excise:.2f}, so no log floor")
    check("(ii) no non-positive excise", int((panel["tax_s"] <= 0).sum()) == 0)


def leg_iii(panel, directory):
    print("(iii) nominal: nothing is deflated")
    with open(os.path.join(directory, "raw", "taxburden.csv"), newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        column = {name: index for index, name in enumerate(header)}
        raw = {}
        for row in reader:
            if row[column["SubMeasureDesc"]] == "Average Cost per pack":
                raw[(row[column["LocationAbbr"]], int(row[column["Year"]]))] = float(row[column["Data_Value"]])
    keys = list(zip(panel["st"], panel["year"], strict=True))
    gap = float(np.max(np.abs(panel["p"] - np.array([raw[key] for key in keys]))))
    check("(iii) loader p == CDC average cost per pack", gap < 1e-12, f"max |diff| {gap:.2e}")
    X = np.log(np.column_stack([panel[name] for name in TREATMENTS]))
    covariance = np.cov(X.T)
    share = float(V @ covariance @ V / len(V) / np.trace(covariance))
    check(
        "(iii) raw share of Var(log X) along v",
        abs(share - RAW_SHARE_V) < 1e-3,
        f"{share:.4f} vs {RAW_SHARE_V} (isotropic 0.25)",
    )


def leg_iv(panel):
    print("(iv) the trend ladder, t = (year - 1994) / 25")
    print(f"      {'spec':>4s} {'K':>4s} {'rho_max':>8s} {'W_iid':>8s} {'W_state':>8s} {'W_year':>8s}")
    for spec, (want_rho, want_iid, want_state, want_year) in LADDER.items():
        design = build_design(panel, spec=spec)
        got = (
            rho_max(design),
            homogeneity_f(design),
            homogeneity_wald(design, "state", "ols"),
            homogeneity_wald(design, "year", "ols"),
        )
        print(f"      {spec:>4s} {design.K:4d} {got[0]:8.4f} {got[1]:8.1f} {got[2]:8.1f} {got[3]:8.1f}")
        check(f"(iv) {spec} rho_max", abs(got[0] - want_rho) < 5e-4, f"{got[0]:.4f} vs {want_rho}")
        check(f"(iv) {spec} W_iid", abs(got[1] - want_iid) < 0.05, f"{got[1]:.1f} vs {want_iid}")
        check(f"(iv) {spec} W_state", abs(got[2] - want_state) < 0.05, f"{got[2]:.1f} vs {want_state}")
        check(f"(iv) {spec} W_year", abs(got[3] - want_year) < 0.05, f"{got[3]:.1f} vs {want_year}")


def leg_v(panel):
    print("(v) FWL residuals are mean zero and the design is sigma-normalised")
    for spec in SPECS:
        design = build_design(panel, spec=spec)
        centred = max(abs(float(design.y.mean())), float(np.abs(design.X.mean(axis=0)).max()))
        variance = sigma_sq_hat(design.X, design.y, intercept=True)
        check(f"(v) {spec} residual means", centred < 1e-10, f"max |mean| {centred:.2e}")
        check(f"(v) {spec} sigma-hat^2 == 1", abs(variance - 1.0) < 1e-12, f"{variance:.15f}")


def leg_vi(panel):
    print(f"(vi) the {SPEC} anchor and the restricted target")
    design = build_design(panel, spec=SPEC)
    for by, want in FIRST_STAGE.items():
        got = first_stage_f(design, by)[0]
        check(f"(vi) first-stage F ({by})", abs(got - want) < 0.05, f"{got:.1f} vs {want}")
    b_r, misfit = restricted_fit(design)
    raw = b_r * design.sigma
    print(f"      b_r (raw log units) {np.array2string(raw, precision=4)}   IV misfit r0 {misfit:.4f}")
    check("(vi) b_r", bool(np.all(np.abs(raw - B_R) < 5e-4)), f"{np.round(raw, 4)} vs {B_R}")
    check("(vi) v' b_r == 0", abs(float(V @ b_r)) < 1e-12, f"{float(V @ b_r):.2e}")
    gap = design.b_ols - b_r
    gamma = float(gap @ design.Sigma @ gap)
    check("(vi) gamma*", abs(gamma - GAMMA_STAR) < 5e-5, f"{gamma:.6f} vs {GAMMA_STAR}")
    anchor = ANCHORS[design.anchor][0]
    check("(vi) Z is the own state excise", anchor == ("tax_s",), f"{anchor}")


def leg_vii():
    print("(vii) no pandas under src/")
    offenders = []
    for root, _, files in os.walk(os.path.join(REPO, "src")):
        for name in files:
            if not name.endswith(".py"):
                continue
            path = os.path.join(root, name)
            with open(path) as handle:
                text = handle.read()
            if "import pandas" in text or "from pandas" in text:
                offenders.append(os.path.relpath(path, REPO))
    check("(vii) src/ is pandas-free", not offenders, f"{offenders}")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    os.chdir(REPO)
    directory = data_directory()
    print(f"panel directory {directory}")
    panel = load_panel()
    leg_i(directory)
    leg_ii(panel)
    leg_iii(panel, directory)
    leg_iv(panel)
    leg_v(panel)
    leg_vi(panel)
    leg_vii()
    if FAIL:
        print(f"A50 FAIL: {FAIL}")
        sys.exit(1)
    print("A50 PASS")
