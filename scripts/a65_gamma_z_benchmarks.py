"""A65: gamma_z benchmarked the way gamma is, and the own-price headline pair.

refactor8 is on top of refactor6's tip M. `benchmark_gamma_z` projects the same
omitted W onto the configured instrument set instead of the treatments (Cinelli
and Hazlett's IV framework, paper [12]); `direct_effect_gamma_z` converts a direct
tax elasticity to gamma_z (Conley, Hansen and Rossi); T3 (`benchmarks_iv.tex`)
prints both; F1 and F2 now come in a pair per coefficient (`beta_pn_*`, `beta_p_*`)
and F2 marks the declared radius and the two benchmark leaks after the bootstrap
quantiles. Legs:

  (i)   T2 does not move: the `benchmark_gamma` refactor keeps lag-q 0.3121 and
        tax-diff 0.1502 to 1e-3 and the OVB / CH identity to 1e-9 on every row.
        Catches: the shared frame changing the sample, the scaling or c.
  (ii)  T3's rows: lag-q gamma_z 0.3113 and tax-diff 0.0905 to 3e-4 (MEASURED on
        the probe with its own algebra); the excise projection never exceeds the
        set's on any row; the row W = log tau_s reads r2 1 to 1e-9 (W is in Z);
        the lag ratio gamma_z / gamma is 0.997 to 1e-2. Catches: the projection
        taken on X, a Z built from the wrong columns, the ratio against the wrong
        table.
  (iii) the declared `GAMMA_Z_DEFAULT` reads back as `DECLARED_DELTA` to 5e-4 and
        the conversion round-trips to 1e-9. Both the label and the tolerance follow
        the live default, so the leg cannot again test one budget while naming
        another. Catches: a var(z_tax) or sigma slipped in the formula, a moved
        default whose delta was not re-derived.
  (iv)  the recipe's query leg at reduced scale (1 experiment, 4 grid points):
        every `beta_p_*` and `beta_pn_*` file and T3 exist; F2's marks are
        (0.1039, 0.2256, 0.0625, 0.5579, 0.3009) to 1e-3, so a61's first two
        stay where they were; F2's grid spans BUDGET_RANGE; on beta_p's F1 the
        F1 and F2 outcomes of both coefficients are keyed by the headline methods
        THE RECIPE LISTS (the intersection, as `cigarettes.py` keys the figure),
        asserted non-empty; whichever of the nesting pairs the block draws holds on
        F1; on beta_pn's F2, when the block lists both, the PI+IV band equals PI's at
        the largest radius to 1e-3, the addiction-stock leak having slackened the
        instrument, and whatever the block lists, every band the r_Z axis moves
        widens along it monotonically and strictly end to end. Catches: the loop writing one
        coefficient twice, the marks reordered, the leak grid not reaching the
        benchmark, a band that stops respecting nesting.
  (D)   the digest leg (scripts/digest_leg.py): with `iv: []` nothing here runs
        and the shipped artifacts hash as before.

    MPLBACKEND=Agg python scripts/a65_gamma_z_benchmarks.py [--reference JSON] [--skip-digest]
"""

import argparse
import os
import pickle
import shutil
import sys
from functools import partial

import numpy as np
import yaml
from loguru import logger

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import digest_leg  # noqa: E402
from munch import munchify  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from src.experiments.cigarettes import (  # noqa: E402
    BUDGET_RANGE,
    HEADLINE_METHODS,
    IV_BENCHMARKS,
    benchmark_covariates,
    benchmark_gamma,
    benchmark_gamma_z,
    direct_effect_gamma_z,
)
from src.experiments.configs import GAMMA_Z_DEFAULT, parse_experiment_plan, resolve_dataset_block  # noqa: E402
from src.experiments.utils import set_seed  # noqa: E402
from src.experiments.utils.constants import (  # noqa: E402
    ARTIFACTS_DIRECTORY,
    IV_MODE_METHODS,
    REAL_Z_METHODS,
    SUBDIR_QUERY,
    iv_mode,
    parse_method,
)
from src.main import ORCHESTRATORS  # noqa: E402
from src.sem.cigarettes import CigaretteSEM, build_design  # noqa: E402

SPEC, IV = "t3", ("tax_s", "y", "cpi")
# the declared budget read back as a direct tax elasticity,
# delta = sqrt(gamma_z / E[z^2]) * sigma  (`cigarettes.py::_write_benchmarks_iv`).
# At the panel's E[z^2] = 0.149006 and sigma = 0.145507 the Conley-scale default
# 0.0177 gives 0.05015. RE-DERIVE this whenever GAMMA_Z_DEFAULT moves: it was 0.0236
# at the old 2^-8 and went stale when f2b1101 adopted the Conley scale.
DECLARED_DELTA = 0.0501
FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAIL.append(name)


# a band the r_Z axis moves carries the OBSERVED instrument's constraint: the
# (Z)-only spellings, plus the bare `IV_MODE_METHODS` defaults, whose mode is (T,Z)
# and so carries Z beside T. A membership test, not a substring one
Z_CONSTRAINED: frozenset[str] = REAL_Z_METHODS | frozenset(IV_MODE_METHODS)


def widens(width):
    """True when `width` rises with the declared radius: monotone within 1e-6 AND
    strictly wider end to end over at least two radii it actually reports.

    The end-to-end clause is what stops a band that never moves, or one that is NaN
    at all but one radius, passing on the monotone clause alone."""
    seen = width[~np.isnan(width)]
    if len(seen) < 2:
        return False
    return bool(np.all(np.diff(seen) > -1e-6) and seen[-1] > seen[0] + 1e-6)


def fold_default_mode(name):
    """`DA+PI+IV(T,Z)` and the bare `DA+PI+IV` are the SAME estimator, spelled two
    ways; `src/experiments/cigarettes.py` folds `(T,Z)` onto the base before the
    headline lookup, so the gate keys the same way."""
    return parse_method(name)[0] if iv_mode(name) == "T,Z" else name


def headline_listed(methods):
    """The headline methods a block lists, in `HEADLINE_METHODS` order. The figures
    draw the INTERSECTION and skip only when it is empty, so the gate derives its
    expectation from the recipe rather than pinning the full five."""
    have = {fold_default_mode(name) for name in methods}
    return tuple(name for name in HEADLINE_METHODS if name in have)


def expected(methods, label):
    """`headline_listed`, asserted non-empty: an intersection-derived expectation
    passes trivially when the intersection is empty."""
    want = headline_listed(methods)
    check(f"{label}: the block lists at least one headline method", bool(want), f"{list(methods)}")
    return want


def both_listed(pairs, available, label):
    """The `(wider, narrower)` pairs both of whose members are present, non-empty."""
    kept = tuple(pair for pair in pairs if pair[0] in available and pair[1] in available)
    check(f"{label}: the block lists at least one comparable pair", bool(kept), f"lists {sorted(available)}")
    return kept


def recipe_block(**overrides):
    with open(os.path.join(REPO, "recipes", "cigarettesFig7.yaml")) as handle:
        config = yaml.safe_load(handle)
    defaults = config.pop("defaults", {}) or {}
    block = {**defaults, **config["cigarettes"]}
    block.pop("experiment", None)
    return resolve_dataset_block("cigarettes", {**block, "n_jobs": 1, **overrides})


def load(folder, name):
    with open(os.path.join(folder, name), "rb") as handle:
        return pickle.load(handle)  # noqa: S301 - our own artifact


# ------------------------------------------------------------------ legs


def leg_d(reference):
    print("(D) the shipped configuration does not move")
    for label, ok, detail in digest_leg.compare(REPO, reference):
        check(f"(D) {label}", ok, detail)


def leg_i():
    print("(i) T2 does not move under the shared frame")
    panel = CigaretteSEM.panel()
    rows = {key: benchmark_gamma(panel, w, SPEC) for key, w in benchmark_covariates(panel).items()}
    check("(i) lag-q gamma 0.3121 to 1e-3", abs(rows["lag_q"][3] - 0.3121) < 1e-3, f"{rows['lag_q'][3]:.4f}")
    check("(i) tax-diff gamma 0.1502 to 1e-3", abs(rows["tax_diff"][3] - 0.1502) < 1e-3, f"{rows['tax_diff'][3]:.4f}")
    check("(i) OVB == CH to 1e-9 on every row", all(abs(r[3] - r[4]) < 1e-9 for r in rows.values()))
    return rows


def leg_ii(gammas):
    print("(ii) T3's rows against the configured set")
    panel = CigaretteSEM.panel()
    rows = {key: benchmark_gamma_z(panel, w, SPEC, IV) for key, w in benchmark_covariates(panel).items()}
    check("(ii) lag-q gamma_z 0.3113 to 3e-4", abs(rows["lag_q"][3] - 0.3113) < 3e-4, f"{rows['lag_q'][3]:.4f}")
    check(
        "(ii) tax-diff gamma_z 0.0905 to 3e-4", abs(rows["tax_diff"][3] - 0.0905) < 3e-4, f"{rows['tax_diff'][3]:.4f}"
    )
    check("(ii) r2(W~tau_s) <= r2(W~Z) on every row", all(r[2] <= r[1] + 1e-12 for r in rows.values()))
    check("(ii) W = log tau_s reads r2(W~Z) 1", abs(rows["log_tax_s"][1] - 1.0) < 1e-9, f"{rows['log_tax_s'][1]:.9f}")
    ratio = rows["lag_q"][3] / gammas["lag_q"][3]
    check("(ii) lag-q gamma_z / gamma 0.997 to 1e-2", abs(ratio - 0.997) < 1e-2, f"{ratio:.3f}")


def leg_iii():
    print("(iii) the direct-effect conversion")
    design = build_design(CigaretteSEM.panel(), spec=SPEC, anchor="own-tax")
    z_var = float(np.mean(design.Z[:, 0] ** 2))
    delta = float(np.sqrt(GAMMA_Z_DEFAULT / z_var) * design.sigma)
    check(
        f"(iii) the declared {GAMMA_Z_DEFAULT:g} reads as delta {DECLARED_DELTA} to 5e-4",
        abs(delta - DECLARED_DELTA) < 5e-4,
        f"{delta:.4f}",
    )
    back = direct_effect_gamma_z(design, delta)
    check("(iii) the conversion round-trips to 1e-9", abs(back - GAMMA_Z_DEFAULT) < 1e-9, f"{back:.10f}")


def leg_iv():
    print("(iv) the query leg at reduced scale, both coefficient pairs")
    block = recipe_block(n_experiments=1, sweep_samples=4)
    folder = os.path.join(ARTIFACTS_DIRECTORY, "cigarettes", SUBDIR_QUERY)
    shutil.rmtree(folder, ignore_errors=True)
    set_seed(block["seed"])
    with threadpool_limits(limits=1):
        ORCHESTRATORS["cigarettes"](**block, hyperparameters=munchify(digest_leg.HYPERPARAMETERS)).run(
            parse_experiment_plan({"query": True})
        )
    for coefficient in ("p", "pn"):
        for stem in (f"beta_{coefficient}_gamma", f"beta_{coefficient}_budget"):
            for suffix in ("sweep.pdf", "values.pkl", "outcomes.pkl", "vlines.pkl"):
                path = os.path.join(folder, f"{stem}_{suffix}")
                check(f"(iv) {stem}_{suffix} exists", os.path.isfile(path) and os.path.getsize(path) > 0)
    check("(iv) benchmarks_iv.tex exists", os.path.isfile(os.path.join(folder, "benchmarks_iv.tex")))
    with open(os.path.join(folder, "benchmarks_iv.tex")) as handle:
        table = handle.read()
    check(
        "(iv) T3 carries the two leaks and the declared delta",
        all(s in table for s in ("0.3113", "0.0905", f"{DECLARED_DELTA:.4f}")),
    )
    marks = load(folder, "beta_pn_budget_vlines.pkl")
    want = (0.1039, 0.2256, np.sqrt(GAMMA_Z_DEFAULT), 0.5579, 0.3009)
    check(
        "(iv) F2 marks (median, p95, declared, lag-q, tax-diff) to 1e-3",
        len(marks) == 3 + len(IV_BENCHMARKS) and np.abs(np.asarray(marks) - want).max() < 1e-3,
        f"{np.round(marks, 4)}",
    )
    check("(iv) beta_p and beta_pn F2 marks agree", np.allclose(marks, load(folder, "beta_p_budget_vlines.pkl")))
    radii = load(folder, "beta_pn_budget_values.pkl")
    check("(iv) F2 grid spans BUDGET_RANGE", radii[0] == BUDGET_RANGE[0] and radii[-1] == BUDGET_RANGE[1])
    want = expected(block["methods"], "(iv) F1")
    p_gamma = load(folder, "beta_p_gamma_outcomes.pkl")
    check("(iv) beta_p F1 keyed by the headline methods the recipe lists", tuple(p_gamma) == want, f"{tuple(p_gamma)}")
    pn_gamma = load(folder, "beta_pn_gamma_outcomes.pkl")
    check("(iv) beta_pn F1 keyed the same way", tuple(pn_gamma) == want, f"{tuple(pn_gamma)}")
    differ = [name for name in want if not np.allclose(p_gamma[name], pn_gamma[name])]
    check(
        "(iv) beta_p and beta_pn F1 are different figures",
        len(differ) == len(want),
        f"same on {set(want) - set(differ)}",
    )
    # (wider, narrower): the narrower solves the same ball with one more constraint
    for wide, narrow in both_listed((("PI", "PI+IV"), ("PI+IV", "PI+INV+IV")), p_gamma, "(iv) beta_p F1 nesting"):
        inside = np.all(p_gamma[narrow][:, 0, 0] >= p_gamma[wide][:, 0, 0] - 1e-6) and np.all(
            p_gamma[narrow][:, 0, 1] <= p_gamma[wide][:, 0, 1] + 1e-6
        )
        check(f"(iv) beta_p F1: {narrow} inside {wide} to 1e-6", bool(inside))
    if {"PI", "PI+IV"} <= set(p_gamma):
        gap = p_gamma["PI"][:, 0, 1] - p_gamma["PI+IV"][:, 0, 1]
        check("(iv) beta_p F1: PI+IV's upper end under PI's by > 0.5 at every gamma", bool(np.all(gap > 0.5)), f"{gap}")
    # F2 sweeps r_Z, so the only pairs it can say anything about are the ones that
    # differ by the Z CONSTRAINT: an IV method against its no-IV parent, which it
    # collapses onto once the radius stops binding. NOT (PI+IV, PI+INV+IV) -- those
    # differ by the INV cone, which this axis never relaxes
    pn_budget = load(folder, "beta_pn_budget_outcomes.pkl")
    # only (PI, PI+IV): these pkls are HEADLINE-keyed, so `PI+INV` and `DA+PI` can
    # never appear and a pair naming them would be dead. The DA'd collapse pair
    # (PI, DA+PI+IV(Z)) needs the recalibration tolerance, so a66 owns it
    z_pairs = (("PI", "PI+IV"),)
    for wide, narrow in [pair for pair in z_pairs if pair[0] in pn_budget and pair[1] in pn_budget]:
        slack = np.abs(pn_budget[narrow][-1, 0] - pn_budget[wide][-1, 0]).max()
        check(f"(iv) beta_pn F2: {narrow} equals {wide} at the largest radius to 1e-3", slack < 1e-3, f"{slack:.5f}")
        tight = pn_budget[narrow][0, 0, 0] - pn_budget[wide][0, 0, 0]
        check(
            f"(iv) beta_pn F2: {narrow}'s lower end above {wide}'s by > 0.5 at the smallest radius",
            tight > 0.5,
            f"{tight:.3f}",
        )
    # the property that holds whatever the block lists, and the leg's non-vacuous
    # anchor when it lists no (parent, +IV) pair at all: a wider declared radius is a
    # weaker constraint, so every band the axis moves widens along it
    # the property that holds whatever the block lists, and the leg's non-vacuous
    # anchor: a wider declared radius is a weaker constraint, so every band the axis
    # moves widens along it -- monotonically AND strictly end to end, so a band that
    # never moves, or that is NaN at all but one radius, does not pass
    # "the r_Z axis moves it" = it carries the OBSERVED instrument's constraint:
    # REAL_Z_METHODS (the (Z)-only spellings) plus the bare defaults, whose mode is
    # (T,Z) and so carries Z beside T. NOT a substring test -- `PI+INV` passes one
    swept = tuple(name for name in pn_budget if name in Z_CONSTRAINED)
    check("(iv) beta_pn F2: the block lists at least one method the r_Z axis moves", bool(swept), f"{tuple(pn_budget)}")
    for name in swept:
        band = pn_budget[name][:, 0]
        width = band[:, 1] - band[:, 0]
        check(
            f"(iv) beta_pn F2: {name} widens with the declared radius", widens(width), f"{np.round(width, 4).tolist()}"
        )


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    os.chdir(REPO)
    os.environ.setdefault("MPLBACKEND", "Agg")
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", default=digest_leg.DEFAULT_REFERENCE)
    parser.add_argument("--skip-digest", action="store_true")
    args = parser.parse_args()
    gammas = {}
    legs = [
        ("(i)", leg_i),
        ("(ii)", lambda: leg_ii(gammas)),
        ("(iii)", leg_iii),
        ("(iv)", leg_iv),
    ]
    if not args.skip_digest:
        legs.append(("(D)", partial(leg_d, args.reference)))
    for tag, leg in legs:
        try:
            out = leg()
            if tag == "(i)" and out:
                gammas.update(out)
        except Exception as error:  # a raise is a FAIL line, and the later legs still report
            check(f"{tag} ran without raising", False, f"{type(error).__name__}: {error}")
    if args.skip_digest:
        print("(D) SKIPPED by --skip-digest: a break-it run, not the committed state")
    if not FAIL:
        print("A65 PASS")
    else:
        print(f"A65 FAIL: {FAIL}")
        sys.exit(1)
