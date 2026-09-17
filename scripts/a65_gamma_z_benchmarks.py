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
  (iii) the declared 2^-8 reads back as delta 0.0236 to 5e-4 and the conversion
        round-trips to 1e-9. Catches: a var(z_tax) or sigma slipped in the formula.
  (iv)  the recipe's query leg at reduced scale (1 experiment, 4 grid points):
        every `beta_p_*` and `beta_pn_*` file and T3 exist; F2's marks are
        (0.1039, 0.2256, 0.0625, 0.5579, 0.3009) to 1e-3, so a61's first two
        stay where they were; F2's grid spans BUDGET_RANGE; on beta_p's F1 the
        PI+IV upper end sits under PI's by more than 0.5 at every gamma and
        PI+INV+IV lies inside PI+IV to 1e-6; on beta_pn's F2 the PI+IV band
        equals PI's at the largest radius to 1e-3, the addiction-stock leak
        having slackened the instrument. Catches: the loop writing one
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
from src.experiments.utils.constants import ARTIFACTS_DIRECTORY, SUBDIR_QUERY  # noqa: E402
from src.main import ORCHESTRATORS  # noqa: E402
from src.sem.cigarettes import CigaretteSEM, build_design  # noqa: E402

SPEC, IV = "t3", ("tax_s", "y", "cpi")
FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAIL.append(name)


def recipe_block(**overrides):
    with open(os.path.join(REPO, "recipes", "neighbour-price_fig12.yaml")) as handle:
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
    check("(iii) declared 2^-8 reads as delta 0.0236 to 5e-4", abs(delta - 0.0236) < 5e-4, f"{delta:.4f}")
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
        all(s in table for s in ("0.3113", "0.0905", "0.0236")),
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
    p_gamma = load(folder, "beta_p_gamma_outcomes.pkl")
    check("(iv) beta_p F1 keyed by HEADLINE_METHODS", tuple(p_gamma) == HEADLINE_METHODS)
    pn_gamma = load(folder, "beta_pn_gamma_outcomes.pkl")
    check(
        "(iv) beta_p and beta_pn F1 are different figures",
        not np.allclose(p_gamma["PI"], pn_gamma["PI"]),
    )
    gap = p_gamma["PI"][:, 0, 1] - p_gamma["PI+IV"][:, 0, 1]
    check("(iv) beta_p F1: PI+IV's upper end under PI's by > 0.5 at every gamma", bool(np.all(gap > 0.5)), f"{gap}")
    inside = np.all(p_gamma["PI+INV+IV"][:, 0, 0] >= p_gamma["PI+IV"][:, 0, 0] - 1e-6) and np.all(
        p_gamma["PI+INV+IV"][:, 0, 1] <= p_gamma["PI+IV"][:, 0, 1] + 1e-6
    )
    check("(iv) beta_p F1: PI+INV+IV inside PI+IV to 1e-6", bool(inside))
    pn_budget = load(folder, "beta_pn_budget_outcomes.pkl")
    slack = np.abs(pn_budget["PI+IV"][-1, 0] - pn_budget["PI"][-1, 0]).max()
    check("(iv) beta_pn F2: PI+IV equals PI at the largest radius to 1e-3", slack < 1e-3, f"{slack:.5f}")
    tight = pn_budget["PI+IV"][0, 0, 0] - pn_budget["PI"][0, 0, 0]
    check("(iv) beta_pn F2: PI+IV's lower end above PI's by > 0.5 at the smallest radius", tight > 0.5, f"{tight:.3f}")


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
