"""A86: the finite-sample budget gamma_n (no experiment is run; seconds).

`finite_sample_budget(d, g, n, a)` is gamma_n(d; g) = F^-1_{chi2_d(n g)}(1 - a) / n,
the per-unit (1 - a) quantile of a noncentral chi-square on d degrees of freedom
with noncentrality n g; a = 0 is the raw (population) budget g. Legs:

  (i)   against scipy's chi2 / ncx2 quantiles on a grid of (d, g, n, a); a = 0
        returns g exactly; g <= 0 is the central chi-square; increasing in d, g
        and a's complement (1 - a), decreasing in n towards g; and the reference
        values at the fixed split a = alpha / 3: sim k 33, n 1843, gamma 1 ->
        1.1194; optical k 55, n 900, gamma 0.66 -> 0.8428; cigarettes k 5, gamma
        0.25 at n 2205 -> 0.2993 and at n 49 -> 0.7493; the mean row (1 + gamma)
        gamma_n(1; 0) at n 1843, gamma 1 -> 0.0062 (each within 1e-4).
        Catches: a quantile at the wrong level, a dropped 1/n, a noncentrality
        not scaled by n, a raw mode that pads.

    uv run python scripts/a86_gamma_n.py [--only LEG]
"""

import argparse
import os
import sys

import numpy as np
from loguru import logger
from scipy.stats import chi2, ncx2

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

ALPHA = 0.05
SPLIT = ALPHA / 3
# (label, d, g, n, a, gamma_n) at the fixed split, within 1e-4
REFERENCES = (
    ("sim ball", 33, 1.0, 1843, SPLIT, 1.1194),
    ("optical ball", 55, 0.66, 900, SPLIT, 0.8428),
    ("cigarettes ball n 2205", 5, 0.25, 2205, SPLIT, 0.2993),
    ("cigarettes ball n 49", 5, 0.25, 49, SPLIT, 0.7493),
)
MEAN_ROW = (1843, 1.0, 0.0062)  # n, gamma, (1 + gamma) gamma_n(1; 0)
FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAIL.append(name)


def leg_i():
    from src.methods.sensitivity_models import finite_sample_budget as fsb

    print("(i) finite_sample_budget against scipy")
    worst = 0.0
    for d in (1, 3, 12, 33):
        for g in (0.0, 2**-8, 0.25, 1.0):
            for n in (49, 400, 2205):
                for a in (SPLIT, ALPHA / 6, ALPHA / 9):
                    want = (chi2.ppf(1 - a, d) if g <= 0 else ncx2.ppf(1 - a, d, n * g)) / n
                    worst = max(worst, abs(fsb(d, g, n, a) - want) / want)
    check("(i) == scipy chi2 / ncx2 quantile / n", worst < 1e-12, f"(worst relative {worst:.1e})")
    raw = [(g, fsb(d, g, 400, 0.0)) for d in (1, 33) for g in (0.0, 2**-8, 1.0)]
    check("(i) a = 0 returns g", all(got == g for g, got in raw), f"{raw}")
    check("(i) g < 0 is the central chi-square", fsb(3, -1.0, 400, SPLIT) == fsb(3, 0.0, 400, SPLIT))
    check("(i) increasing in d", fsb(1, 0.25, 400, SPLIT) < fsb(3, 0.25, 400, SPLIT) < fsb(12, 0.25, 400, SPLIT))
    check("(i) increasing in g", fsb(3, 0.0, 400, SPLIT) < fsb(3, 0.1, 400, SPLIT) < fsb(3, 0.25, 400, SPLIT))
    check(
        "(i) increasing as a falls",
        fsb(3, 0.25, 400, SPLIT) < fsb(3, 0.25, 400, ALPHA / 6) < fsb(3, 0.25, 400, ALPHA / 9),
    )
    ladder = [fsb(5, 0.25, n, SPLIT) for n in (49, 220, 2205, 10**6)]
    check(
        "(i) decreasing in n towards g",
        bool(np.all(np.diff(ladder) < 0)) and 0.25 < ladder[-1] < 0.253,
        f"{np.round(ladder, 4).tolist()}",
    )
    for label, d, g, n, a, want in REFERENCES:
        got = fsb(d, g, n, a)
        check(f"(i) reference {label}", abs(got - want) < 1e-4, f"{got:.6f} (want {want})")
    n, gamma, want = MEAN_ROW
    got = (1 + gamma) * fsb(1, 0.0, n, SPLIT)
    check("(i) reference mean row n 1843", abs(got - want) < 1e-4, f"{got:.6f} (want {want})")


LEGS = {"i": leg_i}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=sorted(LEGS), default=None)
    args = parser.parse_args()
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    os.chdir(REPO)
    for name, leg in LEGS.items():
        if args.only in (None, name):
            leg()
    print(f"\n{'A86 ALL PASS' if not FAIL else 'A86 FAILURES: ' + ', '.join(FAIL)}")
    sys.exit(bool(FAIL))
