"""A71: an empty cell drops out of the coverage rows (D1c).

A (method, step, experiment) cell whose queries are ALL empty, i.e. every interval
has a NaN bound (infeasible, failed or an empty intersection), now records coverage
NaN in `evaluate_queries`, exactly when its `interval_width` is already NaN. The
row's `nanmean` then skips it, so an empty step is a gap in the coverage line, as it
already is in the width line. A partly empty cell keeps the old reading: its NaN
queries count as not covered. `coverage` itself is unchanged.

The reading this settles, CONSISTENCY.md #6:

    Robustness was misread. The analyser read simulation's `epsilon` panel
    (coverage `0 -> 1` across `eps/eps* = 1`) as Thm. 3.A. It is not: the statuses at
    `eps/eps* = 0.5` are `[failure 0, infeasible 204, covered 0, non-covering 0]` -- the
    set is *empty*, and NaN counts as not-covered (`metrics.py:204`). An empty set is not
    a validity failure. The honest demonstration is **cigarettes**, where the same step
    gives genuine non-coverage (`[0, 0, 120, 125]`). Cite that panel, not simulation.

Legs:

  1. the rule, on hand-built cells (4 queries): all four bounds NaN, and one NaN end
     per query, give coverage NaN under INFEASIBLE and FAILURE statuses alike, with
     the status split untouched; a partly empty cell gives `coverage`'s own number
     (RECORDED 0.25 and 0.5); a point estimate never gives NaN; `coverage` itself
     still reads 0 on the all-NaN cell. Catches: the rule keyed on statuses instead
     of the bounds, `any` and `all` swapped, the rule moved into `coverage`.
  2. the equivalence, on 2000 random NaN masks: coverage is NaN iff
     `interval_width` is NaN. Catches: the two rules drifting apart.
  3. the gap: a record of 3 steps x 4 experiments built through `evaluate_queries`,
     step 1 all empty in every experiment, drawn by `plotting._draw_series`; the
     line's y is NaN at that x and finite (RECORDED) at the other two. Catches: an
     empty step drawn as 0.

    MPLBACKEND=Agg python scripts/a71_empty_cells.py [--only LEG]

Nothing here runs a dataset.
"""

import argparse
import os
import sys
import warnings

import numpy as np
from loguru import logger

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import src.experiments.utils.plotting as plotting  # noqa: E402
from src.experiments.utils.metrics import coverage, evaluate_queries, interval_width  # noqa: E402
from src.methods.sensitivity_models import SolveStatus  # noqa: E402

FAIL = []

TRUTH = np.array([[0.0], [1.0], [2.0], [3.0]])
# RECORDED: the two partly empty cells below, read by `coverage`
PARTIAL_ONE = 0.25
PARTIAL_TWO = 0.5
# RECORDED: the gap leg's finite steps (step 0 covers all four queries, step 2 two of four)
GAP_FINITE = (1.0, 0.5)


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAIL.append(name)


def covering(truth):
    """An interval of half-width 0.5 around each target: covers every query."""
    return np.hstack([truth - 0.5, truth + 0.5])


# ------------------------------------------------------------------ legs


def leg_1():
    print("(1) the rule on hand-built cells")
    n = len(TRUTH)
    infeasible = np.full(n, SolveStatus.INFEASIBLE, dtype=int)
    failure = np.full(n, SolveStatus.FAILURE, dtype=int)

    all_nan = np.full((n, 2), np.nan)
    one_end = covering(TRUTH)
    one_end[[0, 2], 0] = np.nan
    one_end[[1, 3], 1] = np.nan
    for name, interval in (("all four bounds NaN", all_nan), ("one NaN end per query", one_end)):
        for tag, statuses in (("INFEASIBLE", infeasible), ("FAILURE", failure)):
            record = evaluate_queries(TRUTH, interval, statuses)
            check(f"(1) {name}, {tag}: coverage NaN", np.isnan(record.coverage), f"{record.coverage}")
            check(f"(1) {name}, {tag}: interval_width NaN", np.isnan(record.interval_width))
    record = evaluate_queries(TRUTH, all_nan, infeasible)
    check(
        "(1) the status split is untouched",
        record.status_counts.tolist() == [0, n, 0, 0],
        f"{record.status_counts.tolist()}",
    )
    record = evaluate_queries(TRUTH, all_nan, failure)
    check(
        "(1) ... and under FAILURE", record.status_counts.tolist() == [n, 0, 0, 0], f"{record.status_counts.tolist()}"
    )
    check("(1) `coverage` itself still reads 0 on the all-NaN cell", coverage(TRUTH, all_nan) == 0.0)

    partial = covering(TRUTH)
    partial[[1, 2, 3]] = np.nan  # one feasible query left, and it covers
    statuses = np.array([SolveStatus.OK, SolveStatus.INFEASIBLE, SolveStatus.INFEASIBLE, SolveStatus.FAILURE])
    record = evaluate_queries(TRUTH, partial, statuses)
    check(
        "(1) partly empty (1 of 4 left): `coverage`'s own number",
        record.coverage == coverage(TRUTH, partial) == PARTIAL_ONE,
        f"{record.coverage} (RECORDED {PARTIAL_ONE})",
    )
    partial = covering(TRUTH)
    partial[3] = np.nan
    partial[2] += 10.0  # feasible but misses
    statuses = np.array([SolveStatus.OK, SolveStatus.OK, SolveStatus.OK, SolveStatus.INFEASIBLE])
    record = evaluate_queries(TRUTH, partial, statuses)
    check(
        "(1) partly empty (3 of 4 left, 1 missing): `coverage`'s own number",
        record.coverage == coverage(TRUTH, partial) == PARTIAL_TWO,
        f"{record.coverage} (RECORDED {PARTIAL_TWO})",
    )

    for name, estimate in (("off target", TRUTH + 1.0), ("on target", TRUTH.copy())):
        record = evaluate_queries(TRUTH, estimate)
        check(f"(1) a point estimate ({name}) is never NaN", not np.isnan(record.coverage), f"{record.coverage}")


def leg_2():
    print("(2) coverage NaN iff interval_width NaN, on random masks")
    rng = np.random.default_rng(0)
    n = 5
    truth = rng.normal(size=(n, 1))
    mismatched, empties = [], 0
    for _ in range(2000):
        interval = covering(truth) + rng.normal(scale=0.3, size=(n, 2))
        interval.sort(axis=1)
        interval[rng.random((n, 2)) < rng.choice([0.2, 0.6, 0.9])] = np.nan
        record = evaluate_queries(truth, interval)
        empties += int(np.isnan(record.coverage))
        if np.isnan(record.coverage) != np.isnan(interval_width(truth, interval)):
            mismatched.append(interval)
    check("(2) the two NaN rules agree on every mask", not mismatched, f"{len(mismatched)} disagree")
    check("(2) the masks reach both sides", 0 < empties < 2000, f"{empties} empty of 2000")


def leg_3():
    print("(3) an empty step is a gap in the drawn line")
    n_exp = 4
    steps = []
    for step in range(3):
        row = []
        for _ in range(n_exp):
            interval = covering(TRUTH)
            if step == 1:
                interval[:] = np.nan
            elif step == 2:
                interval[[0, 1]] += 10.0
            row.append(evaluate_queries(TRUTH, interval).coverage)
        steps.append(row)
    y = np.array(steps)
    x = np.array([0.25, 0.5, 1.0])
    fig, ax = plt.subplots()
    try:
        handles, _ = plotting._draw_series(ax, x, {"PI+INV": y}, has_z=False)
        line = handles.get("PI+INV")
        check("(3) the method is drawn", line is not None)
        if line is not None:
            ydata = np.asarray(line.get_ydata(), dtype=float)
            check("(3) y at the empty step is NaN", np.isnan(ydata[1]), f"{ydata.tolist()}")
            check(
                "(3) y elsewhere is finite and as RECORDED",
                np.allclose(ydata[[0, 2]], GAP_FINITE),
                f"{ydata[[0, 2]].tolist()} (RECORDED {list(GAP_FINITE)})",
            )
    finally:
        plt.close(fig)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    os.environ.setdefault("MPLBACKEND", "Agg")
    warnings.filterwarnings("ignore", message="Mean of empty slice")  # every metric of an empty cell
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=None, help="one leg: 1, 2 or 3")
    args = parser.parse_args()
    print(f"tree: {plotting.__file__}")
    legs = [("1", leg_1), ("2", leg_2), ("3", leg_3)]
    if args.only:
        legs = [(tag, leg) for tag, leg in legs if tag == args.only]
        if not legs:
            sys.exit(f"unknown leg {args.only!r}")
    for tag, leg in legs:
        try:
            leg()
        except Exception as error:  # a raise is a FAIL line, and the later legs still report
            check(f"({tag}) ran without raising", False, f"{type(error).__name__}: {error}")
    if not FAIL:
        print("A71 PASS")
    else:
        print(f"A71 FAIL: {FAIL}")
        sys.exit(1)
