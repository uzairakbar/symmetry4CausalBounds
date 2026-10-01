"""A84: under `im-ci` the sweep figures draw width and worst error off the raw point
bounds and coverage off the IM-CI (no experiment is run; seconds).

The IM-CI is raw +- C SE with the budgets fixed per cell, so a method whose
constraints are a superset of another's (the T-as-IV DA+PI+IV(T) over DA+PI) has
nested raw bounds but can draw a wider CI through its bootstrap SE alone. Sharpness
(width, worst error) is read off the bounds, validity (coverage, approx_error) off
the CI: `MetricSpec.raw`, applied by `configs.sweep_record_for` at both call sites.
Legs:

  (i)   static: the `raw` metrics are exactly width and worst_error.
        Catches: a metric moved to the other record.
  (ii)  `_run_sweeps` on a stub orchestrator holding a synthetic CI record and a
        distinct raw one: the width and worst-error figures are handed the raw
        record, coverage and approx_error the CI's; with no CI record (`im-ci` 0)
        every figure is handed `results`. Mutations: `sweep_record_for` patched to
        always return the CI record, then always the raw one; each must fail a check.
        Catches: a call site that ignores the rule or the fallback.
  (iii) the aggregate `sweep_grid` on a temp tree holding `<param>_results.pkl`
        (CI) and `<param>_results_raw.pkl` (raw): its width and worst-error rows
        draw exactly what a tree holding the raw record alone as `_results.pkl`
        draws, its coverage row exactly what the CI-only tree draws, and the width
        row differs from the CI-only tree's. Same two mutations.
        Catches: the aggregate drifting from the per-recipe figure.
  (iv)  `--artifacts DIR` (read-only): for every `<dataset>/sweep/<param>` record
        (the raw one when written, else `_results.pkl`, which then is raw), each
        T-as-IV method's width and worst error <= its non-T counterpart's
        (DA+PI+IV(T) vs DA+PI, DA+PI+IV(T,Z) vs DA+PI+IV(Z), and the PI& forms) in
        every (step, experiment) cell where both are finite, to a relative
        tolerance of `NESTED_RTOL`. SKIP without `--artifacts`.
        Catches: a raw record whose nesting broke, i.e. the T-as-IV panels would
        read wider than DA's for a reason other than the CI.

Writes only into a fresh directory under `~/scratch/tmp/a84/`, removed when it
passes.

    MPLBACKEND=Agg python scripts/a84_raw_bound_panels.py [--artifacts DIR]
"""

import argparse
import glob
import os
import pickle
import re
import shutil
import sys
import tempfile
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from loguru import logger  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from src import aggregate  # noqa: E402
from src.experiments import base  # noqa: E402
from src.experiments.configs import METRIC_SPECS, PARAM_SPECS, SweepSpec  # noqa: E402
from src.experiments.utils.constants import SUBDIR_SWEEP  # noqa: E402
from src.experiments.utils.data_operations import load  # noqa: E402

N_STEPS = 5
N_EXPERIMENTS = 6
METHODS = ("PI", "DA+PI", "DA+PI+IV(T)")
PARAM = "epsilon"
SWEEP_METRICS = ("coverage", "approx_error", "worst_error", "width")
RAW_METRICS = {"width", "worst_error"}
NESTED_RTOL = 1e-6
TMPROOT = os.path.expanduser("~/scratch/tmp/a84")
FAIL = []
SKIP = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAIL.append(name)


def skip(name, why):
    print(f"  [SKIP] {name}: {why}")
    SKIP.append(name)


def records(n_steps=N_STEPS):
    """(CI, raw) records: every field of every method distinct between the two,
    the CI wider than the raw bounds, PI the widest."""
    rng = np.random.default_rng(0)
    raw, ci = {}, {}
    for k, name in enumerate(METHODS):
        raw[name], ci[name] = {}, {}
        for metric in SWEEP_METRICS:
            key = METRIC_SPECS[metric].key
            value = (3.0 - k) * (1.0 + 0.2 * rng.random((n_steps, N_EXPERIMENTS)))
            if metric == "coverage":
                value = np.clip(0.3 + 0.1 * value, 0, 1)
            raw[name][key] = value
            ci[name][key] = np.clip(value * 1.5, 0, 1) if metric == "coverage" else value * (1.5 + 0.5 * k)
    return ci, raw


def always(which):
    """A `sweep_record_for` that ignores the rule: always the CI record, or always raw."""

    def pick(metric, results, results_raw=None):
        return results if which == "ci" or results_raw is None else results_raw

    return pick


class patched:
    """Patch `sweep_record_for` where both call sites read it."""

    def __init__(self, fn):
        self.fn = fn

    def __enter__(self):
        self.saved = base.sweep_record_for, aggregate.sweep_record_for
        base.sweep_record_for = aggregate.sweep_record_for = self.fn

    def __exit__(self, *exc):
        base.sweep_record_for, aggregate.sweep_record_for = self.saved


def leg_i():
    print("(i) the raw metrics")
    have = {m for m, spec in METRIC_SPECS.items() if spec.raw}
    check(f"(i) the raw metrics are {sorted(RAW_METRICS)}", have == RAW_METRICS, f"{sorted(have)}")
    check("(i) none of them is a perf metric", not any(METRIC_SPECS[m].perf_only for m in have))


def run_sweeps(tmp, ci, raw):
    """{metric: the y_results `_run_sweeps` hands create_sweep_plot}, on a stub
    orchestrator whose CI record is `raw` when `raw` is not None."""
    x = np.asarray(PARAM_SPECS[PARAM].grid_fn("simulation", N_STEPS), dtype=float)
    stub = SimpleNamespace(
        name="simulation",
        has_z=False,
        kwargs={"normalize": True},
        sweep_record=lambda param: (x, ci, {}),
        _sweep_axis={},
        _sweep_ci={PARAM: None if raw is None else {"results_raw": raw, "level": 95.0}},
        _sweep_xlabel={PARAM: PARAM_SPECS[PARAM].xlabel},
        _sweep_vlines={},
        _sweep_xticks={},
    )
    drawn = {}
    original = base.create_sweep_plot

    def spy(x_values, y_results, *args, fname=None, **kwargs):
        drawn[fname[len(PARAM) + 1 :]] = {name: np.array(v) for name, v in y_results.items()}
        plt.close("all")

    cwd = os.getcwd()
    os.chdir(tmp)
    base.create_sweep_plot = spy
    try:
        base.ExperimentOrchestrator._run_sweeps(stub, SweepSpec(param=(PARAM,), metric=SWEEP_METRICS))
    finally:
        base.create_sweep_plot = original
        os.chdir(cwd)
    return drawn


def handed(drawn, metric, record):
    key = METRIC_SPECS[metric].key
    figure = drawn.get(metric, {})
    return bool(figure) and all(np.array_equal(figure[n], record[n][key], equal_nan=True) for n in figure)


def sweeps_ok(tmp, label):
    """[(check name, ok)] of leg (ii) on the current `sweep_record_for`."""
    ci, raw = records()
    drawn = run_sweeps(tmp, ci, raw)
    out = []
    for metric in SWEEP_METRICS:
        want, other = (raw, ci) if metric in RAW_METRICS else (ci, raw)
        source = "raw" if metric in RAW_METRICS else "CI"
        out.append((f"({label}) _run_sweeps hands {metric} the {source} record", handed(drawn, metric, want)))
        out.append((f"({label}) ... not the other one", not handed(drawn, metric, other)))
    alone = run_sweeps(tmp, ci, None)
    out.append(
        (
            f"({label}) with no CI record every figure is handed `results`",
            all(handed(alone, m, ci) for m in SWEEP_METRICS),
        )
    )
    return out


def leg_ii(tmp):
    print("(ii) _run_sweeps")
    for name, ok in sweeps_ok(tmp, "ii"):
        check(name, ok)
    for which in ("ci", "raw"):
        with patched(always(which)):
            failed = [name for name, ok in sweeps_ok(tmp, f"ii, always {which}") if not ok]
        check(f"(ii) mutation: always the {which} record fails a check", bool(failed), f"{failed[:1]}")


def write_tree(root, results, results_raw=None):
    x = np.asarray(PARAM_SPECS[PARAM].grid_fn("simulation", N_STEPS), dtype=float)
    folder = f"{root}/simulation/{SUBDIR_SWEEP}"
    os.makedirs(folder, exist_ok=True)
    stems = [(f"{PARAM}_values", x), (f"{PARAM}_results", results)]
    if results_raw is not None:
        stems.append((f"{PARAM}_results_raw", results_raw))
    for stem, obj in stems:
        with open(f"{folder}/{stem}.pkl", "wb") as handle:
            pickle.dump(obj, handle)
    return root


def cells(root):
    """{row metric: [every line's ydata and fill's vertices]} of the aggregate grid."""
    fig = aggregate.sweep_grid(PARAM, ["simulation"], root, None, hz={"simulation": False})
    out = {}
    for (metric, _), ax in zip(aggregate.ROWS, fig.axes, strict=False):
        drawn = [np.asarray(line.get_ydata(), dtype=float) for line in ax.lines]
        drawn += [c.get_paths()[0].vertices for c in ax.collections if isinstance(c, PolyCollection)]
        out[metric] = drawn
    plt.close(fig)
    return out


def same(a, b):
    return len(a) == len(b) and len(a) > 0 and all(np.allclose(p, q, equal_nan=True) for p, q in zip(a, b, strict=True))


def grid_ok(trees, label):
    both, ci_only, raw_only = (cells(t) for t in trees)
    out = []
    for metric, _ in aggregate.ROWS:
        reference, source = (raw_only, "raw") if metric in RAW_METRICS else (ci_only, "CI")
        out.append((f"({label}) the {metric} row draws the {source} record", same(both[metric], reference[metric])))
    out.append((f"({label}) the width row is not the CI-only tree's", not same(both["width"], ci_only["width"])))
    return out


def leg_iii(tmp):
    print("(iii) the aggregate sweep grid")
    ci, raw = records()
    trees = (
        write_tree(f"{tmp}/both", ci, raw),
        write_tree(f"{tmp}/ci_only", ci),
        write_tree(f"{tmp}/raw_only", raw),
    )
    for name, ok in grid_ok(trees, "iii"):
        check(name, ok)
    for which in ("ci", "raw"):
        with patched(always(which)):
            failed = [name for name, ok in grid_ok(trees, f"iii, always {which}") if not ok]
        check(f"(iii) mutation: always the {which} record fails a check", bool(failed), f"{failed[:1]}")


def counterpart(name):
    """The non-T method a T-as-IV one adds its constraint to: DA+PI+IV(T) -> DA+PI,
    DA+PI+IV(T,Z) -> DA+PI+IV(Z); None for a method without T among its instruments."""
    match = re.fullmatch(r"(.*)\+IV\(([^)]*)\)", name)
    if not match or "T" not in match.group(2).split(","):
        return None
    rest = [m for m in match.group(2).split(",") if m != "T"]
    return f"{match.group(1)}+IV({','.join(rest)})" if rest else match.group(1)


def leg_iv(artifacts):
    print("(iv) the T-as-IV raw bounds nest inside their counterparts'")
    if not artifacts:
        skip("(iv)", "no --artifacts")
        return
    pairs, cells_checked = 0, 0
    found = sorted(glob.glob(f"{artifacts}/*/{SUBDIR_SWEEP}/*_results.pkl"))
    if not found:
        skip("(iv)", f"no sweep pkls under {artifacts}")
        return
    for path in found:
        raw_path = path[: -len(".pkl")] + "_raw.pkl"
        record = load(raw_path if os.path.exists(raw_path) else path)
        where = os.path.relpath(path, artifacts)[: -len("_results.pkl")]
        for name in record:
            other = counterpart(name)
            if other is None or other not in record:
                continue
            pairs += 1
            for metric in sorted(RAW_METRICS):
                key = METRIC_SPECS[metric].key
                t, d = np.asarray(record[name][key], dtype=float), np.asarray(record[other][key], dtype=float)
                both = np.isfinite(t) & np.isfinite(d)
                cells_checked += int(both.sum())
                over = both & (t > d + NESTED_RTOL * np.abs(d))
                excess = float(np.max((t - d)[over] / np.abs(d[over]).clip(1e-300))) if over.any() else 0.0
                check(
                    f"(iv) {where} {name} {metric} <= {other}'s",
                    not over.any(),
                    f"({int(over.sum())}/{int(both.sum())} cells over; max relative excess {excess:.2e})"
                    if over.any()
                    else "",
                )
    check("(iv) some T-as-IV pair compared", pairs > 0, f"{pairs} pairs, {cells_checked} finite cells")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifacts", default=None, help="a real artifacts tree for leg (iv), read only")
    args = parser.parse_args()
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    os.chdir(REPO)
    os.makedirs(TMPROOT, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="a84_", dir=TMPROOT)
    leg_i()
    leg_ii(tmp)
    leg_iii(tmp)
    leg_iv(args.artifacts and os.path.abspath(os.path.expanduser(args.artifacts)))
    skipped = f" ({len(SKIP)} SKIPPED: {', '.join(SKIP)})" if SKIP else ""
    if not FAIL:
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"\nA84 ALL PASS{skipped}")
    else:
        print(f"\nA84 FAILURES: {', '.join(FAIL)} (tree in {tmp}){skipped}")
        sys.exit(1)
