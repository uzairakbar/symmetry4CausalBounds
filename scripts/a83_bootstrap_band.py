"""A83: the sweep band is the 95 % percentile-bootstrap CI of the mean over
experiments (no experiment is run; seconds).

Every param sweep figure, per recipe and in the aggregate grid, draws one mean line
per method and a band of `BAND_PERCENTILES` (2.5, 97.5) taken across
`bootstrap`'s BOOTSTRAP_RESAMPLES resample means (the experiments resampled with
replacement per step and method, seed BOOTSTRAP_SEED), not across the raw
experiments. Legs:

  (i)   `create_sweep_plot(savefig=False)` on a synthetic (n_steps, 10) record per
        method: the drawn band, read off the axes' fill, equals the percentiles of
        `bootstrap(y)` computed here, differs from the raw percentiles, and is
        narrower than them at every step.
        Catches: a band over the raw columns. Misses: a changed level, resample
        count or seed, since the band and its expectation read the same
        constants; (iv) pins those.
  (ii)  the aggregate `sweep_grid` on a temp artifacts tree holding the same kind
        of record: its coverage cell draws the same bootstrap band.
        Catches: an aggregate that drifts from the per-recipe figure.
  (iii) static: `create_sweep_plot` defaults to `bootstrapped=True`, every call
        in `_run_sweeps` leaves it at that, and every other call site in `src/`
        is one of the known non-sweep ones (the perf figures in `_run_perf`, the
        cigarette per-query width ratio); `sweep_grid` calls `bootstrap`.
        Catches: a param sweep switched to raw bands, a new unlisted call site.
  (iv)  `bootstrap` is deterministic under BOOTSTRAP_SEED (another seed moves it),
        returns BOOTSTRAP_RESAMPLES resample means per row, each inside the row's
        range; the constants are pinned: BAND_PERCENTILES is (2.5, 97.5),
        BOOTSTRAP_RESAMPLES 1000 and BOOTSTRAP_SEED 0.
        Catches: a changed level, resample count or seed.
  (v)   the normalised rows (SS10.1): each experiment is divided by its own
        baseline BEFORE the bootstrap. `create_sweep_plot(normalize=True)` on a
        synthetic width record whose experiments differ in scale, and the
        aggregate `sweep_grid`'s width and worst-error cells, draw the
        `BAND_PERCENTILES` of `bootstrap` of the per-experiment ratios; the
        baseline's line and band read exactly 1.0. Mutation: the band of the old
        ratio of means (bootstrap first, then every resample divided by the
        baseline's per-step mean) is computed here and must differ from the
        expected band at some step and fail the same comparison against the drawn
        one. Catches: a normalisation moved back after the bootstrap, a divisor
        pooled over experiments. Misses: nothing the bands do not show.

Writes only into a fresh directory under `~/scratch/tmp/a83/`, removed when it
passes.

    MPLBACKEND=Agg python scripts/a83_bootstrap_band.py
"""

import ast
import inspect
import os
import pickle
import shutil
import sys
import tempfile

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from loguru import logger  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from src import aggregate  # noqa: E402
from src.experiments.configs import METRIC_SPECS, PARAM_SPECS  # noqa: E402
from src.experiments.utils.constants import SUBDIR_SWEEP  # noqa: E402
from src.experiments.utils.data_operations import (  # noqa: E402
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    bootstrap,
)
from src.experiments.utils.plotting import BAND_PERCENTILES, create_sweep_plot, normalize_sweep  # noqa: E402

N_STEPS = 6
N_EXPERIMENTS = 10
METHODS = ("PI", "DA+PI")
PARAM = "epsilon"
METRICS = ("coverage", "width", "worst_error")
ATOL = 1e-12
# the create_sweep_plot call sites that are not param sweeps: (file, function)
NON_SWEEP_SITES = {
    ("src/experiments/base.py", "_run_perf"),
    ("src/experiments/cigarettes.py", "_plot_width_ratio"),
}
TMPROOT = os.path.expanduser("~/scratch/tmp/a83")
FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAIL.append(name)


def synthetic(seed=1, n_steps=N_STEPS):
    """(n_steps, n_experiments) rates per method, each experiment offset by its own
    level so the raw spread is wide and the spread of the mean narrow."""
    rng = np.random.default_rng(seed)
    return {
        name: np.clip(
            0.5 + 0.3 * rng.standard_normal(N_EXPERIMENTS)[None, :] + 0.05 * rng.random((n_steps, N_EXPERIMENTS)), 0, 1
        )
        for name in METHODS
    }


def bands_on(ax, x):
    """{fill index: (low, high)} per x, read off each fill polygon's vertices."""
    out = []
    for collection in ax.collections:
        if not isinstance(collection, PolyCollection):
            continue
        vertices = collection.get_paths()[0].vertices
        low = np.array([vertices[np.isclose(vertices[:, 0], xi), 1].min() for xi in x])
        high = np.array([vertices[np.isclose(vertices[:, 0], xi), 1].max() for xi in x])
        out.append((low, high))
    return out


def expected(y):
    """The bootstrap band and the raw band of each method, in drawing order."""
    boot = bootstrap(y)
    return (
        [tuple(np.nanpercentile(boot[m], BAND_PERCENTILES, axis=1)) for m in y],
        [tuple(np.nanpercentile(np.asarray(y[m]), BAND_PERCENTILES, axis=1)) for m in y],
    )


def compare(leg, drawn, boot, raw):
    check(f"({leg}) one band per method", len(drawn) == len(boot), f"{len(drawn)}")
    if len(drawn) != len(boot):
        return
    same = all(
        np.allclose(d, b, atol=ATOL) for dd, bb in zip(drawn, boot, strict=True) for d, b in zip(dd, bb, strict=True)
    )
    check(f"({leg}) the band is the {BAND_PERCENTILES} percentiles of bootstrap(y)", same)
    differs = all(not np.allclose(dd[0], rr[0]) for dd, rr in zip(drawn, raw, strict=True))
    check(f"({leg}) the band is not the raw percentiles", differs)
    narrower = all(np.all(dd[1] - dd[0] < rr[1] - rr[0]) for dd, rr in zip(drawn, raw, strict=True))
    ratio = np.mean([np.mean((dd[1] - dd[0]) / (rr[1] - rr[0])) for dd, rr in zip(drawn, raw, strict=True)])
    check(f"({leg}) narrower than the raw band at every step", narrower, f"mean width ratio {ratio:.3f}")


def errors_during(fn):
    """Run `fn`, returning its value and the ERROR records it logged (the plotting
    code swallows its exceptions into the log)."""
    records = []
    sink = logger.add(lambda message: records.append(message.record), level="ERROR")
    try:
        value = fn()
    finally:
        logger.remove(sink)
    return value, records


def leg_i():
    print("(i) create_sweep_plot draws the bootstrap band")
    x = np.linspace(0.5, 2.0, N_STEPS)
    y = synthetic()
    plt.close("all")
    _, errors = errors_during(
        lambda: create_sweep_plot(
            x, y, xlabel="r", fname="a83_band_coverage", savefig=False, normalize=False, has_z=False
        )
    )
    check("(i) no plotting error swallowed", not errors, f"{[e['message'][:80] for e in errors]}")
    drawn = bands_on(plt.gcf().axes[0], x)
    compare("i", drawn, *expected(y))
    plt.close("all")


def leg_ii(tmp):
    print("(ii) the aggregate sweep grid draws the same band")
    x = np.asarray(PARAM_SPECS[PARAM].grid_fn("simulation", N_STEPS), dtype=float)
    rng = np.random.default_rng(2)
    record = {
        name: {METRIC_SPECS[m].key: synthetic(seed=int(rng.integers(1 << 30)), n_steps=len(x))[name] for m in METRICS}
        for name in METHODS
    }
    folder = f"{tmp}/simulation/{SUBDIR_SWEEP}"
    os.makedirs(folder, exist_ok=True)
    for stem, obj in ((f"{PARAM}_values", x), (f"{PARAM}_results", record)):
        with open(f"{folder}/{stem}.pkl", "wb") as handle:
            pickle.dump(obj, handle)
    fig, errors = errors_during(
        lambda: aggregate.sweep_grid(PARAM, ["simulation"], tmp, None, hz={"simulation": False})
    )
    check("(ii) no error logged", not errors, f"{[e['message'][:80] for e in errors]}")
    key = METRIC_SPECS["coverage"].key
    y = {name: np.asarray(rec[key]) for name, rec in record.items()}
    order = np.argsort(x, kind="stable")
    drawn = bands_on(fig.axes[0], x[order])
    compare("ii", drawn, *expected({name: v[order] for name, v in y.items()}))
    plt.close(fig)


def calls_in(path, name):
    """[(enclosing function path, e.g. `sweep_grid.load_cell`, {keyword: node})] of
    every call to `name` in `path`."""
    with open(os.path.join(REPO, path)) as handle:
        tree = ast.parse(handle.read())
    found = []

    def visit(node, where):
        for child in ast.iter_child_nodes(node):
            inner = (
                (f"{where}.{child.name}" if where else child.name)
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                else where
            )
            if isinstance(child, ast.Call):
                func = child.func
                called = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
                if called == name:
                    found.append((where, {kw.arg: kw.value for kw in child.keywords}))
            visit(child, inner)

    visit(tree, None)
    return found


def leg_iii():
    print("(iii) every param sweep figure is bootstrapped")
    default = inspect.signature(create_sweep_plot).parameters["bootstrapped"].default
    check("(iii) create_sweep_plot defaults to bootstrapped=True", default is True, f"{default!r}")
    sites = []
    for root, _, files in os.walk(os.path.join(REPO, "src")):
        for file in files:
            if file.endswith(".py"):
                path = os.path.relpath(os.path.join(root, file), REPO)
                sites += [(path, where.split(".")[-1], kws) for where, kws in calls_in(path, "create_sweep_plot")]
    sweep = [(p, w, k) for p, w, k in sites if w == "_run_sweeps"]
    check("(iii) _run_sweeps draws through create_sweep_plot", len(sweep) >= 1, f"{len(sweep)} call(s)")
    raw = [
        (p, w)
        for p, w, k in sweep
        if "bootstrapped" in k and not (isinstance(k["bootstrapped"], ast.Constant) and k["bootstrapped"].value is True)
    ]
    check("(iii) no _run_sweeps call turns the bootstrap off", not raw, f"{raw}")
    others = {(p, w) for p, w, _ in sites if w != "_run_sweeps"}
    check(
        "(iii) every other call site is a known non-sweep one",
        others <= NON_SWEEP_SITES,
        f"{sorted(others - NON_SWEEP_SITES)}",
    )
    grid = [w for w, _ in calls_in("src/aggregate.py", "bootstrap") if w.split(".")[0] == "sweep_grid"]
    check("(iii) the aggregate sweep grid calls bootstrap", bool(grid), f"{grid}")


def leg_iv():
    print("(iv) the resampling itself")
    y = synthetic(seed=3)
    one, two = bootstrap(y), bootstrap(y)
    check("(iv) deterministic under BOOTSTRAP_SEED", all(np.array_equal(one[m], two[m]) for m in y))
    other = bootstrap(y, seed=BOOTSTRAP_SEED + 1)
    check("(iv) another seed moves it", not all(np.array_equal(one[m], other[m]) for m in y))
    default = inspect.signature(bootstrap).parameters["n_samples"].default
    check("(iv) the default resample count is BOOTSTRAP_RESAMPLES", default == BOOTSTRAP_RESAMPLES, f"{default}")
    shapes = {m: one[m].shape for m in y}
    check(
        f"(iv) (n_steps, {BOOTSTRAP_RESAMPLES}) resample means per method",
        all(s == (N_STEPS, BOOTSTRAP_RESAMPLES) for s in shapes.values()),
        f"{shapes}",
    )
    inside = all(
        np.all(one[m] >= y[m].min(axis=1, keepdims=True) - ATOL)
        and np.all(one[m] <= y[m].max(axis=1, keepdims=True) + ATOL)
        for m in y
    )
    check("(iv) every resample mean inside its row's range", inside)
    check("(iv) BAND_PERCENTILES is (2.5, 97.5)", tuple(BAND_PERCENTILES) == (2.5, 97.5), f"{BAND_PERCENTILES}")
    check("(iv) BOOTSTRAP_RESAMPLES is 1000", BOOTSTRAP_RESAMPLES == 1000, f"{BOOTSTRAP_RESAMPLES}")
    check("(iv) BOOTSTRAP_SEED is 0", BOOTSTRAP_SEED == 0, f"{BOOTSTRAP_SEED}")


def scaled(seed=4, n_steps=N_STEPS):
    """(n_steps, n_experiments) widths per method whose experiments differ in scale
    and whose DA+PI ratio to PI is tied to that scale, so the mean of the ratios and
    the ratio of the means part."""
    rng = np.random.default_rng(seed)
    level = np.exp(1.5 * rng.standard_normal(N_EXPERIMENTS))[None, :]
    base = level * (1.0 + 0.1 * rng.random((n_steps, N_EXPERIMENTS)))
    share = 0.3 + 0.6 * (level / level.max()) + 0.05 * rng.random((n_steps, N_EXPERIMENTS))
    return {"PI": base, "DA+PI": base * share}


def expected_normalized(y, fname):
    """The per-experiment band (bootstrap of the ratios) and the old ratio-of-means
    band (bootstrap, then divided by the baseline's per-step mean), in drawing order."""
    ratios, baseline = normalize_sweep(y, fname)
    boot = bootstrap(ratios)
    raw = bootstrap(y)
    divisor = np.nanmean(raw[baseline], axis=1, keepdims=True)
    return (
        [tuple(np.nanpercentile(boot[m], BAND_PERCENTILES, axis=1)) for m in y],
        [tuple(np.nanpercentile(raw[m] / divisor, BAND_PERCENTILES, axis=1)) for m in y],
    )


def same_bands(drawn, want):
    return len(drawn) == len(want) and all(
        np.allclose(d, w, atol=1e-9) for dd, ww in zip(drawn, want, strict=True) for d, w in zip(dd, ww, strict=True)
    )


def compare_normalized(leg, drawn, per_experiment, pooled):
    check(f"({leg}) one band per method", len(drawn) == len(per_experiment), f"{len(drawn)}")
    check(
        f"({leg}) the band is the {BAND_PERCENTILES} percentiles of bootstrap(per-experiment ratios)",
        same_bands(drawn, per_experiment),
    )
    base_low, base_high = drawn[0] if drawn else (np.nan, np.nan)
    check(
        f"({leg}) the baseline's band reads exactly 1.0 (no band)",
        bool(np.all(base_low == 1.0) and np.all(base_high == 1.0)),
        f"{base_low} {base_high}",
    )
    # the mutation: the old ratio of means is a different band, and this very
    # comparison rejects it
    parts = any(
        not np.allclose(e, o, atol=1e-3)
        for ee, oo in zip(per_experiment, pooled, strict=True)
        for e, o in zip(ee, oo, strict=True)
    )
    check(f"({leg}) mutation: the ratio-of-means band differs from the expected one", parts)
    check(f"({leg}) mutation: the drawn band is not the ratio-of-means band", not same_bands(drawn, pooled))


def leg_v(tmp):
    print("(v) normalised rows: each experiment divided by its own baseline, then bootstrapped")
    x = np.linspace(0.5, 2.0, N_STEPS)
    y = scaled()
    plt.close("all")
    _, errors = errors_during(
        lambda: create_sweep_plot(x, y, xlabel="r", fname="a83_band_width", savefig=False, normalize=True, has_z=False)
    )
    check("(v) no plotting error swallowed", not errors, f"{[e['message'][:80] for e in errors]}")
    ax = plt.gcf().axes[0]
    line = np.asarray(ax.lines[0].get_ydata(), dtype=float)
    check("(v) the baseline's line reads exactly 1.0", bool(np.all(line == 1.0)), f"{line}")
    compare_normalized("v", bands_on(ax, x), *expected_normalized(y, "a83_band_width"))
    plt.close("all")

    grid = np.asarray(PARAM_SPECS[PARAM].grid_fn("simulation", N_STEPS), dtype=float)
    record = {
        name: {METRIC_SPECS[m].key: scaled(seed=5 + i, n_steps=len(grid))[name] for i, m in enumerate(METRICS)}
        for name in METHODS
    }
    folder = f"{tmp}/normalized/simulation/{SUBDIR_SWEEP}"
    os.makedirs(folder, exist_ok=True)
    for stem, obj in ((f"{PARAM}_values", grid), (f"{PARAM}_results", record)):
        with open(f"{folder}/{stem}.pkl", "wb") as handle:
            pickle.dump(obj, handle)
    fig, errors = errors_during(
        lambda: aggregate.sweep_grid(PARAM, ["simulation"], f"{tmp}/normalized", None, hz={"simulation": False})
    )
    check("(v) aggregate: no error logged", not errors, f"{[e['message'][:80] for e in errors]}")
    order = np.argsort(grid, kind="stable")
    for row, metric in enumerate(METRICS):
        if metric == "coverage":
            continue
        key = METRIC_SPECS[metric].key
        y = {name: np.asarray(rec[key])[order] for name, rec in record.items()}
        drawn = bands_on(fig.axes[row], grid[order])
        compare_normalized(f"v {metric}", drawn, *expected_normalized(y, f"{PARAM}_{metric}"))
    plt.close(fig)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    os.chdir(REPO)
    os.makedirs(TMPROOT, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="a83_", dir=TMPROOT)
    leg_i()
    leg_ii(tmp)
    leg_iii()
    leg_iv()
    leg_v(tmp)
    if not FAIL:
        shutil.rmtree(tmp, ignore_errors=True)
        print("\nA83 ALL PASS")
    else:
        print(f"\nA83 FAILURES: {', '.join(FAIL)} (tree in {tmp})")
        sys.exit(1)
