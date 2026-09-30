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
        Catches: a band over the raw columns, a changed level or resample count.
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
        range, and BAND_PERCENTILES is (2.5, 97.5).

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
from src.experiments.utils.plotting import BAND_PERCENTILES, create_sweep_plot  # noqa: E402

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
    if not FAIL:
        shutil.rmtree(tmp, ignore_errors=True)
        print("\nA83 ALL PASS")
    else:
        print(f"\nA83 FAILURES: {', '.join(FAIL)} (tree in {tmp})")
        sys.exit(1)
