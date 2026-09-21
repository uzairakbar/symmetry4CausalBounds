"""A70: the feasibility perf sweep, and the failure markers gone from the stability figure.

Round 16 adds `feasibility` beside `seed_var`: per method, grid point and query, the
share of the installed conic backends (the seed_var runs) whose bound is usable,
i.e. status OK, both ends finite and lower <= upper (`perf._failed`'s complement).
The two metrics read ONE set of backend runs (`perf.backend_runs`, built once in
`perf_sweeps`). A method the data refutes reads 0 on the feasibility figure where the
stability figure has a gap, so the stability figure loses its failure markers
(`FAILURE_MARKER`, `_nearest_finite` and every `failures=` argument are removed).
Legs:

  1. config: `feasibility` parses under `perf` and is rejected under `sweep`; its
     MetricSpec is perf-only, linear, labelled "feasible rate (over backends)";
     `normalize` on `epsilon_feasibility` raises in `validate_plot_keys`, the check
     `constants.py` runs at import. Catches: the spec missing or not perf-only, the
     `_feasibility` rejection dropped.
  2. formula, on a64 (iv)'s synthetic record (3 runs, 2 steps, 4 queries; a
     FAILURE status and a lower > upper planted at step 1, then a third failure):
     feasibility by hand ([1, 1, 1, 1] and [1, 2/3, 1, 2/3], then [1, 1/3, 1, 2/3]),
     `R * (1 - feasibility).sum(axis=1)` equal to `solver_stability`'s failure
     count, and `seed_var(runs=...)` equal to `solver_stability` on the same stack.
     Catches: the mean taken over queries instead of runs, a clause of `_failed`
     dropped on one side only.
  3. the simulation perf path end to end (5 methods, 4 sweep samples, 1 experiment,
     serial), run twice under the orchestrator: seed_var alone, then seed_var and
     feasibility. `epsilon_feasibility_results.pkl` is (grid points, n_queries) per
     method with values in [0, 1]; `PI+INV` reads 0 at every step where all its
     runs fail (RECORDED) and 1 at r = 1; the per-dataset figure, as `_run_perf`
     draws it, is on `CLAMP_YLIM` and linear; the seed_var terms, failures and
     statuses of the two runs are `array_equal`, so building the runs once moved no
     number and no RNG draw. Catches: the runs built twice or in another order,
     the clamp missing on `_feasibility`.
  4. the stability figure of the same run has no marker-only line and no count
     text; `create_sweep_plot` and `_draw_series` take no `failures`, and
     `plotting` has no `FAILURE_MARKER` or `_nearest_finite`.
  5. the aggregate at this round, on a synthetic tree (simulation perf with a
     failures pkl, seed_var and feasibility; cigarettes perf with seed_var only):
     the CLI writes `epsilon_feasibility.pdf` beside the other two perf pdfs; the
     feasibility figure is one row, its simulation panel on `CLAMP_YLIM`, its
     cigarette panel blank; `epsilon_seed_var.pdf` draws no marker-only line and
     no count text although the failures pkl is there.
  6. completeness: `git grep -e 'failures=' -e FAILURE_MARKER -e _nearest_finite
     -- src scripts` matches nothing outside this file, apart from `perf.py`'s
     `PerfRecord.failures` field and the `record.failures` save.
  7. do-MNIST perf still logs and skips whatever it is asked, read off the source
     text with `ast` (nothing is imported from `do_mnist`).

    MPLBACKEND=Agg python scripts/a70_perf_feasibility.py [--only LEG]

Leg 3 is the slow one (two perf runs on the simulation block, about a minute). Leg 7
never imports do-MNIST; leg 3 imports `src.main`, which imports it statically as every
recipe run does, and runs the simulation orchestrator only.
"""

import argparse
import ast
import inspect
import os
import pickle
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from loguru import logger

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import digest_leg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from munch import munchify  # noqa: E402

import src.experiments.base as base  # noqa: E402
import src.experiments.perf as perf  # noqa: E402
import src.experiments.utils.plotting as plotting  # noqa: E402
from src import aggregate  # noqa: E402
from src.experiments.configs import (  # noqa: E402
    METRIC_SPECS,
    PARAM_SPECS,
    parse_experiment_plan,
    resolve_dataset_block,
)
from src.experiments.utils import set_seed  # noqa: E402
from src.experiments.utils.constants import (  # noqa: E402
    ARTIFACTS_DIRECTORY,
    CLAMP_YLIM,
    SUBDIR_PERF,
    plot_keys_for,
    validate_plot_keys,
)
from src.methods.sensitivity_models import SolveStatus  # noqa: E402

TMPROOT = os.path.expanduser("~/scratch/tmp/a70")
PERF_METHODS = ["PI", "PI+INV", "DA+PI", "DA+PI+IV(T)", "PI&DA+PI+IV(T)"]
YLABEL = "feasible rate (over backends)"
FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAIL.append(name)


def load(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)  # noqa: S301 - our own artifacts


def dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(obj, handle)


def marker_lines(ax):
    return [line for line in ax.get_lines() if line.get_linestyle() == "None" and line.get_marker() not in ("", "None")]


def sim_block():
    """a64 (viii)'s simulation block: five methods, 4 sweep samples, one experiment, serial."""
    block = {
        **digest_leg.TOGGLES,
        **digest_leg.BLOCKS["simulation"],
        "iv": 0,
        "methods": list(PERF_METHODS),
        "sweep_samples": 4,
        "n_experiments": 1,
        "n_jobs": 1,
    }
    return resolve_dataset_block("simulation", block)


# ------------------------------------------------------------------ legs


def leg_1():
    print("(1) config")
    spec = METRIC_SPECS.get("feasibility")
    check(
        "(1) METRIC_SPECS['feasibility'] is perf-only, linear, labelled by backend rate",
        spec is not None and spec.perf_only and spec.yscale == "linear" and spec.ylabel == YLABEL,
        f"{spec}",
    )
    plan = parse_experiment_plan({"perf": {"metric": ["seed_var", "feasibility"]}})
    check("(1) feasibility parses under perf", plan.perf.metric == ("seed_var", "feasibility"), f"{plan.perf}")
    try:
        parse_experiment_plan({"sweep": {"param": ["epsilon"], "metric": ["feasibility"]}})
        check("(1) feasibility is rejected under sweep", False, "no error")
    except ValueError as error:
        check("(1) feasibility is rejected under sweep", "feasibility" in str(error), str(error))
    try:
        validate_plot_keys("PLOT_CONFIGS['*']", {"epsilon_feasibility": {"normalize": True}}, plot_keys_for)
        check("(1) normalize on epsilon_feasibility raises", False, "no error")
    except ValueError as error:
        check("(1) normalize on epsilon_feasibility raises", "normalize" in str(error), str(error))


def synthetic_runs(lower, upper, status, width_pi, name="M"):
    """`backend_runs`'s shape from stacked (R, n_steps, n_queries) arrays: per run a
    ({name: (lower, upper, status)}, PI (lower, upper, status)) pair, the PI width
    `width_pi[r]` at every step."""
    n_steps = lower.shape[1]
    runs = []
    for r in range(lower.shape[0]):
        pi_lower = np.zeros((n_steps, lower.shape[2]))
        pi = (pi_lower, pi_lower + width_pi[r], np.zeros_like(pi_lower, dtype=int))
        runs.append(({name: (lower[r], upper[r], status[r])}, pi))
    return runs


def leg_2():
    print("(2) the formula on a64 (iv)'s synthetic record")
    n_runs, n_steps, n_queries = 3, 2, 4
    w_pi = np.tile(np.array([1.0, 1.0, 2.0, 2.0]), (n_runs, 1))
    d = np.array([0.05, 0.10, 0.15, 0.20]) * w_pi[0]
    centre_l, centre_u = np.array([0.0, 1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0, 4.0])
    offsets = np.array([-1.0, 0.0, 1.0])
    lower = np.zeros((n_runs, n_steps, n_queries))
    upper = np.zeros((n_runs, n_steps, n_queries))
    for r in range(n_runs):
        for i in range(n_steps):
            lower[r, i] = centre_l + offsets[r] * d
            upper[r, i] = centre_u + offsets[r] * d
    status = np.zeros((n_runs, n_steps, n_queries), dtype=int)

    def both(tag, want):
        runs = synthetic_runs(lower, upper, status, w_pi)
        feasible = perf.feasibility(runs)["M"]
        terms, failures = perf.solver_stability(lower, upper, status, w_pi)
        check(f"(2) {tag}: feasibility by hand", np.allclose(feasible, want, atol=1e-15), f"{np.round(feasible, 4)}")
        check(
            f"(2) {tag}: R * (1 - feasibility) summed over queries is the failure count",
            np.array_equal(np.rint(n_runs * (1.0 - feasible).sum(axis=1)).astype(int), failures),
            f"{n_runs * (1.0 - feasible).sum(axis=1)} vs {failures}",
        )
        got_terms, got_failures, got_status, backends = perf.seed_var(None, None, None, runs=runs, backends=["b"] * 3)
        check(
            f"(2) {tag}: seed_var(runs=...) is solver_stability on the same stack",
            np.array_equal(got_terms["M"], terms, equal_nan=True)
            and np.array_equal(got_failures["M"], failures)
            and np.array_equal(got_status["M"], status)
            and backends == ["b"] * 3,
        )

    both("no failure", np.ones((n_steps, n_queries)))
    status[2, 1, 1] = SolveStatus.FAILURE
    lower[0, 1, 3], upper[0, 1, 3] = upper[0, 1, 3] + 1.0, lower[0, 1, 3]  # lower > upper
    both("a FAILURE and a lower > upper", [[1, 1, 1, 1], [1, 2 / 3, 1, 2 / 3]])
    status[1, 1, 1] = SolveStatus.INFEASIBLE
    both("a third failure (INFEASIBLE)", [[1, 1, 1, 1], [1, 1 / 3, 1, 2 / 3]])
    upper[1, 0, 0] = np.inf  # a non-finite end counts too
    both("an infinite upper end", [[2 / 3, 1, 1, 1], [1, 1 / 3, 1, 2 / 3]])


class Spy:
    """`base.create_sweep_plot` wrapped: each perf figure's frame and artists, read off
    the figure `_run_perf` just drew."""

    def __init__(self):
        self.figures = {}
        self.original = base.create_sweep_plot

    def __call__(self, *args, **kwargs):
        self.original(*args, **kwargs)
        ax = plt.gcf().axes[0]
        self.figures[kwargs.get("fname")] = dict(
            ylim=tuple(float(v) for v in ax.get_ylim()),
            yscale=ax.get_yscale(),
            ylabel=ax.get_ylabel(),
            markers=len(marker_lines(ax)),
            texts=[t.get_text() for t in ax.texts],
            lines=len(ax.get_lines()),
        )
        plt.close("all")

    def __enter__(self):
        base.create_sweep_plot = self
        return self

    def __exit__(self, *exc):
        base.create_sweep_plot = self.original


def perf_run(metrics, workdir):
    """One orchestrator perf run on the simulation block, in `workdir`; returns the
    perf folder and the spy's figure records."""
    from src.main import ORCHESTRATORS

    os.makedirs(workdir, exist_ok=True)
    os.chdir(workdir)
    try:
        block = sim_block()
        plan = parse_experiment_plan({"perf": {"metric": list(metrics)}})
        set_seed(block["seed"])
        with Spy() as spy:
            ORCHESTRATORS["simulation"](**block, hyperparameters=munchify(digest_leg.HYPERPARAMETERS)).run(plan)
    finally:
        os.chdir(REPO)
    return os.path.join(workdir, ARTIFACTS_DIRECTORY, "simulation", SUBDIR_PERF), spy.figures


def leg_3_4():
    print("(3) the simulation perf path end to end; (4) the stability figure")
    os.makedirs(TMPROOT, exist_ok=True)
    root = tempfile.mkdtemp(prefix="perf_", dir=TMPROOT)
    alone, _ = perf_run(["seed_var"], f"{root}/alone")
    folder, figures = perf_run(["seed_var", "feasibility"], f"{root}/both")
    files = sorted(os.listdir(folder))
    check(
        "(3) the feasibility pkl and pdf are written beside seed_var's",
        {"epsilon_feasibility_results.pkl", "epsilon_feasibility_sweep.pdf", "epsilon_seed_var_sweep.pdf"} <= set(files)
        and "epsilon_feasibility_results.pkl" not in os.listdir(alone),
        f"{files}",
    )
    x = load(f"{folder}/epsilon_values.pkl")
    meta = load(f"{folder}/epsilon_perf_meta.pkl")
    n_queries, n_runs, points = meta["n_queries"], len(meta["backends"]), len(x)
    feasible = load(f"{folder}/epsilon_feasibility_results.pkl")
    failures = load(f"{folder}/epsilon_seed_var_failures.pkl")
    check(
        f"(3) feasibility is ({points}, {n_queries}) per method, the five methods",
        list(feasible) == PERF_METHODS and all(v.shape == (points, n_queries) for v in feasible.values()),
        f"{ {k: v.shape for k, v in feasible.items()} }",
    )
    check(
        "(3) every value in [0, 1] and a multiple of 1 / R",
        all(
            np.all((v >= 0) & (v <= 1)) and np.allclose(v * n_runs, np.rint(v * n_runs), atol=1e-12)
            for v in feasible.values()
        ),
        f"R = {n_runs}",
    )
    check(
        "(3) R * (1 - feasibility) summed over queries is the failures pkl, every method",
        all(
            np.array_equal(np.rint(n_runs * (1.0 - feasible[k]).sum(axis=1)).astype(int), failures[k])
            for k in PERF_METHODS
        ),
    )
    refuted = failures["PI+INV"] == n_runs * n_queries
    mid = int(np.argmin(np.abs(np.log(x))))
    print(f"      RECORDED PI+INV refuted at r = {np.round(x[refuted], 3).tolist()}")
    print(
        "      RECORDED feasibility per step: "
        + "; ".join(f"{k} {np.round(v.mean(axis=1), 3).tolist()}" for k, v in feasible.items())
    )
    check(
        "(3) PI+INV reads 0 at every all-failed step (at least one)",
        refuted.any() and np.all(feasible["PI+INV"][refuted] == 0.0),
        f"{int(refuted.sum())} step(s)",
    )
    check(
        "(3) PI+INV reads 1 at r = 1",
        np.isclose(x[mid], 1.0) and np.all(feasible["PI+INV"][mid] == 1.0),
        f"r {x[mid]:.3f}, {feasible['PI+INV'][mid].mean():.3f}",
    )
    fig = figures.get("epsilon_feasibility", {})
    check(
        f"(3) the per-dataset feasibility figure is linear on CLAMP_YLIM {CLAMP_YLIM}",
        fig.get("ylim") == CLAMP_YLIM and fig.get("yscale") == "linear" and fig.get("ylabel") == YLABEL,
        f"{fig}",
    )
    for name, kind in (("terms", "results"), ("failures", "failures"), ("statuses", "statuses")):
        a = load(f"{alone}/epsilon_seed_var_{kind}.pkl")
        b = load(f"{folder}/epsilon_seed_var_{kind}.pkl")
        check(
            f"(3) seed_var {name}: seed_var alone and beside feasibility are array_equal",
            list(a) == list(b) and all(np.array_equal(a[k], b[k], equal_nan=True) for k in a),
        )

    seed = figures.get("epsilon_seed_var", {})
    check(
        "(4) the stability figure has no marker-only line and no count text",
        seed and seed["markers"] == 0 and not seed["texts"],
        f"{seed}",
    )
    params = inspect.signature(plotting.create_sweep_plot).parameters
    check("(4) create_sweep_plot takes no failures", "failures" not in params)
    check("(4) _draw_series takes no failures", "failures" not in inspect.signature(plotting._draw_series).parameters)
    check(
        "(4) FAILURE_MARKER and _nearest_finite are gone",
        not hasattr(plotting, "FAILURE_MARKER") and not hasattr(plotting, "_nearest_finite"),
    )
    shutil.rmtree(root, ignore_errors=True)


def synthetic_tree(root):
    """simulation perf: seed_var, its failures pkl (PI+INV failing at two steps) and
    feasibility; cigarettes perf: seed_var only."""
    x = PARAM_SPECS["epsilon"].grid_fn("simulation", 4)
    points = len(x)
    rng = np.random.default_rng(3)
    marks = np.zeros(points, dtype=int)
    marks[0], marks[1] = 18, 2
    for dataset in ("simulation", "cigarettes"):
        folder = f"{root}/{dataset}/{SUBDIR_PERF}"
        dump(x, f"{folder}/epsilon_values.pkl")
        seed = {"PI": np.full((points, 6), 1e-9), "PI+INV": 1e-8 + 1e-10 * rng.random((points, 6))}
        seed["PI+INV"][0] = np.nan
        dump(seed, f"{folder}/epsilon_seed_var_results.pkl")
        dump({"PI": np.zeros(points, dtype=int), "PI+INV": marks}, f"{folder}/epsilon_seed_var_failures.pkl")
        dump({"xlabel": PARAM_SPECS["epsilon"].xlabel, "repeats": 3}, f"{folder}/epsilon_perf_meta.pkl")
    feasible = {"PI": np.ones((points, 6)), "PI+INV": np.ones((points, 6))}
    feasible["PI+INV"][0] = 0.0
    feasible["PI+INV"][1, :2] = 2 / 3
    dump(feasible, f"{root}/simulation/{SUBDIR_PERF}/epsilon_feasibility_results.pkl")
    return root


def leg_5():
    print("(5) the aggregate at this round, on a synthetic tree")
    os.makedirs(TMPROOT, exist_ok=True)
    root = synthetic_tree(tempfile.mkdtemp(prefix="tree_", dir=TMPROOT))
    out = f"{root}/aggregate"
    proc = subprocess.run(
        [sys.executable, "-m", "src.aggregate", "--artifacts", root, "--out", out],
        capture_output=True,
        text=True,
        env={**os.environ, "MPLBACKEND": "Agg"},
        cwd=REPO,
        timeout=600,
    )
    written = sorted(os.listdir(out)) if os.path.isdir(out) else []
    want = ["epsilon_feasibility.pdf", "epsilon_seed_var.pdf"]
    check(
        "(5) the CLI exits 0 and writes epsilon_feasibility.pdf beside epsilon_seed_var.pdf",
        proc.returncode == 0 and written == want and all(os.path.getsize(f"{out}/{f}") > 0 for f in written),
        f"{written} {proc.stderr.strip().splitlines()[-1][:160] if proc.returncode else ''}",
    )
    datasets = aggregate.columns(root)
    fig = aggregate.perf_row("feasibility", datasets, root)
    panels = [ax for ax in fig.axes if ax.get_title()]
    check(
        "(5) the feasibility figure is one row: simulation, cigarettes",
        len(panels) == 2 and [ax.get_title() for ax in panels] == ["simulation", "cigarettes"],
        f"{[ax.get_title() for ax in panels]}",
    )
    check(
        f"(5) the simulation panel is on CLAMP_YLIM {CLAMP_YLIM}, linear",
        panels[0].axison
        and tuple(float(v) for v in panels[0].get_ylim()) == CLAMP_YLIM
        and panels[0].get_yscale() == "linear",
        f"{panels[0].get_ylim()}",
    )
    check("(5) the cigarette panel is blank (no feasibility pkl)", not panels[1].axison)
    plt.close(fig)
    fig = aggregate.perf_row("seed_var", datasets, root)
    panels = [ax for ax in fig.axes if ax.get_title()]
    check(
        "(5) epsilon_seed_var: no marker-only line and no count text on any panel",
        all(not marker_lines(ax) and not ax.texts for ax in panels) and all(ax.axison for ax in panels),
        f"{[(len(marker_lines(ax)), len(ax.texts)) for ax in panels]}",
    )
    plt.close(fig)
    shutil.rmtree(root, ignore_errors=True)


# the only lines the completeness grep may match: the PerfRecord field and its save
ALLOWED = ("failures: dict[str, np.ndarray] = field(", "save(record.failures,")


def leg_6():
    print("(6) completeness: no failures= / FAILURE_MARKER / _nearest_finite left")
    proc = subprocess.run(
        [
            "git",
            "grep",
            "-n",
            "-e",
            "failures=",
            "-e",
            "FAILURE_MARKER",
            "-e",
            "_nearest_finite",
            "--",
            "src",
            "scripts",
            ":!scripts/a70_perf_feasibility.py",
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    check("(6) git grep ran", proc.returncode in (0, 1), proc.stderr.strip()[:160])
    hits = [line for line in proc.stdout.splitlines() if line]
    stray = [
        line for line in hits if not (line.startswith("src/experiments/perf.py:") and any(a in line for a in ALLOWED))
    ]
    check("(6) the grep matches nothing outside the allowed perf.py lines", not stray, f"{stray[:5]}")


def leg_7():
    print("(7) do-MNIST perf still skips (source text, nothing imported)")
    path = os.path.join(REPO, "src", "experiments", "do_mnist.py")
    with open(path) as handle:
        text = handle.read()
    source = None
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.ClassDef) and node.name == "DoMNISTOrchestrator":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "_run_perf":
                    source = ast.get_source_segment(text, item)
    check("(7) DoMNISTOrchestrator._run_perf is found", source is not None)
    check(
        "(7) it logs a warning and skips, whatever the metrics",
        source is not None
        and "logger.warning" in source
        and "perf skipped" in source
        and not any(s in source for s in ("perf_sweeps", "backend_runs", "feasibility", "seed_var(")),
        f"{(source or '').strip().splitlines()[-1:]}",
    )


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    os.chdir(REPO)
    os.environ.setdefault("MPLBACKEND", "Agg")
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=None, help="one leg: 1, 2, 3 (runs 3 and 4 together), 5, 6 or 7")
    args = parser.parse_args()
    print(f"tree: {perf.__file__}")
    legs = [("1", leg_1), ("2", leg_2), ("3", leg_3_4), ("5", leg_5), ("6", leg_6), ("7", leg_7)]
    if args.only:
        legs = [(tag, leg) for tag, leg in legs if tag == args.only or (args.only == "4" and tag == "3")]
        if not legs:
            sys.exit(f"unknown leg {args.only!r}")
    for tag, leg in legs:
        try:
            leg()
        except Exception as error:  # a raise is a FAIL line, and the later legs still report
            check(f"({tag}) ran without raising", False, f"{type(error).__name__}: {error}")
    if not FAIL:
        print("A70 PASS")
    else:
        print(f"A70 FAIL: {FAIL}")
        sys.exit(1)
