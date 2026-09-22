"""A72: Imbens-Manski confidence intervals on the sweep bounds (`im-ci` in `defaults:`).

Every sweep cell's finalised bounds [L, U] are wrapped in [L - C s_L, U + C s_U],
s_L and s_U the SDs of B bootstrap refits of each method's own fitted rows, C the
root of Phi(C + Delta / sigma) - Phi(-C) = level; the four sweep metrics read that
interval, the raw record sits beside it (`{param}_results_raw.pkl`). Legs:

  0. grids off `sweep_samples`: the n and m grids at 16 equal, bit for bit, the arrays
     RECORDED on `finite` before the change; at 8 they have 8 points with the same
     end points (m is 1..8), at 32 (the DatasetDefaults fallback) 32; the n and m
     strategies take that many steps at sweep_samples 8 without an override; no
     16 / 17 literal left in either `grid_fn`. Catches: a grid that ignores the knob,
     a changed spacing at 16.
  1. the arithmetic on synthetic inputs: C at Delta 0 is z_0.975 and at Delta/sigma
     100 is z_0.95, C decreasing in Delta, level 90 at Delta 0 is z_0.95, the
     equation holds on a d grid, sigma 0 and one zero SE leave those ends, a raw NaN
     stays NaN, a replicate NaN is skipped, 0 or 1 valid replicates give the raw
     interval, containment on 2000 random cells, Delta < 0 by 1e-16 is clamped, a
     symmetric case matches the closed form with the ddof-1 SD. Catches: the two z's
     swapped, an extra sqrt(n), the clamp missing, ddof 0, a NaN leaking into a
     raw-OK query.
  2. resampling: an n-sweep cell has one index vector shared by X/y/GX/G/Z; an m = 4
     sim cell draws 4n tiled rows and n base rows independently, and through
     `fit_model` with recording stubs the baselines are fitted on n rows and the DA+
     methods and intersections on 4n; the same seed gives the same indices, another
     cell others; replicate bounds bit-identical under pools of 1, 2 and -1 (the hard
     assertion); speed on a warmed pool, printed and RECORDED, only > 1 asserted.
  3. `im-ci: 0` reproduces `finite`: the fixture sweep (sim, n on [128, 512], 1
     experiment, PI / DA+PI / PI&DA+PI, the config.yaml toggles, serial solves)
     equals the digests RECORDED on the pre-change tree; at 95 its `results_raw`
     equals them too, its failure / infeasible columns are the raw ones, and the CI
     is never narrower nor less covering than the raw interval.
  4. plumbing: `im-ci` is a toggle, the yaml values accepted and rejected, `im_ci`
     unknown in the yaml and the spelling of the resolved block, a recipe-shaped block
     through main.py's steps, the runner's own validation, the sweep recipes carrying
     the key and the others not (SKIP until they do).
  5. exclusion: (a) static, nothing from do-MNIST is imported or run: the do-MNIST
     block resolves to im_ci 0, the do-MNIST / net / perf / query sources never name
     the CI, and `ParamSweepRunner` is the helper's only caller in src/; (b) on
     simulation, with the helper patched to raise, `_run_perf` (wall_clock, 3 steps)
     and `_run_query_sweep` (sweep_samples 8) complete.
  6. the n and m sweeps at sweep_samples 8: sim, 2 experiments, config.yaml's six
     methods, B = IM_CI_REPLICATES, n_jobs -1, through `_run_sweeps`: CI coverage and
     width never under raw, the statuses' failure / infeasible columns the raw run's,
     the raw run's metrics equal to `results_raw`, the pkls and figures written,
     `aggregate.sweep_params` finds n and m with no warning; the per-method readings
     (coverage, CI/raw width ratio, the log-log slope of the CI excess width in n,
     the valid fractions) printed and compared with RECORDED.
  7. render: the width figure of leg 6's record drawn without a swallowed error; its
     line is the CI record's nanmean in sorted-x order.
  8. the Slurm launcher (`sbatch_sweeps.py --dry-run`), no submission: one yaml per
     (dataset, param) and per perf block, each resolving; no two tasks share a
     (dataset, subdir, stem); only the directives asked for; the task dirs' links;
     no NEW machine-specific value in src/, config.yaml, the recipes or the launcher
     outside its `EPILOG` / docstring. SKIP until the launcher exists.

    MPLBACKEND=Agg uv run python scripts/a72_im_ci.py [--only LEG] [--quick]

`--quick` drops B to 20 (leg 3 at 95 and leg 6) and skips the RECORDED readings of
leg 6. Writes only under $A72_TMPROOT (default ~/scratch/tmp/a72), removed on PASS.
Nothing here instantiates a do-MNIST object.
"""

import argparse
import ast
import contextlib
import glob
import inspect
import os
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
import time
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from loguru import logger  # noqa: E402
from munch import munchify  # noqa: E402
from scipy.special import ndtr  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import digest_leg  # noqa: E402

import src.aggregate as aggregate  # noqa: E402
import src.experiments.base as base  # noqa: E402
import src.experiments.utils.im_ci as im_ci  # noqa: E402
import src.methods.sensitivity_models as sensitivity_models  # noqa: E402
from src.experiments.base import BaseExperimentRunner, QuerySweepRunner, SweepData  # noqa: E402
from src.experiments.configs import (  # noqa: E402
    EPS_TOL,
    IM_CI_REPLICATES,
    IM_CI_SEED_OFFSET,
    PARAM_SPECS,
    TOGGLE_KEYS,
    SweepSpec,
    parse_experiment_plan,
    resolve_dataset_block,
)
from src.experiments.simulation import SimulationOrchestrator  # noqa: E402
from src.experiments.utils import PanelBuilder, fit_model, set_seed  # noqa: E402
from src.experiments.utils.constants import TEX_MAPPER  # noqa: E402
from src.experiments.utils.metrics import STATUS_CATEGORIES  # noqa: E402
from src.experiments.utils.plotting import create_sweep_plot  # noqa: E402

TMPROOT = os.environ.get("A72_TMPROOT", os.path.expanduser("~/scratch/tmp/a72"))
DATASETS = ("simulation", "optical_device", "cigarettes")
FIXTURE_METHODS = ["PI", "DA+PI", "PI&DA+PI"]
FIXTURE_GRID = [128, 512]
QUICK_REPLICATES = 20
FAIL = []
SKIPPED = []
MADE = []  # the directories this run created under TMPROOT, removed on PASS

# RECORDED on `finite` 109e507 before the change: PARAM_SPECS[p].grid_fn(dataset, 16)
PRE_GRIDS = {
    ("simulation", "n"): [128, 187, 247, 307, 366, 426, 486, 546, 605, 665, 725, 785, 844, 904, 964, 1024],
    ("optical_device", "n"): [128, 186, 244, 302, 360, 418, 476, 534, 593, 651, 709, 767, 825, 883, 941, 1000],
    ("cigarettes", "n"): [245, 392, 539, 686, 833, 980, 1127, 1274, 1421, 1568, 1715, 1862, 2009, 2156, 2303, 2450],
    **{(dataset, "m"): list(range(1, 17)) for dataset in DATASETS},
}
# RECORDED on `finite` 109e507: leg 3's fixture, `digest_leg.digest` of every array
# (wall_clock excluded), "__all__" over the whole record
PRE_DIGESTS = {
    "__all__": "9cdaa78a4412d048",
    "results/DA+PI/approximation_error": "5390cc747eeef506",
    "results/DA+PI/coverage": "cf28966d4471fd5d",
    "results/DA+PI/interval_width": "c9c2a911a7da0bbb",
    "results/DA+PI/worst_error": "78ed2c2900576a22",
    "results/PI&DA+PI/approximation_error": "5390cc747eeef506",
    "results/PI&DA+PI/coverage": "cf28966d4471fd5d",
    "results/PI&DA+PI/interval_width": "8bc26d4b44e5d283",
    "results/PI&DA+PI/worst_error": "9512fa1c40eff5ba",
    "results/PI/approximation_error": "5390cc747eeef506",
    "results/PI/coverage": "cf28966d4471fd5d",
    "results/PI/interval_width": "88f2fe30bd3be9c3",
    "results/PI/worst_error": "a6714a83bfe50e15",
    "statuses/DA+PI": "1b7635c09755d456",
    "statuses/PI": "1b7635c09755d456",
    "statuses/PI&DA+PI": "1b7635c09755d456",
    "x": "037b45d15364150c",
}
# RECORDED: leg 2's speedup of the six methods at B = 32 on a warmed pool of -1
# (printed, never asserted as a number: the core count is the machine's)
SPEEDUP_RECORDED = 24.2  # 32 workers, 2026-09-22
# RECORDED 2026-09-22 on a 32-core allocation, re-pinned after the pad tolerance was
# retired under the im-ci (the widths lost 2 EPS_TOL): leg 6's readings at B = IM_CI_REPLICATES
# (per step: CI and raw coverage, CI/raw width ratio, valid fraction; the slope of the
# CI excess width in log x)
RECORDED_6 = {
    "n": {
        "PI+INV": {
            "coverage": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "coverage_raw": [1.0, 0.9069, 0.9951, 1.0, 1.0, 1.0, 1.0, 1.0],
            "ratio": [1.482, 1.6641, 1.5261, 1.4447, 1.3934, 1.3657, 1.343, 1.3034],
            "valid": [0.52, 0.54, 0.76, 0.92, 0.975, 0.99, 1.0, 1.0],
            "slope": -0.2771,
        },
        "PI": {
            "coverage": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "coverage_raw": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "ratio": [1.2657, 1.162, 1.1257, 1.1086, 1.0962, 1.0851, 1.0806, 1.0748],
            "valid": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "slope": -0.5924,
        },
        "DA+PI": {
            "coverage": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "coverage_raw": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "ratio": [1.2932, 1.1854, 1.1463, 1.1261, 1.1106, 1.0977, 1.0931, 1.0868],
            "valid": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "slope": -0.5736,
        },
        "DA+PI+IV": {
            "coverage": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "coverage_raw": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "ratio": [1.3843, 1.2459, 1.1958, 1.1699, 1.1477, 1.1321, 1.1279, 1.116],
            "valid": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "slope": -0.5637,
        },
        "PI&DA+PI": {
            "coverage": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "coverage_raw": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "ratio": [1.2861, 1.1854, 1.1462, 1.1264, 1.1111, 1.0993, 1.0945, 1.087],
            "valid": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "slope": -0.5581,
        },
        "PI&DA+PI+IV": {
            "coverage": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "coverage_raw": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "ratio": [1.3906, 1.2606, 1.2051, 1.1759, 1.154, 1.1383, 1.1346, 1.1202],
            "valid": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "slope": -0.5563,
        },
    },
    "m": {
        "PI+INV": {
            "coverage": [1.0, 1.0, 1.0, 1.0, 0.9167, 0.9167, 0.9167, 0.9167],
            "coverage_raw": [0.6667, 0.6667, 0.6667, 0.6667, 0.6667, 0.6667, 0.6667, 0.6667],
            "ratio": [1.8268, 1.7981, 1.7756, 1.8177, 1.709, 1.6591, 1.6902, 1.626],
            "valid": [0.11, 0.36, 0.43, 0.48, 0.5, 0.59, 0.67, 0.66],
            "slope": -0.1331,
        },
        "PI": {
            "coverage": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "coverage_raw": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "ratio": [1.2672, 1.2611, 1.2558, 1.2623, 1.2665, 1.2615, 1.2546, 1.261],
            "valid": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "slope": -0.0101,
        },
        "DA+PI": {
            "coverage": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "coverage_raw": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "ratio": [1.3017, 1.182, 1.1476, 1.1259, 1.1113, 1.1023, 1.094, 1.0901],
            "valid": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "slope": -0.6297,
        },
        "DA+PI+IV": {
            "coverage": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "coverage_raw": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "ratio": [1.3965, 1.2443, 1.2018, 1.168, 1.1523, 1.1372, 1.1304, 1.125],
            "valid": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "slope": -0.6207,
        },
        "PI&DA+PI": {
            "coverage": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "coverage_raw": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "ratio": [1.2877, 1.1794, 1.149, 1.1267, 1.1107, 1.102, 1.0941, 1.0876],
            "valid": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "slope": -0.6145,
        },
        "PI&DA+PI+IV": {
            "coverage": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "coverage_raw": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "ratio": [1.4124, 1.2535, 1.2158, 1.1754, 1.1612, 1.1445, 1.1386, 1.1301],
            "valid": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "slope": -0.6139,
        },
    },
}
# one query of 204 flipping in one of 2 experiments moves an n-step mean by 0.0025; on
# m (12 queries) one flip is 0.042, so there the tolerance is exact: the run is
# deterministic on one node (seeded draws, seeded bootstrap, serial models)
COVERAGE_ATOL = 0.01
RATIO_RTOL = 0.01
SLOPE_ATOL = 0.05


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        FAIL.append(name)


def skip(tag, reason):
    """A leg that cannot run here: printed and counted in the summary, never a silent PASS."""
    print(f"  [SKIP] {tag} {reason}", flush=True)
    SKIPPED.append(f"{tag} {reason}")


@contextlib.contextmanager
def captured(level="WARNING"):
    """The loguru messages at `level` and above emitted inside the block."""
    lines = []
    handle = logger.add(lambda message: lines.append(message.record["message"]), level=level)
    try:
        yield lines
    finally:
        logger.remove(handle)


@contextlib.contextmanager
def workdir(prefix):
    """A fresh directory under TMPROOT as cwd (the pipeline writes `artifacts/`
    relative to it), with the repo's `data/` linked in."""
    os.makedirs(TMPROOT, exist_ok=True)
    path = tempfile.mkdtemp(prefix=prefix, dir=TMPROOT)
    MADE.append(path)
    if os.path.isdir(os.path.join(REPO, "data")):
        os.symlink(os.path.join(REPO, "data"), os.path.join(path, "data"))
    os.chdir(path)
    try:
        yield path
    finally:
        os.chdir(REPO)


@contextlib.contextmanager
def replicates(count):
    """IM_CI_REPLICATES as the runner reads it."""
    saved = base.IM_CI_REPLICATES
    base.IM_CI_REPLICATES = count
    try:
        yield
    finally:
        base.IM_CI_REPLICATES = saved


def configured():
    """config.yaml's `defaults:`, `hyperparameters:` and simulation block."""
    with open(os.path.join(REPO, "config.yaml")) as handle:
        config = yaml.safe_load(handle) or {}
    return config.get("defaults") or {}, config.get("hyperparameters") or {}, config["simulation"]


def configured_toggles():
    defaults = configured()[0]
    return dict(
        recalibrate=bool(defaults.get("recalibrate", True)),
        pad=bool(defaults.get("pad", False)),
        clipy=bool(defaults.get("clipy", True)),
        mean_match=bool(defaults.get("mean_match", True)),
    )


def sim_orchestrator(methods, n_experiments=1, n_jobs=1, im_ci_level=0.0, sweep_samples=8, **overrides):
    set_seed(42)
    return SimulationOrchestrator(
        seed=42,
        n_samples=2048,
        kernel_dim=0,
        treatment_dim=32,
        augmentation="translate",
        n_experiments=n_experiments,
        sweep_samples=sweep_samples,
        methods=methods,
        hyperparameters={},
        n_jobs=n_jobs,
        im_ci=im_ci_level,
        **{**configured_toggles(), **overrides},
    )


def runner_for(orch, param, **override):
    return orch.get_sweep_runner_cls(param)(
        methods=orch.methods, method_factory=orch.build_methods, **override, **orch._get_clean_kwargs()
    )


def read(path):
    with open(path) as handle:
        return handle.read()


def load_pkl(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)  # noqa: S301 - the gate wrote this file itself


def column(statuses, name):
    return statuses[..., list(STATUS_CATEGORIES).index(name)]


# ------------------------------------------------------------------ legs


def leg_0():
    print("(0) the n and m grids read sweep_samples")
    for (dataset, param), want in PRE_GRIDS.items():
        got = PARAM_SPECS[param].grid_fn(dataset, 16)
        check(
            f"(0) {dataset} {param} at 16 == the RECORDED pre-change grid",
            np.array_equal(got, np.asarray(want)) and got.dtype == np.asarray(want).dtype,
            f"{got.tolist()}",
        )
        for count in (8, 32):
            grid = PARAM_SPECS[param].grid_fn(dataset, count)
            ends = grid[0] == want[0] and (param == "m" or grid[-1] == want[-1])
            check(f"(0) {dataset} {param} at {count}: {count} points, same end points", len(grid) == count and ends)
        if param == "m":
            check(f"(0) {dataset} m at 8 is 1..8", PARAM_SPECS["m"].grid_fn(dataset, 8).tolist() == list(range(1, 9)))
    orch = sim_orchestrator(["PI"])
    for param in ("n", "m"):
        runner = runner_for(orch, param)
        check(f"(0) the sim {param} strategy takes 8 steps at sweep_samples 8", len(runner.get_param_range()) == 8)
    tree = ast.parse(read(os.path.join(REPO, "src/experiments/configs.py")))
    literals = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values, strict=True):
            if isinstance(key, ast.Constant) and key.value in ("n", "m") and isinstance(value, ast.Call):
                for keyword in value.keywords:
                    if keyword.arg == "grid_fn":
                        numbers = [c.value for c in ast.walk(keyword.value) if isinstance(c, ast.Constant)]
                        literals[key.value] = [v for v in numbers if v in (16, 17) and not isinstance(v, bool)]
    check("(0) both grid_fns found in PARAM_SPECS", set(literals) == {"n", "m"}, f"{sorted(literals)}")
    check("(0) no 16 / 17 literal left in either", not any(literals.values()), f"{literals}")


def leg_1():
    print("(1) the Imbens-Manski arithmetic")
    critical = im_ci.imbens_manski_critical

    def C(delta, level=95.0):
        return float(critical(delta, 1.0, level))

    check("(1) Delta 0 -> z_0.975", abs(C(0.0) - 1.959963985) < 1e-9, f"{C(0.0):.10f}")
    check("(1) Delta/sigma 100 -> z_0.95", abs(C(100.0) - 1.644853627) < 1e-9, f"{C(100.0):.10f}")
    check("(1) level 90 at Delta 0 -> z_0.95", abs(C(0.0, 90.0) - 1.644853627) < 1e-9, f"{C(0.0, 90.0):.10f}")
    d = np.linspace(0.0, 20.0, 401)
    c = critical(d, np.ones_like(d), 95.0)
    check("(1) C decreasing in Delta", bool(np.all(np.diff(c) <= 1e-15)))
    error = np.abs(ndtr(c + d) - ndtr(-c) - 0.95).max()
    check("(1) Phi(C + d) - Phi(-C) == 0.95 on a d grid", error < 1e-12, f"{error:.2e}")
    check("(1) the same C for d with sigma scaled", np.allclose(critical(2.0 * d, 2.0 * np.ones_like(d), 95.0), c))

    bounds = im_ci.imbens_manski_bounds
    raw = np.array([[0.0, 1.0]])
    check("(1) sigma 0 leaves the bounds", np.array_equal(bounds(raw, np.tile(raw, (5, 1, 1)), 95.0), raw))
    rep = np.zeros((4, 1, 2))
    rep[:, 0, 0] = 0.0
    rep[:, 0, 1] = 1.0 + np.arange(4)
    ci = bounds(raw, rep, 95.0)
    check("(1) a zero SE leaves its end, the other moves", ci[0, 0] == 0.0 and ci[0, 1] > 1.0, f"{ci}")
    rng = np.random.default_rng(0)
    check("(1) a raw NaN stays NaN", np.all(np.isnan(bounds(np.full((1, 2), np.nan), rng.random((9, 1, 2)), 95.0))))
    rep = rng.random((10, 1, 2))
    rep[3] = np.nan
    check(
        "(1) a replicate NaN is skipped in the SE",
        np.allclose(bounds(raw, rep, 95.0), bounds(raw, np.delete(rep, 3, 0), 95.0), rtol=0, atol=1e-14),
    )
    for valid in (0, 1):
        rep = rng.random((10, 1, 2))
        rep[valid:] = np.nan
        check(
            f"(1) {valid} valid replicate(s): the CI is the raw interval", np.array_equal(bounds(raw, rep, 95.0), raw)
        )
    violations = 0
    for _ in range(2000):
        lower = rng.normal(size=5)
        cell = np.column_stack([lower, lower + rng.exponential(size=5)])
        cell[rng.random((5, 2)) < 0.2] = np.nan
        rep = rng.normal(size=(20, 5, 2))
        rep[rng.random((20, 5, 2)) < 0.3] = np.nan
        ci = bounds(cell, rep, float(rng.choice([5.0, 50.0, 95.0, 99.9])))
        finite = np.isfinite(cell)
        violations += int(np.any(ci[finite[:, 0], 0] > cell[finite[:, 0], 0]))
        violations += int(np.any(ci[finite[:, 1], 1] < cell[finite[:, 1], 1]))
        violations += int(np.any(np.isfinite(ci[~finite])))
        violations += int(not np.all(np.isfinite(ci[finite])))
    check(
        "(1) containment wherever raw is finite, NaN wherever it is not, 2000 cells", violations == 0, f"{violations}"
    )
    rep = rng.normal(size=(10, 1, 2))
    tight = np.array([[1.0, 1.0 - 1e-16]])
    clamped = critical(tight[0, 1] - tight[0, 0], 1.0, 95.0)
    check(
        "(1) Delta < 0 by 1e-16 is clamped: C = z_0.975",
        float(clamped) == float(critical(0.0, 1.0, 95.0)) and np.all(np.isfinite(bounds(tight, rep, 95.0))),
        f"{float(clamped):.10f}",
    )
    # asymmetric SEs: s_L = 1, s_U = 2 (ddof-1 SDs of two replicates each), Delta = 1,
    # so sigma = max = 2, d = 0.5, C = 1.76971, and each end moves by its OWN SE
    half = np.sqrt(0.5)
    rep = np.array([[[1.0 - half, 2.0 - 2 * half]], [[1.0 + half, 2.0 + 2 * half]]])
    ci = bounds(np.array([[1.0, 2.0]]), rep, 95.0)[0]
    want = np.array([1.0 - 1.76971, 2.0 + 2 * 1.76971])
    check(
        "(1) sigma = max(s_L, s_U), each end at its own SE: [L - C, U + 2C]", np.allclose(ci, want, atol=1e-5), f"{ci}"
    )
    rep = rng.normal(size=(50, 1, 2))
    rep[..., 1] = rep[..., 0]
    s = np.std(rep[:, 0, 0], ddof=1)
    ci = bounds(np.array([[0.3, 0.3]]), rep, 95.0)[0]
    closed = np.array([0.3 - 1.959963984540054 * s, 0.3 + 1.959963984540054 * s])
    check("(1) the symmetric closed form, ddof-1 SD, no sqrt(n)", np.allclose(ci, closed, rtol=0, atol=1e-12), f"{ci}")


def leg_2():
    print("(2) resampling: each method's own fitted rows, iid")
    orch = sim_orchestrator(["PI"])
    n_runner = runner_for(orch, "n")
    data = SweepData.coerce(n_runner.generate_data(0, 128))
    arrays = data.fit_arrays
    rows, base_rows = im_ci.replicate_rows(len(arrays["X"]), None, 3, [IM_CI_SEED_OFFSET, 42, 0, 0])
    check("(2) n cell: no base group, base_rows is None", arrays["X_base"] is None and base_rows is None)
    sample = im_ci.resample_fit_arrays(arrays, rows[0], None)
    check(
        "(2) n cell: X / y / GX / G / Z share one index",
        all(np.array_equal(sample[k], arrays[k][rows[0]]) for k in ("X", "y", "GX", "G", "Z")),
    )
    again, _ = im_ci.replicate_rows(len(arrays["X"]), None, 3, [IM_CI_SEED_OFFSET, 42, 0, 0])
    other, _ = im_ci.replicate_rows(len(arrays["X"]), None, 3, [IM_CI_SEED_OFFSET, 42, 0, 1])
    check("(2) the same seed gives the same indices, another cell others", np.array_equal(rows, again))
    check("(2) ... another (j, i) other indices", not np.array_equal(rows, other))

    m_runner = runner_for(orch, "m")
    folds = SweepData.coerce(m_runner.generate_data(0, 4))
    arrays = folds.fit_arrays
    n = len(arrays["X_base"])
    rows, base_rows = im_ci.replicate_rows(len(arrays["X"]), n, 2, [IM_CI_SEED_OFFSET, 42, 0, 3])
    check("(2) m = 4 cell: 4n tiled rows, n base rows", rows.shape == (2, 4 * n) and base_rows.shape == (2, n))
    check("(2) ... drawn independently", not np.array_equal(rows[0][:n], base_rows[0]))

    class Recorder:
        def fit(self, X, y, **kwargs):
            self.rows = len(X)
            self.lengths = {k: len(v) for k, v in kwargs.items() if hasattr(v, "__len__") and not isinstance(v, str)}
            return self

    sample = im_ci.resample_fit_arrays(arrays, rows[0], base_rows[0])
    baselines = ("PI", "PI+IV", "ERM")
    tiled = ("DA+PI", "PI+INV", "DA+PI+IV", "PI&DA+PI", "PI&DA+PI+IV", "DA+ERM")
    rows_seen = {}
    for name in baselines + tiled:
        stub = Recorder()
        fit_model(model=stub, method_name=name, hyperparameters={}, **sample)
        rows_seen[name] = (stub.rows, set(stub.lengths.values()))
    check(
        "(2) the baselines' replicate fits get exactly n rows",
        all(rows_seen[b][0] == n and rows_seen[b][1] <= {n} for b in baselines),
        f"{ {b: rows_seen[b][0] for b in baselines} }",
    )
    check(
        "(2) the DA+ methods and intersections get exactly 4n",
        all(rows_seen[t][0] == 4 * n and rows_seen[t][1] <= {4 * n} for t in tiled),
        f"{ {t: rows_seen[t][0] for t in tiled} }",
    )

    runner = runner_for(sim_orchestrator(FIXTURE_METHODS, im_ci_level=95.0), "n")
    runner.build_models(0, 0, data)
    out = {}
    with replicates(8):
        for n_jobs in (1, 2, -1):
            runner.n_jobs = n_jobs
            out[n_jobs] = runner.bootstrap_bounds(0, 0, data, [128])
    same = all(np.array_equal(out[1][k], out[j][k], equal_nan=True) for j in (2, -1) for k in out[1])
    with replicates(4):
        runner.n_jobs = -1
        first = runner.bootstrap_bounds(0, 0, data, [128])
        second = runner.bootstrap_bounds(0, 1, data, [128])
    check(
        "(2) the runner seeds each cell apart: (0, 0) and (0, 1) replicates differ",
        not np.array_equal(first["PI"], second["PI"], equal_nan=True),
    )
    check(
        "(2) replicate bounds bit-identical under pools of 1, 2 and -1",
        same,
        f"{ {k: v.shape for k, v in out[1].items()} }",
    )

    # speed: the first cell of a process pays the workers' spawn and imports once
    from joblib import Parallel, delayed, parallel_config

    with parallel_config(backend="loky", n_jobs=-1, inner_max_num_threads=1):
        Parallel()(delayed(im_ci.imbens_manski_critical)(0.0, 1.0, 95.0) for _ in range(64))
    six = configured()[2]["methods"]
    runner = runner_for(sim_orchestrator(six, im_ci_level=95.0), "n")
    runner.build_models(0, 0, data)
    seconds = {}
    with replicates(32):
        for n_jobs in (-1, 1):
            runner.n_jobs = n_jobs
            start = time.perf_counter()
            runner.bootstrap_bounds(0, 0, data, [128])
            seconds[n_jobs] = time.perf_counter() - start
    speedup = seconds[1] / seconds[-1]
    print(
        f"      speedup {speedup:.1f}x: serial {seconds[1]:.1f}s, pool of -1 {seconds[-1]:.1f}s "
        f"(RECORDED {SPEEDUP_RECORDED})"
    )
    check("(2) the pool is faster than serial", speedup > 1.0, f"{speedup:.2f}x")


def fixture_run(level, pool=1):
    """Leg 3's fixture, exactly as recorded on `finite`: sim n on [128, 512], one
    experiment, three methods, the config.yaml toggles, serial models and one BLAS
    thread in the parent. `pool` sizes the replicate pool only."""
    orch = sim_orchestrator(FIXTURE_METHODS, n_jobs=1)
    extra = {} if level is None else {"im_ci": level}
    kwargs = {**orch._get_clean_kwargs(), **extra}
    runner = orch.get_sweep_runner_cls("n")(
        methods=orch.methods, method_factory=orch.build_methods, param_grid_override=FIXTURE_GRID, **kwargs
    )
    runner.n_jobs = pool
    with threadpool_limits(limits=1):
        x, results, statuses = runner.run("a72 fixture")
    return runner, x, results, statuses


def fixture_digests(x, results, statuses):
    arrays = {"x": np.asarray(x)}
    for name in FIXTURE_METHODS:
        for metric, values in results[name].items():
            if metric != "wall_clock":
                arrays[f"results/{name}/{metric}"] = np.asarray(values)
        arrays[f"statuses/{name}"] = np.asarray(statuses[name])
    digests = {key: digest_leg.digest(value) for key, value in arrays.items()}
    digests["__all__"] = digest_leg.digest(arrays)
    return digests


def leg_3(quick):
    print("(3) im-ci 0 reproduces finite; im-ci 95 keeps it as results_raw, the pad at eps* alone")
    _, x, results, statuses = fixture_run(0.0)
    got = fixture_digests(x, results, statuses)
    bad = sorted(k for k in PRE_DIGESTS if got.get(k) != PRE_DIGESTS[k])
    check("(3) im_ci 0: every digest == RECORDED on finite", not bad, f"{got['__all__']} {bad}")
    count = QUICK_REPLICATES if quick else IM_CI_REPLICATES
    with replicates(count):
        runner, x95, results95, statuses95 = fixture_run(95.0, pool=-1)
    # under the IM-CI the pad drops EPS_TOL (BoundedSA.pad_tolerance), so the raw record
    # is finite's for the unpadded PI and the padded methods are narrower by at most
    # 2 EPS_TOL, exactly that on a standalone DA+ method; the statuses never move
    raw = runner.im_ci_record["results_raw"]
    got = fixture_digests(x95, raw, statuses)
    unpadded = [k for k in PRE_DIGESTS if k.startswith(("results/PI/", "x"))]
    bad = sorted(k for k in unpadded if got.get(k) != PRE_DIGESTS[k])
    check(f"(3) im_ci 95 (B = {count}): results_raw of the unpadded PI == RECORDED on finite", not bad, f"{bad}")
    narrower = raw["DA+PI"]["interval_width"] - results["DA+PI"]["interval_width"]
    check(
        "(3) ... DA+PI's raw width is finite's minus 2 EPS_TOL (the pad at eps* alone)",
        np.allclose(narrower, -2 * EPS_TOL, rtol=0, atol=1e-12),
        f"{np.round(narrower, 6).tolist()}",
    )
    narrower = raw["PI&DA+PI"]["interval_width"] - results["PI&DA+PI"]["interval_width"]
    check(
        "(3) ... the intersection's by at most 2 EPS_TOL",
        bool(np.all((narrower <= 1e-12) & (narrower >= -2 * EPS_TOL - 1e-12))),
        f"{np.round(narrower, 6).tolist()}",
    )
    for status in ("solver_failure", "infeasible"):
        check(
            f"(3) the {status} column is the solver's",
            all(np.array_equal(column(statuses95[n], status), column(statuses[n], status)) for n in statuses),
        )
    structural("(3)", results95, raw)
    leg_3_pad()


PAD_METHODS = ["DA+PI", "DA+PI+IV", "PI&DA+PI", "PI&DA+PI+IV"]


def leg_3_pad():
    """Under im_ci 95 the sweep pads by eps* alone, on the point models (the DA+ methods
    and both intersections' DA branches, IV included) and in the replicates; under 0 by
    eps* + EPS_TOL."""
    runner, data, budgets = None, None, None
    for level, tol in ((0.0, 0.0), (95.0, EPS_TOL)):
        runner = runner_for(sim_orchestrator(PAD_METHODS, im_ci_level=level), "n")
        data = SweepData.coerce(runner.generate_data(0, 128))
        models = runner.build_models(0, 0, data)
        star = runner.get_oracle(0).epsilon_star
        epsilon = runner.fit_epsilon(0, 0, data)
        budgets = runner.fit_budgets(0, 0, data)
        want = epsilon - tol
        pads = {name: models[name].pad_amount for name in ("DA+PI", "DA+PI+IV")}
        pads |= {f"{name} branch": models[name].augmented.pad_amount for name in ("PI&DA+PI", "PI&DA+PI+IV")}
        check(
            f"(3) im_ci {level:g}: every DA+ method and both intersections' DA branches "
            f"pad by {'eps*' if tol else 'eps* + EPS_TOL'}",
            all(pad == want for pad in pads.values()) and np.isclose(want, star + EPS_TOL - tol, rtol=0, atol=1e-12),
            f"pads {pads}, eps* {star!r}",
        )
        check(
            f"(3) im_ci {level:g}: the intersections' baseline branches still do not pad",
            all(models[name].baseline.pad is False for name in ("PI&DA+PI", "PI&DA+PI+IV")),
        )
    builders = runner.method_factory(n_jobs=1, **budgets)
    rows = np.arange(len(data.X))
    shifted = {}
    for tol in (0.0, EPS_TOL):
        shifted[tol] = {
            name: im_ci._replicate(
                builders, name, data.fit_arrays, rows, None, data.X_test, [{}], {}, runner.get_da(0), tol
            )
            for name in PAD_METHODS
        }
    diff = shifted[0.0]["DA+PI"] - shifted[EPS_TOL]["DA+PI"]
    check(
        "(3) a replicate drops EPS_TOL from each end of its pad (DA+PI)",
        np.allclose(diff[..., 0], -EPS_TOL, rtol=0, atol=1e-12)
        and np.allclose(diff[..., 1], EPS_TOL, rtol=0, atol=1e-12),
    )
    width = lambda b: b[..., 1] - b[..., 0]  # noqa: E731
    for name in ("PI&DA+PI", "PI&DA+PI+IV"):
        narrowing = width(shifted[0.0][name]) - width(shifted[EPS_TOL][name])
        check(
            f"(3) ... and {name}'s replicate DA branch too (narrower, by at most 2 EPS_TOL)",
            np.all(narrowing >= -1e-12) and np.all(narrowing <= 2 * EPS_TOL + 1e-12) and narrowing.max() > 0,
            f"{narrowing.min():.4g}..{narrowing.max():.4g}",
        )


def structural(tag, ci, raw):
    """CI width and coverage never under raw where raw is finite; NaN exactly where raw is."""
    for name in ci:
        if name == "ATE":
            continue
        for metric in ("interval_width", "coverage"):
            a, b = ci[name][metric], raw[name][metric]
            finite = np.isfinite(b)
            ok = np.array_equal(np.isfinite(a), finite) and np.all(a[finite] >= b[finite])
            check(f"{tag} {name} CI {metric} >= raw at every cell", ok)


def leg_4():
    print("(4) plumbing")
    check("(4) `im-ci` is a toggle key", "im-ci" in TOGGLE_KEYS)
    block = {"seed": 42, "kernel_dim": 0}
    for value in (95, 90.5, 0, False):
        resolved = resolve_dataset_block("simulation", {**block, "im-ci": value})
        check(
            f"(4) im-ci {value!r} accepted, resolved as float im_ci",
            resolved.get("im_ci") == float(value) and "im-ci" not in resolved,
        )
    for value in (True, -1, 100, 150, "95", None):
        try:
            resolve_dataset_block("simulation", {**block, "im-ci": value})
            rejected = False
        except ValueError:
            rejected = True
        check(f"(4) im-ci {value!r} rejected", rejected)
    try:
        resolve_dataset_block("simulation", {**block, "im_ci": 95})
        rejected = False
    except ValueError:
        rejected = True
    check("(4) a yaml `im_ci` is unknown (one spelling per side)", rejected)
    with captured("INFO") as lines:
        absent = resolve_dataset_block("simulation", {**block, "experiment": {"sweep": {"param": ["n"]}}})
    check(
        "(4) absent key: raw, with one INFO line",
        absent["im_ci"] == 0.0 and sum("im-ci absent" in ln for ln in lines) == 1,
    )

    defaults, _, sim = configured()
    recipe = {**defaults, **sim}
    plan = parse_experiment_plan(recipe.get("experiment"))
    resolved = resolve_dataset_block("simulation", recipe)
    check(
        "(4) config.yaml's sim block through main.py's steps carries its im-ci",
        resolved["im_ci"] == float(defaults.get("im-ci", 0)) and plan.sweep is not None,
        f"{resolved['im_ci']}",
    )

    class Runner(BaseExperimentRunner):
        def run(self, desc):
            return None

    common = dict(seed=-1, n_samples=1, n_experiments=1, sweep_samples=1, methods={})
    check("(4) the runner keeps a valid level", Runner(**common, im_ci=95).im_ci == 95.0)
    for value in (True, 100, -5, "95"):
        try:
            Runner(**common, im_ci=value)
            rejected = False
        except ValueError:
            rejected = True
        check(f"(4) the runner rejects im_ci {value!r}", rejected)
    check("(4) the runner's default is off, n_jobs 1", Runner(**common).im_ci == 0.0 and Runner(**common).n_jobs == 1)

    merging, unpinned = [], []
    for path in sorted(glob.glob(os.path.join(REPO, "scripts", "*.py"))):
        name = os.path.basename(path)
        if name in PIN_EXEMPT or "defaults" not in read(path):
            continue
        lines = unpinned_merges(path)
        unpinned += [f"{name}:{line}" for line in lines]
        if re.search(r"\{\s*\*\*\(?\s*(defaults|config\.get\(\"defaults\"|cfg\.get\(\"defaults\")", read(path)):
            merging.append(name)
    print(f"      gates merging a yaml's defaults: {merging}")
    check(
        "(4) every such gate pins im-ci off after the merge (today's runtime and numbers)", not unpinned, f"{unpinned}"
    )

    carriers, sweeps = {}, {}
    for path in sorted(glob.glob(os.path.join(REPO, "recipes", "*.yaml"))):
        with open(path) as handle:
            recipe = yaml.safe_load(handle) or {}
        blocks = [v for k, v in recipe.items() if k not in ("defaults", "hyperparameters") and isinstance(v, dict)]
        sweeps[os.path.basename(path)] = any((b.get("experiment") or {}).get("sweep") for b in blocks)
        carriers[os.path.basename(path)] = (recipe.get("defaults") or {}).get("im-ci")
    if not any(v is not None for v in carriers.values()):
        skip("(4) recipes", "no recipe carries im-ci yet (the recipe commit follows)")
    else:
        check(
            "(4) every sweep recipe carries im-ci 95",
            all(carriers[r] == 95 for r in carriers if sweeps[r]),
            f"{ {r: carriers[r] for r in carriers if sweeps[r]} }",
        )
        check("(4) no query / perf recipe carries it", all(carriers[r] is None for r in carriers if not sweeps[r]))


# gates that merge a yaml's `defaults:` into a block but never pin im-ci off: the
# do-MNIST helpers (never run here; the config forces that block off), this gate and
# the tolerance audit (both read the configured level on purpose).
# The scan reads dict displays only: a merge written as `dict.update` or `|` would
# slip past it, and a pin in a gate that never runs a sweep is a harmless no-op
# (the key is simply 0 there, as it was before the toggle existed)
PIN_EXEMPT = ("a72_im_ci.py", "a73_tolerance_audit.py", "smoke_do_mnist.py", "select_domnist_budgets.py")


def unpinned_merges(path):
    """Line numbers of the dict displays in `path` that unpack a yaml's `defaults`
    without a trailing `"im-ci": 0` after that unpack: such a block inherits the
    shipped level and its sweeps would bootstrap."""
    source = read(path)
    bad = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Dict):
            continue
        unpacked = [
            index
            for index, (key, value) in enumerate(zip(node.keys, node.values, strict=True))
            if key is None and "defaults" in (ast.get_source_segment(source, value) or "")
        ]
        if not unpacked:
            continue
        pinned = [
            index
            for index, (key, value) in enumerate(zip(node.keys, node.values, strict=True))
            if isinstance(key, ast.Constant)
            and key.value == "im-ci"
            and isinstance(value, ast.Constant)
            and value.value == 0
            and index > max(unpacked)
        ]
        if not pinned:
            bad.append(node.lineno)
    return bad


def calls_to(tree, names):
    """(enclosing class, enclosing function, name) of every call to one of `names`."""
    found = []

    def walk(node, cls=None, fn=None):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                walk(child, child.name, fn)
            elif isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                walk(child, cls, child.name)
            else:
                if isinstance(child, ast.Call):
                    func = child.func
                    name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
                    if name in names:
                        found.append((cls, fn, name))
                walk(child, cls, fn)

    walk(tree)
    return found


def leg_5a():
    print("(5a) exclusion, static")
    block = dict(seed=42, augmentation="translate", gamma=0.067, epsilon=0.1, methods=["PI"], **{"im-ci": 95})
    check("(5a) the do-MNIST block resolves to im_ci 0.0", resolve_dataset_block("do_mnist", block)["im_ci"] == 0.0)
    tokens = ("im_ci", "imbens", "bootstrap_bounds")
    for rel in (
        "src/experiments/do_mnist.py",
        "src/methods/partial_r2_net.py",
        "src/methods/partial_r2_net_jax.py",
        "src/experiments/perf.py",
    ):
        text = read(os.path.join(REPO, rel))
        check(f"(5a) {rel} names none of {tokens}", not any(t in text for t in tokens))
    # the net backend's intersections do not forward `pad_tolerance` to their branches;
    # do-MNIST is their only caller and its `im-ci` is forced to 0, so it stays 0.0 there
    nets = read(os.path.join(REPO, "src/methods/partial_r2_net.py"))
    check("(5a) partial_r2_net.py never forwards pad_tolerance", "pad_tolerance" not in nets)
    for obj in (QuerySweepRunner, PanelBuilder):
        text = inspect.getsource(obj)
        check(f"(5a) {obj.__name__} names none of {tokens}", not any(t in text for t in tokens))
    callers = []
    for path in glob.glob(os.path.join(REPO, "src", "**", "*.py"), recursive=True):
        if path.endswith(os.path.join("utils", "im_ci.py")):
            continue
        tree = ast.parse(read(path))
        for cls, fn, name in calls_to(tree, {"bootstrap_bounds", "imbens_manski_bounds"}):
            callers.append((os.path.relpath(path, REPO), cls, fn, name))
    want = {
        ("src/experiments/base.py", "ParamSweepRunner", "run", "bootstrap_bounds"),
        ("src/experiments/base.py", "ParamSweepRunner", "run", "imbens_manski_bounds"),
        ("src/experiments/base.py", "ParamSweepRunner", "bootstrap_bounds", "bootstrap_bounds"),
    }
    check("(5a) ParamSweepRunner is the helper's only caller in src/", set(callers) == want, f"{sorted(set(callers))}")


def leg_5b():
    print("(5b) exclusion, dynamic on simulation: perf and query never reach the helper")
    saved = im_ci.bootstrap_bounds

    def refuse(*args, **kwargs):
        raise RuntimeError("the IM-CI helper was reached")

    # every pad the two paths apply, recorded: they must keep EPS_TOL on it (the pad
    # drops it on the im-ci sweeps alone)
    finalize, pads = sensitivity_models.BoundedSA._finalize, []

    def recording(model, bounds):
        if model.pad:
            pads.append(model.pad_tolerance)
        return finalize(model, bounds)

    im_ci.bootstrap_bounds = refuse
    sensitivity_models.BoundedSA._finalize = recording
    try:
        with workdir("perf_"):
            orch = sim_orchestrator(["PI", "DA+PI", "PI&DA+PI"], im_ci_level=95.0, sweep_samples=2)
            plan = parse_experiment_plan({"perf": {"metric": ["wall_clock"]}})
            try:
                orch._run_perf(plan.perf)
                with open("artifacts/simulation/perf/epsilon_values.pkl", "rb") as handle:
                    steps = len(pickle.load(handle))  # noqa: S301 - the gate wrote this file itself
                ok, detail = True, f"{steps} steps"
            except Exception as error:
                ok, detail = False, f"{type(error).__name__}: {error}"
            check("(5b) _run_perf (wall_clock) completes under im_ci 95", ok, detail)
        with workdir("query_"):
            orch = sim_orchestrator(["PI", "DA+PI", "PI&DA+PI"], im_ci_level=95.0)
            try:
                orch._run_query_sweep()
                ok, detail = True, ""
            except Exception as error:
                ok, detail = False, f"{type(error).__name__}: {error}"
            check("(5b) _run_query_sweep (sweep_samples 8) completes under im_ci 95", ok, detail)
    finally:
        im_ci.bootstrap_bounds = saved
        sensitivity_models.BoundedSA._finalize = finalize
    check(
        "(5b) perf and query pad with EPS_TOL kept (pad_tolerance 0.0 on every padded model)",
        bool(pads) and all(tol == 0.0 for tol in pads),
        f"{len(pads)} pads, tolerances {sorted(set(pads))}",
    )


def slope(xs, ys):
    xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    keep = np.isfinite(ys) & (ys > 0)
    if keep.sum() < 2:
        return np.nan
    return float(np.polyfit(np.log(xs[keep]), np.log(ys[keep]), 1)[0])


def readings(x, ci, raw, valid):
    """Per method: the step means of CI and raw coverage, the CI/raw width ratio,
    the valid fractions, and the log-log slope of the CI excess width against x."""
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name in ci:
            if name == "ATE":
                continue
            cov = np.nanmean(ci[name]["coverage"], axis=1)
            cov_raw = np.nanmean(raw[name]["coverage"], axis=1)
            width, width_raw = ci[name]["interval_width"], raw[name]["interval_width"]
            ratio = np.nanmean(width, axis=1) / np.nanmean(width_raw, axis=1)
            excess = np.nanmean(width - width_raw, axis=1)
            out[name] = dict(
                coverage=np.round(cov, 4).tolist(),
                coverage_raw=np.round(cov_raw, 4).tolist(),
                ratio=np.round(ratio, 4).tolist(),
                valid=np.round(np.nanmean(valid[name], axis=1), 4).tolist(),
                slope=round(slope(x, excess), 4),
            )
    return out


def close(a, b, **tolerance):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return a.shape == b.shape and bool(np.allclose(a, b, equal_nan=True, **tolerance))


def leg_6(quick):
    print("(6) the sim n and m sweeps at sweep_samples 8, config.yaml's six methods")
    defaults, hyperparameters, sim = configured()
    metrics = ["coverage", "approx_error", "worst_error", "width"]
    count = QUICK_REPLICATES if quick else IM_CI_REPLICATES

    def block(level):
        resolved = resolve_dataset_block(
            "simulation",
            {**defaults, **sim, "sweep_samples": 8, "n_experiments": 2, "n_jobs": -1, "im-ci": level},
        )
        return resolved

    root = None
    with workdir("sweep_") as path, replicates(count), captured("INFO") as lines:
        root = path
        set_seed(42)
        orch = SimulationOrchestrator(**block(95), hyperparameters=munchify(hyperparameters))
        start = time.perf_counter()
        orch._run_sweeps(SweepSpec(param=("n", "m"), metric=tuple(metrics)))
        print(f"      B = {count}: n and m in {time.perf_counter() - start:.0f}s")
    floors = [ln for ln in lines if "BELOW" in ln]
    print(f"      budgets logged BELOW their floor (F14), {len(floors)} lines:")
    for line in floors:
        print(f"        {line}")
    set_seed(42)
    raw_orch = SimulationOrchestrator(**block(0), hyperparameters=munchify(hyperparameters))
    sweep = os.path.join(root, "artifacts", "simulation", "sweep")
    record = {}
    for param in ("n", "m"):
        x, ci, statuses = orch.sweep_record(param)
        _, plain, plain_statuses = raw_orch.sweep_record(param)
        meta = orch._sweep_ci[param]
        raw = meta["results_raw"]
        record[param] = (x, ci, raw, meta)
        structural(f"(6) {param}", ci, raw)
        for status in ("solver_failure", "infeasible"):
            check(
                f"(6) {param}: the {status} column is the raw run's",
                all(np.array_equal(column(statuses[n], status), column(plain_statuses[n], status)) for n in statuses),
            )
        # the im-ci run pads by eps* alone, the raw run by eps* + EPS_TOL: the unpadded
        # methods agree in every metric, the padded ones are narrower by at most 2 EPS_TOL
        padded = {n for n in raw if n.startswith(("DA+", "PI&DA+"))}
        check(
            f"(6) {param}: the raw run's metrics == results_raw on the unpadded methods",
            all(
                close(raw[n][m], plain[n][m], rtol=1e-9, atol=0)
                for n in set(raw) - padded
                for m in raw[n]
                if m != "wall_clock"
            ),
        )
        gaps = [plain[n]["interval_width"] - raw[n]["interval_width"] for n in padded]
        check(
            f"(6) {param}: ... and the padded ones' raw widths within 2 EPS_TOL below the raw run's",
            all(np.all(np.isnan(g) | ((g >= -1e-9) & (g <= 2 * EPS_TOL + 1e-9))) for g in gaps),
        )
        # a narrower raw interval can only lose coverage, can only be closer to the
        # worst query's target and can only be further from the whole target set
        for metric, sign in (("coverage", 1), ("worst_error", 1), ("approximation_error", -1)):
            moves = [sign * (raw[n][metric] - plain[n][metric]) for n in padded]
            check(
                f"(6) {param}: the padded methods' raw {metric} moves the way a narrower pad must",
                all(np.all(np.isnan(m) | (m <= 1e-9)) for m in moves),
                f"max move {max(np.nanmax(m) for m in moves):.3g}",
            )
        files = set(os.listdir(sweep))
        want = {f"{param}_{stem}.pkl" for stem in ("values", "results", "results_raw", "statuses", "im_ci")}
        want |= {f"{param}_{metric}_sweep.pdf" for metric in metrics}
        check(f"(6) {param}: pkls and figures written", want <= files, f"{sorted(want - files)}")
        saved_ci, saved_raw = (
            load_pkl(os.path.join(sweep, f"{param}_{stem}.pkl")) for stem in ("results", "results_raw")
        )
        check(
            f"(6) {param}: the saved results pkl is the CI record, the raw pkl the raw one",
            all(
                np.array_equal(saved_ci[n][m], ci[n][m], equal_nan=True)
                and np.array_equal(saved_raw[n][m], raw[n][m], equal_nan=True)
                for n in ci
                for m in ci[n]
            ),
        )
        # a nanmean over experiments hides an experiment whose every query is empty
        infeasible = {n: column(statuses[n], "infeasible") / statuses[n].sum(axis=-1) for n in statuses}
        empty = {n: np.isnan(ci[n]["coverage"]).sum(axis=1).tolist() for n in ci}
        share = np.round(infeasible["PI+INV"], 3).tolist()
        print(f"      {param}: PI+INV raw infeasible share per (step, experiment) {share}")
        print(f"      {param}: empty experiments per step {empty}")
        saved = load_pkl(os.path.join(sweep, f"{param}_im_ci.pkl"))
        check(
            f"(6) {param}_im_ci.pkl holds level, B, the seed offset, valid, seconds",
            set(saved) == {"level", "replicates", "seed_offset", "valid", "seconds"}
            and saved["level"] == 95.0
            and saved["replicates"] == count,
        )
    with captured() as warned:
        params = aggregate.sweep_params(os.path.join(root, "artifacts"), ["simulation"])
    check(
        "(6) aggregate.sweep_params finds n and m, no warning", sorted(params) == ["m", "n"] and not warned, f"{params}"
    )

    got = {param: readings(x, ci, raw, meta["valid"]) for param, (x, ci, raw, meta) in record.items()}
    for param, rows in got.items():
        print(f"      {param} = {np.asarray(record[param][0]).tolist()}")
        for name, row in rows.items():
            print(f"        {name}: {row}")
    print(f"      RECORDED_6 = {got!r}")
    if quick:
        skip("(6) RECORDED readings", f"--quick (B = {count})")
    elif RECORDED_6 is None:
        skip("(6) RECORDED readings", "first run: printed above, to be pinned")
    else:
        for param, rows in RECORDED_6.items():
            for name, want in rows.items():
                have = got[param][name]
                check(
                    f"(6) {param} {name}: coverage, ratio, valid, slope == RECORDED",
                    close(have["coverage"], want["coverage"], atol=COVERAGE_ATOL, rtol=0)
                    and close(have["coverage_raw"], want["coverage_raw"], atol=COVERAGE_ATOL, rtol=0)
                    and close(have["ratio"], want["ratio"], rtol=RATIO_RTOL)
                    and close(have["valid"], want["valid"], atol=COVERAGE_ATOL, rtol=0)
                    and close(have["slope"], want["slope"], atol=SLOPE_ATOL, rtol=0),
                    f"{have}",
                )
    return orch, record, root


def leg_7(orch, record, root):
    print("(7) render")
    errors = []
    sink = logger.add(lambda message: errors.append(message.record["message"]), level="ERROR")
    drawn = []
    original = base.create_sweep_plot

    def spy(x_values, y_results, *args, **kwargs):
        drawn.append({name: np.array(values) for name, values in y_results.items()})
        return original(x_values, y_results, *args, **kwargs)

    os.chdir(root)
    base.create_sweep_plot = spy
    try:
        plt.close("all")
        orch._run_sweeps(SweepSpec(param=("n",), metric=("width",)))
    finally:
        base.create_sweep_plot = original
        os.chdir(REPO)
        logger.remove(sink)
    check("(7) _run_sweeps draws without a swallowed error", not errors, errors[0][:120] if errors else "")
    x, ci, raw, _ = record["n"]
    figure = drawn[0] if len(drawn) == 1 else {}
    check(
        "(7) the width figure is handed the CI widths, not the raw ones",
        bool(figure)
        and all(np.array_equal(figure[n], ci[n]["interval_width"], equal_nan=True) for n in figure)
        and any(not np.array_equal(figure[n], raw[n]["interval_width"], equal_nan=True) for n in figure),
        f"{len(drawn)} figure(s)",
    )
    plt.close("all")
    create_sweep_plot(
        x,
        {name: ci[name]["interval_width"] for name in ci},
        xlabel=PARAM_SPECS["n"].xlabel,
        ylabel="w",
        experiment="simulation",
        fname="n_width",
        savefig=False,
        bootstrapped=False,
    )
    ax = plt.gcf().axes[0]
    order = np.argsort(np.asarray(x, dtype=float), kind="stable")
    for name in ("DA+PI", "PI&DA+PI"):
        lines = [ln for ln in ax.lines if ln.get_label() == TEX_MAPPER[name]]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            want = np.nanmean(ci[name]["interval_width"], axis=1)[order]
        ok = len(lines) == 1 and np.array_equal(np.asarray(lines[0].get_ydata(), dtype=float), want, equal_nan=True)
        check(f"(7) the {name} width line is the CI record's nanmean in sorted-x order", ok)
    plt.close("all")


# the machine-specific values of SS0: partitions, accounts, QOS, modules, scratch, hosts
MACHINE = re.compile(r"coc-cpu|oms-csp|coc-ice|ice-cpu|module load|scratch|pace\.gatech|atl1-", re.IGNORECASE)
# pre-existing hits, allowed: the cigarette data fallback, the do-MNIST data dir, two
# log paths quoted in configs.py comments
MACHINE_ALLOWED = (
    ("src/sem/cigarettes.py", 'DATA_ROOT: str = "~/scratch/data"'),
    ("src/sem/do_mnist.py", "~/scratch/data/mnist"),
    ("src/experiments/configs.py", "~/scratch/tmp/impl_v7/logs"),
)


def example_lines(path):
    """The launcher's lines inside its module docstring or its `EPILOG` string, the
    one place a labelled PACE example may appear."""
    tree = ast.parse(read(path))
    spans = []
    if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant):
        spans.append((tree.body[0].lineno, tree.body[0].end_lineno))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "EPILOG" for t in node.targets):
            spans.append((node.lineno, node.end_lineno))
    return {line for start, end in spans for line in range(start, end + 1)}


def leg_8():
    print("(8) the Slurm launcher, dry run")
    launcher = os.path.join(REPO, "scripts", "sbatch_sweeps.py")
    if not os.path.exists(launcher):
        skip("(8)", "scripts/sbatch_sweeps.py not in yet (the launcher commit follows)")
        return
    for source in ("config.yaml", "recipes/nEfficiencyFig13.yaml"):
        with open(os.path.join(REPO, source)) as handle:
            config = yaml.safe_load(handle) or {}
        expected = set()
        for dataset, blk in config.items():
            if dataset in ("defaults", "hyperparameters") or not isinstance(blk, dict):
                continue
            experiment = blk.get("experiment") or {}
            expected |= {f"{dataset}_{p}" for p in (experiment.get("sweep") or {}).get("param", [])}
            if experiment.get("perf"):
                expected.add(f"{dataset}_perf")
        with workdir("launch_") as out:
            done = subprocess.run(  # noqa: S603 - our own script
                [sys.executable, launcher, "--config", os.path.join(REPO, source), "--out", out, "--dry-run"],
                capture_output=True,
                text=True,
                cwd=REPO,
            )
            check(f"(8) {source}: the dry run exits 0", done.returncode == 0, done.stderr[-300:])
            emitted = {os.path.basename(p)[: -len(".yaml")] for p in glob.glob(os.path.join(out, "configs", "*.yaml"))}
            check(
                f"(8) {source}: one yaml per (dataset, param) and perf block", emitted == expected, f"{sorted(emitted)}"
            )
            stems = []
            for task in sorted(emitted):
                with open(os.path.join(out, "configs", f"{task}.yaml")) as handle:
                    cfg = yaml.safe_load(handle)
                check(
                    f"(8) {task}.yaml carries the source's defaults and hyperparameters verbatim",
                    cfg.get("defaults") == config.get("defaults")
                    and cfg.get("hyperparameters") == config.get("hyperparameters"),
                )
                try:
                    for dataset, blk in cfg.items():
                        if dataset in ("defaults", "hyperparameters"):
                            continue
                        merged = {**(cfg.get("defaults") or {}), **blk}
                        plan = parse_experiment_plan(merged.get("experiment"))
                        resolve_dataset_block(dataset, merged)
                        stems += [(dataset, "sweep", p) for p in (plan.sweep.param if plan.sweep else ())]
                        stems += [(dataset, "perf", "epsilon")] if plan.perf else []
                    ok, detail = True, ""
                except Exception as error:
                    ok, detail = False, f"{type(error).__name__}: {error}"
                check(f"(8) {task}.yaml resolves", ok, detail)
            check(f"(8) {source}: no two tasks share a (dataset, subdir, stem)", len(stems) == len(set(stems)))
            script = read(os.path.join(out, "run.sbatch"))
            asked = re.findall(r"^#SBATCH\s+--(partition|account|qos|cpus-per-task|mem)", script, re.MULTILINE)
            check(f"(8) {source}: no partition / account / QOS / cpus / mem directive unasked", not asked, f"{asked}")
            check(f"(8) {source}: no env-setup line unasked", "module load" not in script)
            links = glob.glob(os.path.join(out, "task_*"))
            good = [
                os.path.realpath(os.path.join(t, "artifacts")) == os.path.realpath(os.path.join(REPO, "artifacts"))
                and os.path.realpath(os.path.join(t, "data")) == os.path.realpath(os.path.join(REPO, "data"))
                for t in links
                if os.path.islink(os.path.join(t, "artifacts")) and os.path.islink(os.path.join(t, "data"))
            ]
            check(
                f"(8) {source}: every task dir links artifacts/ and data/ to the repo",
                links and len(good) == len(links) and all(good),
            )
    with workdir("launch_flags_") as out:
        flags = ["--partition", "P0", "--account", "A0", "--qos", "Q0", "--env-setup", "echo setup"]
        flags += ["--cpus-per-task", "3", "--mem", "5G"]
        done = subprocess.run(  # noqa: S603 - our own script
            [
                sys.executable,
                launcher,
                "--config",
                os.path.join(REPO, "config.yaml"),
                "--out",
                out,
                "--dry-run",
                *flags,
            ],
            capture_output=True,
            text=True,
            cwd=REPO,
        )
        check("(8) the flagged dry run exits 0", done.returncode == 0, done.stderr[-300:])
        script = read(os.path.join(out, "run.sbatch"))
        asked = ("--partition=P0", "--account=A0", "--qos=Q0", "--cpus-per-task=3", "--mem=5G", "echo setup")
        check("(8) asked directives are emitted", all(s in script for s in asked))

    with workdir("launch_domnist_") as out:
        synthetic = os.path.join(out, "with_do_mnist.yaml")
        with open(synthetic, "w") as handle:
            yaml.safe_dump(
                {
                    "defaults": {"n_jobs": 1},
                    "simulation": {"seed": 42, "kernel_dim": 0, "experiment": {"sweep": {"param": ["n"]}}},
                    "do_mnist": {
                        "seed": 42,
                        "experiment": {"sweep": {"param": ["m"]}, "perf": {"metric": ["seed_var"]}},
                    },
                },
                handle,
            )
        subprocess.run(  # noqa: S603 - our own script
            [sys.executable, launcher, "--config", synthetic, "--out", out, "--dry-run", "--tasks", "sweep,perf,query"],
            capture_output=True,
            text=True,
            cwd=REPO,
        )
        emitted = sorted(os.path.basename(p) for p in glob.glob(os.path.join(out, "configs", "*.yaml")))
        check("(8) a do_mnist block is never fanned out", emitted == ["simulation_n.yaml"], f"{emitted}")

    hits = []
    paths = glob.glob(os.path.join(REPO, "src", "**", "*.py"), recursive=True)
    paths += [os.path.join(REPO, "config.yaml"), launcher, *glob.glob(os.path.join(REPO, "recipes", "*.yaml"))]
    allowed_launcher = example_lines(launcher)
    for path in paths:
        rel = os.path.relpath(path, REPO)
        for number, line in enumerate(read(path).splitlines(), start=1):
            if not MACHINE.search(line):
                continue
            if path == launcher and number in allowed_launcher:
                continue
            if any(rel == where and text in line for where, text in MACHINE_ALLOWED):
                continue
            hits.append(f"{rel}:{number}")
    check("(8) no NEW machine-specific value outside the launcher's labelled example", not hits, f"{hits[:6]}")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.chdir(REPO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=None, help="one leg: 0, 1, 2, 3, 4, 5a, 5b, 6, 7 (runs 6 first) or 8")
    parser.add_argument("--quick", action="store_true", help="B = 20 in legs 3 and 6, leg 6's RECORDED skipped")
    args = parser.parse_args()
    print(f"tree: {base.__file__}")

    state = {}

    def leg_6_7():
        state["6"] = leg_6(args.quick)
        leg_7(*state["6"])

    legs = [
        ("0", leg_0),
        ("1", leg_1),
        ("2", leg_2),
        ("3", lambda: leg_3(args.quick)),
        ("4", leg_4),
        ("5a", leg_5a),
        ("5b", leg_5b),
        ("6", leg_6_7),
        ("8", leg_8),
    ]
    only = "6" if args.only == "7" else args.only
    if only:
        legs = [(tag, leg) for tag, leg in legs if tag == only]
        if not legs:
            sys.exit(f"unknown leg {args.only!r}")
    for tag, leg in legs:
        start = time.perf_counter()
        try:
            leg()
        except Exception as error:  # a raise is a FAIL line, and the later legs still report
            check(f"({tag}) ran without raising", False, f"{type(error).__name__}: {error}")
        print(f"      leg {tag}: {time.perf_counter() - start:.0f}s", flush=True)
    if SKIPPED:
        print(f"  {len(SKIPPED)} skipped: {SKIPPED}")
    if not FAIL:
        for path in MADE:
            shutil.rmtree(path, ignore_errors=True)
        with contextlib.suppress(OSError):
            os.rmdir(TMPROOT)  # only when nothing else lives there
        print("A72 PASS")
    else:
        print(f"  outputs kept under {TMPROOT}")
        print(f"A72 FAIL: {FAIL}")
        sys.exit(1)
