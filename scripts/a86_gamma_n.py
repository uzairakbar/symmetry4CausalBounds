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
  (vii) one record: `_run_sweeps` writes `<param>_values`, `_results`,
        `_statuses` and `_axis` and nothing else, the aggregate's sweep grid reads
        `_values` and `_results`, the runner takes no CI level, `BoundedSA` has no
        pad tolerance, `src.experiments.utils` has no CI module; and no symbol of
        the retired Imbens-Manski CI is left in src/, scripts/, recipes/,
        config.yaml, README.md or scripts/README.md (this script and a77's
        unknown-key check aside, which name it to reject it).
        Catches: a second sweep record, a leftover of the bootstrap path.
  (x)   grids: the m grid off `sweep_samples` (1..16 at 16, 1..8 at 8, 32 points
        at 32), the n grid `geomspace(10, 100, count)` at every count; the sim n
        strategy takes its 8 row counts (205 .. 2048 of 2048) at sweep_samples 8,
        the m strategy 8 steps at 6.25% of 2048 = 128 rows; no 16 / 17 literal in
        either `grid_fn`. Catches: a grid that ignores the knob.
  (xi)  the Slurm launcher (`sbatch_sweeps.py --dry-run`), no submission: one yaml
        per (dataset, param) and per perf block, each resolving; no two tasks share
        a (dataset, subdir, stem); only the directives asked for; the task dirs'
        links; a do_mnist block never fanned out; no NEW machine-specific value in
        src/, config.yaml, the recipes or the launcher outside its `EPILOG` /
        docstring.

    uv run python scripts/a86_gamma_n.py [--only LEG]

Writes only under $A86_TMPROOT (default ~/scratch/tmp/a86), removed on PASS.
"""

import argparse
import ast
import contextlib
import glob
import importlib.util
import inspect
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import yaml
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
TMPROOT = os.environ.get("A86_TMPROOT", os.path.expanduser("~/scratch/tmp/a86"))
DATASETS = ("simulation", "optical_device", "cigarettes")
# the retired Imbens-Manski CI's symbols (the yaml key, its parsed name, the
# constants, the helpers, the raw record and its reader, the pad tolerance)
IM_CI_SYMBOLS = re.compile(
    r"im-ci|im_ci|IM_CI|imbens|bootstrap_bounds|results_raw|_raw_record|RAW_MTIME_SLACK|sweep_record_for"
    r"|pad_tolerance",
    re.IGNORECASE,
)
# the lines that name the retired key to reject it
IM_CI_ALLOWED = (("scripts/a77_domnist_block.py", "retired"),)
SWEEP_STEMS = {"values", "results", "statuses", "axis"}
# the machine-specific values no file but the launcher's labelled example may carry
MACHINE = re.compile(r"coc-cpu|oms-csp|coc-ice|ice-cpu|module load|scratch|pace\.gatech|atl1-", re.IGNORECASE)
# pre-existing hits, allowed: the cigarette data fallback, the do-MNIST data dir, a
# log path quoted in a configs.py comment
MACHINE_ALLOWED = (
    ("src/sem/cigarettes.py", 'DATA_ROOT: str = "~/scratch/data"'),
    ("src/sem/do_mnist.py", "~/scratch/data/mnist"),
    ("src/experiments/configs.py", "~/scratch/tmp/impl_v7/logs"),
)
FAIL = []
MADE = []  # the directories this run created under TMPROOT, removed on PASS


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


def read(path):
    with open(path) as handle:
        return handle.read()


@contextlib.contextmanager
def workdir(prefix):
    """A fresh directory under TMPROOT as cwd, with the repo's `data/` linked in."""
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


def leg_vii():
    from src.aggregate import sweep_grid
    from src.experiments.base import BaseExperimentRunner, ExperimentOrchestrator
    from src.methods.sensitivity_models import BoundedSA

    print("(vii) one record, no Imbens-Manski CI left")
    stems = set(re.findall(r'f"\{param\}_(\w+)"', inspect.getsource(ExperimentOrchestrator._run_sweeps)))
    check("(vii) _run_sweeps writes values, results, statuses and axis only", stems == SWEEP_STEMS, f"{sorted(stems)}")
    read_back = set(re.findall(r"\{param\}_(\w+)\.pkl", inspect.getsource(sweep_grid)))
    check("(vii) the aggregate's sweep grid reads values, results and axis", read_back == {"values", "results", "axis"})
    check(
        "(vii) the runner takes no CI level",
        not any("ci" in name.split("_") for name in inspect.signature(BaseExperimentRunner.__init__).parameters),
    )
    check("(vii) BoundedSA has no pad tolerance", not any("tolerance" in name for name in vars(BoundedSA)))
    check(
        "(vii) no CI module under src.experiments.utils",
        importlib.util.find_spec("src.experiments.utils.im_ci") is None,
    )
    paths = glob.glob(os.path.join(REPO, "src", "**", "*.py"), recursive=True)
    paths += glob.glob(os.path.join(REPO, "scripts", "*.py")) + glob.glob(os.path.join(REPO, "recipes", "*.yaml"))
    paths += [os.path.join(REPO, name) for name in ("config.yaml", "README.md", os.path.join("scripts", "README.md"))]
    hits = []
    for path in paths:
        rel = os.path.relpath(path, REPO)
        if rel == os.path.join("scripts", os.path.basename(__file__)):
            continue
        for number, line in enumerate(read(path).splitlines(), start=1):
            if IM_CI_SYMBOLS.search(line) and not any(rel == where and text in line for where, text in IM_CI_ALLOWED):
                hits.append(f"{rel}:{number}")
    check("(vii) no Imbens-Manski CI symbol left", not hits, f"{hits[:8]}")


def leg_x():
    from src.experiments.configs import N_PERCENT_RANGE, PARAM_SPECS
    from src.experiments.simulation import SimulationOrchestrator
    from src.experiments.utils import set_seed

    print("(x) the m grid reads sweep_samples, the n grid is log-spaced on 10..100 %")
    for dataset in DATASETS:
        want = list(range(1, 17))
        got = PARAM_SPECS["m"].grid_fn(dataset, 16)
        check(f"(x) {dataset} m at 16 is 1..16", got.tolist() == want, f"{got.tolist()}")
        grid = PARAM_SPECS["m"].grid_fn(dataset, 32)
        check(f"(x) {dataset} m at 32: 32 points from 1", len(grid) == 32 and grid[0] == 1)
        check(f"(x) {dataset} m at 8 is 1..8", PARAM_SPECS["m"].grid_fn(dataset, 8).tolist() == list(range(1, 9)))
        grids = [PARAM_SPECS["n"].grid_fn(dataset, count) for count in (8, 16, 32)]
        check(
            f"(x) {dataset} n at 8, 16, 32 == geomspace({N_PERCENT_RANGE[0]:g}, {N_PERCENT_RANGE[1]:g}, count)",
            all(
                np.array_equal(grid, np.geomspace(*N_PERCENT_RANGE, num=count))
                for grid, count in zip(grids, (8, 16, 32), strict=True)
            ),
            f"{grids[0].tolist()}",
        )
    set_seed(42)
    orch = SimulationOrchestrator(
        seed=42,
        n_samples=2048,
        kernel_dim=0,
        treatment_dim=32,
        augmentation="translate",
        n_experiments=1,
        sweep_samples=8,
        methods=["PI"],
        hyperparameters={},
        n_jobs=1,
        recalibrate=True,
        pad=True,
        clipy=False,
        mean_match=True,
    )

    def runner_for(param):
        return orch.get_sweep_runner_cls(param)(
            methods=orch.methods, method_factory=orch.build_methods, **orch._get_clean_kwargs()
        )

    rows = runner_for("n").get_param_range().tolist()
    want_rows = [205, 285, 395, 549, 763, 1061, 1474, 2048]
    check("(x) the sim n strategy takes the grid's rows of 2048 at 8", rows == want_rows, f"{rows}")
    runner = runner_for("m")
    check("(x) the sim m strategy takes 8 steps at sweep_samples 8", len(runner.get_param_range()) == 8)
    check("(x) ... at 6.25% of 2048 = 128 rows", runner.n_samples_override == 128, f"{runner.n_samples_override}")
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
    check("(x) both grid_fns found in PARAM_SPECS", set(literals) == {"n", "m"}, f"{sorted(literals)}")
    check("(x) no 16 / 17 literal left in either", not any(literals.values()), f"{literals}")


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


def dry_run(launcher, config, out, *flags):
    return subprocess.run(  # noqa: S603 - our own script
        [sys.executable, launcher, "--config", config, "--out", out, "--dry-run", *flags],
        capture_output=True,
        text=True,
        cwd=REPO,
    )


def leg_xi():
    from src.experiments.configs import parse_experiment_plan, resolve_dataset_block

    print("(xi) the Slurm launcher, dry run")
    launcher = os.path.join(REPO, "scripts", "sbatch_sweeps.py")
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
            done = dry_run(launcher, os.path.join(REPO, source), out)
            check(f"(xi) {source}: the dry run exits 0", done.returncode == 0, done.stderr[-300:])
            emitted = {os.path.basename(p)[: -len(".yaml")] for p in glob.glob(os.path.join(out, "configs", "*.yaml"))}
            check(
                f"(xi) {source}: one yaml per (dataset, param) and perf block",
                emitted == expected,
                f"{sorted(emitted)}",
            )
            stems = []
            for task in sorted(emitted):
                with open(os.path.join(out, "configs", f"{task}.yaml")) as handle:
                    cfg = yaml.safe_load(handle)
                check(
                    f"(xi) {task}.yaml carries the source's defaults and hyperparameters verbatim",
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
                check(f"(xi) {task}.yaml resolves", ok, detail)
            check(f"(xi) {source}: no two tasks share a (dataset, subdir, stem)", len(stems) == len(set(stems)))
            script = read(os.path.join(out, "run.sbatch"))
            asked = re.findall(r"^#SBATCH\s+--(partition|account|qos|cpus-per-task|mem)", script, re.MULTILINE)
            check(f"(xi) {source}: no partition / account / QOS / cpus / mem directive unasked", not asked, f"{asked}")
            check(f"(xi) {source}: no env-setup line unasked", "module load" not in script)
            links = glob.glob(os.path.join(out, "task_*"))
            good = [
                os.path.realpath(os.path.join(t, "artifacts")) == os.path.realpath(os.path.join(REPO, "artifacts"))
                and os.path.realpath(os.path.join(t, "data")) == os.path.realpath(os.path.join(REPO, "data"))
                for t in links
                if os.path.islink(os.path.join(t, "artifacts")) and os.path.islink(os.path.join(t, "data"))
            ]
            check(
                f"(xi) {source}: every task dir links artifacts/ and data/ to the repo",
                links and len(good) == len(links) and all(good),
            )
    with workdir("launch_flags_") as out:
        flags = ["--partition", "P0", "--account", "A0", "--qos", "Q0", "--env-setup", "echo setup"]
        flags += ["--cpus-per-task", "3", "--mem", "5G"]
        done = dry_run(launcher, os.path.join(REPO, "config.yaml"), out, *flags)
        check("(xi) the flagged dry run exits 0", done.returncode == 0, done.stderr[-300:])
        script = read(os.path.join(out, "run.sbatch"))
        asked = ("--partition=P0", "--account=A0", "--qos=Q0", "--cpus-per-task=3", "--mem=5G", "echo setup")
        check("(xi) asked directives are emitted", all(s in script for s in asked))
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
        dry_run(launcher, synthetic, out, "--tasks", "sweep,perf,query")
        emitted = sorted(os.path.basename(p) for p in glob.glob(os.path.join(out, "configs", "*.yaml")))
        check("(xi) a do_mnist block is never fanned out", emitted == ["simulation_n.yaml"], f"{emitted}")
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
    check("(xi) no NEW machine-specific value outside the launcher's labelled example", not hits, f"{hits[:6]}")


LEGS = {"i": leg_i, "vii": leg_vii, "x": leg_x, "xi": leg_xi}


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
    if not FAIL:
        for path in MADE:
            shutil.rmtree(path, ignore_errors=True)
        with contextlib.suppress(OSError):
            os.rmdir(TMPROOT)  # only when nothing else lives there
    print(f"\n{'A86 ALL PASS' if not FAIL else 'A86 FAILURES: ' + ', '.join(FAIL)}")
    sys.exit(bool(FAIL))
