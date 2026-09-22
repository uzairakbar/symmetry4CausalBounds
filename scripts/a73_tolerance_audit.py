"""A73: which tolerances were finite-sample patches, read under the IM-CI.

`EPS_TOL` = 2**-5 rides on every auto-set budget twice on the sweep path: (1a) in
the CONSTRAINT budgets (eps* + EPS_TOL, eps_iv* + EPS_TOL, eps_iv_z* + EPS_TOL),
where it keeps PI+INV and the +IV methods off the feasibility knife edge, and (1b)
inside the Thm. 3.A PAD, which is `epsilon` = eps* + EPS_TOL wherever `pad_epsilon`
is unset (sim, cigarettes, and the sweeps of all three orchestrators). With the
metrics reading the Imbens-Manski CI, the sampling allowance on the INTERVAL is the
CI's job, so 1b is the redundancy candidate; an empty set has no CI, so 1a is not.
The audit decides both with data, on the n and m sweeps of the three datasets:

  V0  the raw bounds of the V1 run (its `{param}_results_raw.pkl`)
  V1  the pre-retirement pad, eps* + EPS_TOL, im-ci as configured: an existing tree
      (`--v1`, e.g. a run from before the retirement) or run here with the runner's
      `pad_tolerance` forced to 0
  V2  the pad by `epsilon - EPS_TOL` = eps* alone on every padded model, the
      intersections' DA branches (`_branch` forwards it) and the replicates; the
      constraint budgets unchanged. Since 1b was retired this IS the shipped
      im-ci sweep (`ParamSweepRunner.pad_tolerance`); the patch sets the same
      `pad_tolerance`, so it never subtracts twice and also works under im-ci 0
  V3  V1 with `EPS_TOL = 0` on the modules that add it (base, generic_runner,
      optical_device, cigarettes, and the two `_epsilon_budget` defaults)

Every variant is a patch of module or model attributes inside this script; nothing
under src/ changes. V2 patches the BUILDERS the runner is handed, so the bootstrap
replicates (built in the workers) are padded the same way as the point fit.

  1b retires iff at every (dataset, param, step) and every DA+ method or
     intersection, in any instrument mode (`padded_family`), V2 coverage >=
     min(0.95, V1 coverage - 0.01) and V2 width < V1 width.
  1a retires iff the raw-INFEASIBLE share (the statuses' solver column) of the
     INV methods, DA+PI+IV and the intersections, in any mode (`knife_family`), under
     V3 exceeds V1's by at most 5 points at every step. A verdict with no cell
     checked prints as undecided.

    uv run python scripts/a73_tolerance_audit.py --config YAML --out DIR
        [--v1 ARTIFACTS] [--v2 ARTIFACTS] [--variants V2,V3] [--reuse]

Each criterion is read only when its variant is there, so `--v1 PRE --v2 POST
--variants ""` decides 1b across two finished runs (a pre- and a post-retirement one)
without running anything, and leaves 1a alone.

`--config` is the multi-block yaml the V1 tree was run from, each dataset block with
`experiment.sweep.param` naming the swept params (n and m are audited); each
(variant, dataset, param) runs as its own `src/main.py`-shaped process on the block
with that one param, writing under DIR/<variant>/artifacts; `--reuse` keeps a
finished one. The verdicts print, the per-cell table goes to DIR/audit.json and one
V0 / V1 / V2 / V3 figure per (dataset, param) to DIR/<dataset>_<param>_audit.pdf.
"""

import argparse
import copy
import json
import os
import pickle
import subprocess
import sys
import tempfile
import warnings
from functools import partial

import numpy as np
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

PARAMS = ("n", "m")
VARIANTS = ("V2", "V3")


def padded_family(name: str) -> bool:
    """The DA+ methods and intersections, in any instrument mode: the ones the pad hits."""
    return name.startswith(("DA+", "PI&DA+"))


def knife_family(name: str) -> bool:
    """The methods carrying an invariance or a DA-side IV constraint, in any mode."""
    return "INV" in name or name.startswith(("DA+PI+IV", "PI&DA+"))


COVERAGE_FLOOR = 0.95
COVERAGE_SLACK = 0.01
INFEASIBLE_SLACK = 0.05


# ------------------------------------------------------------------ the variants


def _padded_model(builder, tol):
    """`builder()` padding by its constraint epsilon minus `tol` (eps* alone); an
    intersection hands the setting to its DA branch when it builds it at fit."""
    model = builder()
    if hasattr(model, "pad_tolerance"):
        model.pad_tolerance = tol
    return model


def _padded_factory(factory, tol, **budgets):
    return {name: partial(_padded_model, builder, tol) for name, builder in factory(**budgets).items()}


def _zero_tolerance():
    """EPS_TOL = 0 wherever the sweep path adds it (module attributes and the two
    orchestrators' `_epsilon_budget` defaults, bound at definition time)."""
    import src.experiments.base as base
    import src.experiments.cigarettes as cigarettes
    import src.experiments.generic_runner as generic_runner
    import src.experiments.optical_device as optical_device

    for module in (base, generic_runner, optical_device, cigarettes):
        module.EPS_TOL = 0.0
    optical_device.OpticalOrchestrator._epsilon_budget.__defaults__ = (0.0,)
    cigarettes.CigaretteOrchestrator._epsilon_budget.__defaults__ = (0.0,)


def run_task(config_path, variant, dataset, param, workdir):
    """One (variant, dataset, param) as src/main.py runs a one-param block, in `workdir`."""
    from loguru import logger
    from munch import munchify

    from src.experiments.configs import EPS_TOL, parse_experiment_plan, resolve_dataset_block
    from src.experiments.utils import set_seed
    from src.main import ORCHESTRATORS

    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    with open(config_path) as handle:
        config = yaml.safe_load(handle)
    block = {**(config.get("defaults") or {}), **copy.deepcopy(config[dataset])}
    sweep = dict(block["experiment"]["sweep"])
    block["experiment"] = {"sweep": {**sweep, "param": [param]}}
    plan = parse_experiment_plan(block["experiment"])
    block = resolve_dataset_block(dataset, block)
    if variant == "V1":
        # the pre-retirement pad: nothing dropped, point models and replicates alike
        import src.experiments.base as base

        base.ParamSweepRunner.pad_tolerance = property(lambda runner: 0.0)
    if variant == "V3":
        _zero_tolerance()
    os.chdir(workdir)
    set_seed(block["seed"])
    orchestrator = ORCHESTRATORS[dataset](**block, hyperparameters=munchify(config.get("hyperparameters") or {}))
    if variant == "V2":
        orchestrator.build_methods = partial(_padded_factory, orchestrator.build_methods, EPS_TOL)
    orchestrator.run(plan)


# ------------------------------------------------------------------ reading


def load(root, dataset, param, stem):
    path = os.path.join(root, dataset, "sweep", f"{param}_{stem}.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as handle:
        return pickle.load(handle)  # noqa: S301 - artifacts of our own runs


def infeasible_share(statuses):
    """Per step, the experiment mean of the solver's INFEASIBLE share of the queries."""
    from src.experiments.utils.metrics import STATUS_CATEGORIES

    counts = np.asarray(statuses, dtype=float)
    return (counts[..., list(STATUS_CATEGORIES).index("infeasible")] / counts.sum(axis=-1)).mean(axis=1)


def step_means(results, name, metric):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanmean(results[name][metric], axis=1)


def audit(roots, datasets, out):
    """The two criteria per (dataset, param, step, method), the cells that fail them,
    and one figure per (dataset, param). `roots` maps each variant present -- always
    V1, then whichever of V2 and V3 this run has -- to its artifacts tree, and each
    criterion is evaluated only when its variant is there."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table, failures, checked = {}, {"1b": [], "1a": []}, {"1b": 0, "1a": 0}
    missing = []
    v1_root = roots["V1"]
    for dataset in datasets:
        for param in PARAMS:
            x = load(v1_root, dataset, param, "values")
            runs = {"V0": load(v1_root, dataset, param, "results_raw")}
            runs |= {variant: load(root, dataset, param, "results") for variant, root in roots.items()}
            statuses = {"V1": load(v1_root, dataset, param, "statuses")}
            if "V3" in roots:
                statuses["V3"] = load(roots["V3"], dataset, param, "statuses")
            if x is None or any(r is None for r in runs.values()) or any(s is None for s in statuses.values()):
                missing.append(f"{dataset}/{param}")
                continue
            x = np.asarray(x, dtype=float).tolist()
            cell = table.setdefault(dataset, {}).setdefault(param, {"x": x, "methods": {}})
            for name in runs["V1"]:
                row = {
                    f"{metric}_{v}": np.round(step_means(runs[v], name, key), 5).tolist()
                    for v in runs
                    for metric, key in (("coverage", "coverage"), ("width", "interval_width"))
                }
                for variant, counts in statuses.items():
                    row[f"infeasible_{variant}"] = np.round(infeasible_share(counts[name]), 5).tolist()
                # every metric the retirement can move, for the record
                for metric in ("worst_error", "approximation_error"):
                    for variant, run in runs.items():
                        row[f"{metric}_{variant}"] = np.round(step_means(run, name, metric), 6).tolist()
                cell["methods"][name] = row
                for i, step in enumerate(x):
                    where = f"{dataset} {param}={step:g} {name}"
                    if padded_family(name) and "V2" in roots:
                        c1, c2 = row["coverage_V1"][i], row["coverage_V2"][i]
                        w1, w2 = row["width_V1"][i], row["width_V2"][i]
                        if np.isfinite(c1) or np.isfinite(c2):
                            checked["1b"] += 1
                            ok = c2 >= min(COVERAGE_FLOOR, c1 - COVERAGE_SLACK) and w2 < w1
                            if not ok:
                                failures["1b"].append(
                                    f"{where}: coverage {c1:.4f} -> {c2:.4f}, width {w1:.4g} -> {w2:.4g}"
                                )
                    if knife_family(name) and "V3" in roots:
                        checked["1a"] += 1
                        s1, s3 = row["infeasible_V1"][i], row["infeasible_V3"][i]
                        if s3 - s1 > INFEASIBLE_SLACK:
                            failures["1a"].append(f"{where}: infeasible {s1:.3f} -> {s3:.3f}")
            figure(plt, cell, dataset, param, out)
    return table, failures, checked, missing


def figure(plt, cell, dataset, param, out):
    names = list(cell["methods"])
    fig, axes = plt.subplots(2, len(names), figsize=(3.2 * len(names), 5.6), squeeze=False, sharex=True)
    for column, name in enumerate(names):
        row = cell["methods"][name]
        for line, metric in enumerate(("coverage", "width")):
            ax = axes[line][column]
            for variant, style in (("V0", ":"), ("V1", "-"), ("V2", "--"), ("V3", "-.")):
                if f"{metric}_{variant}" in row:
                    ax.plot(cell["x"], row[f"{metric}_{variant}"], style, label=variant)
            ax.set_title(name if line == 0 else "", fontsize=9)
            ax.set_ylabel(metric if column == 0 else "")
            if param == "n":
                ax.set_xscale("log")
            ax.set_xlabel(param if line == 1 else "")
    axes[0][0].legend(fontsize=8)
    fig.suptitle(f"{dataset}: V0 raw, V1 IM-CI, V2 pad at eps*, V3 EPS_TOL = 0")
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"{dataset}_{param}_audit.pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="the multi-block yaml the V1 tree was run from")
    parser.add_argument("--v1", help="an existing V1 artifacts/ tree; omitted = run V1 here")
    parser.add_argument("--v2", help="an existing V2 artifacts/ tree (a post-retirement run); omitted = run V2 here")
    parser.add_argument("--out", required=True, help="where the V2 / V3 trees, the table and the figures go")
    parser.add_argument("--variants", default="V2,V3")
    parser.add_argument("--reuse", action="store_true", help="keep a (variant, dataset, param) already written")
    parser.add_argument("--task", nargs=3, metavar=("VARIANT", "DATASET", "PARAM"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    config_path = os.path.abspath(args.config)
    out = os.path.abspath(args.out)

    if args.task:
        variant, dataset, param = args.task
        run_task(config_path, variant, dataset, param, os.path.join(out, variant))
        return 0

    with open(config_path) as handle:
        config = yaml.safe_load(handle)
    datasets = [
        d
        for d, block in config.items()
        if d not in ("defaults", "hyperparameters")
        and isinstance(block, dict)
        and set(PARAMS) & set(((block.get("experiment") or {}).get("sweep") or {}).get("param") or [])
    ]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    # `--v2` names a finished V2 tree (a post-retirement run of the same config), so the
    # criterion can be read across two existing runs with nothing rerun here
    if args.v2:
        variants = [v for v in variants if v != "V2"]
    if args.v1 is None and "V1" not in variants:
        variants.insert(0, "V1")
    roots = {"V1": os.path.abspath(args.v1) if args.v1 else os.path.join(out, "V1", "artifacts")}
    if args.v2:
        roots["V2"] = os.path.abspath(args.v2)
    for variant in variants:
        if variant != "V1":
            roots[variant] = os.path.join(out, variant, "artifacts")
        root = os.path.join(out, variant)
        os.makedirs(root, exist_ok=True)
        if not os.path.exists(os.path.join(root, "data")) and os.path.isdir(os.path.join(REPO, "data")):
            os.symlink(os.path.join(REPO, "data"), os.path.join(root, "data"))
        for dataset in datasets:
            for param in PARAMS:
                if args.reuse and load(os.path.join(root, "artifacts"), dataset, param, "results") is not None:
                    print(f"{variant} {dataset} {param}: reused", flush=True)
                    continue
                command = [sys.executable, os.path.abspath(__file__), "--config", config_path]
                command += ["--out", out, "--task", variant, dataset, param]
                with tempfile.TemporaryFile() as log:
                    done = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)  # noqa: S603
                    log.seek(0)
                    tail = log.read().decode(errors="replace")[-800:]
                print(f"{variant} {dataset} {param}: rc {done.returncode}", flush=True)
                if done.returncode:
                    print(tail)
                    return 1

    table, failures, checked, missing = audit(roots, datasets, out)
    with open(os.path.join(out, "audit.json"), "w") as handle:
        json.dump({"table": table, "failures": failures, "checked": checked, "missing": missing}, handle, indent=1)
    if missing:
        print(f"missing runs: {missing}")
    for knob, label in (("1b", "the pad's EPS_TOL (V2)"), ("1a", "the constraint's EPS_TOL (V3)")):
        if knob == "1b" and "V2" not in roots:
            continue
        if knob == "1a" and "V3" not in roots:
            continue
        cells = failures[knob]
        if not checked[knob]:
            verdict = "undecided (no cell checked)"
        else:
            verdict = "RETIRE" if not cells and not missing else "KEEP"
        print(f"{knob} {label}: {verdict} ({len(cells)} failing of {checked[knob]} cells checked)")
        for line in cells[:40]:
            print(f"    {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
