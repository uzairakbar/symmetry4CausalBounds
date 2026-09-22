"""Fan a config's sweeps and perf blocks out over a Slurm job array, one task each.

Every dataset block of the yaml is split into one block per `experiment.sweep.param`
entry (the block with `experiment: {sweep: {param: [p], metric: ...}}`) and one per
`experiment.perf` block, `defaults:` and `hyperparameters:` copied verbatim. Each
task runs `src/main.py` unchanged in its own directory, whose `config.yaml` is the
task's yaml and whose `artifacts` and `data` link to the repo's, so every task writes
into the one shared tree: sweep files are `{param}_*` under `sweep/`, perf files
`epsilon_*` under `perf/`, so no two tasks collide. The query blocks (`--tasks
query`) run as one task per dataset; two simulation query recipes collide by design
(recipes/README.md), so never fan both out into one tree. do-MNIST blocks are never
emitted.

A fanned-out task reproduces a one-param-per-block run of `main.py` (the recipes'
shape), not the sequential multi-param run of a whole `config.yaml`: the global RNG
advances through each param's setup in turn there.

Nothing here assumes a site: every directive is a flag or its `SBATCH_SWEEPS_*`
environment variable, and a directive left unset is not emitted, so the site's
defaults apply. `--env-setup` is a shell line run before `uv run` (e.g. a module
load on a cluster that ships uv as a module). EXAMPLE, on Georgia Tech's PACE ICE
(one site; nothing below is a default):

    uv run python scripts/sbatch_sweeps.py --config recipes/nEfficiencyFig13.yaml \\
        --out ~/scratch/runs/im-ci --partition coc-cpu --account oms-csp --qos coc-ice \\
        --env-setup "module load uv" --cpus-per-task 96 --time 02:00:00

    uv run python scripts/sbatch_sweeps.py --config PATH --out DIR [--dry-run] [...]

`--dry-run` writes the task directories and `DIR/run.sbatch` and prints the `sbatch`
line without submitting.
"""

import argparse
import copy
import os
import shlex
import shutil
import subprocess
import sys

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GLOBAL_KEYS = ("defaults", "hyperparameters")
TASK_TYPES = ("sweep", "perf", "query")
NEVER = ("do_mnist",)  # far too slow to fan out; run it by hand if at all
DEFAULT_TIME = "04:00:00"
# directive -> environment variable; unset and not on the command line = not emitted
DIRECTIVES = {
    "partition": "SBATCH_SWEEPS_PARTITION",
    "account": "SBATCH_SWEEPS_ACCOUNT",
    "qos": "SBATCH_SWEEPS_QOS",
    "time": "SBATCH_SWEEPS_TIME",
    "cpus_per_task": "SBATCH_SWEEPS_CPUS_PER_TASK",
    "mem": "SBATCH_SWEEPS_MEM",
}
EPILOG = """\
EXAMPLE (Georgia Tech PACE ICE, one site; nothing here is a default):
  uv run python scripts/sbatch_sweeps.py --config recipes/nEfficiencyFig13.yaml \\
      --out ~/scratch/runs/im-ci --partition coc-cpu --account oms-csp --qos coc-ice \\
      --env-setup "module load uv" --cpus-per-task 96 --time 02:00:00
"""


def split(config: dict, tasks: tuple[str, ...]) -> list[tuple[str, dict]]:
    """(name, yaml) per task, in file order: one per sweep param and one per perf
    block of every dataset block, one per query block when asked."""
    shared = {key: config[key] for key in GLOBAL_KEYS if config.get(key) is not None}
    out = []
    for dataset, block in config.items():
        if dataset in GLOBAL_KEYS or not isinstance(block, dict):
            continue
        experiment = block.get("experiment") or {}
        if dataset in NEVER:
            if experiment:
                print(f"skipping the {dataset} block: never fanned out", file=sys.stderr)
            continue
        plans = []
        if "sweep" in tasks and experiment.get("sweep"):
            sweep = experiment["sweep"]
            for param in sweep.get("param") or []:
                plans.append((param, {"sweep": {**sweep, "param": [param]}}))
        if "perf" in tasks and experiment.get("perf"):
            plans.append(("perf", {"perf": copy.deepcopy(experiment["perf"])}))
        if "query" in tasks and experiment.get("query"):
            plans.append(("query", {"query": True}))
        for stem, plan in plans:
            task = {k: v for k, v in copy.deepcopy(block).items() if k != "experiment"}
            task["experiment"] = plan
            out.append((f"{dataset}_{stem}", {**copy.deepcopy(shared), dataset: task}))
    return out


def directives(args, count: int, out: str) -> list[str]:
    lines = [
        "#SBATCH --job-name=sweeps",
        f"#SBATCH --array=0-{count - 1}",
        f"#SBATCH --output={os.path.join(out, 'logs', '%A_%a.out')}",
    ]
    for key, env in DIRECTIVES.items():
        value = getattr(args, key) or os.environ.get(env) or (DEFAULT_TIME if key == "time" else None)
        if value:
            lines.append(f"#SBATCH --{key.replace('_', '-')}={value}")
    return lines


def script(args, names: list[str], out: str) -> str:
    env_setup = args.env_setup or os.environ.get("SBATCH_SWEEPS_ENV_SETUP")
    body = [
        "#!/bin/bash",
        *directives(args, len(names), out),
        "set -euo pipefail",
        "# task k runs in DIR/task_k: config.yaml is its yaml, artifacts/ and data/ the repo's",
        f"TASKS=({' '.join(shlex.quote(name) for name in names)})",
        'echo "task ${SLURM_ARRAY_TASK_ID}: ${TASKS[$SLURM_ARRAY_TASK_ID]}"',
        f"mkdir -p {shlex.quote(os.path.join(REPO, 'artifacts'))}",
        f'cd {shlex.quote(out)}/task_"${{SLURM_ARRAY_TASK_ID}}"',
    ]
    if env_setup:
        body.append(env_setup)
    body.append(f"uv run --project {shlex.quote(REPO)} python {shlex.quote(os.path.join(REPO, 'src', 'main.py'))}")
    return "\n".join(body) + "\n"


def write(tasks: list[tuple[str, dict]], out: str) -> None:
    """DIR/configs/<name>.yaml, and DIR/task_k with that yaml as config.yaml and the
    two links the cwd-relative repo needs (`main.py` reads ./config.yaml, the saves
    write ./artifacts, the optical and cigarette loaders read ./data)."""
    os.makedirs(os.path.join(out, "configs"), exist_ok=True)
    os.makedirs(os.path.join(out, "logs"), exist_ok=True)
    for index, (name, config) in enumerate(tasks):
        path = os.path.join(out, "configs", f"{name}.yaml")
        with open(path, "w") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        task_dir = os.path.join(out, f"task_{index}")
        os.makedirs(task_dir, exist_ok=True)
        shutil.copyfile(path, os.path.join(task_dir, "config.yaml"))
        for link in ("artifacts", "data"):
            target = os.path.join(task_dir, link)
            if os.path.islink(target):
                os.remove(target)
            os.symlink(os.path.join(REPO, link), target)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="One Slurm array task per (dataset, sweep param) and per perf block of a config.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default=os.path.join(REPO, "config.yaml"), help="a recipe or config.yaml")
    parser.add_argument("--out", required=True, help="where the task yamls, the sbatch script and the logs go")
    parser.add_argument("--partition", help="$SBATCH_SWEEPS_PARTITION; unset = not emitted")
    parser.add_argument("--account", help="$SBATCH_SWEEPS_ACCOUNT; unset = not emitted")
    parser.add_argument("--qos", help="$SBATCH_SWEEPS_QOS; unset = not emitted")
    parser.add_argument("--time", help=f"$SBATCH_SWEEPS_TIME; default {DEFAULT_TIME}")
    parser.add_argument(
        "--cpus-per-task", dest="cpus_per_task", help="$SBATCH_SWEEPS_CPUS_PER_TASK; unset = not emitted"
    )
    parser.add_argument("--mem", help="$SBATCH_SWEEPS_MEM; unset = not emitted")
    parser.add_argument("--env-setup", help="$SBATCH_SWEEPS_ENV_SETUP: a shell line run before `uv run`")
    parser.add_argument("--tasks", default="sweep,perf", help="block types to fan out, of sweep,perf,query")
    parser.add_argument("--dry-run", action="store_true", help="write everything, submit nothing")
    args = parser.parse_args()

    kinds = tuple(kind.strip() for kind in args.tasks.split(",") if kind.strip())
    unknown = sorted(set(kinds) - set(TASK_TYPES))
    if unknown:
        parser.error(f"--tasks: unknown {unknown}; expected some of {list(TASK_TYPES)}")
    with open(args.config) as handle:
        config = yaml.safe_load(handle) or {}
    tasks = split(config, kinds)
    if not tasks:
        print(f"{args.config}: nothing to fan out for --tasks {args.tasks}", file=sys.stderr)
        return 1

    out = os.path.abspath(os.path.expanduser(args.out))
    stale = [p for p in ("configs", "task_0") if os.path.exists(os.path.join(out, p))]
    if stale:
        print(
            f"warning: {out} already holds {stale}; task dirs beyond this run's count are left as they are",
            file=sys.stderr,
        )
    write(tasks, out)
    sbatch = os.path.join(out, "run.sbatch")
    with open(sbatch, "w") as handle:
        handle.write(script(args, [name for name, _ in tasks], out))
    for index, (name, _) in enumerate(tasks):
        print(f"task {index}: {name}")
    command = ["sbatch", sbatch]
    print(" ".join(shlex.quote(part) for part in command))
    if args.dry_run:
        return 0
    return subprocess.run(command, check=False).returncode  # noqa: S603 - the script this run wrote


if __name__ == "__main__":
    sys.exit(main())
