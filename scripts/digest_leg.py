"""Leg (D): the shipped configuration does not move, in numbers.

One query panel and one gamma sweep step per dataset (simulation, optical_device,
cigarettes) at the SHIPPED configuration of 14509db, no `iv` key anywhere,
1 experiment, a reduced `sweep_samples` and a 2-point gamma grid. Every numeric
artifact the run writes, the pkl and tex files under artifacts/<dataset>/{query,sweep},
is hashed (`wall_clock` excluded) and compared against a reference JSON recorded on
the parent commit with this same file. No tolerance: any difference is a FAIL, and
so is an artifact that appears or disappears.

The blocks are built HERE rather than read from config.yaml: the shipped yaml has
the sim and optical blocks commented out and the live copy carries the user's own
edits, so the leg pins the configuration the plan calls shipped, not whatever the
file says today. The run is the production path (`Orchestrator(**block).run(plan)`,
exactly as src/main.py drives it), figures included; the figures are not hashed
because a PDF carries a timestamp.

    python scripts/digest_leg.py --record [--repo DIR] [--out JSON]
    python scripts/digest_leg.py --compare [--repo DIR] [--reference JSON]

Gates import `compare()` and turn every artifact into one check. Record and compare
on the SAME node: BLAS kernels differ across CPU models at the last ulp, so a lone
leg (D) failure on another node means nothing until the reference is re-recorded
there (the JSON carries the node name and the gate warns on a mismatch).
"""

import argparse
import hashlib
import json
import os
import pickle
import platform
import shutil
import struct
import subprocess
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_REFERENCE = os.path.expanduser("~/scratch/tmp/impl_v10/digests/reference_14509db.json")

DATASETS = ("simulation", "optical_device", "cigarettes")
QUERY_SAMPLES = 8  # even, so the cigarette query grids never touch the origin
SWEEP_SAMPLES = 2  # the gamma grid: geomspace(2^-6, 1, 2)
N_JOBS = 8  # bit-identical to serial (a5); keeps the shared node under 16
HASHED_SUFFIXES = (".pkl", ".tex")
SKIP_KEYS = ("wall_clock",)  # a timer, never reproducible

# config.yaml at 14509db, resolved: the `defaults` toggles, the `hyperparameters`
# block (inert on the linear experiments, passed for fidelity) and the three
# dataset blocks, the sim and optical ones as they read uncommented
TOGGLES = dict(recalibrate=True, pad=True, clipy=False, mean_match=True, n_jobs=N_JOBS)
HYPERPARAMETERS = dict(lr=0.01, batch=256, epochs=1, optimizer="adam", betas=[0.7, 0.9], onecycle=True, loss="mse")
METHODS = ["PI+INV", "PI", "DA+PI", "DA+PI+IV", "PI&DA+PI", "PI&DA+PI+IV"]
BLOCKS = {
    "simulation": dict(
        seed=42, n_samples=2048, kernel_dim=0, treatment_dim=32, methods=METHODS, augmentation="translate"
    ),
    "optical_device": dict(
        seed=42, n_samples=1000, methods=METHODS, augmentation="rotation > hflip > vflip > random-permutation"
    ),
    "cigarettes": dict(
        seed=42,
        target="iv",
        spec="t3",
        anchor="own-tax",
        n_samples=2450,
        da_amplitude=1.0,
        sliver=False,
        methods=METHODS,
        augmentation="translate",
    ),
}
SWEEP_PLAN = {"sweep": {"param": ["gamma"], "metric": ["coverage", "approx_error", "worst_error", "width"]}}
QUERY_PLAN = {"query": True}


# ------------------------------------------------------------------ hashing


def _canonical(obj, out):
    """Append a canonical byte encoding of `obj` to `out`: dtype, shape and raw
    bytes for arrays, sorted keys for dicts, so two pickles of the same numbers
    hash the same and two different numbers never do."""
    if isinstance(obj, np.ndarray):
        array = np.ascontiguousarray(obj)
        out.append(b"A" + array.dtype.str.encode() + repr(array.shape).encode() + array.tobytes())
    elif isinstance(obj, np.generic):
        _canonical(np.asarray(obj), out)
    elif isinstance(obj, bool):
        out.append(b"B" + bytes([obj]))
    elif isinstance(obj, int):
        out.append(b"I" + repr(obj).encode())
    elif isinstance(obj, float):
        out.append(b"F" + struct.pack("<d", obj))
    elif isinstance(obj, str):
        out.append(b"S" + obj.encode())
    elif obj is None:
        out.append(b"N")
    elif isinstance(obj, dict):
        out.append(b"D")
        for key in sorted(obj, key=repr):
            if key in SKIP_KEYS:
                continue
            _canonical(key, out)
            _canonical(obj[key], out)
    elif isinstance(obj, list | tuple):
        out.append(b"L" + repr(len(obj)).encode())
        for item in obj:
            _canonical(item, out)
    else:
        raise TypeError(f"leg (D) cannot hash a {type(obj).__name__}")


def digest(obj) -> str:
    parts = []
    _canonical(obj, parts)
    return hashlib.sha256(b"".join(parts)).hexdigest()[:16]


def digest_file(path: str) -> str:
    if path.endswith(".pkl"):
        with open(path, "rb") as handle:
            return digest(pickle.load(handle))  # noqa: S301 - our own artifacts
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()[:16]


# ------------------------------------------------------------------ the runs


def _enter(repo: str) -> str:
    """Import `src` from `repo` and run in it: data and artifacts are relative paths."""
    repo = os.path.abspath(repo)
    sys.path.insert(0, repo)
    os.chdir(repo)
    os.environ.setdefault("MPLBACKEND", "Agg")
    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    return repo


def run_dataset(name: str, kind: str) -> dict[str, str]:
    """One production run (`kind` is 'query' or 'sweep') at the shipped block;
    returns {artifact: digest} for what it wrote."""
    from munch import munchify

    from src.experiments.configs import parse_experiment_plan, resolve_dataset_block
    from src.experiments.utils import set_seed
    from src.experiments.utils.constants import ARTIFACTS_DIRECTORY, SUBDIR_QUERY, SUBDIR_SWEEP
    from src.main import ORCHESTRATORS

    samples = QUERY_SAMPLES if kind == "query" else SWEEP_SAMPLES
    block = {**TOGGLES, **BLOCKS[name], "n_experiments": 1, "sweep_samples": samples}
    plan = parse_experiment_plan(QUERY_PLAN if kind == "query" else SWEEP_PLAN)
    block = resolve_dataset_block(name, block)

    folder = os.path.join(ARTIFACTS_DIRECTORY, name, SUBDIR_QUERY if kind == "query" else SUBDIR_SWEEP)
    shutil.rmtree(folder, ignore_errors=True)
    set_seed(block["seed"])
    ORCHESTRATORS[name](**block, hyperparameters=munchify(HYPERPARAMETERS)).run(plan)

    files = sorted(f for f in os.listdir(folder) if f.endswith(HASHED_SUFFIXES))
    return {f"{kind}/{f}": digest_file(os.path.join(folder, f)) for f in files}


def run_all(datasets=DATASETS, quiet: bool = False) -> dict[str, dict[str, str]]:
    digests = {}
    for name in datasets:
        start = time.perf_counter()
        digests[name] = {**run_dataset(name, "query"), **run_dataset(name, "sweep")}
        if not quiet:
            print(f"      leg (D) {name}: {len(digests[name])} artifacts in {time.perf_counter() - start:.0f}s")
    return digests


def _commit(repo: str) -> str:
    try:
        command = ["git", "-C", repo, "rev-parse", "--short", "HEAD"]
        return subprocess.run(command, capture_output=True, text=True).stdout.strip()
    except OSError:
        return "unknown"


def record(repo: str, out: str) -> dict:
    repo = _enter(repo)
    reference = {
        "meta": {
            "repo": repo,
            "commit": _commit(repo),
            "node": platform.node(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "recorded": time.strftime("%Y-%m-%d %H:%M:%S"),
            "query_samples": QUERY_SAMPLES,
            "sweep_samples": SWEEP_SAMPLES,
            "n_jobs": N_JOBS,
        },
        "digests": run_all(),
    }
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as handle:
        json.dump(reference, handle, indent=2, sort_keys=True)
    return reference


def compare(repo: str = REPO, reference: str = DEFAULT_REFERENCE, datasets=DATASETS) -> list[tuple[str, bool, str]]:
    """Rerun leg (D) in `repo` and compare with the recorded JSON. Returns one
    (label, ok, detail) per artifact on either side, for the gate to print."""
    if not os.path.isfile(reference):
        return [("leg (D) reference", False, f"{reference} missing; record it on the parent with --record")]
    with open(reference) as handle:
        recorded = json.load(handle)
    meta = recorded["meta"]
    if meta["node"] != platform.node():
        print(f"      leg (D) WARNING: reference recorded on {meta['node']}, this is {platform.node()}")
    repo = _enter(repo)
    got = run_all(datasets)
    rows = []
    for name in datasets:
        want = recorded["digests"][name]
        for artifact in sorted(set(want) | set(got[name])):
            expected, actual = want.get(artifact), got[name].get(artifact)
            ok = expected is not None and expected == actual
            rows.append((f"{name}/{artifact}", ok, f"{expected} vs {actual}" if not ok else actual))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=REPO, help="repo to import `src` from and run in")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--record", action="store_true")
    mode.add_argument("--compare", action="store_true")
    parser.add_argument("--out", default=DEFAULT_REFERENCE, help="--record: where the JSON goes")
    parser.add_argument("--reference", default=DEFAULT_REFERENCE, help="--compare: the JSON to compare against")
    args = parser.parse_args()

    if args.record:
        reference = record(args.repo, args.out)
        print(json.dumps(reference, indent=2, sort_keys=True))
        print(f"recorded {args.out}")
    else:
        rows = compare(args.repo, args.reference)
        for label, ok, detail in rows:
            print(f"  [{'PASS' if ok else 'FAIL'}] {label} {detail}")
        bad = [label for label, ok, _ in rows if not ok]
        print("leg (D) PASS" if not bad else f"leg (D) FAIL: {bad}")
        sys.exit(1 if bad else 0)
