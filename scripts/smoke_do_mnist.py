"""End-to-end do-MNIST smoke run at reduced scale: the query sweep on the copsens
block (perf logs a warning and skips). Run from a SCRATCH cwd: `save` writes
`./artifacts`, and the repo root must stay free of run trees.

    cd ~/scratch/domnist_runs/smoke && PYTHONPATH=$REPO uv run --project $REPO \\
        python $REPO/scripts/smoke_do_mnist.py [--full] [--methods PI DA+PI ...]

A `config.yaml` in the cwd overrides BLOCK (its `do_mnist:` block) and the
hyperparameters; without one the defaults below stand on their own.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from munch import munchify  # noqa: E402

from src.experiments.configs import parse_experiment_plan, resolve_dataset_block  # noqa: E402
from src.experiments.do_mnist import DoMNISTOrchestrator  # noqa: E402
from src.experiments.utils import set_seed  # noqa: E402

SMALL = dict(n_samples=60_000, n_pi=6_000, n_queries=200)
FULL = dict(n_samples=1_200_000, n_pi=60_000, n_queries=2_000)

# Self-contained, so the smoke does not depend on which dataset block happens to be
# uncommented in config.yaml. The values are the shipped block's.
BLOCK = dict(
    seed=42,
    n_experiments=1,
    sweep_samples=10,
    net="domnist-pool",
    n_components=32,
    calibrate_sigma=True,
    gamma=0.062082436071912536,
    target_coverage=0.995,
    epsilon=0.04,
    erm_inv_tau=4e-4,
    gamma_z_star=0.0,
    inv_recenter="inv",
    mix_in=0.05,
    split_seed=420,
    split={"A": 40_000, "B": 10_000, "C": 10_000},
    pop_seed=44,
    exemplar_seed=420,
    augmentation="translate > rotation > contrast > saturation > hue",
    augmentation_amounts=None,
    methods=["ATE", "ERM", "DA+ERM", "PI", "DA+PI", "DA+PI+IV", "PI+INV"],
    recalibrate=False,
    pad=False,
    clipy=False,
    mean_match=True,
    n_jobs=-1,
    experiment={"query": True, "perf": {"metric": ["wall_clock"]}},
)
HYPERPARAMETERS = dict(lr=0.01, batch=256, epochs=1, optimizer="adam", betas=(0.7, 0.9), onecycle=True, loss="mse")


def main(full: bool, methods: list[str] | None):
    config = {}
    if os.path.exists("config.yaml"):
        import yaml

        with open("config.yaml") as fh:
            config = yaml.safe_load(fh) or {}

    defaults = config.pop("defaults", {}) or {}
    hyperparameters = config.pop("hyperparameters", None) or HYPERPARAMETERS
    block = {**defaults, **BLOCK, **(config.get("do_mnist") or {}), **(FULL if full else SMALL), "im-ci": 0}
    if methods:
        block["methods"] = methods

    plan = parse_experiment_plan(block.get("experiment"))
    block = resolve_dataset_block("do_mnist", block)

    set_seed(block["seed"])
    DoMNISTOrchestrator(**block, hyperparameters=munchify(hyperparameters)).run(plan)
    print("\nSMOKE OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="the shipped block's sizes (GPU node, minutes)")
    parser.add_argument("--methods", nargs="+", default=None, help="restrict the method list, e.g. PI DA+PI")
    args = parser.parse_args()
    main(args.full, args.methods)
