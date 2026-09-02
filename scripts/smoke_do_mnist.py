"""End-to-end do-MNIST smoke run at reduced scale: query sweep + perf.

python scripts/smoke_do_mnist.py [--full]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from munch import munchify  # noqa: E402

from src.experiments.configs import (
    parse_experiment_plan,  # noqa: E402
    resolve_dataset_block,
)
from src.experiments.do_mnist import DoMNISTOrchestrator  # noqa: E402
from src.experiments.utils import set_seed  # noqa: E402

SMALL = dict(n_samples=60_000, n_pi=6_000, n_queries=32)
FULL = dict(n_samples=1_200_000, n_pi=60_000, n_queries=512)

# Self-contained, so the gate does not depend on which dataset block happens to be
# uncommented in config.yaml. A `do_mnist:` block, if present, overrides these.
BLOCK = dict(
    seed=42,
    n_experiments=1,
    sweep_samples=10,
    net="domnist-fast",
    unfrozen_layers=1,
    gamma=0.067,
    epsilon=0.1,
    calibrate=True,
    n_jobs=-1,
    augmentation="translate > rotation > contrast > saturation > hue",
    methods=["ATE", "ERM", "DA+ERM", "PI", "DA+PI", "PI+INV", "DA+PI+IV", "PI&DA+PI", "PI&DA+PI+IV"],
    experiment={"query": True, "perf": {"metric": ["wall_clock"]}},
)
HYPERPARAMETERS = dict(lr=0.01, batch=256, epochs=1, optimizer="adam", betas=(0.7, 0.9), onecycle=True, loss="mse")


def main(full: bool):
    import yaml

    with open("config.yaml") as fh:
        config = yaml.safe_load(fh) or {}

    defaults = config.pop("defaults", {}) or {}
    hyperparameters = config.pop("hyperparameters", {}) or HYPERPARAMETERS
    block = {**defaults, **BLOCK, **(config.get("do_mnist") or {}), **(FULL if full else SMALL)}

    plan = parse_experiment_plan(block.get("experiment"))
    block = resolve_dataset_block("do_mnist", block)

    set_seed(block["seed"])
    DoMNISTOrchestrator(**block, hyperparameters=munchify(hyperparameters)).run(plan)
    print("\nSMOKE OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    main(parser.parse_args().full)
