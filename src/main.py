"""
Main entry point for experiments.
"""

import os
import sys
import yaml
from loguru import logger
from munch import munchify

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiments.utils import set_seed
from src.experiments.configs import parse_experiment_plan, resolve_dataset_block
from src.experiments.simulation import SimulationOrchestrator
from src.experiments.optical_device import OpticalOrchestrator
from src.experiments.do_mnist import DoMNISTOrchestrator


ORCHESTRATORS = {
    "simulation": SimulationOrchestrator,
    "optical_device": OpticalOrchestrator,
    "do_mnist": DoMNISTOrchestrator,
}


def main():
    """Run experiments based on config.yaml."""
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # global toggles, overridable per experiment
    defaults = config.pop("defaults", {}) or {}
    hyperparameters = config.pop("hyperparameters", {}) or {}

    unknown = set(config) - set(ORCHESTRATORS)
    if unknown:
        raise ValueError(f"Unknown dataset block(s) {sorted(unknown)} in config.yaml.")

    for name, Orchestrator in ORCHESTRATORS.items():
        if name not in config:
            continue

        logger.info(f"Running {name} experiment.")
        block = {**defaults, **config[name]}
        plan = parse_experiment_plan(block.get("experiment"))
        block = resolve_dataset_block(name, block)

        set_seed(block["seed"])
        Orchestrator(**block, hyperparameters=munchify(hyperparameters)).run(plan)


if __name__ == "__main__":
    main()
