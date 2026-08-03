"""End-to-end do-MNIST smoke run at reduced scale: query sweep + perf.

    python scripts/smoke_do_mnist.py [--full]
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from munch import munchify                                      # noqa: E402
from src.experiments.utils import set_seed                      # noqa: E402
from src.experiments.configs import (parse_experiment_plan,     # noqa: E402
                                     resolve_dataset_block)
from src.experiments.do_mnist import DoMNISTOrchestrator        # noqa: E402

SMALL = dict(n_samples=60_000, n_pi=6_000, n_queries=32)
FULL = dict(n_samples=1_200_000, n_pi=60_000, n_queries=512)


def main(full: bool):
    import yaml
    with open('config.yaml') as fh:
        config = yaml.safe_load(fh)

    defaults = config.pop('defaults', {}) or {}
    hyperparameters = config.pop('hyperparameters', {}) or {}
    block = {**defaults, **config['do_mnist'], **(FULL if full else SMALL)}

    plan = parse_experiment_plan(block.get('experiment'))
    block = resolve_dataset_block('do_mnist', block)

    set_seed(block['seed'])
    DoMNISTOrchestrator(**block, hyperparameters=munchify(hyperparameters)).run(plan)
    print('\nSMOKE OK')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true')
    main(parser.parse_args().full)
