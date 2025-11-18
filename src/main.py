import os
import sys
import yaml
from loguru import logger
from munch import munchify

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiments.simulation import (
    linear as linear_simulation,
)
from src.experiments.real import (
    optical_device as optical_device_experiment,
    # cmnist as colored_mnist_experiment,
)


def main():

    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    config = munchify(config)

    if 'linear_simulation' in config:
        logger.info('Running linear simulation experiment.')
        linear_simulation.run(
            **config.linear_simulation,
            hyperparameters=config.hyperparameters
        )
    
    if 'optical_device' in config:
        logger.info('Running optical device experiment.')
        optical_device_experiment.run(
            **config.optical_device,
            hyperparameters=config.hyperparameters
        )
    
    # if 'colored_mnist' in config:
    #     logger.info('Running colored MNIST experiment.')
    #     colored_mnist_experiment.run(
    #         **config.colored_mnist,
    #         hyperparameters=config.hyperparameters
    #     )


if __name__ == '__main__':
    main()
