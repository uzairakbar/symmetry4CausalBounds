import os
import sys
import yaml
from loguru import logger
from munch import munchify

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiments.simulation import (
    query_sweep as linear_simulation,
    # param_sweep as linear_simulation,
)
from src.experiments.optical_device import (
    query_sweep as optical_device_experiment,
    param_sweep as optical_device_experiment,
)
# from src.experiments.cmnist import (
#     # query_sweep as colored_mnist_experiment,
#     # param_sweep as colored_mnist_experiment,
# )


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
