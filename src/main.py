import os
import sys
import yaml
from loguru import logger
from munch import munchify

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiments import simulation as simulation_experiment
from src.experiments import optical_device as optical_device_experiment

def main():
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    config = munchify(config)

    if 'simulation' in config:
        logger.info('Running linear simulation experiment.')
        simulation_experiment.run(
            **config.simulation,
            hyperparameters=config.hyperparameters
        )
    
    if 'optical_device' in config:
        logger.info('Running optical device experiment.')
        optical_device_experiment.run(
            **config.optical_device,
            hyperparameters=config.hyperparameters
        )

if __name__ == '__main__':
    main()