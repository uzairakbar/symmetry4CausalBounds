"""
src/main.py
"""
import os, sys, yaml
from loguru import logger
from munch import munchify

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiments.simulation import SimulationOrchestrator
from src.experiments.optical_device import OpticalOrchestrator

def main():
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    config = munchify(config)

    if 'simulation' in config:
        logger.info('Running linear simulation experiment.')
        orch = SimulationOrchestrator(
            **config.simulation,
            hyperparameters=config.hyperparameters
        )
        orch.run(sweep_mode=config.simulation.sweep_mode,
                 plot_panel=config.simulation.plot_panel)
    
    if 'optical_device' in config:
        logger.info('Running optical device experiment.')
        orch = OpticalOrchestrator(
            **config.optical_device,
            hyperparameters=config.hyperparameters
        )
        orch.run(sweep_mode=config.optical_device.sweep_mode,
                 plot_panel=config.optical_device.plot_panel)

if __name__ == '__main__':
    main()