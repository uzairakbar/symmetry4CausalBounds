import torch
import numpy as np
from torch import nn
from loguru import logger
from numpy.typing import NDArray
from typing import Optional, Literal, Callable, Dict
from sklearn.model_selection import BaseCrossValidator


Model = Literal['linear', '2-layer', 'cmnist', 'rmnist']


CPU_ONLY: bool = True
MODELS: Dict[Model, Callable[[int], nn.Sequential]] = {
    'linear': lambda input_dim: nn.Sequential(
        nn.Linear(input_dim, 1, bias=False)
    ),
    '2-layer': lambda input_dim: nn.Sequential(
        nn.Linear(input_dim, 20),
        nn.LeakyReLU(0.2),
        nn.Linear(20, 1)
    ),
    'cmnist': lambda input_dim: nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.ReLU(True),
        nn.Linear(256, 256),
        nn.ReLU(True),
        nn.Linear(256, 1),
        nn.Sigmoid()
    ),
    'rmnist': lambda input_dim: nn.Sequential(
        nn.Unflatten(1, torch.Size([1, 28, 28])),
        nn.Conv2d(1, 32, kernel_size=5, stride=1, bias=False),
        nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
        nn.Conv2d(32, 64, kernel_size=5, stride=1, bias=False),
        nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
        nn.Flatten(1, -1),
        nn.Linear(64 * 4 * 4, 128),
        nn.Dropout(0.5),
        nn.ReLU(),
        nn.Linear(128, 10),
        nn.LogSoftmax(dim=1)
    ),
}


def device():
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    if CPU_ONLY:
        device = 'cpu'
    logger.info(f'Using {device} device.')
    return torch.device(device)