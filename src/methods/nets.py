"""Torch backbones for the image experiments."""
import torch
from torch import nn
from loguru import logger
from typing import Callable, Dict

CPU_ONLY: bool = False
_DEVICE = None


def _domnist_fast(input_dim: int, channels: int = 3) -> nn.Sequential:
    """Fast MNIST CNN (99% in ~1 epoch): pool right after conv1, no dropout, k=5, fc 256.

    Sigmoid head, so mu_y lands in (0,1) -- which is what makes clipping bounds to
    [0,1] safe. Pair with Adam(lr=1e-2, betas=(0.7, 0.9)) + OneCycleLR.
    """
    side = int(round((input_dim / channels) ** 0.5))
    flat = 32 * ((((side - 4) // 2) - 2) ** 2)
    return nn.Sequential(
        nn.Unflatten(1, torch.Size([channels, side, side])),
        nn.Conv2d(channels, 24, 5, 1), nn.MaxPool2d(2), nn.ReLU(),
        nn.Conv2d(24, 32, 3, 1), nn.ReLU(),
        nn.Flatten(1),
        nn.Linear(flat, 256), nn.ReLU(),
        nn.Linear(256, 1), nn.Sigmoid(),
    )


NETS: Dict[str, Callable[[int], nn.Sequential]] = {
    'domnist-fast': _domnist_fast,
}


def device():
    """Cached: the DA calls this once per chunk and the log line is not news."""
    global _DEVICE
    if _DEVICE is None:
        name = 'cpu'
        if not CPU_ONLY:
            if torch.cuda.is_available():
                name = 'cuda'
            elif torch.backends.mps.is_available():
                name = 'mps'
        logger.info(f'Using {name} device.')
        _DEVICE = torch.device(name)
    return _DEVICE
