import torch
import numpy as np
from typing import Tuple
from torch import FloatTensor
from numpy.typing import NDArray
from torchvision import datasets
from torch.utils.data import Dataset

from src.sem.abstract import StructuralEquationModel as SEM


def torch_xor(a: FloatTensor, b: FloatTensor) -> FloatTensor:
    # assumes both inputs are either 0 or 1
    return (a - b).abs()


def torch_bernoulli(p: float, size: int):
    # flip coin `size` times
    return (torch.rand(size) < p).float()


def colour_image(image_grey: FloatTensor, color: FloatTensor) -> FloatTensor:
    N = len(image_grey)
    zeros = torch.zeros_like(image_grey)
    image_rgb = torch.stack([image_grey, image_grey, zeros], dim=1)
    image_rgb[torch.tensor(range(N)), (1 - color).long(), :, :] *= 0
    return image_rgb


class ColoredDigitsSEM(SEM):
    @staticmethod
    def load_dataset(directory: str="data/mnist", train: bool=True) -> Dataset:
        mnist = datasets.MNIST(directory, train=train, download=True)
        return mnist
    
    _TRAIN: Dataset = load_dataset.__func__()
    _TEST: Dataset = load_dataset.__func__(train=False)

    def __init__(self, train: bool=True):
        self.train = train
        if train:
            self.images = self._TRAIN.data
            self.targets = self._TRAIN.targets
        else:
            self.images = self._TEST.data
            self.targets = self._TEST.targets
    
    def __len__(self) -> int:
        return len(self.images)
    
    def sample(self, N: int=1, **kwargs) -> Tuple[NDArray, NDArray]:
        N_max = len(self.images)
        indices = np.arange(N_max)
        if N == -1:
            N = N_max
        replace = N > N_max
        sampled = np.random.choice(
            indices, N, replace
        )
        images, targets = self.images[sampled], self.targets[sampled]
        
        # get MNIST image and ground truth label
        N_X = images.reshape((-1, 28, 28))[:, ::2, ::2] # MNIST image with 2x subsample for computational convenience
        fX = (targets < 5).float()                      # assign ground truth lables based on image
        
        # add noise to labelling function -- flip label with probability 0.25
        n_y = torch_bernoulli(0.25, N)
        y = torch_xor(fX, n_y)
        
        # assign a color based on the label; flip the color with probability e
        if self.train:
            e_space = torch.tensor([0.1, 0.2])
        else:
            e_space = torch.tensor([0.5])
        idx = torch.multinomial(e_space, num_samples=N, replacement=True)
        e = e_space[idx]
        C = torch_xor(y, torch_bernoulli(e, N))         # color C confounds X and y
        
        # apply the color to the image by zeroing out the other color channel
        X = colour_image(image_grey=N_X, color=C)
        return (
            (X.float() / 255.).numpy(), # treatment
            y[:, None].numpy(),         # outcome
        )

