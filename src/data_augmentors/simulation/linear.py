import numpy as np
from typing import Tuple
from loguru import logger
from numpy.typing import NDArray
from typing import Dict, Literal, List, Optional

from src.data_augmentors.abstract import DataAugmenter as DA


DA_STD: float=1.0
BASIS_SELECTOIN_PROBABILITY: float=2.0/3.0


class NullSpaceTranslation(DA):
    def __init__(
            self,
            W_XY: NDArray,
            kernel_dim: int,
            std: float=DA_STD,
        ):
        null_basis = self.null_space(W_XY.T).T

        k_max, _ = null_basis.shape

        assert k_max >= kernel_dim, \
            f'`kernel_dim`={kernel_dim} cannot be greater than `k_max`={k_max}.'

        if kernel_dim < 0:
            logger.info(
                '`kernel_dim`<0 means DA is constructed from full bases of ker(f).'
            )
            sample = np.ones(k_max, dtype='bool')
        elif kernel_dim == 0:
            logger.info(
                '`kernel_dim`=0 means DA is constructed from randomly picked bases of ker(f).'
            )
            sample = (
                np.random.random(k_max) < BASIS_SELECTOIN_PROBABILITY
            )
        else:
            logger.info(
                f'Selecting `kernel_dim`={kernel_dim} bases of ker(f) to construct DA.'
            )
            sample = np.zeros(k_max, dtype='bool')
            sample[:kernel_dim] = True
        
        # randomize bases ordering
        np.random.shuffle(sample)
        
        self.std = std
        self.W_ZXtilde = null_basis[sample]
        self.param_dimension, _ = self.W_ZXtilde.shape
    
    @property
    def augmentation(self):
        return 'translate'
    
    def augment(
            self, X: NDArray, gamma: float=16.0
        ) -> Tuple[NDArray, NDArray]:
        N = len(X)
        G = np.random.randn(N, self.param_dimension) * self.std

        GX = X + gamma * G @ self.W_ZXtilde
        
        return GX, G
    
    @staticmethod
    def null_space(
            W: NDArray,
            absolute_tolerance: float=1e-13,
            relative_tolerance: float=0.0
        ) -> NDArray:
        U, s, VT = np.linalg.svd(W)
        
        max_singular = s[0]
        tolerance = max(absolute_tolerance,
                        relative_tolerance * max_singular)
        
        num_singular = (s >= tolerance).sum()
        null_space_basis = VT[num_singular:].T
        
        return null_space_basis


class Identity(DA):
    @property
    def augmentation(self):
        return 'identity'
    
    def augment(self, X):
        GX, G = X, X
        return GX, G


Augmentation = Literal['translate']


class LinearSimulationDA(DA):
    def __init__(
            self,
            W_XY: NDArray,
            augmentations: Optional[str]='all'
        ):
        all_augmentations: Dict[Augmentation, DA] = {
            augmenter.augmentation: augmenter for augmenter in ([
                NullSpaceTranslation(W_XY=W_XY),
            ])
        }
        
        if augmentations == 'all':
            augmentations: List[Augmentation] = list(all_augmentations.keys())
        elif augmentations:
            augmentations: List[Augmentation] = augmentations.replace(' ','').split('>')

        if augmentations:        
            self._augmentations: List[DA] = ([
                all_augmentations[augmentation] for augmentation in augmentations
            ])
        else:
            self._augmentations: List[DA] = [Identity()]
        
    @property
    def augmentation(self):
        return 'linear_simulation'
    
    def augment(
            self,
            X: NDArray
        ) -> Tuple[NDArray, NDArray]:

        GX: NDArray = X.copy()
        G_list: List[NDArray] = []
        for i, augmentation in enumerate(self._augmentations):
            GX, G = augmentation(GX)
            G_list.append(G)
        G: NDArray = np.hstack(G_list)
        
        return GX, G