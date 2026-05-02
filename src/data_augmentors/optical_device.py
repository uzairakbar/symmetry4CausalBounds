import numpy as np
from abc import abstractmethod
from numpy.typing import NDArray
from typing import Dict, Literal, List, Optional, Tuple

from src.data_augmentors.abstract import DataAugmenter as DA
from src.data_augmentors.utils import BernoulliStandardScaler


P = 0.5


class Permutation(DA):
    """Base class for permutation-based augmentations."""
    
    def __init__(self, p=P):
        self.p = p
        self.param_scaler = BernoulliStandardScaler(p=p)
        super().__init__()

    def __call__(self, X):
        PERMUTE = [1.0, 0.0]
        
        N = len(X)
        G = np.random.choice(PERMUTE, size=N, p=[self.p, (1.0 - self.p)])
        
        GX = np.zeros_like(X)
        for i in range(len(X)):
            x = X[i, :]
            g = G[i]
            GX[i, :] = self.augment(x, g)
        
        return GX, self.param_scaler(G).reshape(-1, 1)
    
    @property
    @abstractmethod
    def augmentation(self):
        pass
    
    @abstractmethod
    def augment(self, x, g):
        pass

    @staticmethod
    def permute(x, g, permutation):
        """Apply permutation if g == 1.0."""
        if g == 1.0:
            return x[permutation]
        return x


class RandomPermutation(Permutation):
    """Random permutation augmentation."""
    
    def __call__(self, X):
        N, M = X.shape
        permutation_vector = np.arange(M, dtype=int)

        GX = np.zeros_like(X)
        G = np.zeros_like(X, dtype=int)
        
        for i in range(N):
            np.random.shuffle(permutation_vector)
            G[i, :] = permutation_vector

            x, g = X[i, :], G[i, :]
            GX[i, :] = self.augment(x, g)
        
        # Standardize
        G = (G - G.mean(axis=1)[:, np.newaxis]) / G.std(axis=1)[:, np.newaxis]
        return GX, G
    
    @property
    def augmentation(self):
        return 'random-permutation'
    
    def augment(self, x, g):
        return self.permute(x, 1.0, g)


class RandomRotation(Permutation):
    """90-degree rotation augmentation."""
    
    @property
    def augmentation(self):
        return 'rotation'
    
    def augment(self, x, g):
        ROTATION90 = np.array([6, 3, 0, 7, 4, 1, 8, 5, 2])
        return self.permute(x, g, ROTATION90)


class RandomHorizontalFlip(Permutation):
    """Horizontal flip augmentation."""
    
    @property
    def augmentation(self):
        return 'hflip'
    
    def augment(self, x, g):
        HORIZONTAL_FLIP = np.array([2, 1, 0, 5, 4, 3, 8, 7, 6])
        return self.permute(x, g, HORIZONTAL_FLIP)


class RandomVerticalFlip(Permutation):
    """Vertical flip augmentation."""
    
    @property
    def augmentation(self):
        return 'vflip'
    
    def augment(self, x, g):
        VERTICAL_FLIP = np.array([6, 7, 8, 3, 4, 5, 0, 1, 2])
        return self.permute(x, g, VERTICAL_FLIP)


class GaussianNoise(DA):
    """Gaussian noise augmentation."""
    
    def __call__(self, X):
        N, M = X.shape
        G = np.random.randn(N, M)
        GX = self.augment(X, G)
        return GX, G
    
    @property
    def augmentation(self):
        return 'gaussian-noise'
    
    def augment(self, X, G):
        return X + np.sqrt(0.01) * np.std(X) * G


class Identity(DA):
    """Identity augmentation (no change)."""
    
    @property
    def augmentation(self):
        return 'identity'
    
    def augment(self, X):
        return X, X


Augmentation = Literal['rotation', 'hflip', 'vflip', 'gaussian-noise', 'random-permutation']

ALL_AUGMENTATIONS: Dict[Augmentation, DA] = {
    augmenter.augmentation: augmenter for augmenter in [
        RandomRotation(),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        GaussianNoise(),
        RandomPermutation()
    ]
}


class OpticalDeviceDA(DA):
    """Data augmenter for optical device experiments."""
    
    def __init__(self, augmentations: Optional[str] = 'all'):
        if augmentations == 'all':
            augmentations: List[Augmentation] = list(ALL_AUGMENTATIONS.keys())
        elif augmentations:
            augmentations: List[Augmentation] = augmentations.replace(' ', '').split('>')

        if augmentations:
            self._augmentations: List[DA] = [
                ALL_AUGMENTATIONS[augmentation] for augmentation in augmentations
            ]
        else:
            self._augmentations: List[DA] = [Identity()]
    
    @property
    def augmentation(self):
        return 'optical_device'
    
    def augment(self, X: NDArray) -> Tuple[NDArray, NDArray]:
        """Apply augmentations sequentially."""
        GX: NDArray = X.copy()
        G_list: List[NDArray] = []
        
        for augmentation in self._augmentations:
            GX, G = augmentation(GX)
            G_list.append(G)
        
        G: NDArray = np.hstack(G_list)
        
        return GX, G





























# import numpy as np
# from abc import abstractmethod
# from numpy.typing import NDArray
# from typing import Dict, Literal, List, Optional, Tuple

# from src.data_augmentors.abstract import DataAugmenter as DA
# from src.data_augmentors.utils import BernoulliStandardScaler


# P = 0.5


# class Permutation(DA):
#     """Base class for permutation-based augmentations (Rotations, Flips)."""
    
#     def __init__(self, p=P):
#         self.p = p
#         self.param_scaler = BernoulliStandardScaler(p=p)
#         super().__init__()

#     def __call__(self, X, p: Optional[float] = None, **kwargs):
#         """
#         Apply binary augmentation (Flip/Rotate) with probability p.
#         Returns G as a binary vector (1.0 or 0.0).
#         """
#         PERMUTE = [1.0, 0.0]
        
#         # Use dynamic p if provided, else default
#         current_p = p if p is not None else self.p
        
#         N = len(X)
#         # Sample binary G: 1 (apply) or 0 (identity)
#         G = np.random.choice(PERMUTE, size=N, p=[current_p, (1.0 - current_p)])
        
#         GX = np.zeros_like(X)
#         for i in range(len(X)):
#             x = X[i, :]
#             g = G[i]
#             GX[i, :] = self.augment(x, g)
        
#         return GX, self.param_scaler(G).reshape(-1, 1)
    
#     @property
#     @abstractmethod
#     def augmentation(self):
#         pass
    
#     @abstractmethod
#     def augment(self, x, g):
#         pass

#     @staticmethod
#     def permute(x, g, permutation):
#         """
#         Apply permutation if g == 1.0. 
#         Helper used by both binary augmentations (g is scalar switch) 
#         and RandomPermutation (g is vector indices).
#         """
#         if g == 1.0:
#             return x[permutation]
#         return x


# class RandomPermutation(Permutation):
#     """
#     Random permutation augmentation.
#     Overrides __call__ because G is a vector of indices, not a binary scalar.
#     """
    
#     def __call__(self, X, p: Optional[float] = None, **kwargs):
#         # Handle dynamic p
#         current_p = p if p is not None else self.p
        
#         N, M = X.shape
#         base_indices = np.arange(M, dtype=int)

#         GX = np.zeros_like(X)
#         G = np.zeros_like(X, dtype=int)
        
#         # Determine which samples to permute based on p
#         do_permute = np.random.choice([True, False], size=N, p=[current_p, 1 - current_p])
        
#         for i in range(N):
#             if do_permute[i]:
#                 perm = base_indices.copy()
#                 np.random.shuffle(perm)
#                 G[i, :] = perm
#             else:
#                 # Identity permutation
#                 G[i, :] = base_indices

#             x, g = X[i, :], G[i, :]
#             # augment expects g to be the permutation vector
#             GX[i, :] = self.augment(x, g)
        
#         # Standardize G
#         means = G.mean(axis=1)[:, np.newaxis]
#         stds = G.std(axis=1)[:, np.newaxis]
#         stds[stds == 0] = 1.0 # Safety for constant rows
        
#         G_out = (G - means) / stds
#         return GX, G_out
    
#     @property
#     def augmentation(self):
#         return 'random-permutation'
    
#     def augment(self, x, g):
#         # Here g is the permutation vector. 
#         # We pass 1.0 as the 'switch' to force permute() to use 'g' as indices.
#         return self.permute(x, 1.0, g)


# class RandomRotation(Permutation):
#     """90-degree rotation augmentation."""
    
#     @property
#     def augmentation(self):
#         return 'rotation'
    
#     def augment(self, x, g):
#         # g is binary switch. permutation is fixed.
#         ROTATION90 = np.array([6, 3, 0, 7, 4, 1, 8, 5, 2])
#         return self.permute(x, g, ROTATION90)


# class RandomHorizontalFlip(Permutation):
#     """Horizontal flip augmentation."""
    
#     @property
#     def augmentation(self):
#         return 'hflip'
    
#     def augment(self, x, g):
#         HORIZONTAL_FLIP = np.array([2, 1, 0, 5, 4, 3, 8, 7, 6])
#         return self.permute(x, g, HORIZONTAL_FLIP)


# class RandomVerticalFlip(Permutation):
#     """Vertical flip augmentation."""
    
#     @property
#     def augmentation(self):
#         return 'vflip'
    
#     def augment(self, x, g):
#         VERTICAL_FLIP = np.array([6, 7, 8, 3, 4, 5, 0, 1, 2])
#         return self.permute(x, g, VERTICAL_FLIP)


# class GaussianNoise(DA):
#     """Gaussian noise augmentation."""
    
#     def __call__(self, X, noise_coeff: Optional[float] = None, **kwargs):
#         N, M = X.shape
#         G = np.random.randn(N, M)
#         GX = self.augment(X, G, noise_coeff=noise_coeff)
#         return GX, G
    
#     @property
#     def augmentation(self):
#         return 'gaussian-noise'
    
#     def augment(self, X, G, noise_coeff: Optional[float] = None):
#         # Default to sqrt(0.1) if not provided
#         if noise_coeff is None:
#             noise_coeff = np.sqrt(0.1)
            
#         return X + noise_coeff * np.std(X) * G


# class Identity(DA):
#     """Identity augmentation (no change)."""
    
#     @property
#     def augmentation(self):
#         return 'identity'
    
#     def augment(self, X, **kwargs):
#         return X, X


# Augmentation = Literal['rotation', 'hflip', 'vflip', 'gaussian-noise', 'random-permutation']

# ALL_AUGMENTATIONS: Dict[Augmentation, DA] = {
#     augmenter.augmentation: augmenter for augmenter in [
#         RandomRotation(),
#         RandomHorizontalFlip(),
#         RandomVerticalFlip(),
#         GaussianNoise(),
#         RandomPermutation()
#     ]
# }


# class OpticalDeviceDA(DA):
#     """Data augmenter for optical device experiments."""
    
#     def __init__(self, augmentations: Optional[str] = 'all'):
#         if augmentations == 'all':
#             augmentations: List[Augmentation] = list(ALL_AUGMENTATIONS.keys())
#         elif augmentations:
#             augmentations: List[Augmentation] = augmentations.replace(' ', '').split('>')

#         if augmentations:
#             self._augmentations: List[DA] = [
#                 ALL_AUGMENTATIONS[augmentation] for augmentation in augmentations
#             ]
#         else:
#             self._augmentations: List[DA] = [Identity()]
    
#     @property
#     def augmentation(self):
#         return 'optical_device'
    
#     def augment(self, X: NDArray, **kwargs) -> Tuple[NDArray, NDArray]:
#         """Apply augmentations sequentially."""
#         GX: NDArray = X.copy()
#         G_list: List[NDArray] = []
        
#         for augmentation in self._augmentations:
#             # Pass dynamic sweep parameters (p, noise_coeff) to children
#             # kwargs allows other parameters to pass through safely
#             GX, G = augmentation(GX, **kwargs)
#             G_list.append(G)
        
#         G: NDArray = np.hstack(G_list)
        
#         return GX, G