import numpy as np
from abc import abstractmethod
from numpy.typing import NDArray
from typing import Callable, Dict, Literal, List, Optional, Tuple

from src.data_augmentors.abstract import DataAugmenter as DA
from src.data_augmentors.utils import BernoulliStandardScaler


P = 0.5
NOISE_COEFF = np.sqrt(0.01)


class Permutation(DA):
    """Base class for permutation-based augmentations. Assumed exactly invariant."""

    def __init__(self, p=P):
        self.p = p
        super().__init__()

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, p):
        self._p = p
        self.param_scaler = BernoulliStandardScaler(p=p)

    def __call__(self, X, p: Optional[float] = None, **kwargs):
        if p is not None:
            self.p = p

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

    def __call__(self, X, **kwargs):
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
        return "random-permutation"

    def augment(self, x, g):
        return self.permute(x, 1.0, g)


class RandomRotation(Permutation):
    """90-degree rotation augmentation."""

    @property
    def augmentation(self):
        return "rotation"

    def augment(self, x, g):
        ROTATION90 = np.array([6, 3, 0, 7, 4, 1, 8, 5, 2])
        return self.permute(x, g, ROTATION90)


class RandomHorizontalFlip(Permutation):
    """Horizontal flip augmentation."""

    @property
    def augmentation(self):
        return "hflip"

    def augment(self, x, g):
        HORIZONTAL_FLIP = np.array([2, 1, 0, 5, 4, 3, 8, 7, 6])
        return self.permute(x, g, HORIZONTAL_FLIP)


class RandomVerticalFlip(Permutation):
    """Vertical flip augmentation."""

    @property
    def augmentation(self):
        return "vflip"

    def augment(self, x, g):
        VERTICAL_FLIP = np.array([6, 7, 8, 3, 4, 5, 0, 1, 2])
        return self.permute(x, g, VERTICAL_FLIP)


class GaussianNoise(DA):
    """Gaussian noise augmentation: the only source of invariance error here."""

    exact_invariance = False

    def __init__(self, noise_coeff: float = NOISE_COEFF):
        self._strength = noise_coeff
        super().__init__()

    @property
    def strength(self) -> float:
        return self._strength

    @strength.setter
    def strength(self, value: float):
        assert value >= 0.0, "`strength` must be non-negative."
        self._strength = float(value)

    def __call__(self, X, noise_coeff: Optional[float] = None, **kwargs):
        if noise_coeff is not None:
            self.strength = noise_coeff

        N, M = X.shape
        G = np.random.randn(N, M)
        GX = self.augment(X, G)
        return GX, G

    @property
    def augmentation(self):
        return "gaussian-noise"

    def augment(self, X, G):
        return X + self._strength * np.std(X) * G


class Identity(DA):
    """Identity augmentation (no change)."""

    @property
    def augmentation(self):
        return "identity"

    def augment(self, X, **kwargs):
        return X, X


Augmentation = Literal["rotation", "hflip", "vflip", "gaussian-noise", "random-permutation"]

# constructors, NOT instances: augmenters carry per-DA state (p, strength)
ALL_AUGMENTATIONS: Dict[Augmentation, Callable[[], DA]] = {
    "rotation": RandomRotation,
    "hflip": RandomHorizontalFlip,
    "vflip": RandomVerticalFlip,
    "gaussian-noise": GaussianNoise,
    "random-permutation": RandomPermutation,
}


class OpticalDeviceDA(DA):
    """Data augmenter for optical device experiments."""

    exact_invariance = False

    def __init__(self, augmentations: Optional[str] = "all"):
        if augmentations == "all":
            augmentations: List[Augmentation] = list(ALL_AUGMENTATIONS.keys())
        elif augmentations:
            augmentations: List[Augmentation] = augmentations.replace(" ", "").split(">")

        if augmentations:
            self._augmentations: List[DA] = [ALL_AUGMENTATIONS[augmentation]() for augmentation in augmentations]
        else:
            self._augmentations: List[DA] = [Identity()]

    @property
    def augmentation(self):
        return "optical_device"

    @property
    def _inexact(self) -> List[DA]:
        """Components carrying invariance error (flips/rotations assumed exact)."""
        return [a for a in self._augmentations if not a.exact_invariance]

    @property
    def strength(self) -> Optional[float]:
        inexact = self._inexact
        return inexact[0].strength if inexact else None

    @strength.setter
    def strength(self, value: float):
        inexact = self._inexact
        if not inexact:
            raise NotImplementedError("No invariance-error carrying augmentation.")
        for augmentation in inexact:
            augmentation.strength = value

    def augment(self, X: NDArray, **kwargs) -> Tuple[NDArray, NDArray]:
        """Apply augmentations sequentially."""
        GX: NDArray = X.copy()
        G_list: List[NDArray] = []

        for augmentation in self._augmentations:
            GX, G = augmentation(GX, **kwargs)
            G_list.append(G)

        G: NDArray = np.hstack(G_list)

        return GX, G

    def perturb(self, X: NDArray, **kwargs) -> NDArray:
        GX: NDArray = X.copy()
        for augmentation in self._inexact:
            GX, _ = augmentation(GX, **kwargs)
        return GX
