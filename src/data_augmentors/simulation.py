import numpy as np
from typing import Tuple
from loguru import logger
from numpy.typing import NDArray

from src.data_augmentors.abstract import DataAugmenter as DA


DA_STD: float = 1.0
BASIS_SELECTOIN_PROBABILITY: float = 8.0 / 10.0


class NullSpaceTranslation(DA):
    """
    Translation along ker(f): exactly T-invariant, hence eps = 0.
    `strength` > 0 adds an out-of-kernel component, the only source of
    invariance error (misspecified symmetry).
    """

    exact_invariance = False

    def __init__(
        self,
        W_XY: NDArray,
        kernel_dim: int,
        std: float = DA_STD,
        strength: float = 0.0,
    ):
        null_basis = self.null_space(W_XY.T).T

        k_max, _ = null_basis.shape

        assert k_max >= kernel_dim, f"`kernel_dim`={kernel_dim} cannot be greater than `k_max`={k_max}."

        if kernel_dim < 0:
            logger.info("`kernel_dim`<0 means DA is constructed from full bases of ker(f).")
            sample = np.ones(k_max, dtype="bool")
        elif kernel_dim == 0:
            logger.info("`kernel_dim`=0 means DA is constructed from randomly picked bases of ker(f).")
            sample = np.random.random(k_max) < BASIS_SELECTOIN_PROBABILITY
        else:
            logger.info(f"Selecting `kernel_dim`={kernel_dim} bases of ker(f) to construct DA.")
            sample = np.zeros(k_max, dtype="bool")
            sample[:kernel_dim] = True

        # randomize bases ordering
        np.random.shuffle(sample)

        self.std = std
        self.W_XY = W_XY
        self.W_ZXtilde = null_basis[sample]
        self.param_dimension, _ = self.W_ZXtilde.shape

        # unit direction outside ker(f): carries the invariance error
        self.W_perp = (W_XY / np.linalg.norm(W_XY)).T
        self._strength = strength

    @property
    def augmentation(self):
        return "translate"

    @property
    def strength(self) -> float:
        return self._strength

    @strength.setter
    def strength(self, value: float):
        assert value >= 0.0, "`strength` must be non-negative."
        self._strength = float(value)

    def augment(self, X: NDArray, scale: float = 1.0, **kwargs) -> Tuple[NDArray, NDArray]:
        N = len(X)
        G = np.random.randn(N, self.param_dimension) * self.std

        GX = X + scale * G @ self.W_ZXtilde

        if self._strength > 0.0:
            G_perp = np.random.randn(N, len(self.W_perp)) * self.std
            GX = GX + self._strength * G_perp @ self.W_perp
            G = np.hstack([G, G_perp])

        return GX, G

    def perturb(self, X: NDArray, **kwargs) -> NDArray:
        """Null-space part is exactly invariant; only the perp part matters."""
        if self._strength <= 0.0:
            return X
        G_perp = np.random.randn(len(X), len(self.W_perp)) * self.std
        return X + self._strength * G_perp @ self.W_perp

    @staticmethod
    def null_space(W: NDArray, absolute_tolerance: float = 1e-13, relative_tolerance: float = 0.0) -> NDArray:
        U, s, VT = np.linalg.svd(W)

        max_singular = s[0]
        tolerance = max(absolute_tolerance, relative_tolerance * max_singular)

        num_singular = (s >= tolerance).sum()
        null_space_basis = VT[num_singular:].T

        return null_space_basis
