from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class StructuralEquationModel(ABC):
    @abstractmethod
    def sample(self, N: int = 1, **kwargs) -> tuple[NDArray, NDArray]:
        pass

    def __call__(self, N: int = 1, **kwargs) -> tuple[NDArray, NDArray]:
        return self.sample(N=N, **kwargs)

    def f(self, X) -> NDArray:
        return X @ self.W_XY

    def extent(self, X) -> NDArray:
        """Half-width of the target SET at each query. Zero means a POINT target.

        `f` is the centre either way, so a point target is a set target of extent 0
        and that is the default: every SEM here has been that case. A SEM whose
        target is identified only up to a set (the cigarette leaky-IV sliver)
        overrides this; `coverage` and `approximation_error` then ask whether the
        whole set is inside the interval, not just its centre.
        """
        return np.zeros(len(X))

    @property
    def solution(self) -> NDArray:
        return self.W_XY

    @property
    def pool(self) -> tuple[NDArray, NDArray] | None:
        """(X, y) when the SEM is a FINITE recorded dataset, else None.

        A simulation SEM can be asked for any number of fresh draws, so oracle
        quantities are estimated from a draw. A recorded one cannot: asking it for
        more rows than it has returns a bootstrap resample, which adds variance to
        every oracle number and makes them depend on the RNG state. Exposing the
        pool lets the oracle use the data itself, which for a recorded device IS
        the population it is estimating.
        """
        return None

    @property
    def bias_sq(self) -> float:
        """|| h_erm - h_* ||^2_X = Var(E[xi|X]). Population, not last draw."""
        raise NotImplementedError

    @property
    def sigma_sq(self) -> float:
        """sigma^2 = E[Var(Y|X)]. Population, not last draw."""
        raise NotImplementedError
