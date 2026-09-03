from abc import ABC, abstractmethod

from numpy.typing import NDArray


class StructuralEquationModel(ABC):
    @abstractmethod
    def sample(self, N: int = 1, **kwargs) -> tuple[NDArray, NDArray]:
        pass

    def __call__(self, N: int = 1, **kwargs) -> tuple[NDArray, NDArray]:
        return self.sample(N=N, **kwargs)

    def f(self, X) -> NDArray:
        return X @ self.W_XY

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
