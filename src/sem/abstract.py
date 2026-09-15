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

    # ---------------------------------------------------------- instruments

    @property
    def iv_width(self) -> int:
        """How many TRAILING columns of a draw are the instrument Z.

        `sample` keeps its (X, y) signature; a SEM with instruments returns
        X_full = [X_treatment | Z] and the runner splits it at the two places a
        draw is made (`split_instruments`). 0, the default, means the draw is
        treatment only and every consumer sees exactly what it saw before.
        """
        return 0

    @property
    def iv_pool(self) -> NDArray | None:
        """The instrument rows beside `pool`, row-aligned with it; None when the SEM
        is not a recorded dataset or carries no instrument. `pool` itself stays
        (X_treatment, y), so the oracle never sees a wider X."""
        return None

    def split_instruments(self, X) -> tuple[NDArray, NDArray]:
        """(X_treatment, Z) of a draw: the last `iv_width` columns are Z.

        At width 0 the draw comes back untouched and Z is (n, 0): empty is spelled
        as an array with no columns everywhere downstream, never None, so that
        `column_stack([G, Z])` is G elementwise.
        """
        width = int(self.iv_width)
        if width == 0:
            return X, np.zeros((len(X), 0))
        X = np.asarray(X)
        return np.ascontiguousarray(X[:, :-width]), np.ascontiguousarray(X[:, -width:])

    @property
    def bias_sq(self) -> float:
        """|| h_erm - h_* ||^2_X = Var(E[xi|X]). Population, not last draw."""
        raise NotImplementedError

    @property
    def sigma_sq(self) -> float:
        """sigma^2 = E[Var(Y|X)]. Population, not last draw."""
        raise NotImplementedError
