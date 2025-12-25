from typing import Tuple
from numpy.typing import NDArray
from abc import ABC, abstractmethod

class StructuralEquationModel(ABC):
    @abstractmethod
    def sample(self, N: int=1, **kwargs) -> Tuple[NDArray, NDArray]:
        pass
    
    def __call__(self, N: int=1, **kwargs) -> Tuple[NDArray, NDArray]:
        return self.sample(N=N, **kwargs)
    
    def f(self, X) -> NDArray:
        return X @ self.W_XY
    
    @property
    def solution(self) -> NDArray:
        return self.W_XY

