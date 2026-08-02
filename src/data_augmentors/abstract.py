from typing import Optional
from abc import ABC, abstractmethod


class DataAugmenter(ABC):
    # True: assumed to leave h_* invariant, i.e. contributes no invariance error
    exact_invariance: bool = True

    @property
    @abstractmethod
    def augmentation(self):
        pass

    @abstractmethod
    def augment(self, X, **kwargs):
        pass

    def __call__(self, X, **kwargs):
        return self.augment(X = X, **kwargs)

    @property
    def strength(self) -> Optional[float]:
        """Scalar knob driving the invariance error; None if the DA has none."""
        return None

    @strength.setter
    def strength(self, value: float):
        raise NotImplementedError(f'{type(self).__name__} has no strength knob.')

    def perturb(self, X, **kwargs):
        """Apply only the invariance-error carrying part of the augmentation."""
        return self.augment(X, **kwargs)[0]
