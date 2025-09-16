from abc import ABC, abstractmethod

class DataAugmenter(ABC):
    @property
    @abstractmethod
    def augmentation(self):
        pass

    @abstractmethod
    def augment(self, X, **kwargs):
        pass
    
    def __call__(self, X, **kwargs):
        return self.augment(X = X, **kwargs)