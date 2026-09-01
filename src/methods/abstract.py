import numpy as np
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator


class pointEstimator(ABC, BaseEstimator):
    """Base class for point estimators."""

    def fit(self, X, y, **kwargs):
        """Fit the model. Subclasses extract what they need from kwargs."""
        X = X.reshape(*X.shape[:1], -1)
        return self._fit(X, y, **kwargs)

    @abstractmethod
    def _fit(self, X, y, **kwargs):
        pass

    @property
    def solution(self):
        return self._W

    def predict(self, X, **kwargs):
        """Predict outcomes. Subclasses extract what they need from kwargs."""
        X = X.reshape(*X.shape[:1], -1)
        return self._predict(X, **kwargs)

    @abstractmethod
    def _predict(self, X, **kwargs):
        pass


class partialIdentifier(ABC, BaseEstimator):
    """Base class for partial identification methods."""

    def fit(self, X, y, **kwargs):
        """Fit the model. Subclasses extract what they need from kwargs."""
        X = X.reshape(*X.shape[:1], -1)
        return self._fit(X, y, **kwargs)

    @abstractmethod
    def _fit(self, X, y, **kwargs):
        pass

    @property
    def solution(self):
        return self._W

    def predict(self, X, **kwargs):
        """Predict outcomes. Subclasses extract what they need from kwargs."""
        X = X.reshape(*X.shape[:1], -1)
        return self._predict(X, **kwargs)

    @abstractmethod
    def _predict(self, X, **kwargs):
        pass


class sensitivityAnalyzer(partialIdentifier):
    """Base class for sensitivity analysis methods."""

    def __init__(self, gamma=1.0):
        self._gamma = gamma
        super().__init__()

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, gamma):
        self._gamma = gamma
