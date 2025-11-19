import numpy as np
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer
from sklearn.model_selection._search import BaseSearchCV


class pointEstimator(ABC, BaseEstimator):
    def fit(self, X, y, **kwargs):
        method_name = self.__class__.__name__
        if method_name == 'RICE':
            return self._fit(X, y, **kwargs)
        
        X = X.reshape(*X.shape[:1], -1)
        return self._fit(X, y, **kwargs)
    
    @abstractmethod
    def _fit(self, X, y):
        pass
    
    @property
    def solution(self):
        return self._W

    def predict(self, X, **kwargs):
        X = X.reshape(*X.shape[:1], -1)
        return self._predict(X, **kwargs)
    
    @abstractmethod
    def _predict(self, X):
        pass


class partialIdentifier(ABC, BaseEstimator):
    def fit(self, X, y, **kwargs):
        X = X.reshape(*X.shape[:1], -1)
        return self._fit(X, y, **kwargs)
    
    @abstractmethod
    def _fit(self, X, y):
        pass
    
    @property
    def solution(self):
        return self._W

    def predict(self, X, **kwargs):
        X = X.reshape(*X.shape[:1], -1)
        return self._predict(X, **kwargs)
    
    @abstractmethod
    def _predict(self, X):
        pass


class sensitivityAnalyzer(partialIdentifier):
    def __init__(self, gamma=1.0):
        self._gamma = gamma
        super(sensitivityAnalyzer, self).__init__()
    
    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, gamma):
        self._gamma = gamma

