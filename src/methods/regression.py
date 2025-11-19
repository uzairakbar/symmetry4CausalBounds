import torch
import numpy as np
import cvxpy as cp
from loguru import logger
import torch.nn.functional as F
import torch.utils.data as data_utils

from src.methods.utils import MODELS, Model, device

from src.methods.abstract import pointEstimator


DEVICE: str=device()
MAX_BATCH: int=256
LOG_FREQUENCY: int=100


class LeastSquaresClosedForm(pointEstimator):
    def _fit(self, X, y, **kwargs):
        self._W = np.linalg.pinv(X) @ y
        return self
    
    def _predict(self, X, **kwargs):
        return X @ self._W

