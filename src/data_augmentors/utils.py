"""
Utilities for data augmentation including standard scalers.
"""

from abc import ABC, abstractmethod

import numpy as np


class StandardScaler(ABC):
    """Abstract base class for standard scalers."""

    @abstractmethod
    def __init__(self, **kwargs):
        pass

    def __call__(self, sample):
        return (sample - self.mean) / self.std


class UniformStandardScaler(StandardScaler):
    """Standard scaler for uniform distributions."""

    def __init__(self, low, high):
        self.mean = (low + high) / 2.0
        self.std = (high - low) / (12.0**0.5)


class BernoulliStandardScaler(StandardScaler):
    """Standard scaler for Bernoulli distributions."""

    def __init__(self, p=0.5):
        self.mean = p
        self.std = np.sqrt(p * (1.0 - p))


class BetaStandardScaler(StandardScaler):
    """Standard scaler for Beta distributions."""

    def __init__(self, min_val, max_val, alpha=2, beta=2):
        self.mean = alpha / (alpha + beta)
        self.std = np.sqrt(alpha * beta) / ((alpha + beta) * np.sqrt(alpha + beta + 1))
        self.min = min_val
        self.max = max_val

    def rescale(self, sample):
        """Rescale from [0,1] to [min, max]."""
        return sample * (self.max - self.min) + self.min

    def __call__(self, sample):
        """Normalize and standardize."""
        sample = (sample - self.min) / (self.max - self.min)
        return super().__call__(sample)
