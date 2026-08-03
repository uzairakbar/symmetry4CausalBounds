"""
Oracle sensitivity parameters for an (SEM, DA) pair.

Computed in sequence gamma* -> epsilon* -> gamma_z*, and returned to the
experiment scripts, which may or may not use them.
"""
import numpy as np
from loguru import logger
from contextlib import contextmanager
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Callable, Optional
from numpy.typing import NDArray

from src.methods.regression import LeastSquaresClosedForm as OLS


CALIBRATION_SAMPLES: int = 2048
STRENGTH_BRACKET: tuple = (0.0, 1e3)
STRENGTH_TOLERANCE: float = 1e-9


@dataclass(frozen=True)
class OracleParameters:
    """Oracle values; `calibrate` fixes the units of gamma_star."""
    gamma_star: float
    epsilon_star: float
    gamma_z_star: Optional[float]
    bias_sq: float
    sigma_sq: float
    rho: Optional[float]


@contextmanager
def preserve_rng():
    """Oracle draws must not shift the global stream the experiments use."""
    state = np.random.get_state()
    try:
        yield
    finally:
        np.random.set_state(state)


def _identity(X: NDArray) -> NDArray:
    return X


# =============================================================================
# gamma*
# =============================================================================

class GammaStarStrategy(ABC):
    """Selection strategy for the confounding budget gamma*."""

    @abstractmethod
    def __call__(self, sem, calibrate: bool = False) -> float:
        pass


class ValidityForBaselinePI(GammaStarStrategy):
    """Tightest gamma keeping h_* inside the baseline PI set (Lem. 2)."""

    def __call__(self, sem, calibrate: bool = False) -> float:
        bias_sq = sem.bias_sq
        if not calibrate:
            return float(bias_sq)
        sigma_sq = sem.sigma_sq
        if sigma_sq <= 0.0:
            logger.warning('sigma^2 = 0 (fully confounded): gamma* is unbounded.')
            return float(np.inf)
        return float(bias_sq / sigma_sq)


DEFAULT_GAMMA_STAR = ValidityForBaselinePI()


def gamma_star(sem, calibrate: bool = False, strategy: GammaStarStrategy = DEFAULT_GAMMA_STAR) -> float:
    return strategy(sem, calibrate=calibrate)


# =============================================================================
# epsilon*
# =============================================================================

def invariance_error(
    sem,
    da,
    X: Optional[NDArray] = None,
    features: Optional[Callable] = None,
    n_samples: int = CALIBRATION_SAMPLES,
) -> float:
    """
    eps* = sqrt( E[ (h_*(X) - h_*(X~))^2 ] ), over the DA components that are
    not assumed exactly invariant.
    """
    features = features or _identity

    with preserve_rng():
        if X is None:
            X, _ = sem(N=n_samples)
        GX = da.perturb(X)

        residuals = sem.f(features(X)) - sem.f(features(GX))

    return float(np.sqrt(np.mean(residuals ** 2)))


def calibrate_da_epsilon(
    sem,
    da,
    epsilon_target: float,
    X: Optional[NDArray] = None,
    features: Optional[Callable] = None,
    n_samples: int = CALIBRATION_SAMPLES,
) -> float:
    """
    Inverse of `invariance_error`: set the DA strength knob achieving
    `epsilon_target`. Returns the achieved eps*.
    """
    if epsilon_target < 0.0:
        raise ValueError('`epsilon_target` must be non-negative.')

    if da.strength is None:
        raise NotImplementedError(
            f'{type(da).__name__} has no strength knob to hit eps={epsilon_target}.'
        )

    # freeze the sample so the 1-D solve sees a deterministic objective
    if X is None:
        with preserve_rng():
            X, _ = sem(N=n_samples)

    def error(strength: float) -> float:
        da.strength = strength
        return invariance_error(sem, da, X=X, features=features) - epsilon_target

    low, high = STRENGTH_BRACKET
    if error(low) >= 0.0:
        da.strength = low
        logger.warning(f'eps* floor exceeds target {epsilon_target}; strength set to {low}.')
    else:
        while error(high) < 0.0:
            high *= 2.0
        for _ in range(200):
            mid = 0.5 * (low + high)
            if error(mid) < 0.0:
                low = mid
            else:
                high = mid
            if high - low < STRENGTH_TOLERANCE:
                break
        da.strength = 0.5 * (low + high)

    achieved = invariance_error(sem, da, X=X, features=features)
    logger.info(f'DA strength {da.strength:.6g} -> eps* {achieved:.6g} (target {epsilon_target:.6g}).')
    return achieved


# =============================================================================
# Thm. 1 threshold
# =============================================================================

def thm1_gamma_min(oracle: 'OracleParameters', calibrate: bool = False) -> float:
    """
    Smallest gamma at which the DA+PI set still contains h_* (Thm. 1), i.e. the
    budget the augmentation buys back. Below gamma* by an amount set by rho.

    calibrated:   sqrt(gamma_min) = max(0, sqrt(gamma*) - (rho - 1)/sqrt(rho))
    uncalibrated: gamma_min       = max(0, bias^2 - sigma^2 (rho - 1))
    """
    rho = oracle.rho
    if rho is None or not np.isfinite(rho):
        logger.warning('rho unavailable; Thm. 1 threshold falls back to gamma*.')
        return float(oracle.gamma_star)

    if calibrate:
        slack = (rho - 1.0) / np.sqrt(rho)
        return float(max(0.0, np.sqrt(oracle.gamma_star) - slack) ** 2)

    return float(max(0.0, oracle.bias_sq - oracle.sigma_sq * (rho - 1.0)))


# =============================================================================
# gamma_z*
# =============================================================================

def gamma_z_star(sem, da, X=None, features=None, calibrate: bool = False) -> Optional[float]:
    """
    Oracle IV budget (Asm. 3): Var(E[Y - h_*(X) | Z]) <= sigma^2 gamma_z.
    Not implemented: no experiment uses instruments.
    """
    return None


# =============================================================================
# entry point
# =============================================================================

def _noise_ratio(sem, da, X, y, features, n_samples: int = CALIBRATION_SAMPLES) -> Optional[float]:
    """rho = sigma-tilde^2 / sigma^2, the information-loss factor (DPI: >= 1)."""
    sigma_sq = sem.sigma_sq
    if sigma_sq <= 0.0:
        return None

    with preserve_rng():
        if y is None:       # X given without outcomes: rho needs its own draw
            X, y = sem(N=n_samples)
        GX, _ = da(X)
        Phi = features(GX)
        residuals = y.flatten() - Phi @ OLS().fit(Phi, y).solution.flatten()

    return float(np.mean(residuals ** 2) / sigma_sq)


def compute_oracle_parameters(
    sem,
    da,
    X: Optional[NDArray] = None,
    y: Optional[NDArray] = None,
    features: Optional[Callable] = None,
    calibrate: bool = False,
    n_samples: int = CALIBRATION_SAMPLES,
    strategy: GammaStarStrategy = DEFAULT_GAMMA_STAR,
) -> OracleParameters:
    """Oracle parameters for one (SEM, DA) pair, in the given budget units."""
    features = features or _identity

    if X is None:
        with preserve_rng():
            X, y = sem(N=n_samples)

    return OracleParameters(
        gamma_star=gamma_star(sem, calibrate=calibrate, strategy=strategy),
        epsilon_star=invariance_error(sem, da, X=X, features=features),
        gamma_z_star=gamma_z_star(sem, da, X=X, features=features, calibrate=calibrate),
        bias_sq=float(sem.bias_sq),
        sigma_sq=float(sem.sigma_sq),
        rho=_noise_ratio(sem, da, X, y, features, n_samples),
    )
