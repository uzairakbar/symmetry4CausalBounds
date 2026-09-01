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
STRENGTH_DOUBLINGS: int = 20  # bracket expansions before declaring the target unreachable


@dataclass(frozen=True)
class OracleParameters:
    """Oracle values; `calibrate` fixes the units of gamma_star."""

    gamma_star: float
    epsilon_star: float
    gamma_z_star: Optional[float]
    bias_sq: float
    sigma_sq: float
    rho: Optional[float]
    # IV budget (Thm. 3.B, exact at gamma_z* = 0) and its byproducts
    eps_iv_star: Optional[float] = None
    eps_rms: Optional[float] = None
    eta: Optional[float] = None
    # SHELVED: eps* under the perturb convention (exactly-invariant components
    # excluded). Recorded, never consumed -- lets the padding choice be revisited.
    epsilon_star_pointwise: Optional[float] = None


@contextmanager
def preserve_rng():
    """Oracle draws must not shift the global streams the experiments use.

    numpy AND torch: the image DAs draw from torch, so restoring numpy alone lets
    an oracle call change the experiment's augmentation realisation -- same seed,
    different GX, depending only on whether the oracle ran.
    """
    numpy_state = np.random.get_state()
    torch_state = cuda_state = None
    try:
        import torch

        torch_state = torch.random.get_rng_state()
        if torch.cuda.is_available():
            cuda_state = torch.cuda.get_rng_state_all()
    except ImportError:
        pass

    try:
        yield
    finally:
        np.random.set_state(numpy_state)
        if torch_state is not None:
            import torch

            torch.random.set_rng_state(torch_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)


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
            logger.warning("sigma^2 = 0 (fully confounded): gamma* is unbounded.")
            return float(np.inf)
        return float(bias_sq / sigma_sq)


DEFAULT_GAMMA_STAR = ValidityForBaselinePI()


def gamma_star(sem, calibrate: bool = False, strategy: GammaStarStrategy = DEFAULT_GAMMA_STAR) -> float:
    return strategy(sem, calibrate=calibrate)


# =============================================================================
# epsilon*
# =============================================================================


def _invariance_signal(
    sem,
    da,
    X: Optional[NDArray] = None,
    features: Optional[Callable] = None,
    n_samples: int = CALIBRATION_SAMPLES,
    **augment_kwargs,
) -> tuple:
    """
    (w, Phi, G) for ONE augmentation draw, where w = h_*(X) - h_*(X~) is the
    paper's W (C.3) over the FULL augmentation.

    Sole definition of W: both `epsilon_star` (Thm. 3.A, needs ||W||) and
    `eps_iv_star` (Thm. 3.B, needs the residual W# = W - E[W|X~]) build on it,
    so the two can never drift apart.
    """
    features = features or _identity

    with preserve_rng():
        if X is None:
            X, _ = sem(N=n_samples)
        GX, G = da(X, **augment_kwargs)
        Phi = features(GX)
        w = (sem.f(features(X)) - sem.f(Phi)).flatten()

    return w, Phi, np.asarray(G).reshape(len(G), -1)


def epsilon_star(
    sem,
    da,
    X: Optional[NDArray] = None,
    features: Optional[Callable] = None,
    n_samples: int = CALIBRATION_SAMPLES,
    **augment_kwargs,
) -> float:
    """
    eps* = || W || = sqrt( E[ (h_*(X) - h_*(X~))^2 ] ) over the FULL augmentation.

    This is the paper's ε (C.3: |W| <= ε a.s.), relaxed from the pointwise sup
    to RMS. Two properties that make it the right budget:
      - Thm. 3.A padding needs ||E[W|X~]||, and ||W||^2 = ||E[W|X~]||^2 + ||W#||^2
        by orthogonality, so ||W|| dominates it.
      - it equals the PI_INV constraint evaluated at h_* exactly, since
        (Phi(GX) - Phi(X)) h_* = -w. So eps* + EPS_TOL admits h_* by construction.
    """
    w, _, _ = _invariance_signal(sem, da, X, features, n_samples, **augment_kwargs)
    return float(np.sqrt(np.mean(w**2)))


def invariance_error(
    sem,
    da,
    X: Optional[NDArray] = None,
    features: Optional[Callable] = None,
    n_samples: int = CALIBRATION_SAMPLES,
) -> float:
    """
    SHELVED -- not wired into the pipeline; kept for comparison and recorded on
    `OracleParameters.epsilon_star_pointwise`.

    Same functional as `epsilon_star` but over `da.perturb`, i.e. only the DA
    components not assumed exactly invariant. Retained so the padding decision
    (RMS on the full augmentation vs this) can be revisited once the
    epsilon/epsilon* sweep results are in.
    """
    features = features or _identity

    with preserve_rng():
        if X is None:
            X, _ = sem(N=n_samples)
        GX = da.perturb(X)

        residuals = sem.f(features(X)) - sem.f(features(GX))

    return float(np.sqrt(np.mean(residuals**2)))


def calibrate_da_epsilon(
    sem,
    da,
    epsilon_target: float,
    X: Optional[NDArray] = None,
    features: Optional[Callable] = None,
    n_samples: int = CALIBRATION_SAMPLES,
) -> float:
    """
    Inverse of `epsilon_star`: set the DA strength knob achieving
    `epsilon_target`. Returns the achieved eps*.
    """
    if epsilon_target < 0.0:
        raise ValueError("`epsilon_target` must be non-negative.")

    if da.strength is None:
        raise NotImplementedError(f"{type(da).__name__} has no strength knob to hit eps={epsilon_target}.")

    # freeze the sample so the 1-D solve sees a deterministic objective
    if X is None:
        with preserve_rng():
            X, _ = sem(N=n_samples)

    def error(strength: float) -> float:
        da.strength = strength
        return epsilon_star(sem, da, X=X, features=features) - epsilon_target

    low, high = STRENGTH_BRACKET
    if error(low) >= 0.0:
        da.strength = low
        logger.warning(
            f"eps* floor {epsilon_star(sem, da, X=X, features=features):.6g} exceeds "
            f"target {epsilon_target:.6g}; strength clamped to {low}. The sweep "
            "will not vary the DA -- raise the target above the floor."
        )
    else:
        # eps* saturates in strength, so a target above the reachable ceiling
        # would send the bracket to ~1e6 and "converge" on a wrong value
        for _ in range(STRENGTH_DOUBLINGS):
            if error(high) >= 0.0:
                break
            high *= 2.0
        else:
            da.strength = high
            achieved = epsilon_star(sem, da, X=X, features=features)
            raise ValueError(
                f"eps* target {epsilon_target:.6g} is above the reachable ceiling "
                f"(~{achieved:.6g} at strength {high:.6g}); it saturates in strength."
            )
        for _ in range(200):
            mid = 0.5 * (low + high)
            if error(mid) < 0.0:
                low = mid
            else:
                high = mid
            if high - low < STRENGTH_TOLERANCE:
                break
        da.strength = 0.5 * (low + high)

    achieved = epsilon_star(sem, da, X=X, features=features)
    logger.info(f"DA strength {da.strength:.6g} -> eps* {achieved:.6g} (target {epsilon_target:.6g}).")
    return achieved


# =============================================================================
# Thm. 3.B diagnostics
# =============================================================================


def eps_iv_star(
    sem,
    da,
    X: Optional[NDArray] = None,
    features: Optional[Callable] = None,
    n_samples: int = CALIBRATION_SAMPLES,
) -> tuple:
    """
    The IV budget: || E-hat[W# | Z-tilde] || / sqrt(N).

    Exact expansion of C.3 Part (B) at gamma_z* = 0. Every experiment here uses
    T as the instrument with T independent of (U, xi) by construction, so
    E[U + xi | Z-tilde] = 0, the Minkowski cross-term vanishes identically, and
    this IS the budget h# requires -- no correlation (r) assumption anywhere.

        w    = f(Phi(X)) - f(Phi(GX))
        W#   = w - OLS fit of w on Phi(GX)
        eps_iv_star = RMS of the projection of W# onto span(G)

    Returns:
        (eps_iv_star, eps_rms, eta) -- eps_rms = RMS(W#) and eta =
        eps_iv_star / eps_rms are free byproducts, logged for the record.
    """
    w, Phi, G = _invariance_signal(sem, da, X, features, n_samples)

    # W#: what the augmented design cannot explain. The -f(Phi(GX)) term is
    # exactly linear in Phi(GX), so OLS absorbs it and this is the part of
    # f(Phi(X)) unreachable from the augmented data.
    W_sharp = w - Phi @ np.linalg.lstsq(Phi, w, rcond=None)[0]
    eps_rms = float(np.sqrt(np.mean(W_sharp**2)))

    # project onto span(G): same QR geometry as the IV constraint itself
    Q, _ = np.linalg.qr(G)
    budget = float(np.sqrt(np.mean((Q @ (Q.T @ W_sharp)) ** 2)))

    eta = float(budget / eps_rms) if eps_rms > 0.0 else 0.0
    return budget, eps_rms, eta


# =============================================================================
# Thm. 1 threshold
# =============================================================================


def thm1_gamma_min(oracle: "OracleParameters", calibrate: bool = False) -> float:
    """
    Smallest gamma at which the DA+PI set still contains h_* (Thm. 1), i.e. the
    budget the augmentation buys back. Below gamma* by an amount set by rho.

    calibrated:   sqrt(gamma_min) = max(0, sqrt(gamma*) - (rho - 1)/sqrt(rho))
    uncalibrated: gamma_min       = max(0, bias^2 - sigma^2 (rho - 1))
    """
    rho = oracle.rho
    if rho is None or not np.isfinite(rho):
        logger.warning("rho unavailable; Thm. 1 threshold falls back to gamma*.")
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
        if y is None:  # X given without outcomes: rho needs its own draw
            X, y = sem(N=n_samples)
        GX, _ = da(X)
        Phi = features(GX)
        residuals = y.flatten() - Phi @ OLS().fit(Phi, y).solution.flatten()

    return float(np.mean(residuals**2) / sigma_sq)


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

    iv_budget, eps_rms, eta = eps_iv_star(sem, da, X=X, features=features, n_samples=n_samples)

    return OracleParameters(
        gamma_star=gamma_star(sem, calibrate=calibrate, strategy=strategy),
        epsilon_star=epsilon_star(sem, da, X=X, features=features),
        gamma_z_star=gamma_z_star(sem, da, X=X, features=features, calibrate=calibrate),
        bias_sq=float(sem.bias_sq),
        sigma_sq=float(sem.sigma_sq),
        rho=_noise_ratio(sem, da, X, y, features, n_samples),
        eps_iv_star=iv_budget,
        eps_rms=eps_rms,
        eta=eta,
        epsilon_star_pointwise=invariance_error(sem, da, X=X, features=features),
    )
