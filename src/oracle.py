"""
Oracle sensitivity parameters for an (SEM, DA) pair.

Computed in sequence gamma* -> epsilon* -> gamma_z*, and returned to the
experiment scripts, which may or may not use them.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import contextmanager, suppress
from dataclasses import dataclass, replace

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from src.methods.regression import LeastSquaresClosedForm as OLS
from src.methods.regression import residual_variance

CALIBRATION_SAMPLES: int = 2048
# `epsilon_star_q95`, the sweeps' one reading of |W|: how many augmentation
# realisations to concatenate, the stream they are drawn from (draw d at
# W_SEED + d) and the quantile taken. A quantile rather than the sup because a DA
# with a Gaussian component has no finite sup; see the function.
W_DRAWS: int = 8
W_SEED: int = 0
W_QUANTILE: float = 0.95
STRENGTH_BRACKET: tuple = (0.0, 1e3)
STRENGTH_TOLERANCE: float = 1e-9
STRENGTH_DOUBLINGS: int = 20  # bracket expansions before declaring the target unreachable


@dataclass(frozen=True)
class OracleParameters:
    """Oracle values; gamma_star is in the paper's sigma-scaled units, bias^2/sigma^2."""

    gamma_star: float
    epsilon_star: float
    gamma_z_star: float | None
    bias_sq: float
    sigma_sq: float
    rho: float | None


# Draw-pooled fields and how they pool: a NORM pools in the square (an RMS of RMSs),
# a ratio pools as a plain mean. The rest are functions of the SEM alone.
_ORACLE_RMS_FIELDS = ("epsilon_star",)
_ORACLE_MEAN_FIELDS = ("rho", "gamma_z_star")


def pool_oracles(oracles: list[OracleParameters]) -> OracleParameters:
    """One `OracleParameters` from several independent DA realisations.

    Only worth doing for a SEM whose pool is FIXED: there every replicate sees the
    same rows, so the sole randomness in `rho`/`eps*` is the augmentation draw, and
    a single draw is not the population quantity the figures annotate. It matters
    most for rho, because Thm. 1's plotted threshold
    gamma_min/gamma* = (gamma* - (rho - 1)) / (rho gamma*) differences two similar
    numbers in its numerator: near rho - 1 ~ gamma*, where the optical device sits,
    d ln(threshold) / d ln rho is about -8, so a 3% draw-to-draw wobble in rho
    becomes a 20% wobble in the annotation.
    """
    if len(oracles) == 1:
        return oracles[0]

    values = {}
    for field in _ORACLE_RMS_FIELDS:
        found = [getattr(o, field) for o in oracles if getattr(o, field) is not None]
        values[field] = float(np.sqrt(np.mean(np.square(found)))) if found else None
    for field in _ORACLE_MEAN_FIELDS:
        found = [getattr(o, field) for o in oracles if getattr(o, field) is not None]
        values[field] = float(np.mean(found)) if found else None
    return replace(oracles[0], **values)


@contextmanager
def preserve_rng():
    """Oracle draws must not shift the global streams the experiments use.

    numpy AND torch: the image DAs draw from torch, so restoring numpy alone lets
    an oracle call change the experiment's augmentation realisation -- same seed,
    different GX, depending only on whether the oracle ran.
    """
    numpy_state = np.random.get_state()
    torch_state = cuda_state = None
    # no torch, no torch stream to preserve -- the sentinel None already says so
    with suppress(ImportError):
        import torch

        torch_state = torch.random.get_rng_state()
        if torch.cuda.is_available():
            cuda_state = torch.cuda.get_rng_state_all()

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


def _draw(sem, n_samples: int) -> tuple:
    """(X, y, Z) of one draw. A SEM with instruments emits them as trailing
    columns of X (`SEM.iv_width`); they are split off HERE so no oracle routine
    ever sees a wider design, and the empty case is (n, 0), never None."""
    X, y = sem(N=n_samples)
    split = getattr(sem, "split_instruments", None)
    if split is None:
        return X, y, np.zeros((len(X), 0))
    X, Z = split(X)
    return X, y, Z


# =============================================================================
# gamma*
# =============================================================================


class GammaStarStrategy(ABC):
    """Selection strategy for the confounding budget gamma*."""

    @abstractmethod
    def __call__(self, sem) -> float:
        pass


class ValidityForBaselinePI(GammaStarStrategy):
    """Tightest gamma keeping h_* inside the baseline PI set (Lem. 2), in the
    paper's units: bias^2 / sigma^2."""

    def __call__(self, sem) -> float:
        bias_sq = sem.bias_sq
        sigma_sq = sem.sigma_sq
        if sigma_sq <= 0.0:
            logger.warning("sigma^2 = 0 (fully confounded): gamma* is unbounded.")
            return float(np.inf)
        return float(bias_sq / sigma_sq)


DEFAULT_GAMMA_STAR = ValidityForBaselinePI()


def gamma_star(sem, strategy: GammaStarStrategy = DEFAULT_GAMMA_STAR) -> float:
    return strategy(sem)


# =============================================================================
# epsilon*
# =============================================================================


def _invariance_signal(
    sem,
    da,
    X: NDArray | None = None,
    features: Callable | None = None,
    n_samples: int = CALIBRATION_SAMPLES,
    **augment_kwargs,
) -> tuple:
    """
    (w, Phi, G) for ONE augmentation draw, where w = h_*(X) - h_*(X~) is the
    paper's W (C.3) over the FULL augmentation.

    Sole definition of W: every invariance-error reading (`epsilon_star`,
    `epsilon_star_q95`, the DA's tuning) builds on it, so none can drift apart.
    """
    features = features or _identity

    with preserve_rng():
        if X is None:
            X, _, _ = _draw(sem, n_samples)
        GX, G = da(X, **augment_kwargs)
        Phi = features(GX)
        w = (sem.f(features(X)) - sem.f(Phi)).flatten()

    return w, Phi, np.asarray(G).reshape(len(G), -1)


def epsilon_star(
    sem,
    da,
    X: NDArray | None = None,
    features: Callable | None = None,
    n_samples: int = CALIBRATION_SAMPLES,
    **augment_kwargs,
) -> float:
    """
    eps* = || W || = sqrt( E[ (h_*(X) - h_*(X~))^2 ] ) over the FULL augmentation.

    This is the paper's ε (C.3: |W| <= ε a.s.), relaxed from the pointwise sup
    to RMS. Two properties that make it the right budget:
      - Thm. 3.A padding needs ||E[W|X~]||, and ||W||^2 = ||E[W|X~]||^2 + ||W#||^2
        by orthogonality, so ||W|| dominates it.
      - it equals the PI+INV constraint evaluated at h_* exactly, since
        (Phi(GX) - Phi(X)) h_* = -w. So eps* + EPS_TOL admits h_* by construction.
    """
    w, _, _ = _invariance_signal(sem, da, X, features, n_samples, **augment_kwargs)
    return float(np.sqrt(np.mean(w**2)))


def epsilon_star_q95(
    sem,
    da,
    X: NDArray | None = None,
    features: Callable | None = None,
    n_samples: int = CALIBRATION_SAMPLES,
    **augment_kwargs,
) -> float:
    """The sweeps' eps*: the W_QUANTILE (0.95) of |W| over W_DRAWS concatenated
    augmentation realisations, draw d seeded W_SEED + d, under `preserve_rng`.

    One reading of the defect W = h_*(X) - h_*(X~) for everything a sweep reads
    off it: the INV budget eps (+ EPS_TOL), the +-eps pad (= eps), the omega
    knob's per-step eps*, the robustness DA's tuning target and the sweeps'
    fallback budgets. SS2.4 states T-invariance as sup |W| <= eps, and Thm. 3.A's
    pad carries that bound pointwise; a quantile and not the sup because under a
    DA with a Gaussian component the sup is infinite, so the pad holds with
    probability >= W_QUANTILE per query. The query panels keep the RMS
    (`epsilon_star`).

    The draws are advanced explicitly: `_invariance_signal` restores the RNG on
    exit, so a bare loop would evaluate the SAME realisation every time. A SEM
    without a fixed pool draws its rows once, at W_SEED.
    """
    features = features or _identity
    with preserve_rng():
        np.random.seed(W_SEED)
        if X is None:
            X, _, _ = _draw(sem, n_samples)
        pooled = []
        for draw in range(W_DRAWS):
            np.random.seed(W_SEED + draw)
            w, _, _ = _invariance_signal(sem, da, X, features, n_samples, **augment_kwargs)
            pooled.append(np.abs(w))
    return float(np.quantile(np.concatenate(pooled), W_QUANTILE))


def sweep_epsilon_star(sem, da, X: NDArray | None = None, features: Callable | None = None, **augment_kwargs) -> float:
    """eps* in the norm the SEM is routed to: `epsilon_star_q95` on a sweep's SEM
    (`epsilon_quantile` set), the RMS `epsilon_star` on a query panel's."""
    if getattr(sem, "epsilon_quantile", None) is not None:
        return epsilon_star_q95(sem, da, X=X, features=features, **augment_kwargs)
    return epsilon_star(sem, da, X=X, features=features, **augment_kwargs)


def recalibrated_da_epsilon(
    sem,
    da,
    epsilon_target: float,
    X: NDArray | None = None,
    features: Callable | None = None,
    n_samples: int = CALIBRATION_SAMPLES,
) -> float:
    """
    Inverse of `sweep_epsilon_star`: set the DA strength knob so the recalibrated
    DA achieves `epsilon_target` in the SEM's norm of W. Returns the achieved eps*.
    """
    if epsilon_target < 0.0:
        raise ValueError("`epsilon_target` must be non-negative.")

    if da.strength is None:
        raise NotImplementedError(f"{type(da).__name__} has no strength knob to hit eps={epsilon_target}.")

    # freeze the sample so the 1-D solve sees a deterministic objective
    if X is None:
        with preserve_rng():
            X, _, _ = _draw(sem, n_samples)

    def error(strength: float) -> float:
        da.strength = strength
        return sweep_epsilon_star(sem, da, X=X, features=features) - epsilon_target

    low, high = STRENGTH_BRACKET
    if error(low) >= 0.0:
        da.strength = low
        logger.warning(
            f"eps* floor {sweep_epsilon_star(sem, da, X=X, features=features):.6g} exceeds "
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
            achieved = sweep_epsilon_star(sem, da, X=X, features=features)
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

    achieved = sweep_epsilon_star(sem, da, X=X, features=features)
    logger.info(f"DA strength {da.strength:.6g} -> eps* {achieved:.6g} (target {epsilon_target:.6g}).")
    return achieved


# =============================================================================
# Thm. 1 threshold
# =============================================================================


def _thm1_r_base(oracle: "OracleParameters", gamma: float) -> float:
    """Radius of the BASELINE ball, sigma sqrt(gamma) (paper's units). No rho."""
    gamma = max(float(gamma), 0.0)
    return float(np.sqrt(float(oracle.sigma_sq) * gamma))


def _thm1_r_da(oracle: "OracleParameters", gamma: float) -> float:
    """Radius of the POST-DA ball at the inherited gamma: sigma-tilde sqrt(gamma)
    = sigma sqrt(rho gamma). The only radius that needs rho -- callers guard it."""
    gamma = max(float(gamma), 0.0)
    return float(np.sqrt(float(oracle.sigma_sq) * float(oracle.rho) * gamma))


def thm1_eps_ceiling(oracle: "OracleParameters", gamma: float) -> float:
    """
    eps+ of Thm. 1: the largest approximation error of the BASELINE set at which
    DA still cannot lose h_* (App. F.1).

        eps+ = ( sqrt(C^2 + r_DA^2) - r_base )^2,      C^2 = sigma^2 (rho - 1)

    with C^2 = sigma-tilde^2 - sigma^2 the information the DA gives up (Lem. 4:
    A^2 = B^2 + C^2 for the pre-/post-DA bias magnitudes A, B). In the paper's
    units this is the statement of Thm. 1,

        eps+ = sigma^2 ( sqrt(rho - 1 + gamma rho) - sqrt(gamma) )^2.

    TIGHT, not merely sufficient: for A >= r_base every step of the F.1 chain
    (eps <= eps+  =>  A^2 <= C^2 + r_DA^2  =>  B^2 <= r_DA^2) is reversible, so
    eps <= eps+ is EQUIVALENT to post-DA membership h_* in H_pi-tilde, Eq. (dagger).
    """
    rho = oracle.rho
    if rho is None or not np.isfinite(rho):
        logger.warning("rho unavailable; Thm. 1 ceiling is undefined.")
        return float("nan")

    r_base = _thm1_r_base(oracle, gamma)
    r_da = _thm1_r_da(oracle, gamma)
    c_sq = float(oracle.sigma_sq) * (float(rho) - 1.0)
    return float((np.sqrt(max(c_sq + r_da**2, 0.0)) - r_base) ** 2)


def thm1_eps_valid(oracle: "OracleParameters", gamma: float) -> float:
    """
    Approximation error of the BASELINE set, App. F.1 Eq. (*): for a ball of
    radius r_base around h_erm, E^-(H_pi) = (A - r_base)_+^2 with A = ||h_erm - h_*||.

    The left-hand side of Thm. 1's hypothesis; `thm1_gamma_min` is exactly where
    it meets `thm1_eps_ceiling`. Needs no rho -- it is a statement about the
    BASELINE set alone, and stays defined when the DA's noise ratio does not.
    """
    r_base = _thm1_r_base(oracle, gamma)
    return float(max(0.0, np.sqrt(max(float(oracle.bias_sq), 0.0)) - r_base) ** 2)


def thm1_gamma_min(oracle: "OracleParameters") -> float:
    """
    Smallest gamma at which the DA+PI set still contains h_* (Thm. 1), i.e. the
    budget the augmentation buys back.

    The inversion of `thm1_eps_ceiling`: eps_valid(gamma) <= eps+(gamma) reduces
    to Eq. (dagger) B^2 <= r_DA^2 with B^2 = bias^2 - sigma^2 (rho - 1), giving

        gamma_min = max(0, (gamma* - (rho - 1)) / rho),

    gamma* at rho = 1 and decreasing as the DA gives up more information.
    """
    rho = oracle.rho
    if rho is None or not np.isfinite(rho):
        logger.warning("rho unavailable; Thm. 1 threshold falls back to gamma*.")
        return float(oracle.gamma_star)

    return float(max(0.0, (oracle.gamma_star - (rho - 1.0)) / rho))


# =============================================================================
# gamma_z*
# =============================================================================


def gamma_z_star(sem, da, X=None, features=None) -> float | None:
    """
    Oracle IV leakiness (Asm. 3): Var(E[Y - h_*(X) | Z]) <= sigma^2 gamma_z.

    None on purpose. The real-Z leak gamma_z is DECLARED (SS2.6: the cigarette
    block's `gamma_z`, `SimulationConfig.gamma_z`), never estimated: a non-empty
    `iv:` asserts the instruments are near perfect. The slot stays so
    `OracleParameters` keeps its shape.
    """
    return None


# =============================================================================
# entry point
# =============================================================================


def _noise_ratio(
    sem, da, X, y, features, n_samples: int = CALIBRATION_SAMPLES, mean_match: bool = False
) -> float | None:
    """rho = sigma-tilde^2 / sigma^2, the information-loss factor (DPI: >= 1).

    `mean_match` takes the post-DA MMSE over Lem. 2's class (free intercept), the
    same class the solver's sigma-tilde-hat comes from, on the same SSR / (n - k):
    the numerator is a fit over a population sigma^2, and 1/n reads it low by k/n.
    """
    sigma_sq = sem.sigma_sq
    if sigma_sq <= 0.0:
        return None

    with preserve_rng():
        if y is None:  # X given without outcomes: rho needs its own draw
            X, y, _ = _draw(sem, n_samples)
        GX, _ = da(X)
        Phi = features(GX)
        fit = OLS(fit_intercept=mean_match).fit(Phi, y)
        residuals = y.flatten() - fit.predict(Phi).flatten()

    return float(
        residual_variance(residuals, np.asarray(Phi).reshape(len(Phi), -1).shape[1] + int(mean_match)) / sigma_sq
    )


def compute_oracle_parameters(
    sem,
    da,
    X: NDArray | None = None,
    y: NDArray | None = None,
    features: Callable | None = None,
    n_samples: int = CALIBRATION_SAMPLES,
    strategy: GammaStarStrategy = DEFAULT_GAMMA_STAR,
    mean_match: bool = False,
) -> OracleParameters:
    """Oracle parameters for one (SEM, DA) pair; budgets in the paper's units."""
    features = features or _identity

    if X is None:
        with preserve_rng():
            X, y, _ = _draw(sem, n_samples)

    return OracleParameters(
        gamma_star=gamma_star(sem, strategy=strategy),
        epsilon_star=epsilon_star(sem, da, X=X, features=features),
        gamma_z_star=gamma_z_star(sem, da, X=X, features=features),
        bias_sq=float(sem.bias_sq),
        sigma_sq=float(sem.sigma_sq),
        rho=_noise_ratio(sem, da, X, y, features, n_samples, mean_match=mean_match),
    )
