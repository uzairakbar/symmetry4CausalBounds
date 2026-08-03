"""
Error metrics for evaluating causal estimation methods.
"""
import numpy as np
from dataclasses import dataclass
from numpy.typing import NDArray

from src.methods.sensitivity_models import SolveStatus
from .constants import DEFAULT_NORMALIZE_ERROR


def _as_interval(estimate: NDArray) -> NDArray:
    """Point estimates become degenerate intervals."""
    if estimate.shape[-1] == 1:
        return np.repeat(estimate, 2, axis=1)
    return estimate


def _compute_squared_norm(x: NDArray) -> float:
    """Compute squared L2 norm divided by sample size."""
    return np.nanmean(x**2)


def estimation_error(
    estimand: NDArray,
    estimate: NDArray,
    normalize: bool = DEFAULT_NORMALIZE_ERROR,
) -> float:
    """
    Compute estimation error between estimand and estimate.
    
    Args:
        estimand: Ground truth target f or f(x)
        estimate: Hypothesis h or h(x)
        normalize: Whether to normalize by baseline error
        
    Returns:
        Squared error, optionally normalized
    """
    sq_error = _compute_squared_norm(estimate - estimand)
    
    if normalize:
        baseline = _compute_squared_norm(estimand)
        sq_error = sq_error / (sq_error + baseline)
    
    return sq_error


def approximation_error(
    estimand: NDArray,
    estimate: NDArray,
    normalize: bool = DEFAULT_NORMALIZE_ERROR,
) -> float:
    """
    Compute approximation error for interval estimates.
    
    For points inside the interval, error is 0.
    For points outside, error is squared distance to nearest bound.
    
    Args:
        estimand: Ground truth target f or f(x)
        estimate: Interval estimates [lower, upper] or point estimates
        normalize: Whether to normalize by baseline error
        
    Returns:
        Approximation error
    """
    assert estimate.ndim >= estimand.ndim, \
        f'Estimate dimension {estimate.ndim} less than estimand dimension {estimand.ndim}.'
    assert estimate.shape[0] == estimand.shape[0], \
        f'Estimate sample size {estimate.shape[0]} != estimand sample size {estimand.shape[0]}.'
    
    estimate = _as_interval(estimate)
    
    lower_bound = estimate[:, 0]
    upper_bound = estimate[:, 1]
    estimand_flat = estimand.squeeze()
    
    # Check if points are inside intervals
    inside_interval = (estimand_flat >= lower_bound) & (estimand_flat <= upper_bound)
    
    # For points outside, compute squared distance to nearest bound
    distance_squared = np.minimum(
        (lower_bound - estimand_flat)**2,
        (upper_bound - estimand_flat)**2
    )
    
    # Combine: 0 if inside, distance_squared if outside
    errors = np.where(inside_interval, 0, distance_squared)
    approx_sq_error = np.nanmean(errors[:, None])
    
    if normalize:
        baseline = estimation_error(estimand, np.zeros_like(estimand), normalize=False)
        approx_sq_error = approx_sq_error / (approx_sq_error + baseline)
    
    return approx_sq_error


def worst_error(
    estimand: NDArray,
    estimate: NDArray,
    normalize: bool = DEFAULT_NORMALIZE_ERROR,
) -> float:
    """
    Compute worst-case error for interval estimates.
    
    Takes the maximum squared error across both bounds.
    
    Args:
        estimand: Ground truth target f or f(x)
        estimate: Interval estimates [lower, upper] or point estimates
        normalize: Whether to normalize by baseline error
        
    Returns:
        Worst-case squared error
    """
    assert estimate.ndim >= estimand.ndim, \
        f'Estimate dimension {estimate.ndim} less than estimand dimension {estimand.ndim}.'
    assert estimate.shape[0] == estimand.shape[0], \
        f'Estimate sample size {estimate.shape[0]} != estimand sample size {estimand.shape[0]}.'
    
    estimate = _as_interval(estimate)
    
    difference = estimate - estimand
    squared_errors = difference**2
    worst_sq_error = np.nanmean(squared_errors.max(axis=1))
    
    if normalize:
        baseline = estimation_error(estimand, np.zeros_like(estimand), normalize=False)
        worst_sq_error = worst_sq_error / (worst_sq_error + baseline)
    
    return worst_sq_error


def interval_width(
    estimand: NDArray,
    estimate: NDArray,
    normalize: bool = DEFAULT_NORMALIZE_ERROR,
) -> float:
    """
    Compute average width of interval estimates.
    
    Args:
        estimand: Ground truth (used for normalization only)
        estimate: Interval estimates [lower, upper] or point estimates
        normalize: Whether to normalize by standard deviation of estimand
        
    Returns:
        Average interval width
    """
    assert estimate.ndim >= estimand.ndim, \
        f'Estimate dimension {estimate.ndim} less than estimand dimension {estimand.ndim}.'
    assert estimate.shape[0] == estimand.shape[0], \
        f'Estimate sample size {estimate.shape[0]} != estimand sample size {estimand.shape[0]}.'
    
    estimate = _as_interval(estimate)
    
    widths = estimate[:, 1] - estimate[:, 0]
    width = np.nanmean(widths)
    
    assert np.all(widths[~np.isnan(widths)] >= 0), \
        'Upper bound should be greater than lower bound for all samples.'
    
    if normalize:
        width = width / (width + np.std(estimand))
    
    return width


def coverage(
    estimand: NDArray,
    estimate: NDArray,
    normalize: bool = DEFAULT_NORMALIZE_ERROR,
) -> float:
    """
    Coverage rate: mean 1{lower <= truth <= upper}.

    NaN bounds (solver failure / infeasible / empty intersection) count as
    NOT covered, so this stays consistent with the 4-way perf split.
    Point estimators give a degenerate interval, hence coverage ~ 0.

    Args:
        estimand: Ground truth target f or f(x)
        estimate: Interval estimates [lower, upper] or point estimates
        normalize: Unused; kept for a uniform metric signature

    Returns:
        Fraction of queries whose interval contains the truth
    """
    assert estimate.shape[0] == estimand.shape[0], \
        f'Estimate sample size {estimate.shape[0]} != estimand sample size {estimand.shape[0]}.'

    estimate = _as_interval(estimate)
    estimand_flat = estimand.squeeze()

    covered = (estimand_flat >= estimate[:, 0]) & (estimand_flat <= estimate[:, 1])
    return float(np.mean(np.where(np.isnan(estimate).any(axis=1), False, covered)))


def rho_hat(X: NDArray, GX: NDArray, y: NDArray) -> float:
    """
    Information-loss factor rho = sigma-tilde^2 / sigma^2 measured on a sample:
    MSE of OLS on GX over MSE of OLS on X. >= 1 by the DPI.

    Args:
        X: Original data in the feature space the methods use
        GX: Augmented data in the same space
        y: Outcomes

    Returns:
        rho_hat, or NaN if the baseline MSE vanishes
    """
    def mse(A):
        residuals = y.flatten() - A @ np.linalg.lstsq(A, y.flatten(), rcond=None)[0]
        return float(np.mean(residuals ** 2))

    denominator = mse(X)
    if denominator <= 0.0:
        return np.nan
    return mse(GX) / denominator


# =============================================================================
# PER-(METHOD, STEP, EXPERIMENT) RECORD
# =============================================================================

# status_counts layout; precedence is first match wins
STATUS_CATEGORIES = ('solver_failure', 'infeasible',
                     'feasible_and_covers', 'feasible_and_noncovering')


@dataclass
class QueryEval:
    """All metrics for one fit+predict, computed in a single pass."""
    approximation_error: float
    worst_error: float
    interval_width: float
    coverage: float
    wall_clock: float               # predict seconds per query
    status_counts: NDArray          # counts over STATUS_CATEGORIES, sums to n


def evaluate_queries(
    estimand: NDArray,
    estimate: NDArray,
    statuses: NDArray = None,
    elapsed: float = 0.0,
) -> QueryEval:
    """
    Build the full record for one (method, step, experiment).

    Args:
        estimand: Ground truth f(x) per query
        estimate: Interval estimates [lower, upper] or point estimates
        statuses: Per-query SolveStatus ints; None means all-OK (point estimators)
        elapsed: Wall-clock seconds for the whole predict call

    Returns:
        QueryEval
    """
    interval = _as_interval(estimate)
    n_queries = len(interval)

    if statuses is None:
        statuses = np.full(n_queries, SolveStatus.OK, dtype=int)
    statuses = np.asarray(statuses, dtype=int)

    estimand_flat = estimand.squeeze()
    covered = (
        (estimand_flat >= interval[:, 0]) & (estimand_flat <= interval[:, 1])
        & ~np.isnan(interval).any(axis=1)
    )

    # mutually exclusive, ordered: failure -> infeasible -> covers -> non-covering
    failure = statuses == SolveStatus.FAILURE
    infeasible = ~failure & (statuses == SolveStatus.INFEASIBLE)
    feasible = ~failure & ~infeasible
    counts = np.array([
        failure.sum(),
        infeasible.sum(),
        (feasible & covered).sum(),
        (feasible & ~covered).sum(),
    ], dtype=int)

    return QueryEval(
        approximation_error=approximation_error(estimand, estimate),
        worst_error=worst_error(estimand, estimate),
        interval_width=interval_width(estimand, estimate),
        coverage=coverage(estimand, estimate),
        wall_clock=elapsed / max(n_queries, 1),
        status_counts=counts,
    )


def trace_S_over_k(X: NDArray, GX: NDArray, keep: float = 1.0) -> float:
    """
    Augmentation strength as tr(S)/k (Prop. 2), where S = Sigma_GX^-1 Sigma_X
    is the variance-shift operator. Sharpening iff tr(S)/k <= 1/rho.

    Args:
        X: Original data in the feature space the methods use
        GX: Augmented data in the same space
        keep: Fraction of Sigma_GX's variance retained before inverting. 1.0 (the
            default) inverts the whole spectrum, which is right at the low d the
            linear experiments use. In a HIGH-dimensional feature space the
            near-null eigendirections of Sigma_GX are noise and 1/w blows them up:
            at d=588, n=60k two IDENTICALLY distributed samples score ~10 instead
            of 1. Pass 0.999 there.

    Returns:
        tr(S)/k, in (0, 1] for variance-inflating DA.
    """
    X = X - X.mean(axis=0)
    GX = GX - GX.mean(axis=0)
    k = X.shape[1]

    Sigma_X = X.T @ X / len(X)
    Sigma_GX = GX.T @ GX / len(GX)

    try:
        if keep >= 1.0:
            # pseudo-inverse for stability against rank-deficient feature spaces
            S = np.linalg.pinv(Sigma_GX) @ Sigma_X
            return float(np.trace(S) / k)

        w, V = np.linalg.eigh(Sigma_GX)
        order = np.argsort(w)[::-1]
        w, V = w[order], V[:, order]
        positive = w > max(w.max(), 1e-30) * 1e-10
        w, V = w[positive], V[:, positive]
        rank = int(np.searchsorted(np.cumsum(w) / w.sum(), keep) + 1)
        w, V = w[:rank], V[:, :rank]

        whiten = V / np.sqrt(w)
        return float(np.trace(whiten.T @ Sigma_X @ whiten) / max(len(w), 1))
    except Exception:
        return np.nan