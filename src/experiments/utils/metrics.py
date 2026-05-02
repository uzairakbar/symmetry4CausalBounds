"""
Error metrics for evaluating causal estimation methods.
"""
import numpy as np
from numpy.typing import NDArray
from .constants import DEFAULT_NORMALIZE_ERROR


def _compute_squared_norm(x: NDArray) -> float:
    """Compute squared L2 norm divided by sample size."""
    return (x**2).mean()


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
    
    # Convert point estimates to intervals
    if estimate.shape[-1] == 1:
        estimate = np.repeat(estimate, 2, axis=1)
    
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
    approx_sq_error = errors[:, None].mean()
    
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
    
    # Convert point estimates to intervals
    if estimate.shape[-1] == 1:
        estimate = np.repeat(estimate, 2, axis=1)
    
    difference = estimate - estimand
    squared_errors = difference**2
    worst_sq_error = squared_errors.max(axis=1).mean()
    
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
    
    # Convert point estimates to intervals
    if estimate.shape[-1] == 1:
        estimate = np.repeat(estimate, 2, axis=1)
    
    width = (estimate[:, 1] - estimate[:, 0]).mean()
    
    assert np.all(width >= 0), \
        'Upper bound should be greater than lower bound for all samples.'
    
    if normalize:
        width = width / (width + np.std(estimand))
    
    return width


def augmentation_strength_metric(Sigma_X: NDArray, Sigma_GX: NDArray) -> float:
    """
    Compute a scalar metric representing the strength of data augmentation.
    
    Current implementation: Trace( (Sigma_GX - Sigma_X) @ Sigma_X^-1 )
    
    Args:
        Sigma_X: Covariance/Second-moment of original data (Ambient Space)
        Sigma_GX: Covariance/Second-moment of augmented data (Ambient Space)
        
    Returns:
        Scalar strength metric.
    """
    Delta = Sigma_GX - Sigma_X
    
    try:
        # Use pseudo-inverse for stability against rank-deficient ambient spaces
        Sigma_X_inv = np.linalg.pinv(Sigma_X)
        metric = np.trace(Delta @ Sigma_X_inv)
        return float(metric)
    except Exception:
        return np.nan