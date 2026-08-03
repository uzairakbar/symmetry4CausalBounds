"""
Utilities for fitting causal estimation models.
"""
from typing import Optional, Dict, Any


def fit_model(
    model,
    method_name: str,
    X,
    y,
    GX=None,
    G=None,
    X_base=None,
    y_base=None,
    hyperparameters: Optional[Dict[str, Any]] = None,
    **kwargs
):
    """
    Fit a causal estimation model with appropriate data.

    The key insight: Methods are the same class but used differently:
    - PI uses original data X
    - DA+PI uses augmented data GX
    - PI_INV uses both X and GX

    Args:
        model: Model instance to fit
        method_name: Name of the method ('PI', 'DA+PI', 'PI_INV', 'ERM', 'DA+ERM', 'ATE')
        X: Original treatment data
        y: Outcome data
        GX: Augmented treatment data (optional)
        G: Augmentation parameters (optional)
        X_base: Untiled X (m-sweep); baselines that are exactly tiling-invariant
            fit on it instead, to avoid a QR on the m-fold matrix
        y_base: Untiled y, paired with X_base
        hyperparameters: Training hyperparameters (optional)
        **kwargs: Additional arguments (e.g., pbar_manager, da)
    """
    if method_name == 'ATE':
        # ATE is computed analytically, no fitting required
        return

    # Prepare fit kwargs with hyperparameters
    fit_kwargs = {**(hyperparameters or {}), **kwargs}

    # baselines ignore GX: tiling them is exactly a no-op, so use the base copy
    X_solo = X if X_base is None else X_base
    y_solo = y if y_base is None else y_base

    # Dispatch based on method name to use correct data
    if method_name == 'PI':
        # PI uses original data only
        model.fit(X=X_solo, y=y_solo, **fit_kwargs)

    elif method_name == 'DA+PI':
        # DA+PI uses augmented data only
        model.fit(X=GX, y=y, **fit_kwargs)

    elif method_name == 'PI_INV':
        # PI_INV uses both original and augmented data
        model.fit(X=X, y=y, GX=GX, G=G, **fit_kwargs)

    elif method_name == 'PI_IV':
        # baseline IV PI: no instrument in the original data
        model.fit(X=X_solo, y=y_solo, Z=None, **fit_kwargs)

    elif method_name == 'DA+PI_IV':
        # DA+PI_IV uses both original and augmented data
        model.fit(X=GX, y=y, Z=G, **fit_kwargs)

    elif method_name in ('PI&DA+PI', 'PI&DA+PI_IV'):
        # intersections fit a baseline branch on X and a DA branch on GX
        model.fit(X=X, y=y, GX=GX, G=G, **fit_kwargs)

    elif method_name == 'ERM':
        # ERM uses original data
        model.fit(X=X_solo, y=y_solo, **fit_kwargs)
    
    elif method_name == 'DA+ERM':
        # DA+ERM uses augmented data
        model.fit(X=GX, y=y, **fit_kwargs)
    
    elif method_name == 'DA+IV':
        # DA+ERM uses augmented data
        model.fit(X=GX, y=y, Z=G, **fit_kwargs)
    
    else:
        # Fallback for any custom methods - pass everything
        model.fit(X=X, y=y, GX=GX, G=G, **fit_kwargs)