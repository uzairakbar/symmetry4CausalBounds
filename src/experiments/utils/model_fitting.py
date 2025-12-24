"""
Utilities for fitting causal estimation models.
"""
from typing import Optional, Dict, Any


def fit_model(
    model,
    method_name: str,
    X,
    y,
    GX,
    G=None,
    hyperparameters: Optional[Dict[str, Any]] = None,
    da=None,
    pbar_manager=None
):
    """
    Fit a causal estimation model with appropriate data and hyperparameters.
    
    Handles different method requirements:
    - PI: Fits on original data X
    - DA+PI: Fits on augmented data GX
    - INV+PI: Fits with both X and GX (invariance constraints)
    - ERM: Fits on X with optional progress bar
    - DA+ERM: Fits on GX with optional progress bar
    - ATE: No fitting required (analytical)
    
    Args:
        model: Model instance to fit
        method_name: Name of the method ('PI', 'DA+PI', 'INV+PI', 'ERM', 'DA+ERM', 'ATE')
        X: Original treatment data
        y: Outcome data
        GX: Augmented treatment data
        G: Augmentation parameters (optional)
        hyperparameters: Training hyperparameters (learning rate, epochs, etc.)
        da: Data augmentor instance (optional)
        pbar_manager: Progress bar manager (optional)
    """
    if method_name == 'ATE':
        # ATE is computed analytically, no fitting required
        return
    
    fit_kwargs = {**(hyperparameters or {})}
    
    # Add progress bar if available and method supports it
    if pbar_manager and method_name in ['ERM', 'DA+ERM']:
        fit_kwargs['pbar_manager'] = pbar_manager
    
    # Fit based on method requirements
    if method_name == 'PI':
        model.fit(X=X, y=y, **fit_kwargs)
    
    elif method_name == 'DA+PI':
        model.fit(X=GX, y=y, **fit_kwargs)
    
    elif method_name == 'INV+PI':
        # Invariance-constrained methods need both X and GX
        model.fit(X=X, y=y, GX=GX, G=G, **fit_kwargs)
    
    elif method_name == 'ERM':
        model.fit(X=X, y=y, **fit_kwargs)
    
    elif method_name == 'DA+ERM':
        model.fit(X=GX, y=y, **fit_kwargs)
    
    else:
        # Fallback for custom methods - pass everything available
        model.fit(X=X, y=y, GX=GX, G=G, **fit_kwargs)