"""
Utilities for fitting causal estimation models.
"""

from typing import Any

import numpy as np

from .constants import parse_method


def instrument_columns(Z, n: int) -> np.ndarray:
    """`Z` as the (n, m) float array the solvers take; None or an empty array is
    (n, 0). The one spelling of an empty instrument set: `column_stack([G, Z])`
    is then G elementwise with an identical QR, and nothing downstream needs a
    None branch."""
    if Z is None:
        return np.zeros((n, 0))
    Z = np.asarray(Z, dtype=float)
    if Z.ndim != 2:
        Z = Z.reshape(n, -1) if Z.size else Z.reshape(n, 0)
    if len(Z) != n:
        raise ValueError(f"instrument has {len(Z)} rows, the design {n}.")
    return Z


def joint_instrument(G, Z) -> np.ndarray:
    """Z-tilde = (T, Z) of Asm. 3 for the DA+ methods: the translation amounts G
    first, then the real instrument. With Z (n, 0) this is exactly G (SS2.5)."""
    G = np.asarray(G).reshape(len(G), -1)
    return np.column_stack([G, instrument_columns(Z, len(G))])


def fit_model(
    model,
    method_name: str,
    X,
    y,
    GX=None,
    G=None,
    X_base=None,
    y_base=None,
    Z=None,
    Z_base=None,
    hyperparameters: dict[str, Any] | None = None,
    **kwargs,
):
    """
    Fit a causal estimation model with appropriate data.

    The key insight: Methods are the same class but used differently:
    - PI uses original data X
    - DA+PI uses augmented data GX
    - PI+INV uses both X and GX
    - the +IV methods add an instrument: non-DA ones see the real Z alone,
      DA+ ones the joint Z-tilde = (G, Z), and the intersection takes the raw
      Z and stacks G onto its DA branch itself
    - a DA+ IV method spelled `(Z)` (`parse_method`) sees the real Z alone,
      no G; bare and `(T,Z)` are the joint instrument

    Args:
        model: Model instance to fit
        method_name: Name of the method ('PI', 'DA+PI', 'PI+INV', 'ERM', 'DA+ERM', 'ATE', ...)
        X: Original treatment data
        y: Outcome data
        GX: Augmented treatment data (optional)
        G: Augmentation parameters (optional)
        X_base: Untiled X (m-sweep); baselines that are exactly tiling-invariant
            fit on it instead, to avoid a QR on the m-fold matrix
        y_base: Untiled y, paired with X_base
        Z: the real instrument, (n, m); None or (n, 0) is no instrument
        Z_base: Untiled Z, paired with X_base
        hyperparameters: Training hyperparameters (optional)
        **kwargs: Additional arguments (e.g., pbar_manager, da)
    """
    base, mode = parse_method(method_name)
    if base == "ATE":
        # ATE is computed analytically, no fitting required
        return

    # A prefit model was trained once on the FULL draw and handed to every variant.
    # Refitting it here would train it on whatever subset this step passes and break
    # the matched ERM/DA-ERM pairing the coupling depends on.
    if getattr(model, "prefit_", False):
        return

    # Prepare fit kwargs with hyperparameters
    fit_kwargs = {**(hyperparameters or {}), **kwargs}

    # baselines ignore GX: tiling them is exactly a no-op, so use the base copy
    X_solo = X if X_base is None else X_base
    y_solo = y if y_base is None else y_base
    Z = instrument_columns(Z, len(X))
    if X_base is not None and Z_base is None:
        # an untiled X without its untiled Z would fit PI+IV and IV with no
        # instrument and no warning; `SweepData` always derives one, so this is a
        # hand-written call and it must say what it means
        raise ValueError("X_base without Z_base: pass the untiled instrument beside the untiled design.")
    Z_solo = Z if X_base is None else instrument_columns(Z_base, len(X_base))
    # the DA+ IV methods' instrument: Z-tilde = (T, Z) by default, the real Z
    # alone in the (Z) mode
    z_da = Z if mode == "Z" else _joint(G, Z)

    # Dispatch based on the base name to use correct data
    if base == "PI":
        # PI uses original data only
        model.fit(X=X_solo, y=y_solo, **fit_kwargs)

    elif base == "DA+PI":
        # DA+PI uses augmented data only
        model.fit(X=GX, y=y, **fit_kwargs)

    elif base == "PI+INV":
        # PI+INV uses both original and augmented data
        model.fit(X=X, y=y, GX=GX, G=G, **fit_kwargs)

    elif base == "PI+IV":
        # the real instrument alone, never G; empty reduces it to PI exactly
        model.fit(X=X_solo, y=y_solo, Z=Z_solo, **fit_kwargs)

    elif base == "PI+INV+IV":
        # the INV cone needs GX, the IV cone the real Z; empty is PI+INV exactly
        model.fit(X=X, y=y, GX=GX, G=G, Z=Z, **fit_kwargs)

    elif base == "DA+PI+IV":
        # the DA ball on GX with the joint instrument Z-tilde = (T, Z), or Z alone
        model.fit(X=GX, y=y, Z=z_da, **fit_kwargs)

    elif base == "PI&DA+PI":
        # intersections fit a baseline branch on X and a DA branch on GX
        model.fit(X=X, y=y, GX=GX, G=G, **fit_kwargs)

    elif base == "PI&DA+PI+IV":
        # raw Z: the class hands it to its baseline and stacks G onto its DA
        # branch itself (or not, in the (Z) mode it was built for), so
        # pre-stacking here would give that branch [G, G, Z]
        model.fit(X=X, y=y, GX=GX, G=G, Z=Z, **fit_kwargs)

    elif base == "ERM":
        # ERM uses original data
        model.fit(X=X_solo, y=y_solo, **fit_kwargs)

    elif base == "DA+ERM":
        # DA+ERM uses augmented data
        model.fit(X=GX, y=y, **fit_kwargs)

    elif base == "IV":
        # 2SLS on the real instrument. With no instrument it would return W = 0 and
        # predict ybar at every query (regression.py); the config rejects that
        # upstream, and this stays loud in case a caller skips the config
        if Z_solo.shape[1] == 0:
            raise ValueError("IV needs an instrument; the instrument set is empty.")
        model.fit(X=X_solo, y=y_solo, Z=Z_solo, **fit_kwargs)

    elif base == "DA+IV":
        # 2SLS on the augmented data with the joint instrument, or Z alone in the
        # (Z) mode, where an empty set is the same silent W = 0 as for IV
        if z_da.shape[1] == 0:
            raise ValueError("DA+IV needs an instrument; the instrument set is empty.")
        model.fit(X=GX, y=y, Z=z_da, **fit_kwargs)

    else:
        # Fallback for any custom methods - pass everything
        model.fit(X=X, y=y, GX=GX, G=G, Z=Z, **fit_kwargs)


def _joint(G, Z):
    """Z-tilde for a DA+ method; with no translation amounts the real Z alone,
    which under an empty Z is no instrument, as before."""
    return Z if G is None else joint_instrument(G, Z)
