"""
Imbens-Manski confidence intervals around the sweep bounds (App. E's gamma_n, put
back as sampling error).

Per sweep cell, B nonparametric bootstrap replicates of the FITTED rows are refit
and re-solved on the cell's queries at the cell's budgets; per query the replicate
spread gives s_L and s_U, and the interval [L - C s_L, U + C s_U] at the `im-ci`
level replaces the finalised bounds before the metrics read them. Only
`ParamSweepRunner.run` calls this module.
"""

import numpy as np
from joblib import Parallel, delayed, parallel_config
from loguru import logger
from numpy.typing import NDArray
from scipy.special import ndtr, ndtri
from threadpoolctl import threadpool_limits

from src.methods.sensitivity_models import SolveStatus

from .metrics import _as_interval
from .model_fitting import fit_model

# the fit arrays one row index applies to (tiled, m n rows on the m sweep) and the
# untiled base group the baselines fit on instead (m sweep only, `X_solo`)
TILED_KEYS: tuple[str, ...] = ("X", "y", "GX", "G", "Z")
BASE_KEYS: tuple[str, ...] = ("X_base", "y_base", "Z_base")
# halvings of the bracket [z_{1-alpha}, z_{1-alpha/2}]: a few units wide at most, so
# 64 halvings reach machine precision
BISECTION_STEPS: int = 64


def imbens_manski_critical(delta, sigma, level: float) -> NDArray:
    """The Imbens-Manski critical value C per query: the root of

        Phi(C + delta / sigma) - Phi(-C) = level / 100,

    which runs from z_{1-alpha/2} at delta = 0 (a point: the ordinary two-sided
    CI) down to z_{1-alpha} as delta / sigma grows (the two ends are then
    one-sided each). The left side is increasing in C, so a vectorised bisection
    on that bracket finds it with no per-query loop. `sigma == 0` is the
    `delta / sigma = inf` branch, taken without dividing (C is then irrelevant:
    both SEs are zero); a NaN delta is read as 0, whose C is the larger one. C is
    floored at 0 so the CI never narrows the raw interval (below a 50% level
    z_{1-alpha} is negative)."""
    alpha = 1.0 - float(level) / 100.0
    delta = np.asarray(delta, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    delta, sigma = np.broadcast_arrays(np.nan_to_num(np.maximum(delta, 0.0), nan=0.0), sigma)
    positive = sigma > 0
    d = np.where(positive, delta / np.where(positive, sigma, 1.0), np.inf)

    lower = np.full(d.shape, ndtri(1.0 - alpha))
    upper = np.full(d.shape, ndtri(1.0 - alpha / 2.0))
    for _ in range(BISECTION_STEPS):
        middle = 0.5 * (lower + upper)
        above = ndtr(middle + d) + ndtr(middle) - 1.0 >= 1.0 - alpha
        upper = np.where(above, middle, upper)
        lower = np.where(above, lower, middle)
    return np.maximum(0.5 * (lower + upper), 0.0)


def _bootstrap_se(values: NDArray) -> NDArray:
    """ddof-1 SD over the finite replicates, per query (axis 0); 0 where fewer
    than 2 are finite: the SE is then not estimable and that end does not move."""
    finite = np.isfinite(values)
    count = finite.sum(axis=0)
    filled = np.where(finite, values, 0.0)
    mean = filled.sum(axis=0) / np.maximum(count, 1)
    squares = np.where(finite, (values - mean) ** 2, 0.0).sum(axis=0)
    return np.where(count >= 2, np.sqrt(squares / np.maximum(count - 1, 1)), 0.0)


def imbens_manski_bounds(estimate: NDArray, replicate_bounds: NDArray, level: float) -> NDArray:
    """The IM-CI (n, 2) around the finalised bounds `estimate` (n, 2), from the
    cell's replicate bounds (B, n, 2) at `level` percent.

    A replicate NaN (not OK on that query) is skipped in the SE; with fewer than 2
    valid replicates the SE is 0 and the CI is the raw interval. A raw NaN stays
    NaN: an empty set has no sampling error to add. Wherever a raw bound is
    finite the CI contains it, by construction."""
    estimate = np.asarray(estimate, dtype=float)
    replicate_bounds = np.asarray(replicate_bounds, dtype=float)
    lower, upper = estimate[:, 0], estimate[:, 1]
    se_lower = _bootstrap_se(replicate_bounds[..., 0])
    se_upper = _bootstrap_se(replicate_bounds[..., 1])
    # floating noise can leave upper a hair under lower; the solvers give min <= max
    critical = imbens_manski_critical(upper - lower, np.maximum(se_lower, se_upper), level)
    return np.column_stack([lower - critical * se_lower, upper + critical * se_upper])


def replicate_rows(n_rows: int, n_base: int | None, replicates: int, seed) -> tuple[NDArray, NDArray | None]:
    """The iid row indices of every replicate, drawn in a fixed order: per
    replicate `rows` over the tiled group, then `base_rows` over the base group
    when there is one (`n_base` is None otherwise, and so is `base_rows`). Each
    method thereby resamples exactly the rows it is fitted on (SS4.2); the two
    draws are independent, since no method consumes both groups."""
    rng = np.random.default_rng(seed)
    rows, base_rows = [], []
    for _ in range(replicates):
        rows.append(rng.integers(0, n_rows, n_rows))
        if n_base is not None:
            base_rows.append(rng.integers(0, n_base, n_base))
    return np.asarray(rows), (None if n_base is None else np.asarray(base_rows))


def resample_fit_arrays(fit_arrays: dict, rows: NDArray, base_rows: NDArray | None) -> dict:
    """`SweepData.fit_arrays` at one replicate's rows: ONE index over the tiled
    group, so row i of GX stays the augmentation of row i of X, and `base_rows`
    over the base group (None when there is no base group). None stays None."""
    out = {key: (None if fit_arrays.get(key) is None else fit_arrays[key][rows]) for key in TILED_KEYS}
    for key in BASE_KEYS:
        value = fit_arrays.get(key)
        out[key] = None if base_rows is None or value is None else value[base_rows]
    return out


def _replicate(
    builders, name, arrays, rows, base_rows, X_test, predict_kwargs, hyperparameters, da, pad_tolerance=0.0
) -> NDArray:
    """The loky worker: ONE method refit on one replicate's rows, then predicted at
    every step in step order (the models are stateful in `epsilon` / `recalibrate`,
    as in the production loop). Returns (n_steps, n_queries, 2), NaN on every
    query that step did not solve OK."""
    # an empty intersection or a budget line would otherwise print B times a cell
    logger.disable("src.methods")
    try:
        # one BLAS thread in the fit too, so the bounds do not depend on the pool size
        with threadpool_limits(limits=1):
            model = builders[name]()
            if hasattr(model, "n_jobs"):
                model.n_jobs = 1  # the intersections build their branches from it at fit
            if pad_tolerance and hasattr(model, "pad_tolerance"):
                model.pad_tolerance = pad_tolerance  # the point model's pad, branches included
            fit_model(
                model=model,
                method_name=name,
                hyperparameters=hyperparameters,
                da=da,
                **resample_fit_arrays(arrays, rows, base_rows),
            )
            steps = []
            for kwargs in predict_kwargs:
                bounds = _as_interval(np.asarray(model.predict(X_test, **kwargs), dtype=float)).copy()
                status = getattr(model, "query_status", None)
                if status is not None:
                    bounds[np.asarray(status) != SolveStatus.OK] = np.nan
                steps.append(bounds)
        return np.stack(steps)
    finally:
        # n_jobs = 1 runs this in the parent: give it its logging back
        logger.enable("src.methods")


def bootstrap_bounds(
    builders,
    method_names,
    data,
    predict_kwargs,
    replicates: int,
    seed,
    n_jobs: int = 1,
    hyperparameters=None,
    da=None,
    pad_tolerance: float = 0.0,
) -> dict[str, NDArray]:
    """Replicate bounds of one sweep cell: {name: (n_steps, B, n_queries, 2)}.

    `builders` are the cell's method builders at the cell's budgets (held fixed
    across replicates), `data` its `SweepData`, `predict_kwargs` one dict per step
    (one entry on a data-varying sweep, every step on a data-constant one). The
    task is one (replicate, method), dispatched as ONE flat list in `method_names`
    order, method-major, so a caller that passes the slowest method first leaves a
    short tail. Every index is drawn here, before dispatch, so the replicates do not
    depend on the scheduling or on `n_jobs`; the models inside solve serially, and
    the pool is joined before this returns. `pad_tolerance` is set on every replicate
    model as the runner sets it on the point models."""
    names = [name for name in method_names if name != "ATE"]
    arrays = data.fit_arrays
    n_base = None if arrays.get("X_base") is None else len(arrays["X_base"])
    rows, base_rows = replicate_rows(len(arrays["X"]), n_base, replicates, seed)
    tasks = [(name, b) for name in names for b in range(replicates)]
    with parallel_config(backend="loky", n_jobs=n_jobs, inner_max_num_threads=1):
        out = Parallel()(
            delayed(_replicate)(
                builders,
                name,
                arrays,
                rows[b],
                None if base_rows is None else base_rows[b],
                data.X_test,
                predict_kwargs,
                hyperparameters,
                da,
                pad_tolerance,
            )
            for name, b in tasks
        )
    stacked = {name: [None] * replicates for name in names}
    for (name, b), bounds in zip(tasks, out, strict=True):
        stacked[name][b] = bounds
    # B arrays of (n_steps, n_queries, 2), stacked on the replicate axis
    return {name: np.stack(stacked[name], axis=1) for name in names}
