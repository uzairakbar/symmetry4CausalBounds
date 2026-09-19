"""
The perf sweeps: wall clock and solver stability along the epsilon grid.

Both run on the robustness sweep's own runner (`get_sweep_runner_cls("epsilon")`,
one experiment, serial): one data draw, one augmentation draw, one query set, the
same ratio grid, x-label and r = 1 line as the `epsilon` sweep figure.

Wall clock times what a method has to do at each grid point and cumulates it:
a fresh build + fit + predict at point 0, then a re-solve where the fitted program
reads a predict-time budget (`solves_on_epsilon`: the INV cone reads epsilon, a
T-as-IV constraint reads the T budget) and a `repad` (the last raw bounds
finalised at the new epsilon, no solve) everywhere else, each point the median of
`repeats` repeats, the total in units of one baseline PI solve (`normaliser`).
The grid is centred on the oracle budget, so point 0 is an UNDER-budget solve and
can be an infeasibility proof, which is cheaper than a solve; the cumulative curve
is what the figure reads, not the first increment. Seed var holds everything fixed and varies the conic backend
over the installed subset of BACKENDS at tight, comparable tolerances; the metric
is D(eps) of `solver_stability`, failures counted per (run, query) pair.
"""

import time
import warnings
from dataclasses import dataclass, field
from typing import Any

import cvxpy as cp
import numpy as np
from loguru import logger

from src.experiments.utils.constants import parse_method
from src.experiments.utils.model_fitting import fit_model
from src.methods.regression import LeastSquaresIterative, MomentConstrainedLeastSquares, TwoStageLeastSquaresIV
from src.methods.sensitivity_models import BoundedSA, PartialR2, SolveStatus

# the seed_var backends, in this order, filtered by what the venv has installed;
# every tolerance at 1e-8 so the three read the same problem to the same accuracy
BACKENDS: tuple[tuple[str, dict[str, float | int]], ...] = (
    ("CLARABEL", {"tol_gap_abs": 1e-8, "tol_gap_rel": 1e-8, "tol_feas": 1e-8}),
    ("ECOS", {"abstol": 1e-8, "reltol": 1e-8, "feastol": 1e-8}),
    ("SCS", {"eps_abs": 1e-8, "eps_rel": 1e-8, "max_iters": 100_000}),
)
# the point estimators whose fit is itself a conic solve (the moment-constrained ERM)
CONIC_FITS: tuple[str, ...] = ("ERM+IV", "DA+ERM+IV")


@dataclass
class PerfRecord:
    x: np.ndarray  # the epsilon grid
    results: dict[str, dict[str, np.ndarray]]  # metric -> method -> 2-D array
    failures: dict[str, np.ndarray] = field(default_factory=dict)  # method -> (n_steps,) int
    statuses: dict[str, np.ndarray] = field(default_factory=dict)  # method -> (R, n_steps, n_queries) int
    meta: dict[str, Any] = field(default_factory=dict)


def installed_backends() -> list[tuple[str, dict[str, float | int]]]:
    have = set(cp.installed_solvers())
    return [(name, opts) for name, opts in BACKENDS if name in have]


def budgets(runner, data) -> dict[str, Any]:
    """The builder kwargs `ParamSweepRunner.build_models` computes at experiment 0."""
    return dict(
        gamma=runner.fit_gamma(0),
        epsilon=runner.fit_epsilon(0, 0, data),
        epsilon_iv=runner.fit_epsilon_iv(0, 0, data),
        epsilon_iv_z=runner.fit_epsilon_iv_z(0, data),
        rho=runner.fit_rho(0, data),
        **runner.method_kwargs(0),
    )


def builders(runner, data, b=None) -> dict[str, Any]:
    b = budgets(runner, data) if b is None else b
    return runner.method_factory(**b) if runner.method_factory else runner.methods


def baseline_pi(runner, data, b=None) -> PartialR2:
    """A fresh baseline PI with the registry's PI kwargs, serial whatever the
    block's `n_jobs` says; it exists whether or not `PI` is in `methods`. `b` is
    the budgets dict, computed once by the caller when the build is timed."""
    b = budgets(runner, data) if b is None else b
    return PartialR2(
        gamma=b["gamma"],
        epsilon=b["epsilon"],
        pad=False,
        recalibrate=runner.recalibrate,
        clipy=runner.clipy,
        n_jobs=1,
        mean_match=runner.mean_match,
    )


def set_backend(model, backend) -> None:
    """Pin one `(name, opts)` conic backend on a model and on its branches; None
    restores the CLARABEL-then-ECOS chain."""
    for part in (model, getattr(model, "baseline", None), getattr(model, "augmented", None)):
        if isinstance(part, BoundedSA | LeastSquaresIterative | MomentConstrainedLeastSquares | TwoStageLeastSquaresIV):
            part.backend = backend


def _fit(runner, data, name, model):
    fit_model(
        model=model, method_name=name, hyperparameters=runner.hyperparameters, da=runner.get_da(0), **data.fit_arrays
    )
    return model


def _as_bounds(estimate) -> np.ndarray:
    """(n, 2) float; a point estimate is a degenerate interval."""
    estimate = np.asarray(estimate, dtype=float).reshape(len(estimate), -1)
    return np.repeat(estimate, 2, axis=1) if estimate.shape[1] == 1 else estimate[:, :2]


def _status(model, n) -> np.ndarray:
    status = getattr(model, "query_status", None)
    return np.full(n, SolveStatus.OK, dtype=int) if status is None else np.asarray(status, dtype=int)


def bounds_along(models: dict[str, Any], runner, data, x) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Per method (lower, upper, status), each (n_steps, n_queries): a predict at
    step 0 and wherever the program reads epsilon, a `repad` elsewhere, and the
    step-0 answer repeated for a point estimator, which never reads it."""
    out = {}
    for name, model in models.items():
        lower, upper, status = [], [], []
        bounds = None
        for i, r in enumerate(x):
            kw = runner.get_predict_kwargs(r, 0)
            if i == 0 or getattr(model, "solves_on_epsilon", False):
                bounds = _as_bounds(model.predict(data.X_test, **kw))
            elif hasattr(model, "repad"):
                bounds = _as_bounds(model.repad(kw["epsilon"]))
            lower.append(bounds[:, 0])
            upper.append(bounds[:, 1])
            status.append(_status(model, len(bounds)))
        out[name] = (np.array(lower), np.array(upper), np.array(status))
    return out


def normaliser(runner, data, repeats: int) -> tuple[list[float], float]:
    """Seconds of a single baseline PI solve, problem construction included: `repeats`
    fresh build + fit + predict calls after one warm-up call that is dropped;
    returns every call's seconds and the median of the kept ones. The oracle
    budgets are fitted once, outside the timer: they are the runner's, not the
    solve's."""
    kw = runner.get_predict_kwargs(runner.get_param_range()[0], 0)
    b = budgets(runner, data)
    seconds = []
    for _ in range(repeats + 1):
        start = time.perf_counter()
        _fit(runner, data, "PI", baseline_pi(runner, data, b)).predict(data.X_test, **kw)
        seconds.append(time.perf_counter() - start)
    return seconds, float(np.median(seconds[1:]))


def wall_clock(runner, data, x, repeats: int, seconds_per_solve: float):
    """Cumulative seconds along the grid per method, in baseline-solve equivalents:
    `{name: (n_steps, 1)}` and the raw increments `{name: (n_steps, repeats)}`."""
    build = builders(runner, data)
    results, increments = {}, {}
    for name in runner.methods:
        if name == "ATE":
            continue
        seconds = np.zeros((len(x), repeats))
        model = None
        for i, r in enumerate(x):
            kw = runner.get_predict_kwargs(r, 0)
            for k in range(repeats):
                if i == 0:
                    start = time.perf_counter()
                    model = _fit(runner, data, name, build[name]())
                    model.predict(data.X_test, **kw)
                elif getattr(model, "solves_on_epsilon", False):
                    start = time.perf_counter()
                    model.predict(data.X_test, **kw)
                elif hasattr(model, "repad"):
                    start = time.perf_counter()
                    model.repad(kw["epsilon"])
                else:
                    continue  # a point estimator: nothing to do after step 0
                seconds[i, k] = time.perf_counter() - start
        increments[name] = seconds
        results[name] = (np.cumsum(np.median(seconds, axis=1)) / seconds_per_solve)[:, None]
    return results, increments


def solver_stability(lower, upper, status, width_pi):
    """D(eps) per query. lower, upper, status: (R, n_steps, n_queries) over the backend runs r;
    width_pi: (R, n_queries), the baseline PI width per run. Returns the per-query terms
    (n_steps, n_queries), whose nanmean over queries is
        D(eps) = 1/(2|Q|) sum_x [sd_r L_r(x, eps) + sd_r U_r(x, eps)] / mean_r W_PI(x),
    and the failure count per step: (run, query) pairs with a non-OK status or lower > upper,
    excluded from both sd. A query with fewer than two surviving runs has no sd and drops out
    of the mean (ddof 1 leaves NaN there)."""
    lower, upper, status = (np.asarray(a, dtype=float) for a in (lower, upper, status))
    failed = (status != SolveStatus.OK) | ~np.isfinite(lower) | ~np.isfinite(upper) | (lower > upper)
    L = np.where(failed, np.nan, lower)
    U = np.where(failed, np.nan, upper)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        sd_l = np.nanstd(L, axis=0, ddof=1)
        sd_u = np.nanstd(U, axis=0, ddof=1)
        w = np.nanmean(np.asarray(width_pi, dtype=float), axis=0)
    terms = (sd_l + sd_u) / (2.0 * w[None, :])
    return terms, failed.sum(axis=(0, 2)).astype(int)


def seed_var(runner, data, x):
    """Every method's bounds along the grid under each installed backend, then D(eps).
    The PI family is fit once (its fit is solver-free) and re-solved per backend;
    ERM+IV and DA+ERM+IV are re-fitted per backend (their fit is the conic solve);
    ERM and DA+ERM are fit once and held fixed. Returns (terms, failures, statuses,
    backends)."""
    backends = installed_backends()
    if len(backends) < 2:
        logger.warning(f"seed_var: {len(backends)} conic backend(s) installed; D(eps) needs two or more.")
    b = budgets(runner, data)
    build = builders(runner, data, b)
    fixed = {
        name: _fit(runner, data, name, build[name]())
        for name in runner.methods
        if name != "ATE" and parse_method(name)[0] not in CONIC_FITS
    }
    pi = _fit(runner, data, "PI", baseline_pi(runner, data, b))

    runs = []
    for backend in backends:
        models = dict(fixed)
        for name in runner.methods:
            if parse_method(name)[0] in CONIC_FITS:
                model = build[name]()
                set_backend(model, backend)
                models[name] = _fit(runner, data, name, model)
        for model in (*models.values(), pi):
            set_backend(model, backend)
        runs.append((bounds_along(models, runner, data, x), bounds_along({"PI": pi}, runner, data, x)["PI"]))
    for model in (*fixed.values(), pi):
        set_backend(model, None)

    # W_PI at step 0: the baseline does not read epsilon
    width_pi = np.array([run_pi[1][0] - run_pi[0][0] for _, run_pi in runs])
    terms, failures, statuses = {}, {}, {}
    for name in runs[0][0]:
        lower = np.array([run[name][0] for run, _ in runs])
        upper = np.array([run[name][1] for run, _ in runs])
        status = np.array([run[name][2] for run, _ in runs])
        terms[name], failures[name] = solver_stability(lower, upper, status, width_pi)
        statuses[name] = status
    return terms, failures, statuses, backends


def perf_sweeps(runner, metrics, repeats: int = 3) -> PerfRecord:
    """Both perf sweeps on the runner's grid and its experiment-0 data."""
    x = np.asarray(runner.get_param_range(), dtype=float)
    data = runner.generate_data(0, x[0])
    n_queries = len(data.X_test)

    # first, so its warm-up call absorbs the process-level first-solve cost
    seconds, per_solve = normaliser(runner, data, repeats)
    logger.info(
        f"perf: baseline PI solve {per_solve:.3f} s over {n_queries} queries (calls {np.round(seconds, 3).tolist()})"
    )
    record = PerfRecord(
        x=x,
        results={},
        meta={
            "repeats": int(repeats),
            "n_experiments": int(runner.n_experiments),
            "normaliser_seconds": seconds,
            "seconds_per_solve": per_solve,
            "n_queries": int(n_queries),
            "xlabel": runner.xlabel,
            "vlines": tuple(runner.vlines),
        },
    )
    if "wall_clock" in metrics:
        record.results["wall_clock"], record.meta["increments_seconds"] = wall_clock(
            runner, data, x, repeats, per_solve
        )
    if "seed_var" in metrics:
        terms, record.failures, record.statuses, backends = seed_var(runner, data, x)
        record.results["seed_var"] = terms
        record.meta["backends"] = backends
    return record
