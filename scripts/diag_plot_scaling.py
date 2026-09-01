"""Diagnose the sweep/scatter/perf rescale rules: what do they pick, and on what?

Prints the resolved scale / linear_width / limits for every checked-in sweep pkl, so
PLOT_CONFIGS entries get chosen from measurements rather than guesses, then renders
the pathological cases the artifacts do not cover. The three plot functions swallow
exceptions, so this installs an ERROR sink -- without it a total failure passes green.

    python scripts/diag_plot_scaling.py
"""

import glob
import os
import pickle
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger  # noqa: E402

from src.experiments.configs import METRIC_SPECS, PARAM_SPECS  # noqa: E402
from src.experiments.utils import plotting  # noqa: E402
from src.experiments.utils.constants import (
    ARTIFACTS_DIRECTORY,
    PLOT_CONFIGS,  # noqa: E402
)
from src.experiments.utils.data_operations import bootstrap  # noqa: E402

DIAG_EXPERIMENT = "_diag"
SEEDS = (0, 1, 2, 3, 4)


# --------------------------------------------------------------- error capture
_errors = []
logger.add(lambda m: _errors.append(m), level="ERROR")


def _means(results, key, seed):
    """The series as PLOTTED: bootstrapped, then nanmean over replicates."""
    raw = {name: record[key] for name, record in results.items() if key in record}
    if not raw:
        return []
    series = []
    for values in bootstrap(raw, seed=seed).values():
        clean = np.array(values, dtype=np.float64)
        clean[np.isinf(clean)] = np.nan
        series.append(np.nanmean(clean, axis=1))
    return series


def report_real():
    """Table 1: what the rules resolve to on the pkls that already exist."""
    paths = sorted(glob.glob(f"{ARTIFACTS_DIRECTORY}/*/sweep/*_results.pkl"))
    if not paths:
        print("no sweep pkls under artifacts/; skipping the measured table\n")
        return

    print(f"=== resolved on the checked-in sweeps (median over {len(SEEDS)} bootstrap seeds) ===")
    header = (
        f"{'experiment':<16}{'param':<8}{'metric':<14}{'range':>22} "
        f"{'scale':>7} {'knee':>10} {'limits':>22} {'clip':>6} {'seed-jitter':>12}"
    )
    print(header)
    print("-" * len(header))

    for path in paths:
        experiment = path.split(os.sep)[-3]
        param = os.path.basename(path).replace("_results.pkl", "")
        if param not in PARAM_SPECS:
            continue
        with open(path, "rb") as handle:
            results = pickle.load(handle)  # noqa: S301 - our own artifacts, no untrusted input

        for metric, spec in METRIC_SPECS.items():
            if spec.perf_only:
                continue
            per_seed = [_means(results, spec.key, seed) for seed in SEEDS]
            if not per_seed[0]:
                continue

            tops, rows = [], []
            for series in per_seed:
                limits = plotting._limits(series)
                if limits is None:
                    continue
                tops.append(limits[1])
                rows.append((series, limits))
            if not rows:
                continue

            series, limits = rows[len(rows) // 2]
            cfg = plotting._plot_config(experiment, f"{param}_{metric}")
            scale, kwargs = plotting._resolve_scale(spec.yscale, plotting._finite(*series), cfg, "y", limits)
            pooled = plotting._finite(*series)
            clipped = int((pooled > limits[1]).sum())
            jitter = max(tops) / min(tops) if min(tops) > 0 else float("nan")
            knee = next(iter(kwargs.values()), float("nan"))

            print(
                f"{experiment:<16}{param:<8}{metric:<14}"
                f"{pooled.min():>10.4g}..{pooled.max():<11.4g} {scale:>7} "
                f"{knee:>10.4g} {limits[0]:>10.4g}..{limits[1]:<11.4g} "
                f"{clipped:>6d} {jitter:>11.2f}x"
            )
    print()


def synthetic_cases():
    """The shapes the artifacts do not contain."""
    x = np.linspace(0.1, 10.0, 12)
    steps = len(x)

    def replicate(values):
        return np.repeat(np.asarray(values, dtype=float)[:, None], 4, axis=1)

    return {
        "all_nan": {"PI": replicate(np.full(steps, np.nan))},
        "flat": {"PI": replicate(np.full(steps, 2.0))},
        "zero_crossing": {"PI": replicate(np.linspace(-5, 5, steps)), "DA+PI": replicate(np.zeros(steps))},
        "six_decades": {"PI": replicate(np.logspace(-3, 3, steps)), "DA+PI": replicate(np.full(steps, 1e-3))},
        "single_method": {"PI": replicate(np.linspace(1, 2, steps))},
        "huge_ci": {"PI": np.column_stack([np.linspace(1, 2, steps) + 100 * s for s in range(4)])},
        "partial_nan": {
            "PI": replicate([np.nan] * 6 + list(np.linspace(1, 5, 6))),
            "DA+PI": replicate(np.linspace(0.001, 0.002, steps)),
        },
    }, x


def render():
    """Table 2: every plot function against every pathological case."""
    cases, x = synthetic_cases()
    print(f"=== synthetic cases (rendered to {ARTIFACTS_DIRECTORY}/{DIAG_EXPERIMENT}/) ===")

    for name, y_results in cases.items():
        before = len(_errors)
        for yscale in ("linear", "log", "asinh"):
            plotting.create_sweep_plot(
                x,
                {k: v.copy() for k, v in y_results.items()},
                xlabel=name,
                ylabel="y",
                yscale=yscale,
                experiment=DIAG_EXPERIMENT,
                fname=f"{name}_{yscale}",
            )
        record = {k: {"a": v, "b": v} for k, v in y_results.items()}
        plotting.create_scatter_plot(
            record,
            x,
            metric_x="a",
            metric_y="b",
            xlabel="a",
            ylabel="b",
            experiment=DIAG_EXPERIMENT,
            fname=f"scatter_{name}",
            crosshairs=True,
        )
        print(f"  {name:<16} {'ok' if len(_errors) == before else 'ERRORS'}")

    perf = {
        "PI": {"rates": np.array([0.1, 0.2, 0.6, 0.1]), "wall_clock": 0.5, "seed_var": 0.02},
        "DA+PI": {"rates": np.array([0.0, 0.0, 0.9, 0.1]), "wall_clock": 1e-4, "seed_var": 0.5},
    }
    for overlays in ([], ["wall_clock"], ["wall_clock", "seed_var"]):
        before = len(_errors)
        plotting.create_perf_plot(perf, overlay_metrics=overlays, experiment=DIAG_EXPERIMENT)
        print(f"  {'perf ' + (','.join(overlays) or 'no-overlay'):<16} {'ok' if len(_errors) == before else 'ERRORS'}")
    print()


def verdicts():
    print("=== verdicts ===")
    cases, _ = synthetic_cases()

    # every series keeps its median inside the limits
    blanked = []
    for name, y_results in cases.items():
        series = [np.nanmean(np.where(np.isinf(v), np.nan, v), axis=1) for v in y_results.values()]
        limits = plotting._limits(series)
        if limits is None:
            continue
        for method, values in zip(y_results, series):
            finite = plotting._finite(values)
            if len(finite) and not (limits[0] <= np.median(finite) <= limits[1]):
                blanked.append(f"{name}/{method}")
    print(f"  no series blanked:        {'PASS' if not blanked else 'FAIL ' + str(blanked)}")

    # a resolved log scale never coexists with a non-positive value
    bad_log = []
    for name, y_results in cases.items():
        series = [np.nanmean(np.where(np.isinf(v), np.nan, v), axis=1) for v in y_results.values()]
        pooled = plotting._finite(*series)
        for requested in ("linear", "log", "asinh"):
            scale, _ = plotting._resolve_scale(requested, pooled, {}, "y", plotting._limits(series))
            if scale == "log" and len(pooled) and np.any(pooled <= 0):
                bad_log.append(f"{name}:{requested}")
    print(f"  log never over non-pos:   {'PASS' if not bad_log else 'FAIL ' + str(bad_log)}")

    print(f"  no logged plot failure:   {'PASS' if not _errors else f'FAIL ({len(_errors)})'}")
    for message in _errors[:5]:
        print(f"      {str(message).strip()[:160]}")


def stale_ids():
    """PLOT_CONFIGS ids nothing can consume -- typos stay inert otherwise."""
    valid = {"perf"}
    for param in PARAM_SPECS:
        for metric, spec in METRIC_SPECS.items():
            if spec.perf_only:
                continue
            valid.add(f"{param}_{metric}")
            for other in METRIC_SPECS:
                valid.add(f"scatter_{param}_{metric}_vs_{other}")

    stale = [
        f"{experiment}.{plot_id}"
        for experiment, plots in PLOT_CONFIGS.items()
        for plot_id in plots
        if plot_id not in valid
    ]
    print(f"  no stale PLOT_CONFIGS id: {'PASS' if not stale else 'FAIL ' + str(stale)}")


if __name__ == "__main__":
    report_real()
    render()
    verdicts()
    stale_ids()
