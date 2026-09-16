"""
Aggregate the sweep and perf pkls under `artifacts/` into grids under `artifacts/aggregate/`.

    python -m src.aggregate [--artifacts ./artifacts] [--out ./artifacts/aggregate]

Reads pkls only, runs nothing, and is not called from `src/main.py`, so the grids
can be redrawn without rerunning an experiment. Per sweep parameter found under any
`<dataset>/sweep/`, one `<param>_grid.pdf`: rows coverage, width, worst error (the
last two divided by the baseline PI as the sweep figures are under `normalize`),
columns the datasets in `DATASET_ORDER` that are present, x shared within a column,
y shared across the grid on `CLAMP_YLIM`, one x-label, three y-labels without the
"/ PI" suffix, one legend above the titles. A missing pkl leaves its cells blank.
Per perf metric, one row of panels (`epsilon_wall_clock.pdf`, `epsilon_seed_var.pdf`)
with a shared y axis, drawn only where some dataset ran the perf sweep.
"""

import argparse
import glob
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from loguru import logger

from src.experiments.configs import ALL_METHODS, METRIC_SPECS, PARAM_SPECS
from src.experiments.utils.constants import (
    CLAMP_YLIM,
    DATASET_ORDER,
    DATASET_TITLES,
    FS_LABEL,
    FS_TICK,
    IV_MODES,
    PLOT_DPI,
    PLOT_FORMAT,
    RC_PARAMS,
    SUBDIR_PERF,
    SUBDIR_SWEEP,
    TEX_MAPPER,
    parse_method,
    spelled_method,
)
from src.experiments.utils.data_operations import bootstrap, load
from src.experiments.utils.plotting import (
    X_MARGIN,
    _at_least_two_major_ticks,
    _draw_series,
    _label_major_ticks_only,
    _pad,
    normalize_sweep,
)

# the grid's rows, in order: (metric id, y-label)
ROWS: tuple[tuple[str, str], ...] = (("coverage", "coverage"), ("width", "width"), ("worst_error", "worst error"))
PERF_METRICS: tuple[str, ...] = ("wall_clock", "seed_var")
# one legend row holds this many entries; more wrap
LEGEND_MAX_COLS: int = 6
PANEL_WIDTH: float = 4.0
GRID_HEIGHT: float = 8.0
PERF_ROW_HEIGHT: float = 4.2  # room for the two-line wall-clock label


# ------------------------------------------------------------------ discovery


def columns(artifacts: str) -> list[str]:
    """The datasets present, in the fixed order: any sweep results pkl or a perf grid."""
    return [
        d
        for d in DATASET_ORDER
        if glob.glob(f"{artifacts}/{d}/{SUBDIR_SWEEP}/*_results.pkl")
        or os.path.exists(f"{artifacts}/{d}/{SUBDIR_PERF}/epsilon_values.pkl")
    ]


def sweep_params(artifacts: str, datasets: list[str]) -> list[str]:
    """Every sweep parameter with a results pkl under some dataset; a stem outside
    PARAM_SPECS is warned about and skipped."""
    stems = set()
    for d in datasets:
        for path in glob.glob(f"{artifacts}/{d}/{SUBDIR_SWEEP}/*_results.pkl"):
            stems.add(os.path.basename(path)[: -len("_results.pkl")])
    unknown = sorted(stems - set(PARAM_SPECS))
    if unknown:
        logger.warning(f"aggregate: sweep pkls {unknown} are not PARAM_SPECS params; skipped.")
    return sorted(stems & set(PARAM_SPECS))


def _has_perf(artifacts: str, dataset: str, metric: str) -> bool:
    folder = f"{artifacts}/{dataset}/{SUBDIR_PERF}"
    return os.path.exists(f"{folder}/epsilon_values.pkl") and os.path.exists(f"{folder}/epsilon_{metric}_results.pkl")


# ------------------------------------------------------------------ drawing


def _frame(ax, x, xscale: str, vlines) -> None:
    """The exact grid plus the sweep figures' margin, and the in-frame reference lines."""
    ax.set_xscale(xscale)
    ax.set_xlim(_pad(ax.xaxis, float(x.min()), float(x.max()), frac=X_MARGIN))
    ax.tick_params(labelsize=FS_TICK)
    x_lo, x_hi = ax.get_xlim()
    for v in vlines:
        if np.isfinite(v) and x_lo <= v <= x_hi:
            ax.axvline(v, color="0.4", linestyle=":", linewidth=1.0, zorder=0)


def _legend(fig, handles: dict) -> int:
    """One legend for the figure, keyed on (base, mode) so a bare name and its
    `(T,Z)` spelling are one entry, in the repo's method order; returns the rows."""
    keys = sorted(
        handles,
        key=lambda k: (ALL_METHODS.index(k[0]) if k[0] in ALL_METHODS else len(ALL_METHODS), IV_MODES.index(k[1])),
    )
    labels = [TEX_MAPPER.get(spelled_method(handles[k][1]), handles[k][1]) for k in keys]
    ncol = min(len(keys), LEGEND_MAX_COLS)
    if keys:
        fig.legend(
            [handles[k][0] for k in keys],
            labels,
            loc="upper center",
            ncol=ncol,
            bbox_to_anchor=(0.5, 1.0),
            fontsize=FS_TICK,
            frameon=True,
            edgecolor="black",
            fancybox=False,
        )
    return math.ceil(len(keys) / ncol) if keys else 0


def _finish(fig, axes, xlabel: str, rows: int, path: str | None):
    fig.supxlabel(xlabel, fontsize=FS_LABEL)
    _label_major_ticks_only(*axes)
    _at_least_two_major_ticks(*axes)
    fig.tight_layout(rect=(0, 0, 1, 0.93 if rows <= 1 else 0.90))
    if path is not None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, format=PLOT_FORMAT, dpi=PLOT_DPI)
        logger.info(f"aggregate: wrote {path}")
    return fig


def _xlabel(current: str | None, found: str, where: str) -> str:
    if current is not None and found != current:
        logger.warning(f"aggregate: {where} labels its x {found!r}, the first column {current!r}; the first wins.")
        return current
    return found


def sweep_grid(param: str, datasets: list[str], artifacts: str, out: str | None = None):
    """The 3 x [datasets] grid of one sweep parameter; saved under `out` when given."""
    plt.rcParams.update(RC_PARAMS)
    sns.set_palette("deep")
    spec = PARAM_SPECS[param]
    fig, axes = plt.subplots(
        len(ROWS),
        len(datasets),
        figsize=(PANEL_WIDTH * len(datasets), GRID_HEIGHT),
        sharex="col",
        sharey=True,
        squeeze=False,
    )
    handles, xlabel = {}, None
    for c, dataset in enumerate(datasets):
        axes[0, c].set_title(DATASET_TITLES[dataset], fontsize=FS_LABEL)
        folder = f"{artifacts}/{dataset}/{SUBDIR_SWEEP}"
        values, results = f"{folder}/{param}_values.pkl", f"{folder}/{param}_results.pkl"
        if not (os.path.exists(values) and os.path.exists(results)):
            for r in range(len(ROWS)):
                axes[r, c].axis("off")
            continue
        # the one sort of the sweep path (a measured trS axis is not ascending)
        x = np.asarray(load(values), dtype=float)
        order = np.argsort(x, kind="stable")
        x = x[order]
        record = load(results)
        axis_pkl = f"{folder}/{param}_axis.pkl"
        found = load(axis_pkl).get("xlabel", spec.xlabel) if os.path.exists(axis_pkl) else spec.xlabel
        xlabel = _xlabel(xlabel, found, f"{dataset} {param}")
        for r, (metric, _) in enumerate(ROWS):
            ax, mspec = axes[r, c], METRIC_SPECS[metric]
            include_ate = spec.include_ate and mspec.include_ate
            y = {
                name: np.asarray(rec[mspec.key])[order]
                for name, rec in record.items()
                if mspec.key in rec and (include_ate or name != "ATE")
            }
            if not y:
                ax.axis("off")
                continue
            y = bootstrap(y)
            if metric != "coverage":
                # the same PI, else PI+IV, division the sweep figure applies under
                # `normalize`; a rate is drawn as it is
                y, _ = normalize_sweep(y, f"{param}_{metric}")
            drawn, _ = _draw_series(ax, x, y)
            for name, handle in drawn.items():
                handles.setdefault(parse_method(name), (handle, name))
            _frame(ax, x, spec.xscale, spec.vlines)
            ax.set_yscale("linear")
            ax.set_ylim(*CLAMP_YLIM)
    for r, (_, label) in enumerate(ROWS):
        axes[r, 0].set_ylabel(label, fontsize=FS_LABEL)
    rows = _legend(fig, handles)
    path = None if out is None else f"{out}/{param}_grid.{PLOT_FORMAT}"
    return _finish(fig, axes.ravel(), xlabel or spec.xlabel, rows, path)


def perf_row(metric: str, datasets: list[str], artifacts: str, out: str | None = None):
    """One row of [datasets] panels of a perf metric on a shared y; saved under `out`."""
    plt.rcParams.update(RC_PARAMS)
    sns.set_palette("deep")
    spec, pspec = METRIC_SPECS[metric], PARAM_SPECS["epsilon"]
    fig, axes = plt.subplots(
        1, len(datasets), figsize=(PANEL_WIDTH * len(datasets), PERF_ROW_HEIGHT), sharey=True, squeeze=False
    )
    axes = axes[0]
    handles, xlabel = {}, None
    for ax, dataset in zip(axes, datasets, strict=True):
        ax.set_title(DATASET_TITLES[dataset], fontsize=FS_LABEL)
        if not _has_perf(artifacts, dataset, metric):
            ax.axis("off")
            continue
        folder = f"{artifacts}/{dataset}/{SUBDIR_PERF}"
        x = np.asarray(load(f"{folder}/epsilon_values.pkl"), dtype=float)
        y = load(f"{folder}/epsilon_{metric}_results.pkl")
        meta = f"{folder}/epsilon_perf_meta.pkl"
        found = load(meta).get("xlabel", pspec.xlabel) if os.path.exists(meta) else pspec.xlabel
        xlabel = _xlabel(xlabel, found, f"{dataset} perf")
        failures = None
        if metric == "seed_var":
            # the D(eps) terms per query: the mean line and the band over queries
            y = bootstrap(y)
            path = f"{folder}/epsilon_seed_var_failures.pkl"
            failures = load(path) if os.path.exists(path) else None
        drawn, _ = _draw_series(ax, x, y, failures)
        for name, handle in drawn.items():
            handles.setdefault(parse_method(name), (handle, name))
        _frame(ax, x, pspec.xscale, pspec.vlines)
        ax.set_yscale(spec.yscale)
    axes[0].set_ylabel(spec.ylabel, fontsize=FS_LABEL)
    rows = _legend(fig, handles)
    path = None if out is None else f"{out}/epsilon_{metric}.{PLOT_FORMAT}"
    return _finish(fig, axes, xlabel or pspec.xlabel, rows, path)


# ------------------------------------------------------------------ cli


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifacts", default="./artifacts", help="the tree to read (default ./artifacts)")
    parser.add_argument("--out", default=None, help="where the grids go (default <artifacts>/aggregate)")
    args = parser.parse_args(argv)
    artifacts = os.path.abspath(args.artifacts)
    out = os.path.abspath(args.out) if args.out else f"{artifacts}/aggregate"

    datasets = columns(artifacts)
    if not datasets:
        logger.warning(f"aggregate: nothing to draw under {artifacts}.")
        return
    for param in sweep_params(artifacts, datasets):
        plt.close(sweep_grid(param, datasets, artifacts, out))
    for metric in PERF_METRICS:
        if any(_has_perf(artifacts, d, metric) for d in datasets):
            plt.close(perf_row(metric, datasets, artifacts, out))


if __name__ == "__main__":
    main()
