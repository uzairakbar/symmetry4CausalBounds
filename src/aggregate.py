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
with a shared y axis, drawn only where some dataset ran the perf sweep. From the
cigarette query pkls, the 2 x 2 elasticity grid (`cigarettes_elasticities.pdf`):
rows the state and neighbour price coefficients, columns the confounding budget
and the leak radius, x shared within a column, y within a row, the reference marks
of each panel, one legend inside the first panel.
"""

import argparse
import glob
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from loguru import logger

from src.experiments.configs import ALL_METHODS, ANNOTATE_SWEEP_PLOT, METRIC_SPECS, PARAM_SPECS
from src.experiments.utils.constants import (
    ALPHA_MAP,
    CLAMP_YLIM,
    COEFFICIENT_LABELS,
    COLOR_MAP,
    DATASET_ORDER,
    DATASET_TITLES,
    FS_LABEL,
    FS_TICK,
    IV_MODES,
    PAIR_ORDER,
    PLOT_DPI,
    PLOT_FORMAT,
    RC_PARAMS,
    SUBDIR_PERF,
    SUBDIR_QUERY,
    SUBDIR_SWEEP,
    TEX_MAPPER,
    parse_method,
    spelled_method,
)
from src.experiments.utils.data_operations import bootstrap, load
from src.experiments.utils.plotting import (
    X_MARGIN,
    _at_least_two_major_ticks,
    _draw_bands,
    _draw_series,
    _label_major_ticks_only,
    _line_style,
    _mark_frame,
    _pad,
    normalize_sweep,
)

# the grid's rows, in order: (metric id, y-label)
ROWS: tuple[tuple[str, str], ...] = (("coverage", "coverage"), ("width", "width"), ("worst_error", "worst error"))
PERF_METRICS: tuple[str, ...] = ("wall_clock", "seed_var")
# one legend row holds this many entries; more wrap
LEGEND_MAX_COLS: int = 6
LEGEND_GAP: float = 0.01  # figure fraction between the legend and the column titles
PANEL_WIDTH: float = 4.0
GRID_HEIGHT: float = 8.0
PERF_ROW_HEIGHT: float = 4.2  # room for the two-line wall-clock label
# the elasticity grid: (coefficient id, row label) and (axis id, column stem)
ELASTICITY_ROWS: tuple[tuple[str, str], ...] = (("p", COEFFICIENT_LABELS["p"]), ("pn", COEFFICIENT_LABELS["pn"]))
ELASTICITY_AXES: tuple[str, ...] = ("gamma", "budget")
ELASTICITY_LEGEND_PANEL: tuple[int, int] = (0, 0)
ELASTICITY_Y_PAD: float = 0.05  # of the row's span


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


def _elasticity_stem(artifacts: str, coefficient: str, axis: str) -> str:
    return f"{artifacts}/cigarettes/{SUBDIR_QUERY}/beta_{coefficient}_{axis}"


def _has_elasticities(artifacts: str) -> bool:
    return any(
        os.path.exists(f"{_elasticity_stem(artifacts, c, a)}_outcomes.pkl")
        for c, _ in ELASTICITY_ROWS
        for a in ELASTICITY_AXES
    )


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


def _legend(fig, handles: dict):
    """One legend for the figure, one PAIR_ORDER group per column. Two collapses:
    `parse_method` already keys a bare name and its `(T,Z)` spelling alike, and
    entries that would render as the same pixels (label, hue, alpha, dash) fold to
    one, which is what blends a T-as-IV method's three mode spellings. `ncol` is the
    number of surviving GROUPS and paired groups sort first, so each column holds one
    group: member 0 on top, member 1 below. Returns the legend, or None with nothing
    drawn."""
    if not handles:
        return None
    spelled = {k: spelled_method(handles[k][1]) for k in handles}
    labels = {k: TEX_MAPPER.get(spelled[k], handles[k][1]) for k in handles}

    def pair(k):
        # a name no table knows lands past the end, deterministically, rather than
        # raising here
        return PAIR_ORDER.get(spelled[k], (len(PAIR_ORDER), 0))

    def order(k):
        return (
            pair(k),
            ALL_METHODS.index(k[0]) if k[0] in ALL_METHODS else len(ALL_METHODS),
            IV_MODES.index(k[1]),
        )

    # the render signature carries NO base term: two entries merge iff a reader
    # cannot tell them apart. `.get(s, s)` keeps a stale name distinct instead of
    # collapsing every unknown onto one sentinel
    keys, seen = [], set()
    for k in sorted(handles, key=order):
        s = spelled[k]
        signature = (labels[k], COLOR_MAP.get(s, s), ALPHA_MAP.get(s, s), _line_style(s))
        if signature not in seen:
            seen.add(signature)
            keys.append(k)
    sizes = Counter(pair(k)[0] for k in keys)
    # n = g + p over groups of size 1 or 2, so divmod(n, g) = (1, p) and matplotlib
    # fills column-major: the p paired groups take the two-entry columns and the rest
    # take the one-entry ones. Nothing has to be padded
    keys.sort(key=lambda k: (0 if sizes[pair(k)[0]] == 2 else 1, order(k)))
    # one group per column needs g <= LEGEND_MAX_COLS AND every group of size <= 2,
    # so that divmod(n, g) = (1, p). A group of three sorts as a singleton and the
    # guarantee goes quietly; unreachable with today's PAIR_ORDER, so say it, do not
    # raise -- a wrapped legend is still readable
    crowded = sorted(g for g, size in sizes.items() if size > 2)
    if len(sizes) > LEGEND_MAX_COLS or crowded:
        logger.warning(
            f"aggregate: {len(sizes)} legend groups against LEGEND_MAX_COLS {LEGEND_MAX_COLS}, "
            f"groups over two entries {crowded}; the legend may put two groups in one column."
        )
    return fig.legend(
        [handles[k][0] for k in keys],
        [labels[k] for k in keys],
        loc="upper center",
        ncol=min(len(sizes), LEGEND_MAX_COLS),
        bbox_to_anchor=(0.5, 1.0),
        fontsize=FS_TICK,
        frameon=True,
        edgecolor="black",
        fancybox=False,
    )


def _label_rows(axes_rows, labels) -> None:
    """The row label and the y tick numbers on the first axes of each row that is
    on (`sharey` blanks the inner columns' tick labels; a blank first column must
    not take the labels with it)."""
    for row, label in zip(axes_rows, labels, strict=True):
        first = next((ax for ax in row if ax.axison), None)
        if first is not None:
            first.set_ylabel(label, fontsize=FS_LABEL)
            first.tick_params(labelleft=True)


def _finish(fig, axes, xlabel: str, legend, path: str | None):
    fig.supxlabel(xlabel, fontsize=FS_LABEL)
    _label_major_ticks_only(*axes)
    _at_least_two_major_ticks(*axes)
    # the room the legend takes, as rendered, so a second or third row never sits
    # on the column titles
    top = 0.97
    if legend is not None:
        fig.canvas.draw()
        box = legend.get_window_extent().transformed(fig.transFigure.inverted())
        top = max(0.5, box.y0 - LEGEND_GAP)
    fig.tight_layout(rect=(0, 0, 1, top))
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
        # the one sort of the sweep path (a measured omega axis is not ascending)
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
    _label_rows(axes, [label for _, label in ROWS])
    legend = _legend(fig, handles)
    path = None if out is None else f"{out}/{param}_grid.{PLOT_FORMAT}"
    return _finish(fig, axes.ravel(), xlabel or spec.xlabel, legend, path)


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
    _label_rows([axes], [spec.ylabel])
    legend = _legend(fig, handles)
    path = None if out is None else f"{out}/epsilon_{metric}.{PLOT_FORMAT}"
    return _finish(fig, axes, xlabel or pspec.xlabel, legend, path)


def elasticity_grid(artifacts: str, out: str | None = None):
    """The 2 x 2 grid of the cigarette price elasticities against gamma and r_Z;
    saved under `out` when given. A missing pkl pair leaves its panel blank."""
    plt.rcParams.update(RC_PARAMS)
    sns.set_palette("deep")
    fig, axes = plt.subplots(
        len(ELASTICITY_ROWS),
        len(ELASTICITY_AXES),
        figsize=(PANEL_WIDTH * len(ELASTICITY_AXES), PANEL_WIDTH * len(ELASTICITY_ROWS)),
        sharex="col",
        sharey="row",
        squeeze=False,
    )
    handles = {}
    for r, (coefficient, _) in enumerate(ELASTICITY_ROWS):
        lo, hi = float("inf"), float("-inf")
        for c, axis in enumerate(ELASTICITY_AXES):
            ax, stem = axes[r, c], _elasticity_stem(artifacts, coefficient, axis)
            spec = ANNOTATE_SWEEP_PLOT[f"beta_{coefficient}_{axis}"]
            if not (os.path.exists(f"{stem}_values.pkl") and os.path.exists(f"{stem}_outcomes.pkl")):
                ax.axis("off")
                continue
            x = np.asarray(load(f"{stem}_values.pkl"), dtype=float)
            drawn, panel_lo, panel_hi = _draw_bands(ax, x, load(f"{stem}_outcomes.pkl"))
            for name, handle in drawn.items():
                handles.setdefault(name, handle)
            lo, hi = min(lo, panel_lo), max(hi, panel_hi)
            marks = load(f"{stem}_vlines.pkl") if os.path.exists(f"{stem}_vlines.pkl") else ()
            x_lo, x_hi, marks = _mark_frame(x, marks, spec["xscale"])
            ax.set_xscale(spec["xscale"])
            ax.set_xlim(x_lo, x_hi)
            ax.tick_params(labelsize=FS_TICK)
            for v in marks:
                ax.axvline(v, color="0.4", linestyle=":", linewidth=1.0, zorder=0)
            if r == len(ELASTICITY_ROWS) - 1:
                ax.set_xlabel(spec["xlabel"], fontsize=FS_LABEL)
        if np.isfinite(lo) and np.isfinite(hi):
            pad = ELASTICITY_Y_PAD * (hi - lo)
            axes[r, 0].set_ylim(lo - pad, hi + pad)
    _label_rows(axes, [label for _, label in ELASTICITY_ROWS])
    legend_ax = axes[ELASTICITY_LEGEND_PANEL]
    if handles and legend_ax.axison:
        legend_ax.legend(
            list(handles.values()),
            [TEX_MAPPER.get(name, name) for name in handles],
            loc="best",
            fontsize=FS_TICK,
            frameon=True,
            edgecolor="black",
            fancybox=False,
        )
    live = [ax for ax in axes.ravel() if ax.axison]
    _label_major_ticks_only(*live)
    _at_least_two_major_ticks(*live)
    fig.tight_layout()
    path = None if out is None else f"{out}/cigarettes_elasticities.{PLOT_FORMAT}"
    if path is not None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, format=PLOT_FORMAT, dpi=PLOT_DPI)
        logger.info(f"aggregate: wrote {path}")
    return fig


# ------------------------------------------------------------------ cli


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifacts", default="./artifacts", help="the tree to read (default ./artifacts)")
    parser.add_argument("--out", default=None, help="where the grids go (default <artifacts>/aggregate)")
    args = parser.parse_args(argv)
    artifacts = os.path.abspath(args.artifacts)
    out = os.path.abspath(args.out) if args.out else f"{artifacts}/aggregate"

    datasets = columns(artifacts)
    if not datasets and not _has_elasticities(artifacts):
        logger.warning(f"aggregate: nothing to draw under {artifacts}.")
        return
    for param in sweep_params(artifacts, datasets):
        plt.close(sweep_grid(param, datasets, artifacts, out))
    for metric in PERF_METRICS:
        if any(_has_perf(artifacts, d, metric) for d in datasets):
            plt.close(perf_row(metric, datasets, artifacts, out))
    if _has_elasticities(artifacts):
        plt.close(elasticity_grid(artifacts, out))


if __name__ == "__main__":
    main()
