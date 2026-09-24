"""
Aggregate the sweep and perf pkls under `artifacts/` into grids under `artifacts/aggregate/`.

    python -m src.aggregate [--artifacts ./artifacts] [--out ./artifacts/aggregate]

Reads pkls only, runs nothing, and is not called from `src/main.py`, so the grids
can be redrawn without rerunning an experiment. Per sweep parameter found under any
`<dataset>/sweep/`, one `<param>_grid.pdf`: rows coverage, width, worst error (the
last two divided by the baseline PI as the sweep figures are under `normalize`),
columns the datasets in `DATASET_ORDER` that are present, x shared within a column,
y shared across the grid on `CLAMP_YLIM`, one x-label (under the middle column when
the column count is odd, else centred), three y-labels without the "/ PI" suffix,
one legend above the titles: one row of up to `LEGEND_FLAT_MAX` entries, else
`LEGEND_ROWS` rows with a paired family in one column and the singletons stacked
two to a column. A missing pkl leaves its cells blank.
The perf sweeps against epsilon, laid out by the same grid code: rows the metrics,
columns the datasets, x shared within a column, y within a row, the same x-label and
legend rules. `epsilon_wall_clock.pdf` is the wall-clock row alone;
`epsilon_seed_var.pdf` stacks the feasible rate (on `CLAMP_YLIM`) under the
stability. A row is drawn only where some dataset ran its metric. From the
cigarette query pkls, the 2 x 2 elasticity grid (`cigarettes_elasticities.pdf`):
rows the state and neighbour price coefficients, columns the confounding budget
gamma and the leakiness budget gamma_z, x shared within a column, y within a row,
the reference marks of each panel, one legend inside the top-right panel, pinned
upper left. From the do-MNIST tint sweep pkls, the stacked tint figure
(`do_mnist_tint.pdf`): one row per digit, 0 at the top, the bounds along the tint
grid in the middle with the digit's blue-tint image on the left and its red-tint
image on the right (full resolution, white background), titled $h({\bm{x}})$, the
x-axis `tint` shared down to the bottom row, the tint histogram of the B rows
before (blue) and after (red) DA, its legend at the bottom right.
"""

import argparse
import glob
import json
import os
import re
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
# the perf figures, `epsilon_{stem}`, each a grid of its metric rows: the wall clock
# alone, the stability with the feasible rate stacked below it
PERF_FIGURES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("wall_clock", ("wall_clock",)),
    ("seed_var", ("seed_var", "feasibility")),
)
PERF_OVER_QUERIES: tuple[str, ...] = ("seed_var", "feasibility")  # bootstrapped over queries
PERF_RATES: tuple[str, ...] = ("feasibility",)  # framed on CLAMP_YLIM
# a row label the panel height cannot take on one line, broken for the grid
PERF_ROW_LABELS: dict[str, str] = {"feasibility": "feasible rate\n(over backends)"}
# the figure legend by entry count n: up to LEGEND_FLAT_MAX entries in one row of n,
# more in LEGEND_ROWS rows of ceil(n / LEGEND_ROWS) columns; past LEGEND_GRID_COLS
# columns it still widens, with a warning
LEGEND_FLAT_MAX: int = 6
LEGEND_ROWS: int = 2
LEGEND_GRID_COLS: int = 5
LEGEND_GAP: float = 0.01  # figure fraction between the legend and the column titles
PANEL_WIDTH: float = 4.0
GRID_HEIGHT: float = 8.0
PERF_ROW_HEIGHT: float = 4.2  # per perf row, the legend and the x-label included
# the elasticity grid: (coefficient id, row label) and (axis id, column stem)
ELASTICITY_ROWS: tuple[tuple[str, str], ...] = (("p", COEFFICIENT_LABELS["p"]), ("pn", COEFFICIENT_LABELS["pn"]))
ELASTICITY_AXES: tuple[str, ...] = ("gamma", "budget")
ELASTICITY_LEGEND_PANEL: tuple[int, int] = (0, 1)
ELASTICITY_LEGEND_LOC: str = "upper left"
ELASTICITY_Y_PAD: float = 0.05  # of the row's span
# the do-MNIST tint stack: the image | bounds | image width ratios, the height per
# digit row and of the histogram row, the legend strip, the bounds' y frame (the
# digit sweep's), and the before / after-DA colours (`deep` blue and red)
TINT_WIDTHS: tuple[float, float, float] = (0.14, 1.0, 0.14)
TINT_ROW_HEIGHT: float = 1.1
TINT_DENSITY_HEIGHT: float = 1.5
TINT_LEGEND_HEIGHT: float = 1.1
TINT_WIDTH: float = 2 * PANEL_WIDTH
TINT_YLIM: tuple[float, float] = (-0.05, 1.05)
TINT_DENSITY_COLORS: tuple[int, int] = (0, 3)
TINT_TITLE: str = r"$h({\bm{x}})$"


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


def _tint_folder(artifacts: str) -> str:
    return f"{artifacts}/do_mnist/{SUBDIR_QUERY}"


def tint_digits(artifacts: str) -> list[int]:
    """The digits with a tint sweep (`tint_{d}_outcomes.pkl`), ascending."""
    found = []
    for path in glob.glob(f"{_tint_folder(artifacts)}/tint_*_outcomes.pkl"):
        match = re.fullmatch(r"tint_(\d)_outcomes\.pkl", os.path.basename(path))
        if match:
            found.append(int(match.group(1)))
    return sorted(found)


def _has_tint(artifacts: str) -> bool:
    return bool(tint_digits(artifacts))


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
    """One legend for the figure, laid out by entry count. Two collapses first:
    `parse_method` already keys a bare name and its `(T,Z)` spelling alike, and
    entries that would render as the same pixels (label, hue, alpha, dash) fold to
    one, which is what blends a T-as-IV method's three mode spellings. Then n
    entries up to LEGEND_FLAT_MAX are one row of n in the repo's order; more are
    LEGEND_ROWS rows of ceil(n / LEGEND_ROWS) columns, paired groups first, each in
    one column (member 0 on top, member 1 below), and the singletons after them in
    PAIR_ORDER order, two to a column. Returns the legend, or None with nothing
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
    n = len(keys)
    ncol = n
    if n > LEGEND_FLAT_MAX:
        # matplotlib fills column-major and gives the first n % ncol columns the
        # extra entry, so with ncol = ceil(n / 2) every column holds two entries but
        # the last one when n is odd. The paired groups go first and each lands on
        # a column of its own; the singletons stack behind them. Nothing is padded
        keys.sort(key=lambda k: (0 if sizes[pair(k)[0]] == 2 else 1, order(k)))
        ncol = -(-n // LEGEND_ROWS)
        if ncol > LEGEND_GRID_COLS:
            logger.warning(
                f"aggregate: {n} legend entries in LEGEND_ROWS {LEGEND_ROWS} rows take {ncol} columns, "
                f"past LEGEND_GRID_COLS {LEGEND_GRID_COLS}; the legend widens."
            )
    # a pair per column needs every group of size <= 2. A group of three sorts as a
    # singleton and the guarantee goes quietly; unreachable with today's PAIR_ORDER,
    # so say it, do not raise -- a wrapped legend is still readable
    crowded = sorted(g for g, size in sizes.items() if size > 2)
    if crowded:
        logger.warning(
            f"aggregate: legend groups over two entries {crowded}; the legend may split a group across columns."
        )
    return fig.legend(
        [handles[k][0] for k in keys],
        [labels[k] for k in keys],
        loc="upper center",
        ncol=ncol,
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
    """`axes` is the 2-D grid; the one x-label sits under the middle column when
    the column count is odd (one included), else at the figure's centre."""
    label = fig.supxlabel(xlabel, fontsize=FS_LABEL)
    _label_major_ticks_only(*axes.ravel())
    _at_least_two_major_ticks(*axes.ravel())
    # the room the legend takes, as rendered, so a second or third row never sits
    # on the column titles
    top = 0.97
    if legend is not None:
        fig.canvas.draw()
        box = legend.get_window_extent().transformed(fig.transFigure.inverted())
        top = max(0.5, box.y0 - LEGEND_GAP)
    fig.tight_layout(rect=(0, 0, 1, top))
    ncols = axes.shape[1]
    if ncols % 2:
        box = axes[-1, ncols // 2].get_position()
        label.set_x((box.x0 + box.x1) / 2)
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


def _metric_grid(
    rows,
    datasets: list[str],
    load_cell,
    xlabel_default: str,
    xscale: str,
    vlines,
    sharey,
    row_ylim,
    *,
    row_yscale,
    height: float,
    where: str,
    path: str | None,
):
    """The [rows] x [datasets] grid: `rows` is [(metric, row label)], `load_cell(dataset,
    metric)` gives (x, {method: y}, x-label) or None for a blank cell (an empty dict
    blanks the cell but still offers its x-label), x shared within a column, y as
    `sharey` says, each row on `row_yscale(metric)` and, when `row_ylim(metric)` is not
    None, framed there. Titles on the first row, one x-label, one legend; saved at
    `path` when given."""
    plt.rcParams.update(RC_PARAMS)
    sns.set_palette("deep")
    fig, axes = plt.subplots(
        len(rows),
        len(datasets),
        figsize=(PANEL_WIDTH * len(datasets), height),
        sharex="col",
        sharey=sharey,
        squeeze=False,
    )
    handles, xlabel = {}, None
    for c, dataset in enumerate(datasets):
        axes[0, c].set_title(DATASET_TITLES[dataset], fontsize=FS_LABEL)
        labelled = False
        for r, (metric, _) in enumerate(rows):
            ax, cell = axes[r, c], load_cell(dataset, metric)
            if cell is None:
                ax.axis("off")
                continue
            x, y, found = cell
            if not labelled:
                # one x-label per column, the first cell's
                xlabel, labelled = _xlabel(xlabel, found, f"{dataset} {where}"), True
            if not y:
                ax.axis("off")
                continue
            drawn, _ = _draw_series(ax, x, y)
            for name, handle in drawn.items():
                handles.setdefault(parse_method(name), (handle, name))
            _frame(ax, x, xscale, vlines)
            ax.set_yscale(row_yscale(metric))
            ylim = row_ylim(metric)
            if ylim is not None:
                ax.set_ylim(*ylim)
    _label_rows(axes, [label for _, label in rows])
    legend = _legend(fig, handles)
    return _finish(fig, axes, xlabel or xlabel_default, legend, path)


def sweep_grid(param: str, datasets: list[str], artifacts: str, out: str | None = None):
    """The 3 x [datasets] grid of one sweep parameter; saved under `out` when given."""
    spec = PARAM_SPECS[param]
    loaded = {}

    def column(dataset):
        # the pkls of one column, read once for its three rows
        if dataset not in loaded:
            folder = f"{artifacts}/{dataset}/{SUBDIR_SWEEP}"
            values, results = f"{folder}/{param}_values.pkl", f"{folder}/{param}_results.pkl"
            if not (os.path.exists(values) and os.path.exists(results)):
                loaded[dataset] = None
            else:
                # the one sort of the sweep path (a measured omega axis is not ascending)
                x = np.asarray(load(values), dtype=float)
                order = np.argsort(x, kind="stable")
                axis_pkl = f"{folder}/{param}_axis.pkl"
                found = load(axis_pkl).get("xlabel", spec.xlabel) if os.path.exists(axis_pkl) else spec.xlabel
                loaded[dataset] = (x[order], order, load(results), found)
        return loaded[dataset]

    def load_cell(dataset, metric):
        if column(dataset) is None:
            return None
        x, order, record, found = column(dataset)
        mspec = METRIC_SPECS[metric]
        include_ate = spec.include_ate and mspec.include_ate
        y = {
            name: np.asarray(rec[mspec.key])[order]
            for name, rec in record.items()
            if mspec.key in rec and (include_ate or name != "ATE")
        }
        if y:
            y = bootstrap(y)
            if metric != "coverage":
                # the same PI, else PI+IV, division the sweep figure applies under
                # `normalize`; a rate is drawn as it is
                y, _ = normalize_sweep(y, f"{param}_{metric}")
        return x, y, found

    return _metric_grid(
        ROWS,
        datasets,
        load_cell,
        spec.xlabel,
        spec.xscale,
        spec.vlines,
        True,
        lambda metric: CLAMP_YLIM,
        row_yscale=lambda metric: "linear",
        height=GRID_HEIGHT,
        where=param,
        path=None if out is None else f"{out}/{param}_grid.{PLOT_FORMAT}",
    )


def perf_grid(stem: str, metrics, datasets: list[str], artifacts: str, out: str | None = None):
    """The [metrics] x [datasets] grid of the perf sweeps against epsilon, a y shared
    within each row, the rates on `CLAMP_YLIM`; saved as `epsilon_{stem}` under `out`."""
    pspec = PARAM_SPECS["epsilon"]

    def load_cell(dataset, metric):
        if not _has_perf(artifacts, dataset, metric):
            return None
        folder = f"{artifacts}/{dataset}/{SUBDIR_PERF}"
        x = np.asarray(load(f"{folder}/epsilon_values.pkl"), dtype=float)
        y = load(f"{folder}/epsilon_{metric}_results.pkl")
        meta = f"{folder}/epsilon_perf_meta.pkl"
        found = load(meta).get("xlabel", pspec.xlabel) if os.path.exists(meta) else pspec.xlabel
        if metric in PERF_OVER_QUERIES:
            # the D(eps) terms or the feasible rates per query: the mean line and
            # the band over queries
            y = bootstrap(y)
        return x, y, found

    return _metric_grid(
        [(metric, PERF_ROW_LABELS.get(metric, METRIC_SPECS[metric].ylabel)) for metric in metrics],
        datasets,
        load_cell,
        pspec.xlabel,
        pspec.xscale,
        pspec.vlines,
        "row",
        # a rate, on the coverage rows' frame
        lambda metric: CLAMP_YLIM if metric in PERF_RATES else None,
        row_yscale=lambda metric: METRIC_SPECS[metric].yscale,
        height=PERF_ROW_HEIGHT * len(metrics),
        where="perf",
        path=None if out is None else f"{out}/epsilon_{stem}.{PLOT_FORMAT}",
    )


def elasticity_grid(artifacts: str, out: str | None = None):
    """The 2 x 2 grid of the cigarette price elasticities against gamma and gamma_z;
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
            loc=ELASTICITY_LEGEND_LOC,
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


def _tint_provenance(artifacts: str, digits: list[int], grids: dict[int, np.ndarray]) -> None:
    """Warn when the stacked sweeps do not all come from the run whose `run.json`
    sits beside them: a digit it did not sweep, or a grid it did not use."""
    path = f"{_tint_folder(artifacts)}/run.json"
    if not os.path.exists(path):
        logger.warning(f"aggregate: no run.json beside the tint sweeps under {_tint_folder(artifacts)}.")
        return
    with open(path) as fh:
        tint = json.load(fh).get("tint")
    if not tint:
        logger.warning("aggregate: the run.json beside the tint sweeps records no tint sweep; they may be stale.")
        return
    stale = [d for d in digits if d not in tint["digits"]]
    if stale:
        logger.warning(f"aggregate: tint sweeps {stale} are not in run.json's digits {tint['digits']}; stale pkls?")
    grid = np.asarray(tint["grid"], dtype=float)
    moved = [d for d, x in grids.items() if x.shape != grid.shape or not np.allclose(x, grid)]
    if moved:
        logger.warning(f"aggregate: tint sweeps {moved} use another grid than run.json's; stale pkls?")


def _tint_image(ax, image) -> None:
    """The exemplar thumbnail rule: RGB with the ink as alpha, so the background is
    white; full resolution, no axes."""
    rgb = np.clip(np.transpose(np.asarray(image), (1, 2, 0)), 0.0, 1.0)
    ax.imshow(np.dstack([rgb, np.clip(rgb.sum(-1), 0.0, 1.0)]), interpolation="nearest")
    ax.axis("off")


def tint_stack(artifacts: str, out: str | None = None):
    """The do-MNIST tint sweeps stacked, one row per digit (0 at the top); saved
    under `out` when given. Returns the figure, or None with no sweep."""
    from matplotlib.gridspec import GridSpec

    digits = tint_digits(artifacts)
    if not digits:
        return None
    folder = _tint_folder(artifacts)
    plt.rcParams.update(RC_PARAMS)
    sns.set_palette("deep")
    colors = sns.color_palette("deep")
    density_path = f"{folder}/tint_density.pkl"
    has_density = os.path.exists(density_path)
    rows = len(digits) + int(has_density)
    heights = [TINT_ROW_HEIGHT] * len(digits) + ([TINT_DENSITY_HEIGHT] if has_density else [])
    fig = plt.figure(figsize=(TINT_WIDTH, sum(heights) + TINT_LEGEND_HEIGHT))
    grid = GridSpec(rows, 3, figure=fig, width_ratios=TINT_WIDTHS, height_ratios=heights, hspace=0.12, wspace=0.04)

    handles, grids, shared, bounds_axes = {}, {}, None, []
    for r, digit in enumerate(digits):
        x = np.asarray(load(f"{folder}/tint_{digit}_values.pkl"), dtype=float)
        grids[digit] = x
        ax = fig.add_subplot(grid[r, 1], sharex=shared)
        shared = shared or ax
        drawn, _, _ = _draw_bands(ax, x, load(f"{folder}/tint_{digit}_outcomes.pkl"))
        for name, handle in drawn.items():
            handles.setdefault(parse_method(name), (handle, name))
        ax.set_ylim(*TINT_YLIM)
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.tick_params(labelsize=FS_TICK - 4, labelbottom=False)
        bounds_axes.append(ax)
        images = f"{folder}/tint_{digit}_images.pkl"
        if os.path.exists(images):
            blue, red = load(images)
            _tint_image(fig.add_subplot(grid[r, 0]), blue)
            _tint_image(fig.add_subplot(grid[r, 2]), red)
    bounds_axes[0].set_title(TINT_TITLE, fontsize=FS_LABEL)
    _tint_provenance(artifacts, digits, grids)

    bottom = bounds_axes[-1]
    if has_density:
        density = load(density_path)
        bottom = fig.add_subplot(grid[-1, 1], sharex=shared)
        bars = []
        for key, color in zip(("before", "after"), TINT_DENSITY_COLORS, strict=True):
            bars.append(
                bottom.hist(np.asarray(density[key]), bins=50, density=True, alpha=0.45, color=colors[color])[2][0]
            )
        bottom.set_ylabel("density", fontsize=FS_TICK)
        bottom.tick_params(labelsize=FS_TICK - 4)
        side = fig.add_subplot(grid[-1, 2])
        side.axis("off")
        side.legend(
            bars,
            ["pre-DA", "post-DA"],
            loc="lower left",
            bbox_to_anchor=(0.0, 0.0),
            fontsize=FS_TICK - 5,
            handlelength=1.0,
            frameon=True,
            edgecolor="black",
            fancybox=False,
        )
    bottom.tick_params(labelbottom=True)
    bottom.set_xlabel(ANNOTATE_SWEEP_PLOT["tint"]["xlabel"], fontsize=FS_LABEL)
    x = grids[digits[0]]
    shared.set_xlim(float(x.min()), float(x.max()))
    _label_major_ticks_only(*bounds_axes, bottom)

    legend = _legend(fig, handles)
    top = 0.97
    if legend is not None:
        fig.canvas.draw()
        box = legend.get_window_extent().transformed(fig.transFigure.inverted())
        top = max(0.5, box.y0 - LEGEND_GAP)
    fig.subplots_adjust(left=0.02, right=0.98, top=top - 0.02, bottom=0.6 / fig.get_figheight())
    path = None if out is None else f"{out}/do_mnist_tint.{PLOT_FORMAT}"
    if path is not None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, format=PLOT_FORMAT, dpi=PLOT_DPI, bbox_inches="tight")
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
    if not datasets and not _has_elasticities(artifacts) and not _has_tint(artifacts):
        logger.warning(f"aggregate: nothing to draw under {artifacts}.")
        return
    for param in sweep_params(artifacts, datasets):
        plt.close(sweep_grid(param, datasets, artifacts, out))
    for stem, metrics in PERF_FIGURES:
        # the rows some dataset ran; a figure none of whose metrics ran is not drawn
        ran = [m for m in metrics if any(_has_perf(artifacts, d, m) for d in datasets)]
        if ran:
            plt.close(perf_grid(stem, ran, datasets, artifacts, out))
    if _has_elasticities(artifacts):
        plt.close(elasticity_grid(artifacts, out))
    if _has_tint(artifacts):
        plt.close(tint_stack(artifacts, out))


if __name__ == "__main__":
    main()
