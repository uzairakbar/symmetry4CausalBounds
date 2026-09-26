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
one legend inside the top panel of the middle column, pinned lower right, in the
repo's legend order, the single intervals in its left column and PI+INV and the
intersections (`SWEEP_LEGEND_RIGHT`) in its right. A missing pkl leaves its cells
blank.
The perf sweeps against epsilon, laid out by the same grid code: rows the metrics,
columns the datasets, x shared within a column, y within a row, the same x-label
rule, one legend above the titles: one row of up to `LEGEND_FLAT_MAX` entries, else
`LEGEND_ROWS` rows with a paired family in one column and the singletons stacked
two to a column. `epsilon_wall_clock.pdf` is the wall-clock row alone;
`epsilon_seed_var.pdf` stacks the feasible rate (on `CLAMP_YLIM`) under the
stability. A row is drawn only where some dataset ran its metric. From the
cigarette query pkls, the 2 x 2 elasticity grid (`cigarettes_elasticities.pdf`):
rows the state and neighbour price coefficients, columns the confounding budget
gamma and the leakiness budget gamma_z, x shared within a column, y within a row,
the reference marks of each panel, one legend inside the top-right panel, pinned
upper left, in the repo's legend order (`_legend_key`: PI+INV+IV, PI+IV, then the
DA+ methods). From the do-MNIST tint sweep pkls, the stacked tint figure
(`do_mnist_tint.pdf`): one row per digit, 0 at the top, the bounds along the tint
grid in the middle with the digit's blue-tint image on the left and its red-tint
image on the right (full resolution, white background), titled $h({\bm{x}})$, the
x-axis `tint` shared down to the bottom row, the tint histogram of the B rows
before (blue) and after (red) DA, its legend at the bottom right. From the
do-MNIST population pkls and `run.json`, the results table (`do_mnist_table.tex`):
one row per interval method, coverage and width with bootstrap bands over the
population queries, Omega-hat on the DA+ rows, worst error, the one-time fit, the
mean per-query solve and the latency (the fit plus the per-query solve).
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
from matplotlib.lines import Line2D

from src.experiments.configs import ALL_METHODS, ANNOTATE_SWEEP_PLOT, METRIC_SPECS, PARAM_SPECS
from src.experiments.utils.constants import (
    CLAMP_YLIM,
    COEFFICIENT_LABELS,
    DATASET_ORDER,
    DATASET_TITLES,
    FS_LABEL,
    FS_TICK,
    IV_MODES,
    NULL_Z_TITLE_SUFFIX,
    PLOT_DPI,
    PLOT_FORMAT,
    RC_PARAMS,
    SUBDIR_PERF,
    SUBDIR_QUERY,
    SUBDIR_SWEEP,
    method_style,
    parse_method,
)
from src.experiments.utils.constants import label as method_label
from src.experiments.utils.data_operations import bootstrap, load
from src.experiments.utils.plotting import (
    X_MARGIN,
    _at_least_two_major_ticks,
    _draw_bands,
    _draw_series,
    _label_major_ticks_only,
    _mark_frame,
    _pad,
    draw_da_density,
    fix_x_ticks,
    fold_by_signature,
    mark_failed,
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
# the sweep grids' legend: inside the top panel of the middle column, pinned lower
# right, at the tick size less SWEEP_LEGEND_SHRINK so ten entries keep to a panel;
# two columns, the single intervals on the left and the SWEEP_LEGEND_RIGHT bases
# (PI+INV and the intersections) on the right, each in the repo's legend order
SWEEP_LEGEND_LOC: str = "lower right"
SWEEP_LEGEND_SHRINK: int = 4
SWEEP_LEGEND_RIGHT: tuple[str, ...] = ("PI+INV", "PI+INV+IV", "PI&DA+PI", "PI&DA+PI+IV")
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
TINT_ROW_HEIGHT: float = 1.1
TINT_DENSITY_HEIGHT: float = 1.5
TINT_WIDTH: float = 2 * PANEL_WIDTH
TINT_YLIM: tuple[float, float] = (-0.05, 1.05)
TINT_TITLE: str = r"$h({\bm{x}})$"
# columns: blue image | the y tick labels | bounds | red image. Each image keeps
# TINT_IMAGE_SCALE of the width share it had in the former three-column grid (ratios
# 0.14 : 1 : 0.14 at wspace 0.04); the images are square and width-bound, so their
# height follows. The bounds take the rest
TINT_IMAGE_SCALE: float = 2 / 3
TINT_TICK_PAD: float = 0.06
TINT_WSPACE: float = 0.03
_TINT_FORMER: float = 0.14 / (1 + 2 * 0.14) * 3 / (3 + 2 * 0.04)
_TINT_SHARE: float = TINT_IMAGE_SCALE * _TINT_FORMER
# a GridSpec cell's share of the row is ratio / sum(ratios) * n / (n + (n - 1) wspace)
_TINT_CELLS: float = 4 / (4 + 3 * TINT_WSPACE)
_TINT_IMAGE: float = _TINT_SHARE * (1 + TINT_TICK_PAD) / (_TINT_CELLS - 2 * _TINT_SHARE)
TINT_WIDTHS: tuple[float, float, float, float] = (_TINT_IMAGE, TINT_TICK_PAD, 1.0, _TINT_IMAGE)
# the bounds' y ticks and their mid reference line, and the gap (figure fraction)
# between a blue image and the tick labels, and between a panel and its red image
TINT_YTICKS: tuple[float, float] = (0.0, 1.0)
# the 0.5 line: thin, light grey and dashed, behind the bound lines
TINT_MIDLINE: dict = {"y": 0.5, "color": "0.8", "linestyle": "--", "linewidth": 0.6, "zorder": 1}
TINT_LABEL_GAP: float = 0.003
# each interval method is its two bounds as solid lines, no fill; TINT_UNDER lowest
TINT_BOUND_WIDTH: float = 1.4
TINT_UNDER: str = "PI+INV"
# the point estimators the stack leaves out (the bands and the target are what it compares)
TINT_OMIT: tuple[str, ...] = ("ERM", "DA+ERM")
# the do-MNIST table: the bootstrap band's level (%) and resample count
TABLE_BAND: float = 95.0
TABLE_BOOTSTRAP: int = 1000


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


def _has_domnist_table(artifacts: str) -> bool:
    folder = _tint_folder(artifacts)
    return all(
        os.path.exists(f"{folder}/{name}") for name in ("run.json", "population_values.pkl", "population_outcomes.pkl")
    )


def _has_perf(artifacts: str, dataset: str, metric: str) -> bool:
    folder = f"{artifacts}/{dataset}/{SUBDIR_PERF}"
    return os.path.exists(f"{folder}/epsilon_values.pkl") and os.path.exists(f"{folder}/epsilon_{metric}_results.pkl")


def column_has_z(artifacts: str, dataset: str) -> bool:
    """Whether the run behind a column had a real Z: `has_z` of the `labels.json`
    every run writes beside its pkls. A missing or unreadable file (a tree from
    before the file, a torn write) is drawn as null Z, with a warning."""
    path = f"{artifacts}/{dataset}/labels.json"
    try:
        with open(path) as fh:
            value = json.load(fh)["has_z"]
        if isinstance(value, bool):
            return value
        logger.warning(f"aggregate: {path} has a non-bool has_z {value!r}; its column is drawn as null Z.")
    except FileNotFoundError:
        logger.warning(f"aggregate: no labels.json under {artifacts}/{dataset}; its column is drawn as null Z.")
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        logger.warning(f"aggregate: unreadable {path} ({error}); its column is drawn as null Z.")
    return False


# ------------------------------------------------------------------ drawing


def _frame(ax, x, xscale: str, vlines, xticks=()) -> None:
    """The exact grid plus the sweep figures' margin, the in-frame reference lines,
    and `xticks` as the only labelled x ticks when given (`fix_x_ticks`)."""
    ax.set_xscale(xscale)
    ax.set_xlim(_pad(ax.xaxis, float(x.min()), float(x.max()), frac=X_MARGIN))
    fix_x_ticks(ax, xticks)
    ax.tick_params(labelsize=FS_TICK)
    x_lo, x_hi = ax.get_xlim()
    for v in vlines:
        if np.isfinite(v) and x_lo <= v <= x_hi:
            ax.axvline(v, color="0.4", linestyle=":", linewidth=1.0, zorder=0)


def _legend(fig, handles: dict):
    """One legend for the figure, laid out by entry count. `handles` is
    {signature: (artist, MethodStyle, name)}: entries that render as the same
    pixels (label, hue, alpha, dash) are one entry, which is what blends a T-as-IV
    method's three mode spellings and, in a merged grid, a null-Z method with its Z
    counterpart. Then n entries up to LEGEND_FLAT_MAX are one row of n in the
    repo's order; more are LEGEND_ROWS rows of ceil(n / LEGEND_ROWS) columns, paired
    groups first, each in one column (member 0 on top, member 1 below), and the
    singletons after them in legend order, two to a column. Returns the legend, or
    None with nothing drawn."""
    entries = _legend_entries(handles)
    if entries is None:
        return None
    artists, labels, ncol = entries
    return fig.legend(
        artists,
        labels,
        loc="upper center",
        ncol=ncol,
        bbox_to_anchor=(0.5, 1.0),
        fontsize=FS_TICK,
        frameon=True,
        edgecolor="black",
        fancybox=False,
    )


def _panel_legend(ax, handles: dict):
    """The sweep grids' legend, inside `ax` at SWEEP_LEGEND_LOC: `_legend`'s render
    fold in the repo's legend order, split into two columns, the single intervals on
    the left and the SWEEP_LEGEND_RIGHT bases on the right (one column when either
    side is empty). Returns the legend, or None with nothing drawn."""
    if not handles:
        return None
    keys = sorted(handles, key=lambda k: _legend_key(handles[k][2], handles[k][1]))
    keys, _ = fold_by_signature((k, handles[k][1]) for k in keys)
    right = [k for k in keys if parse_method(handles[k][2]).base in SWEEP_LEGEND_RIGHT]
    left = [k for k in keys if k not in right]
    columns = [side for side in (left, right) if side]
    depth = max(len(side) for side in columns)
    # matplotlib fills column-major, so the shorter column is padded with blank
    # slots to keep each side to its own column
    blank = Line2D([], [], linestyle="none")
    artists, labels = [], []
    for side in columns:
        artists += [handles[k][0] for k in side] + [blank] * (depth - len(side))
        labels += [handles[k][1].label for k in side] + [""] * (depth - len(side))
    return ax.legend(
        artists,
        labels,
        loc=SWEEP_LEGEND_LOC,
        ncol=len(columns),
        fontsize=FS_TICK - SWEEP_LEGEND_SHRINK,
        handlelength=1.2,
        columnspacing=0.8,
        frameon=True,
        edgecolor="black",
        fancybox=False,
    )


def _legend_key(name: str, style) -> tuple:
    """The repo's legend order: the family slot (`constants.legend_order`, member 0
    above member 1), then ALL_METHODS, then the IV mode."""
    base, mode = parse_method(name)
    return (
        style.order,
        ALL_METHODS.index(base) if base in ALL_METHODS else len(ALL_METHODS),
        IV_MODES.index(mode),
    )


def _legend_entries(handles: dict):
    """`_legend`'s entries without the legend: (artists, labels, ncol) in the
    repo's order after the render fold, or None with nothing drawn. `handles` is
    {key: (artist, MethodStyle, name)}."""
    if not handles:
        return None

    def pair(k):
        return handles[k][1].order

    def order(k):
        return _legend_key(handles[k][2], handles[k][1])

    # the fold keeps the first key of every render signature, in legend order
    keys, _ = fold_by_signature((k, handles[k][1]) for k in sorted(handles, key=order))

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
    # singleton and the guarantee goes quietly; unreachable with today's legend order,
    # so say it, do not raise -- a wrapped legend is still readable
    crowded = sorted(g for g, size in sizes.items() if size > 2)
    if crowded:
        logger.warning(
            f"aggregate: legend groups over two entries {crowded}; the legend may split a group across columns."
        )
    return [handles[k][0] for k in keys], [handles[k][1].label for k in keys], ncol


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
    hz: dict[str, bool],
    legend_panel: bool = False,
):
    """The [rows] x [datasets] grid: `rows` is [(metric, row label)], `load_cell(dataset,
    metric)` gives (x, {method: y}, x-label[, x ticks]) or None for a blank cell (an
    empty dict blanks the cell but still offers its x-label), x shared within a column, y as
    `sharey` says, each row on `row_yscale(metric)` and, when `row_ylim(metric)` is not
    None, framed there. Titles on the first row, one x-label, one legend: above the
    titles, or with `legend_panel` inside the middle column's top panel
    (`_panel_legend`); saved at `path` when given. `hz` is each column's `has_z`: when a column that draws has a
    real Z, the grid is merged and every null-Z column draws each method as its Z
    counterpart (`constants.method_style`), so each cross-dataset pair is one
    legend entry, and each null-Z column title gets `NULL_Z_TITLE_SUFFIX`."""
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
    # every cell read once, in drawing order; a column draws when a cell of it has a
    # series, and only a drawing column's Z merges the grid (a Z dataset present as
    # a blank column for this parameter must not relabel the others)
    cells = {(dataset, metric): load_cell(dataset, metric) for dataset in datasets for metric, _ in rows}
    drawing = [d for d in datasets if any(cells[d, m] is not None and cells[d, m][1] for m, _ in rows)]
    merged = any(hz[d] for d in drawing)
    handles, xlabel = {}, None
    for c, dataset in enumerate(datasets):
        # a merged grid marks each column without a Z, blank or not: its entries in
        # the shared legend read the Z labels
        suffix = NULL_Z_TITLE_SUFFIX if merged and not hz[dataset] else ""
        axes[0, c].set_title(DATASET_TITLES[dataset] + suffix, fontsize=FS_LABEL)
        labelled = False
        for r, (metric, _) in enumerate(rows):
            ax, cell = axes[r, c], cells[dataset, metric]
            if cell is None:
                ax.axis("off")
                continue
            # a fourth element, when given, is the column's fixed x ticks
            x, y, found, *ticks = cell
            if not labelled:
                # one x-label per column, the first cell's
                xlabel, labelled = _xlabel(xlabel, found, f"{dataset} {where}"), True
            if not y:
                ax.axis("off")
                continue
            drawn, _ = _draw_series(ax, x, y, has_z=hz[dataset], merged=merged)
            for name, handle in drawn.items():
                # keyed by render: one method drawn two ways in two columns keeps both
                style = method_style(name, hz[dataset], merged=merged)
                handles.setdefault(style.signature, (handle, style, name))
            _frame(ax, x, xscale, vlines, ticks[0] if ticks else ())
            ax.set_yscale(row_yscale(metric))
            ylim = row_ylim(metric)
            if ylim is not None:
                ax.set_ylim(*ylim)
    _label_rows(axes, [label for _, label in rows])
    if legend_panel:
        # inside a panel it takes no room above the titles
        _panel_legend(axes[0, len(datasets) // 2], handles)
        return _finish(fig, axes, xlabel or xlabel_default, None, path)
    legend = _legend(fig, handles)
    return _finish(fig, axes, xlabel or xlabel_default, legend, path)


def _columns_has_z(artifacts: str, datasets: list[str], hz: dict[str, bool] | None) -> dict[str, bool]:
    """`hz` when the caller read it (`main`, once per column), else each column's."""
    return hz if hz is not None else {d: column_has_z(artifacts, d) for d in datasets}


def sweep_grid(param: str, datasets: list[str], artifacts: str, out: str | None = None, hz=None):
    """The 3 x [datasets] grid of one sweep parameter; saved under `out` when given.
    `hz` is {dataset: has_z}, read from each column's `labels.json` when None."""
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
                # the axis pkl says what x is; a tree written before the param had
                # one reads the legacy label (n: absolute rows, not the percentage)
                axis_pkl = f"{folder}/{param}_axis.pkl"
                axis = load(axis_pkl) if os.path.exists(axis_pkl) else None
                found = axis.get("xlabel", spec.xlabel) if axis else (spec.legacy_xlabel or spec.xlabel)
                ticks = tuple(axis.get("xticks", ())) if axis else ()
                loaded[dataset] = (x[order], order, load(results), found, ticks)
        return loaded[dataset]

    # one x-label per figure: the current spec's meaning wins when any column carries
    # it, else the first column's; a stale column (an absolute-n tree beside a
    # percentage one) is blanked with a warning, never drawn under the wrong label
    drawn = [d for d in datasets if column(d) is not None]
    labels = [column(d)[3] for d in drawn]
    reference = spec.xlabel if spec.xlabel in labels else (labels[0] if labels else None)
    for dataset in drawn:
        if column(dataset)[3] != reference:
            logger.warning(
                f"aggregate: {dataset} {param} x is {column(dataset)[3]!r}, the figure's "
                f"{reference!r}; {dataset} left blank."
            )
            loaded[dataset] = None

    def load_cell(dataset, metric):
        if column(dataset) is None:
            return None
        x, order, record, found, ticks = column(dataset)
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
        return x, y, found, ticks

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
        hz=_columns_has_z(artifacts, datasets, hz),
        legend_panel=True,
    )


def perf_grid(stem: str, metrics, datasets: list[str], artifacts: str, out: str | None = None, hz=None):
    """The [metrics] x [datasets] grid of the perf sweeps against epsilon, a y shared
    within each row, the rates on `CLAMP_YLIM`; saved as `epsilon_{stem}` under `out`.
    `hz` as `sweep_grid`'s."""
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
        hz=_columns_has_z(artifacts, datasets, hz),
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
    has_z = column_has_z(artifacts, "cigarettes")
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
            drawn, panel_lo, panel_hi = _draw_bands(ax, x, load(f"{stem}_outcomes.pkl"), has_z=has_z)
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
        # in the repo's legend order, not the pkls' method order
        names = sorted(handles, key=lambda n: _legend_key(n, method_style(n, has_z)))
        legend_ax.legend(
            *fold_by_signature((handles[n], method_style(n, has_z)) for n in names),
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
    ax.set_aspect("equal")
    ax.axis("off")


def _tint_bounds(ax, x, outcomes: dict, *, has_z: bool) -> dict:
    """The stack's own rendering: each interval method's lower and upper bound as
    two solid lines in its hue at one width (TINT_UNDER below the rest), the point
    estimates as the query figures draw them. Returns {method: legend handle}."""
    points = {name: p for name, p in outcomes.items() if np.ndim(p) != 3}
    drawn = dict(_draw_bands(ax, x, points, has_z=has_z)[0]) if points else {}
    for name, prediction in outcomes.items():
        if name in points:
            continue
        prediction = np.asarray(prediction, dtype=float)
        style = dict(
            color=method_style(name, has_z).colour,
            linestyle="-",
            linewidth=TINT_BOUND_WIDTH,
            zorder=2.0 if name == TINT_UNDER else 2.1,
        )
        lower = ax.plot(x, prediction[:, :, 0].mean(axis=1), label=method_label(name, has_z), **style)[0]
        ax.plot(x, prediction[:, :, 1].mean(axis=1), **style)
        drawn[name] = lower
    # in `outcomes` order, keyed like `_draw_bands`
    return {name: drawn[name] for name in outcomes if name in drawn}


def _clear_tick_labels(fig, bounds_axes, image_axes) -> None:
    """Set the blue images' ink `TINT_LABEL_GAP` left of the bounds' y tick labels
    and the red images the same gap right of their panel, each image keeping its size."""
    if not image_axes:
        return
    fig.canvas.draw()
    to_figure = fig.transFigure.inverted()
    labels = [t.get_window_extent() for ax in bounds_axes for t in ax.get_yticklabels() if t.get_text()]
    if not labels:
        return
    left = to_figure.transform((min(b.x0 for b in labels), 0))[0]
    right = max(ax.get_position().x1 for ax in bounds_axes)
    # the gap is to the INK: the image is 28 pixels wide and the digit sits in its
    # middle, so the box edge would leave a margin of blank pixels on top of the gap
    ink = [np.flatnonzero(blue.images[0].get_array()[..., 3].max(axis=0) > 0.1) for blue, _ in image_axes]
    width = max(blue.images[0].get_array().shape[1] for blue, _ in image_axes)
    inked = max((cols[-1] + 1) / width for cols in ink if len(cols)) if any(len(c) for c in ink) else 1.0
    for blue, red in image_axes:
        box = blue.get_position()
        blue.set_position([left - TINT_LABEL_GAP - inked * box.width, box.y0, box.width, box.height])
        box = red.get_position()
        red.set_position([right + TINT_LABEL_GAP, box.y0, box.width, box.height])


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
    density_path = f"{folder}/tint_density.pkl"
    has_density = os.path.exists(density_path)
    rows = len(digits) + int(has_density)
    heights = [TINT_ROW_HEIGHT] * len(digits) + ([TINT_DENSITY_HEIGHT] if has_density else [])
    fig = plt.figure(figsize=(TINT_WIDTH, sum(heights)))
    grid = GridSpec(
        rows, 4, figure=fig, width_ratios=TINT_WIDTHS, height_ratios=heights, hspace=0.12, wspace=TINT_WSPACE
    )

    has_z = column_has_z(artifacts, "do_mnist")
    handles, grids, shared, bounds_axes, image_axes = {}, {}, None, [], []
    for r, digit in enumerate(digits):
        x = np.asarray(load(f"{folder}/tint_{digit}_values.pkl"), dtype=float)
        grids[digit] = x
        ax = fig.add_subplot(grid[r, 2], sharex=shared)
        shared = shared or ax
        outcomes = {k: v for k, v in load(f"{folder}/tint_{digit}_outcomes.pkl").items() if k not in TINT_OMIT}
        outcomes = dict(sorted(outcomes.items(), key=lambda item: item[0] != TINT_UNDER))
        drawn = _tint_bounds(ax, x, outcomes, has_z=has_z)
        mark_failed(ax, x, outcomes, has_z=has_z)
        for name, handle in drawn.items():
            style = method_style(name, has_z)
            handles.setdefault(style.signature, (handle, style, name))
        ax.set_ylim(*TINT_YLIM)
        ax.set_yticks(TINT_YTICKS, [f"{t:g}" for t in TINT_YTICKS])
        ax.axhline(**TINT_MIDLINE)
        ax.tick_params(labelsize=FS_TICK - 4, labelbottom=False)
        bounds_axes.append(ax)
        images = f"{folder}/tint_{digit}_images.pkl"
        if os.path.exists(images):
            blue, red = load(images)
            image_axes.append((fig.add_subplot(grid[r, 0]), fig.add_subplot(grid[r, 3])))
            _tint_image(image_axes[-1][0], blue)
            _tint_image(image_axes[-1][1], red)
    # the title and the x-label at the size of the "density" label
    bounds_axes[0].set_title(TINT_TITLE, fontsize=FS_TICK)
    _tint_provenance(artifacts, digits, grids)

    bottom = bounds_axes[-1]
    if has_density:
        density = load(density_path)
        bottom = fig.add_subplot(grid[-1, 2], sharex=shared)
        # the query panel's density row, as the simulation and optical figures draw it
        draw_da_density(bottom, np.asarray(density["before"]), np.asarray(density["after"]))
        bottom.set_ylabel("density", fontsize=FS_TICK)
        bottom.tick_params(labelsize=FS_TICK - 4)
    bottom.tick_params(labelbottom=True)
    bottom.set_xlabel(ANNOTATE_SWEEP_PLOT["tint"]["xlabel"], fontsize=FS_TICK)
    x = grids[digits[0]]
    shared.set_xlim(float(x.min()), float(x.max()))
    _label_major_ticks_only(*bounds_axes, bottom)

    # the one legend, the methods', in the bottom-right cell beside the histogram
    entries = _legend_entries(handles)
    if entries is not None:
        side = fig.add_subplot(grid[-1, 3])
        side.axis("off")
        side.legend(
            entries[0],
            entries[1],
            loc="lower left",
            bbox_to_anchor=(0.0, 0.0),
            ncol=1,
            fontsize=FS_TICK - 5,
            handlelength=1.2,
            frameon=True,
            edgecolor="black",
            fancybox=False,
        )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.6 / fig.get_figheight())
    _clear_tick_labels(fig, bounds_axes, image_axes)
    path = None if out is None else f"{out}/do_mnist_tint.{PLOT_FORMAT}"
    if path is not None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, format=PLOT_FORMAT, dpi=PLOT_DPI, bbox_inches="tight")
        logger.info(f"aggregate: wrote {path}")
    return fig


def fit_parts(method: str, inv_recenter: str) -> tuple[str, ...]:
    """The `run.json` seconds one method needs before its first solve: the nets it
    is centred on, the DA passes it consumes (the A-draw pass for a centre trained
    on GX, the B-row pass for a method that fits on or pairs with GX), its own
    `fit_model` and its floor. A net shared by several methods is charged in full
    to each, so every row is the cost of running that method alone."""
    fit = (f"fit_seconds_{method}", f"floor_seconds_{method}")
    da = ("augment_seconds_A", "augment_seconds_B")
    if method == "PI":
        return ("train_seconds_X", *fit)
    if method in ("DA+PI", "DA+PI+IV"):
        return ("train_seconds_GX", *da, *fit)
    if method in ("PI&DA+PI", "PI&DA+PI+IV"):
        return ("train_seconds_X", "train_seconds_GX", *da, *fit)
    if method == "PI+INV":
        centre = {"off": ("train_seconds_X",), "on": ("train_seconds_GX", "augment_seconds_A")}.get(
            inv_recenter, ("train_seconds_INV", "augment_seconds_A")
        )
        return (*centre, "augment_seconds_B", *fit)
    raise ValueError(f"no fit charge for {method!r}")


def _band(values, fmt: str) -> tuple[str, str]:
    """(`mean`, `[lo, hi]`) cells of a per-query series: the nan-mean and its
    bootstrap band, the band set small for the line under the row."""
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).any():
        return "--", ""
    mean = float(np.nanmean(values))
    draws = np.asarray(bootstrap({"PI": values}, n_samples=TABLE_BOOTSTRAP)["PI"]).ravel()
    tail = (100.0 - TABLE_BAND) / 2
    lo, hi = np.nanpercentile(draws, [tail, 100.0 - tail])
    return f"${mean:{fmt}}$", rf"{{\scriptsize$[{lo:{fmt}}, {hi:{fmt}}]$}}"


#: the do-MNIST net of records written before `net` was recorded
FLAT_NET = "domnist-fast"


def _calibration(artifacts: str, run: dict) -> list[str]:
    """The comment lines on where gamma came from. The calibration sentence and
    the split-C coverage are written only when the `gamma_selection.json` beside
    the run selected this very gamma on this very split with the same nets;
    otherwise a WARNING."""
    net = run.get("net", FLAT_NET)
    head = f"% gamma = {float(run['gamma']):.6g} for every method, nets {net}"
    path = f"{artifacts}/do_mnist/select/gamma_selection.json"
    if not os.path.exists(path):
        return [f"{head}; no gamma_selection.json beside the run, so its calibration is not checked."]
    with open(path) as fh:
        selection = json.load(fh)
    shared, method = selection.get("shared_gamma"), selection.get("calibrated_on")
    same_gamma = shared is not None and bool(np.isclose(float(run["gamma"]), float(shared), rtol=1e-9, atol=0.0))
    same_split = selection.get("split_key") is not None and selection.get("split_key") == run.get("split_key")
    same_net = selection.get("net", FLAT_NET) == net
    if not (same_gamma and same_split and same_net and method in selection):
        logger.warning(
            f"aggregate: the run's gamma {float(run['gamma']):.6g} (split {run.get('split_key')}, nets {net}) is "
            f"not the selection's shared gamma {shared} (split {selection.get('split_key')}, nets "
            f"{selection.get('net', FLAT_NET)}); the table says so."
        )
        return [
            f"{head}; gamma not from the selection beside it (shared_gamma {shared}, split_key "
            f"{'matches' if same_split else 'differs'}, nets {'match' if same_net else 'differ'})."
        ]
    lines = [
        f"{head}, calibrated on {method} to target_coverage {selection.get('target_coverage')} on split C "
        f"({selection.get('n_select')} rows)."
    ]
    cells = [f"{method} {selection[method]['coverage']:.4f}"]
    for other, entry in selection.items():
        if isinstance(entry, dict) and "at_shared_gamma" in entry:
            cells.append(f"{other} {entry['at_shared_gamma']['coverage']:.4f}")
    lines.append(f"% split-C coverage at that gamma: {', '.join(cells)}.")
    return lines


def _load_line(run: dict, method: str, parts: list[str]) -> str | None:
    """One method's load averages around its timed phases: the net it trains
    (`train_load_*`), its `fit_model` plus floor, and its population solve. None
    for a run.json that predates the load records."""

    def span(pair) -> str:
        return f"{pair[0]:.1f} -> {pair[1]:.1f}"

    cells = [
        f"net {key.removeprefix('train_seconds_')} {span(run[load])}"
        for key in parts
        if key.startswith("train_seconds_") and (load := key.replace("train_seconds_", "train_load_")) in run
    ]
    phases = run.get(f"load_{method}") or {}
    for label, phase in (("fit", "fit"), ("solve", "predict")):
        if f"{phase}_before" in phases and f"{phase}_after" in phases:
            cells.append(f"{label} {span((phases[f'{phase}_before'], phases[f'{phase}_after']))}")
    return f"{method}: {', '.join(cells)}." if cells else None


def _seconds(run: dict, key: str) -> float:
    """A `run.json` time, 0 when absent (a method without a floor has none)."""
    value = run.get(key)
    return float(value) if value is not None and np.isfinite(value) else 0.0


def _omega_line(run: dict) -> str:
    """The table's comment on Omega-hat: which rho it multiplies. A run.json from
    before the nets' rho has no `rho_linear`, and its `rho` IS the linear one."""
    rho, trace = float(run.get("rho", float("nan"))), float(run.get("tr_S_over_k", float("nan")))
    head = f"% Omega-hat = rho-hat tr(S)/k on the mixed B rows (Prop. 2), tr(S)/k {trace:.4f} from the linear shift "
    if "rho_linear" not in run:
        return head + f"operators and rho-hat {rho:.4f} the pixel-logistic ratio (a run before the nets' rho)."
    band = run.get("rho_band")
    band = f" (95% paired bootstrap band [{band[0]:.4f}, {band[1]:.4f}])" if band else ""
    return head + (
        f"operators and rho-hat {rho:.4f}{band} = DA+ERM's / ERM's held-out squared error on those rows (the paper's "
        f"best-achievable errors, via the nets); the pixel-logistic rho_linear {float(run['rho_linear']):.4f} is "
        "kept for provenance only."
    )


def domnist_table(artifacts: str, out: str | None = None) -> str | None:
    """The do-MNIST results table on the evaluation population, written to
    `out/do_mnist_table.tex` when given; returns the tex, or None without the pkls.

    One row per interval method in `ALL_METHODS` order. Columns: coverage of h_*
    (NaN bounds count as not covered, as in `run.json`) and mean width, each with a
    bootstrap band over the queries; Omega-hat = rho-hat tr(S)/k from the prescreen,
    on the DA+ rows, with rho-hat the nets' ratio (`diagnostics.net_noise`); worst
    error (E+, the mean over queries of the larger squared bound error); the one-time
    fit in seconds (every part in `fit_parts`); the mean per-query solve in ms (the
    population predict at the run's `n_jobs`); and the latency, the fit plus the mean
    per-query solve, in seconds.
    """
    if not _has_domnist_table(artifacts):
        return None
    folder = _tint_folder(artifacts)
    with open(f"{folder}/run.json") as fh:
        run = json.load(fh)
    target = np.asarray(load(f"{folder}/population_values.pkl"), dtype=float).ravel()
    outcomes = load(f"{folder}/population_outcomes.pkl")
    recenter = run.get("inv_recenter", "off")
    has_z = column_has_z(artifacts, "do_mnist")
    methods = [m for m in ALL_METHODS if m in outcomes and np.ndim(outcomes[m]) == 2 and np.shape(outcomes[m])[1] == 2]
    omega = float(run["rho"]) * float(run["tr_S_over_k"]) if "rho" in run and "tr_S_over_k" in run else float("nan")

    rows, charges, solved, loads = [], [], [], []
    for m in methods:
        bounds = np.asarray(outcomes[m], dtype=float)
        lo, hi = bounds[:, 0], bounds[:, 1]
        empty = ~np.isfinite(bounds).all(axis=1)
        covered = np.where(empty, 0.0, ((lo <= target) & (target <= hi)).astype(float))
        if empty.all():
            covered = np.full(len(target), np.nan)
        width = np.where(empty, np.nan, hi - lo)
        parts = [k for k in fit_parts(m, recenter) if k in run]
        fit = sum(_seconds(run, k) for k in parts)
        solve = float(run.get(f"wall_clock_{m}", float("nan")))
        charges.append(f"{m}: " + " + ".join(parts))
        loads.append(_load_line(run, m, parts))
        solved.append(f"{m} {int((~empty).sum())}/{len(empty)}")
        (coverage, coverage_band), (mean_width, width_band) = _band(covered, ".3f"), _band(width, ".3f")
        rows.append(
            [
                method_label(m, has_z),
                coverage,
                mean_width,
                f"${omega:.3f}$" if "DA+" in m and np.isfinite(omega) else "--",
                f"${float(run.get(f'worst_error_{m}', float('nan'))):.4f}$",
                f"${fit:.2f}$",
                f"${1e3 * solve:.2f}$",
                f"${fit + solve:.2f}$",
            ]
        )
        # the bands on a line of their own under the row, so the table keeps to a page
        rows.append(["", coverage_band, width_band, "", "", "", "", ""])

    status = run.get("status_PI+INV")
    feasible = (
        (status["feasible_and_covers"] + status["feasible_and_noncovering"]) / max(sum(status.values()), 1)
        if status
        else float("nan")
    )
    nets = ", ".join(
        f"{k} {_seconds(run, f'train_seconds_{k}'):.2f}" for k in ("X", "GX", "INV") if f"train_seconds_{k}" in run
    )
    rmse = ", ".join(f"{m} {run[f'rmse_{m}']:.4f}" for m in ("ERM", "DA+ERM", "ERM+INV") if f"rmse_{m}" in run)
    lines = [
        f"% do-MNIST results on the evaluation population: {run.get('n_queries')} MNIST-test draws at pop_seed "
        f"{run.get('pop_seed')}, never in A/B/C.",
        *_calibration(artifacts, run),
        f"% epsilon {run.get('epsilon')}; inv_recenter {recenter}; erm_inv_tau {run.get('erm_inv_tau')}.",
        f"% coverage and width: mean, and under it a {TABLE_BAND:g}% bootstrap band over the queries "
        f"({TABLE_BOOTSTRAP} resamples). Coverage counts a query without a bound (NaN) as not covered; width, "
        "worst error and the band of width are over the solved queries only.",
        f"% solved queries: {', '.join(solved)}.",
        _omega_line(run),
        "% solve (ms) is the mean per query; lat. (s) = the one-time fit (s) + that mean solve; the fit charges "
        "every part a method "
        "needs before its first solve, a shared net in full to each row:",
        *[f"%   {charge}" for charge in charges],
        f"% solves at n_jobs {run.get('toggle_n_jobs')} on {run.get('cpu_count')} cores; the CopSens fits cap "
        f"BLAS at {run.get('blas_threads')} threads.",
        f"% net training (s): {nets}; augmentation (s): A {_seconds(run, 'augment_seconds_A'):.2f}, "
        f"B {_seconds(run, 'augment_seconds_B'):.2f}.",
        *(
            [f"% 1-min load average, before -> after each timed phase, on {run.get('cpu_count')} cores:"]
            + [f"%   {line}" for line in loads if line]
            if any(loads)
            else []
        ),
        f"% PI+INV floor {run.get('inv_floor', float('nan')):.4g} vs budget eps^2 "
        f"{run.get('inv_budget', float('nan')):.4g}, feasible share {feasible:.3f}.",
        f"% point estimators, population RMSE to h_*: {rmse}.",
    ]
    head = ["", "coverage", "width", r"$\hat{\Omega}$", "worst err.", "fit (s)", "solve (ms)", "lat. (s)"]
    lines += [
        # small type and narrow gaps, so the eight columns keep to an article text width
        r"{\small\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{l" + "r" * (len(head) - 1) + "}",
        r"\toprule",
        " & ".join(head) + r" \\",
        r"\midrule",
        *[" & ".join(row) + r" \\" for row in rows],
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
    ]
    tex = "\n".join(lines) + "\n"
    if out is not None:
        os.makedirs(out, exist_ok=True)
        path = f"{out}/do_mnist_table.tex"
        with open(path, "w") as fh:
            fh.write(tex)
        logger.info(f"aggregate: wrote {path}")
    return tex


# ------------------------------------------------------------------ cli


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifacts", default="./artifacts", help="the tree to read (default ./artifacts)")
    parser.add_argument("--out", default=None, help="where the grids go (default <artifacts>/aggregate)")
    args = parser.parse_args(argv)
    artifacts = os.path.abspath(args.artifacts)
    out = os.path.abspath(args.out) if args.out else f"{artifacts}/aggregate"

    datasets = columns(artifacts)
    drawn = _has_elasticities(artifacts) or _has_tint(artifacts) or _has_domnist_table(artifacts)
    if not datasets and not drawn:
        logger.warning(f"aggregate: nothing to draw under {artifacts}.")
        return
    # each column's has_z, read (and warned about) once
    hz = {d: column_has_z(artifacts, d) for d in datasets}
    for param in sweep_params(artifacts, datasets):
        plt.close(sweep_grid(param, datasets, artifacts, out, hz=hz))
    for stem, metrics in PERF_FIGURES:
        # the rows some dataset ran; a figure none of whose metrics ran is not drawn
        ran = [m for m in metrics if any(_has_perf(artifacts, d, m) for d in datasets)]
        if ran:
            plt.close(perf_grid(stem, ran, datasets, artifacts, out, hz=hz))
    if _has_elasticities(artifacts):
        plt.close(elasticity_grid(artifacts, out))
    if _has_tint(artifacts):
        plt.close(tint_stack(artifacts, out))
    if _has_domnist_table(artifacts):
        domnist_table(artifacts, out)


if __name__ == "__main__":
    main()
