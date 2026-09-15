"""
Plotting utilities for experiment results.
"""

import warnings
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from loguru import logger
from matplotlib.ticker import LogLocator, MaxNLocator, NullFormatter, ScalarFormatter
from numpy.typing import NDArray

from .constants import (
    _STYLE_KEYS,
    ALPHA_MAP,
    COLOR_MAP,
    DEFAULT_HILIGHT_OURS,
    DEFAULT_NORMALIZE_SWEEP,
    FS_LABEL,
    FS_TICK,
    NORMALIZE_BASELINES,
    NORMALIZED_SWEEP_SUFFIXES,
    PAGE_WIDTH,
    PANEL_CONFIGS,
    PARTIAL_IDENTIFICATION_STYLE,
    PLOT_CONFIGS,
    PLOT_DPI,
    PLOT_FORMAT,
    POINT_ESTIMATE_STYLE,
    POINT_ESTIMATES,
    RC_PARAMS,
    SUBDIR_PERF,
    SUBDIR_QUERY,
    SUBDIR_SWEEP,
    TEX_MAPPER,
)
from .data_operations import bootstrap, save

PlotScale = Literal["linear", "log", "symlog", "asinh"]
# margin beyond the outermost mark on a query figure that carries marks, as a
# fraction of the framed span, so no mark lies on the frame
X_MARK_MARGIN: float = 0.02

# clip the top tail of the pooled means, y only. Errors/widths: small is the signal,
# large is the runaway. A symmetric floor crops the TIGHTEST method, which is the
# result. x is a grid: every point is the result, so it is never clipped; it gets
# X_MARGIN of its transformed span on each side, enough to lift the r = 1 line off
# the frame, too little to read as widening (the 5 % `_pad` did).
CLIP_PERCENTILE: float = 98.0
X_MARGIN: float = 0.02
# asinh knee, as a fraction of the upper limit. PANEL_CONFIGS' own ylim/linear_width.
LINEAR_WIDTH_RATIO: float = 40.0
# promote linear -> log past this dynamic range. Fires on nothing today; a guard.
LOG_PROMOTE_RATIO: float = 100.0


def _plot_config(experiment: str, fname: str | None) -> dict[str, object]:
    """PLOT_CONFIGS['*'] under PLOT_CONFIGS[experiment], key by key."""
    if not fname:
        return {}
    return {**PLOT_CONFIGS.get("*", {}).get(fname, {}), **PLOT_CONFIGS.get(experiment, {}).get(fname, {})}


def _style(cfg: dict[str, object], **given) -> dict[str, object]:
    """
    The five style keys (constants._STYLE_KEYS), resolved: the plot config wins
    over the function's own argument, which wins over the default.

    `legend` is three-valued: None leaves the function's `hide_legend` /
    `legend_loc` in charge; False hides; True shows; a str or (x, y) tuple shows
    at that loc.
    """
    return {key: cfg.get(key, given[key]) for key in _STYLE_KEYS}


def _apply_style(ax, style: dict[str, object], xlabel: str, ylabel: str):
    """
    Axis labels and their major tick labels in their colours; a title only when
    one is given. `plt.setp` on the tick labels is what `plt.xticks(**kwargs)`
    does; ticks created later copy the first tick's properties.
    """
    ax.set_xlabel(xlabel, fontsize=FS_LABEL, color=style["x_color"])
    plt.setp(ax.get_xticklabels(), fontsize=FS_TICK, color=style["x_color"])
    ax.set_ylabel(ylabel, fontsize=FS_LABEL, color=style["y_color"])
    plt.setp(ax.get_yticklabels(), fontsize=FS_TICK, color=style["y_color"])
    if style["title"]:
        ax.set_title(style["title"], fontsize=FS_LABEL, color=style["title_color"])


def _legend_choice(style: dict[str, object], hide_legend: bool, legend_loc) -> tuple[bool, object]:
    """(hide, loc) after the `legend` key; when it is None the arguments decide."""
    legend = style["legend"]
    if legend is None:
        return hide_legend, legend_loc
    if isinstance(legend, str | tuple):
        return False, legend
    return legend is False, legend_loc


def _finite(*arrays) -> NDArray:
    """Pool the finite values. Empty is a valid answer; every caller handles it."""
    if not arrays:
        return np.array([], dtype=float)
    values = np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrays])
    return values[np.isfinite(values)]


def _limits(series: list[NDArray], clip: bool = True) -> tuple[float, float] | None:
    """
    Clip the top tail of the pooled means, then guarantee no series goes blank.

    Point estimates only -- CI bands and SE crosshairs are deliberately excluded and
    left to clip against the frame. `clip=False` is the exact pooled [min, max]:
    the x grid, whose last point (r = 1, n = 1024, m = 16) the clip used to drop
    along with the r = 1 reference line (`_rescale` adds the X_MARGIN).
    """
    pooled = _finite(*series)
    if not len(pooled):
        return None

    lo, hi = float(pooled.min()), float(np.percentile(pooled, CLIP_PERCENTILE) if clip else pooled.max())
    # a series may lose points, it can never vanish
    for values in series:
        finite = _finite(values)
        if len(finite):
            hi = max(hi, float(np.median(finite)))
    # flat: no range to set, let matplotlib expand around it
    return (lo, hi) if hi > lo else None


def _scale_kwargs(scale: str, cfg: dict[str, object], upper: float | None = None) -> dict[str, float]:
    """asinh/symlog knee. Defaults to `upper / 40` -- at 1.0 an asinh axis over data
    at ~1e-3 is asinh in name and linear in fact."""
    if scale not in ("asinh", "symlog"):
        return {}

    key = "linear_width" if scale == "asinh" else "linthresh"
    default = 1.0 if scale == "asinh" else 0.1
    knee = cfg.get(key)
    if knee is None and upper is not None and np.isfinite(upper) and upper > 0:
        knee = upper / LINEAR_WIDTH_RATIO
    return {key: float(knee) if knee and knee > 0 else default}


def _resolve_scale(
    scale: PlotScale,
    values: NDArray,
    cfg: dict[str, object],
    axis: str,
    limits: tuple[float, float] | None = None,
    promote: bool = True,
) -> tuple[PlotScale, dict[str, float]]:
    """cfg > the caller's spec > auto-promote, then one safety clamp over the winner."""
    scale = cfg.get(f"{axis}scale", scale)
    finite = _finite(values)

    if promote and scale == "linear" and len(finite) and np.all(finite > 0):
        lo, hi = np.percentile(finite, [0.5, 99.5])
        if lo > 0 and hi / lo >= LOG_PROMOTE_RATIO:
            scale = "log"

    # never crash, never blank: a hand-set log over signed data becomes asinh, which
    # still shows every point, rather than linear, which flattens it
    if scale == "log" and len(finite) and np.any(finite <= 0):
        logger.warning(f"{axis}scale=log with non-positive data; using asinh.")
        scale = "asinh"
    elif scale == "log" and not len(finite):
        scale = "linear"

    if scale not in ("asinh", "symlog"):
        for key in ("linear_width", "linthresh"):
            if key in cfg:
                logger.warning(f"{key} ignored: {axis}scale is {scale!r}.")

    return scale, _scale_kwargs(scale, cfg, limits[1] if limits else None)


def _pad(axis, lo: float, hi: float, frac: float = 0.05) -> tuple[float, float]:
    """Pad in the axis's own transformed space -- one expression for every scale."""
    transform = axis.get_transform()
    try:
        t_lo, t_hi = transform.transform([lo, hi])
        if not (np.isfinite(t_lo) and np.isfinite(t_hi)) or t_hi <= t_lo:
            return lo, hi
        margin = frac * (t_hi - t_lo)
        padded = transform.inverted().transform([t_lo - margin, t_hi + margin])
    except (ValueError, FloatingPointError):
        return lo, hi
    return tuple(padded) if np.all(np.isfinite(padded)) else (lo, hi)


def _apply_cfg_limits(limits: tuple[float, float] | None, cfg_limits, where: str) -> tuple[float, float] | None:
    """Element-wise override; None on either end keeps the computed edge."""
    if cfg_limits is None:
        return limits
    lo, hi = cfg_limits
    if limits is not None:
        lo = limits[0] if lo is None else lo
        hi = limits[1] if hi is None else hi
    if lo is None or hi is None:
        return limits
    if lo >= hi:
        raise ValueError(f"{where}: lo must be < hi, got {(lo, hi)}.")
    return float(lo), float(hi)


def _rescale(
    ax,
    cfg: dict[str, object],
    x_series: list[NDArray],
    y_series: list[NDArray],
    xscale: PlotScale,
    yscale: PlotScale,
    pad_x: bool = True,
    promote_x: bool = True,
):
    """Limits -> cfg -> scale -> pad -> set. Limits never depend on the scale.
    x is the exact grid plus X_MARGIN (5 % when `pad_x`), y is top-clipped
    (see CLIP_PERCENTILE)."""
    x_limits = _apply_cfg_limits(_limits(x_series, clip=False), cfg.get("xlim"), "xlim")
    y_limits = _apply_cfg_limits(_limits(y_series), cfg.get("ylim"), "ylim")

    xscale, x_kwargs = _resolve_scale(xscale, _finite(*x_series), cfg, "x", x_limits, promote=promote_x)
    yscale, y_kwargs = _resolve_scale(yscale, _finite(*y_series), cfg, "y", y_limits)
    ax.set_xscale(xscale, **x_kwargs)
    ax.set_yscale(yscale, **y_kwargs)

    if x_limits:
        ax.set_xlim(_pad(ax.xaxis, *x_limits, frac=0.05 if pad_x else X_MARGIN))
    if y_limits:
        ax.set_ylim(_pad(ax.yaxis, *y_limits))


def _label_major_ticks_only(*axes):
    """
    Minor tick marks stay; their labels go.

    A log scale installs a labelling minor formatter (matplotlib scale.py,
    LogScale.set_default_locators_and_formatters) that writes 2, 3, 4, 6 x 10^k
    whenever at most one major tick is in view and the span exceeds 0.4 decades,
    which the ratio grids and the n sweep satisfy and the m sweep does not. The
    figures then look uneven. Every figure calls this on every axes AFTER its last
    set_xscale / set_yscale, because a later scale change reinstalls the formatter.
    """
    for ax in axes:
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())


def _major_ticks_in_view(axis) -> int:
    lo, hi = sorted(axis.get_view_interval())
    return int(np.sum([(lo <= t <= hi) for t in axis.get_majorticklocs()]))


def _at_least_two_major_ticks(*axes, skip_x=()):
    """
    Every axis shows at least two labelled major ticks, or the reader cannot
    size the scale.

    A log axis spanning under a decade (the n sweep: 128..1024 on sim,
    128..1000 on optical) holds at most one decade tick, so the default
    LogLocator leaves zero or one major in view. Where fewer than two majors fall
    inside the view interval, a log axis gets a LogLocator on (1, 2, 5) x 10^k
    with plain-number labels (LogFormatterSciNotation labels only one of the (1,
    2, 5) ticks in view), and a linear axis a MaxNLocator that insists on two.
    Called AFTER the last limit or scale change at every figure site; axes in
    `skip_x` keep their x ticks (the digit sweep's thumbnails ARE its ticks).
    Ends by re-blanking the minor labels, which a new locator would reinstate.
    """
    for ax in axes:
        for axis, scale in ((ax.xaxis, ax.get_xscale()), (ax.yaxis, ax.get_yscale())):
            if axis is ax.xaxis and ax in skip_x:
                continue
            if _major_ticks_in_view(axis) >= 2:
                continue
            if scale == "log":
                axis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
                formatter = ScalarFormatter()
                formatter.set_scientific(False)
                formatter.set_useOffset(False)
                axis.set_major_formatter(formatter)
            else:
                axis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=2))
    _label_major_ticks_only(*axes)


def _get_method_color(method_name: str) -> str:
    """Get color for a method from the color palette."""
    palette = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3", "C4", "C5"])
    color_index = COLOR_MAP.get(method_name, 0) % len(palette)
    return palette[color_index]


def _apply_tex_highlighting(labels: list[str], hilight_ours: bool) -> list[str]:
    """Apply bold formatting to our methods in labels."""
    if not hilight_ours:
        return labels

    highlighted = []
    for label in labels:
        if "IVL" in label or "average" in label:
            # Apply bold formatting
            bold = label.replace(r"\alpha", r"{\boldsymbol{\alpha}}")
            bold = bold.replace(r"\Pi", r"{\boldsymbol{\Pi}}")
            label = rf"\textbf{{{bold}}}"
        highlighted.append(label)

    return highlighted


def normalize_sweep(y_results: dict[str, NDArray], fname: str | None) -> tuple[dict[str, NDArray], str | None]:
    """Every series divided by the baseline's per-step mean (SS10.1); returns the
    new dict and the baseline's name, or the input untouched and None.

    Honoured for the `_width`, `_worst_error` and `_approx_error` ids only; a
    `_coverage` id is a rate and is ignored with one warning. The baseline is
    deterministic, `NORMALIZE_BASELINES` in order, and its absence means no
    normalisation and one warning naming the methods present. The divisor is the
    baseline's nanmean per step (over experiments, or over the bootstrap
    resamples once `bootstrap` has run), so the baseline's own mean reads 1.0.
    Where that mean is exactly 0.0 (`approx_error` at and above gamma*, where
    the baseline misses nothing) every method's ratio is NaN, never inf and never
    a floor: 0/0 and x/0 both read as a gap, and one INFO line counts the steps.
    """
    fname = fname or ""
    if not fname.endswith(NORMALIZED_SWEEP_SUFFIXES):
        if fname.endswith("_coverage"):
            logger.warning(f"{fname}: `normalize` ignored, coverage is a rate.")
        return y_results, None
    baseline = next((name for name in NORMALIZE_BASELINES if name in y_results), None)
    if baseline is None:
        logger.warning(f"{fname}: `normalize` needs one of {NORMALIZE_BASELINES} among {list(y_results)}; drawn as is.")
        return y_results, None
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        base = np.nanmean(np.asarray(y_results[baseline], dtype=np.float64), axis=1)
    positive = base > 0
    dropped = int(np.sum(~positive))
    if dropped:
        logger.info(f"{fname}: {dropped} of {len(base)} steps have a zero {baseline} baseline and read NaN.")
    divisor = np.where(positive, base, np.nan)[:, None]
    return {name: np.asarray(series, dtype=np.float64) / divisor for name, series in y_results.items()}, baseline


def create_sweep_plot(
    x_values: NDArray,
    y_results: dict[str, NDArray],
    xlabel: str,
    ylabel: str = "nCER",
    xscale: PlotScale = "linear",
    yscale: PlotScale = "linear",
    savefig: bool = True,
    format: str = PLOT_FORMAT,
    legend_items: list[str] | None = None,
    legend_loc: str | tuple[float, float] = "best",
    y_color: str = "k",
    hide_legend: bool = False,
    hilight_ours: bool = DEFAULT_HILIGHT_OURS,
    bootstrapped: bool = True,
    experiment: str = "simulation",
    fname: str | None = None,
    subdir: str = SUBDIR_SWEEP,
    vlines: tuple[float, ...] = (),
    legend: bool | str | tuple[float, float] | None = None,
    x_color: str = "k",
    title: str | None = None,
    title_color: str = "k",
    normalize: bool = DEFAULT_NORMALIZE_SWEEP,
):
    """
    Create a parameter sweep plot showing method performance across parameter values.
    Aggressively robust to NaN/Inf values.

    `vlines` marks reference values on the x-axis (budget ratio 1, Prop. 2
    threshold).

    `normalize` divides every series by the baseline's (`normalize_sweep`, SS10.1)
    on the width, worst-error and approx-error figures and appends the baseline's
    name to the y-label; `PLOT_CONFIGS[experiment][fname]["normalize"]` overrides it
    per figure. The pkls are written before this function runs and never move.

    `subdir` is the artifacts folder the figure lands in; the default is where every
    param sweep goes. A per-query curve on a shared x grid is this function's shape
    too (the cigarette width-ratio figure), and it belongs beside the query figures.

    Limits/scales come from PLOT_CONFIGS[experiment][fname], else automatically from
    the mean lines -- see _rescale. The style keys `legend`, `x_color`, `y_color`,
    `title`, `title_color` come from the same config entry, else from the arguments
    of the same name (_style); `legend`, when given, overrides `hide_legend` and
    `legend_loc`.
    """
    try:
        # derived HERE, not inside `if savefig`, so the config id and the filename
        # cannot drift apart
        fname = fname or "".join(c for c in xlabel if c.isalnum())
        cfg = _plot_config(experiment, fname)

        # x can be MEASURED rather than a designed grid (trS plots tr(S)/k when
        # recalibrated, rho tr(S)/k otherwise), so it is not
        # guaranteed ascending. matplotlib draws segments in array order, so an
        # out-of-order x makes the line double back on itself and read as
        # jitter. This is the one sort on the sweep path: the runner and the
        # pkls stay in knob order. Reorder (x, y) pairs together; a no-op for
        # the sweeps whose grid is already ascending.
        x_values = np.asarray(x_values, dtype=float)
        order = np.argsort(x_values, kind="stable")
        if not np.array_equal(order, np.arange(len(x_values))):
            x_values = x_values[order]
            y_results = {k: np.asarray(v)[order] for k, v in y_results.items()}

        if bootstrapped:
            y_results = bootstrap(y_results)

        # after the bootstrap, so the bands are divided by the same per-step
        # number as the mean
        if cfg.get("normalize", normalize):
            y_results, baseline = normalize_sweep(y_results, fname)
            if baseline is not None:
                ylabel = rf"{ylabel} / {TEX_MAPPER[baseline]}"

        legend_items = [item for item in (legend_items or []) if item in y_results]

        plt.rcParams.update(RC_PARAMS)
        sns.set_palette("deep")
        colors = sns.color_palette()
        fig = plt.figure()

        # the mean lines, which alone decide the limits: the CI band is contextual
        # and is left to clip against the frame
        all_means = []

        all_labels = []
        plot_handles = []

        for method_name, errors in y_results.items():
            # 1. Sanitize Data: Convert to float64, replace Infs with NaNs
            clean_data = np.array(errors, dtype=np.float64)
            clean_data[np.isinf(clean_data)] = np.nan

            # 2. Compute Mean (ignoring NaNs)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                mean_error = np.nanmean(clean_data, axis=1)

            # 3. Check if we have ANYTHING valid to plot
            if np.all(np.isnan(mean_error)):
                continue

            all_means.append(mean_error)

            # Labeling
            label = TEX_MAPPER.get(method_name, method_name)
            all_labels.append(label)
            if method_name in legend_items:
                legend_items[legend_items.index(method_name)] = label

            # Plot
            linestyle = POINT_ESTIMATE_STYLE if method_name in POINT_ESTIMATES else PARTIAL_IDENTIFICATION_STYLE
            color = colors[COLOR_MAP[method_name]]

            handle = plt.plot(x_values, mean_error, color=color, label=label, linestyle=linestyle)[0]
            plot_handles.append(handle)

            # Confidence Intervals
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                low = np.nanpercentile(clean_data, 2.5, axis=1)
                high = np.nanpercentile(clean_data, 97.5, axis=1)

            # Fill between requires matching shapes; if all NaNs, skip fill
            if not np.all(np.isnan(low)) and not np.all(np.isnan(high)):
                plt.fill_between(x_values, low, high, color=color, alpha=0.2)

        # Formatting
        style = _style(cfg, legend=legend, x_color=x_color, y_color=y_color, title=title, title_color=title_color)
        _apply_style(plt.gca(), style, xlabel, ylabel)

        # x is the exact grid plus a 2 % margin (_limits clip=False, X_MARGIN):
        # the grid's last point and the r = 1 reference line stay in view and
        # off the frame; the 5 % pad would visibly widen every sweep.
        # x is also never auto-promoted: PARAM_SPECS.xscale is an author's choice
        # (trS opts out to linear on purpose), not a default to be second-guessed.
        _rescale(plt.gca(), cfg, [x_values], all_means, xscale, yscale, pad_x=False, promote_x=False)
        _label_major_ticks_only(plt.gca())
        _at_least_two_major_ticks(plt.gca())

        # Reference thresholds (budget ratio 1, Prop. 2 threshold): one unlabelled
        # line each. AFTER _rescale: gated on the resolved xlim, so a narrowing
        # override cannot draw off-frame. zorder=0 keeps these behind the data.
        x_lo, x_hi = plt.gca().get_xlim()
        for x in vlines:
            if not (np.isfinite(x) and x_lo <= x <= x_hi):
                continue
            plt.axvline(x, color="0.4", linestyle=":", linewidth=1.0, zorder=0)

        # Legend
        hide_legend, legend_loc = _legend_choice(style, hide_legend, legend_loc)
        if not hide_legend and plot_handles:
            # Reconstruct legend based on what actually plotted
            final_handles = []
            final_labels = []

            # Use requested order if possible
            targets = legend_items if legend_items else all_labels

            for target_lbl in targets:
                if target_lbl in all_labels:
                    idx = all_labels.index(target_lbl)
                    if idx < len(plot_handles):
                        final_handles.append(plot_handles[idx])
                        final_labels.append(target_lbl)

            final_labels = _apply_tex_highlighting(final_labels, hilight_ours)

            plt.legend(
                handles=final_handles,
                labels=final_labels,
                fontsize=FS_TICK,
                loc=legend_loc,
                frameon=True,
                edgecolor="black",
                fancybox=False,
            )

        plt.tight_layout()
        plt.show()

        if savefig:
            save(fig, f"{fname}_sweep", experiment, format, subdir=subdir, dpi=PLOT_DPI)

    except Exception as e:
        # Fallback so one plot failure doesn't kill the whole experiment batch
        logger.error(f"Failed to plot sweep {fname or xlabel}: {e}")
        import traceback

        logger.error(traceback.format_exc())


def create_query_sweep_plot(
    x_values: NDArray,
    y_results: dict[str, NDArray],
    xlabel: str,
    ylabel: str = r"$h({\bm{x}})$",
    xscale: Literal["linear", "log"] = "linear",
    savefig: bool = True,
    format: str = PLOT_FORMAT,
    legend_items: list[str] | None = None,
    legend_loc: str | tuple[float, float] = "best",
    y_color: str = "k",
    hide_legend: bool = False,
    hilight_ours: bool = DEFAULT_HILIGHT_OURS,
    experiment: str = "simulation",
    legend: bool | str | tuple[float, float] | None = None,
    x_color: str = "k",
    title: str | None = None,
    title_color: str = "k",
    fname: str | None = None,
    vlines: tuple[float, ...] = (),
):
    """
    Create a query sweep plot showing predictions across treatment values.

    Handles both point estimates and interval estimates (PI methods).
    `vlines` marks reference x positions, as on `create_sweep_plot`, and the frame
    widens to cover them (`X_MARK_MARGIN`); empty by default, so every existing
    figure is drawn as before.

    Args:
        x_values: Query values for x-axis
        y_results: Dictionary mapping method names to prediction arrays
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        xscale: Scale for x-axis
        savefig: Whether to save the figure
        format: File format for saving
        legend_items: Specific methods to show in legend
        legend_loc: Legend location
        y_color: Color for y-axis
        hide_legend: Whether to hide legend
        hilight_ours: Whether to highlight our methods
        experiment: Experiment name for file organization
        legend, x_color, title, title_color: style keys, see constants._STYLE_KEYS.
            The orchestrator passes ANNOTATE_SWEEP_PLOT["pc12"] here;
            PLOT_CONFIGS[experiment]["query"] wins over these arguments key by key.
        fname: filename stem, '_sweep' appended. Default: the alphanumerics of
            `xlabel`, which is what every shipped figure is named by. Given when
            several figures share an axis label, or when the derived name is
            unreadable (a TeX label reduces to e.g. 'logmathrmCPI').
    """
    legend_items = [item for item in (legend_items or []) if item in y_results]
    cfg = _plot_config(experiment, "query")
    style = _style(cfg, legend=legend, x_color=x_color, y_color=y_color, title=title, title_color=title_color)

    # Setup plot
    plt.rcParams.update(RC_PARAMS)
    sns.set_palette("deep")
    colors = sns.color_palette()
    fig = plt.figure()

    # Track bounds
    max_mean = float("-inf")
    min_mean = float("inf")
    all_labels = []
    plot_handles = []

    # Plot each method
    for method_name, predictions in y_results.items():
        # Handle interval estimates (PI methods) vs point estimates
        if "PI" in method_name:
            lower_bound = predictions[:, :, 0].mean(axis=1)
            upper_bound = predictions[:, :, 1].mean(axis=1)
            mean_pred = None
        else:
            mean_pred = predictions.mean(axis=1)
            lower_bound = upper_bound = mean_pred

        label = TEX_MAPPER.get(method_name, method_name)
        all_labels.append(label)
        if method_name in legend_items:
            legend_items[legend_items.index(method_name)] = label

        # Update bounds. nan-aware: an interval-against-a-budget figure carries
        # a gap where a solve was INFEASIBLE, and a NaN limit would raise
        max_mean = max(max_mean, float(np.nanmax(upper_bound)))
        min_mean = min(min_mean, float(np.nanmin(lower_bound)))

        color = colors[COLOR_MAP[method_name]]

        # Plot based on method type
        if "PI" in method_name:
            alpha = ALPHA_MAP.get(method_name, 0.2)
            handle = plt.fill_between(x_values, lower_bound, upper_bound, color=color, alpha=alpha)
        else:
            linestyle = POINT_ESTIMATE_STYLE if method_name in POINT_ESTIMATES else PARTIAL_IDENTIFICATION_STYLE
            line_color = "black" if method_name == "ATE" else color
            handle = plt.plot(
                x_values,
                mean_pred,
                color=line_color,
                label=label,
                linestyle=linestyle,
                linewidth=2,
                solid_capstyle="round",
            )[0]

        plot_handles.append(handle)

    # Formatting
    _apply_style(plt.gca(), style, xlabel, ylabel)
    # the frame is the grid, as always; with marks it widens to cover every one
    # of them, with a small margin, so a mark beyond the solved grid (F1's
    # feasibility floor, its 3x benchmark) sits inside the frame over empty axis
    marks = [x for x in vlines if np.isfinite(x)]
    x_lo, x_hi = min(x_values), max(x_values)
    if marks:
        x_lo, x_hi = min(x_lo, min(marks)), max(x_hi, max(marks))
        margin = X_MARK_MARGIN * (x_hi - x_lo)
        x_lo, x_hi = x_lo - margin, x_hi + margin
    plt.xlim([x_lo, x_hi])

    padding = 0.05 * max_mean
    plt.ylim([min_mean - padding, max_mean + padding])
    plt.xscale(xscale)
    _label_major_ticks_only(plt.gca())
    _at_least_two_major_ticks(plt.gca())
    for x in marks:
        plt.axvline(x, color="0.4", linestyle=":", linewidth=1.0, zorder=0)

    # Legend
    hide_legend, legend_loc = _legend_choice(style, hide_legend, legend_loc)
    if not hide_legend:
        labels = legend_items if legend_items else all_labels
        handles = [plot_handles[all_labels.index(item)] for item in labels]
        labels = _apply_tex_highlighting(labels, hilight_ours)

        plt.legend(
            handles=handles,
            labels=labels,
            fontsize=FS_TICK,
            loc=legend_loc,
            frameon=True,
            edgecolor="black",
            fancybox=False,
        )

    plt.tight_layout()
    plt.show()

    if savefig:
        fname = fname or "".join(c for c in xlabel if c.isalnum())
        save(fig, f"{fname}_sweep", experiment, format, subdir=SUBDIR_QUERY, dpi=PLOT_DPI)


def create_panel_plot(
    experiment_name: str,
    column_data: list[tuple[dict[str, NDArray], NDArray, NDArray]],
    histograms: dict[str, tuple[NDArray, NDArray]],
    legend_ncols: int = 2,
):
    plt.rcParams.update(RC_PARAMS)

    column_titles = [
        r"principal direction 1" + "\n" + r"${\bm{x}} := t\cdot {\bm{\nu}}_1$",
        r"radial sweep" + "\n" + r"${\bm{x}} := s_1\sin(\vartheta){\bm{\nu}}_1 + s_2 \cos(\vartheta){\bm{\nu}}_2$",
        r"principal direction 2" + "\n" + r"${\bm{x}} := t\cdot {\bm{\nu}}_2$",
    ]
    x_labels = [r"$t$", r"$\vartheta$", r"$t$"]

    # 1. Share the y axis for each row
    fig, axes = plt.subplots(
        4,
        3,
        figsize=(15, 8),
        sharex="col",
        sharey="row",
        gridspec_kw={"height_ratios": [0.2, 0.2, 0.2, 0.7]},
        constrained_layout=True,
    )

    orig_color = _get_method_color("ERM")
    aug_color = _get_method_color("DA+ERM")
    legend_handles = {}

    # Define a small epsilon to prevent log(0) errors on fills
    LOG_EPS = 1e-9

    for col_idx in range(3):
        results_dict, ground_truth, x_grid = column_data[col_idx]
        exp_cfg = PANEL_CONFIGS.get(experiment_name, {})

        # === ROW 3: Predictions ===
        ax_pred = axes[3, col_idx]
        for method_name, predictions in results_dict.items():
            label = TEX_MAPPER.get(method_name, method_name)
            if predictions.ndim == 3:
                lower = predictions[:, :, 0].mean(axis=1)
                upper = predictions[:, :, 1].mean(axis=1)
                y_mean = None
            else:
                y_mean = predictions.mean(axis=1)
                lower = upper = None

            if "PI" in method_name:
                alpha = ALPHA_MAP.get(method_name, 0.2)
                handle = ax_pred.fill_between(
                    x_grid, lower, upper, alpha=alpha, color=_get_method_color(method_name), zorder=-1
                )
            else:
                linestyle = POINT_ESTIMATE_STYLE if method_name in POINT_ESTIMATES else PARTIAL_IDENTIFICATION_STYLE
                line_color = "black" if method_name == "ATE" else _get_method_color(method_name)
                zorder = 1 if method_name == "ATE" else 0
                handle = ax_pred.plot(
                    x_grid, y_mean, linestyle=linestyle, linewidth=2, color=line_color, zorder=zorder
                )[0]
            if label not in legend_handles:
                legend_handles[label] = handle

        ax_pred.set_xlabel(x_labels[col_idx], fontsize=FS_LABEL)
        if col_idx == 0:
            ax_pred.set_ylabel(r"$h({\bm{x}})$", fontsize=FS_LABEL)
        ax_pred.tick_params(labelsize=FS_TICK)
        ax_pred.set_xlim([x_grid.min(), x_grid.max()])

        # === ROW 1: Interval Width ===
        ax_width = axes[1, col_idx]
        row_cfg = exp_cfg.get(1, {})
        baseline = LOG_EPS if row_cfg.get("scale") in ["log", "asinh", "symlog"] else 0

        for method_name, predictions in results_dict.items():
            if "PI" in method_name:
                width = (predictions[:, :, 1] - predictions[:, :, 0]).mean(axis=1)
                if baseline > 0:
                    width = np.maximum(width, baseline)
                color = _get_method_color(method_name)
                ax_width.fill_between(x_grid, baseline, width, alpha=ALPHA_MAP.get(method_name, 0.2), color=color)
                ax_width.plot(x_grid, width, linewidth=0.5, color=color)

        if col_idx == 0:
            ax_width.set_ylabel("width", fontsize=FS_LABEL)
        ax_width.tick_params(labelsize=FS_TICK)

        # === ROW 0: Worst-Case Error ===
        ax_worst = axes[0, col_idx]
        row_cfg = exp_cfg.get(0, {})
        baseline = LOG_EPS if row_cfg.get("scale") in ["log", "asinh", "symlog"] else 0
        gt_for_broadcast = ground_truth[:, None] if ground_truth.ndim == 1 else ground_truth

        for method_name, predictions in results_dict.items():
            if "PI" in method_name:
                lower, upper = predictions[:, :, 0], predictions[:, :, 1]
                worst_err = np.maximum((lower - gt_for_broadcast) ** 2, (upper - gt_for_broadcast) ** 2).max(axis=1)
                if baseline > 0:
                    worst_err = np.maximum(worst_err, baseline)
                color = _get_method_color(method_name)
                ax_worst.fill_between(x_grid, baseline, worst_err, alpha=ALPHA_MAP.get(method_name, 0.2), color=color)
                ax_worst.plot(x_grid, worst_err, linewidth=0.5, color=color)

        if col_idx == 0:
            ax_worst.set_ylabel(r"$E^+_{\bm{x}}$", fontsize=FS_LABEL)
        ax_worst.set_title(column_titles[col_idx], fontsize=FS_LABEL, pad=8)
        ax_worst.tick_params(labelsize=FS_TICK)

        # === ROW 2: Histograms ===
        ax_hist = axes[2, col_idx]
        hist_key = "pc1" if col_idx == 0 else ("pc2" if col_idx == 2 else None)
        if hist_key and hist_key in histograms:
            orig_proj, aug_proj = histograms[hist_key]
            ax_hist.hist(orig_proj, bins=50, density=True, alpha=0.45, color=orig_color)
            ax_hist.hist(aug_proj, bins=50, density=True, alpha=0.45, color=aug_color)
            if col_idx == 0:
                ax_hist.set_ylabel("density", fontsize=FS_LABEL)
        else:
            ax_hist.axis("off")
        ax_hist.tick_params(labelsize=FS_TICK)

    # 2, 3, 4: Apply row-specific scales, limits, and log-params from constants.py
    for row_idx in range(4):
        cfg = PANEL_CONFIGS.get(experiment_name, {}).get(row_idx, {})
        ax = axes[row_idx, 0]  # Applied via sharey

        if "scale" in cfg:
            ax.set_yscale(cfg["scale"], **_scale_kwargs(cfg["scale"], cfg))

        if "ylim" in cfg:
            ax.set_ylim(cfg["ylim"])
    _label_major_ticks_only(*axes.ravel())
    _at_least_two_major_ticks(*axes.ravel())

    # === Legend ===
    ax_legend = axes[2, 1]
    ax_legend.axis("off")
    label_order = [TEX_MAPPER.get(n, n) for n in results_dict]
    handles = [legend_handles[label] for label in label_order if label in legend_handles]
    ax_legend.legend(
        handles=handles,
        labels=label_order,
        loc="center",
        ncol=legend_ncols,
        fontsize=FS_TICK + 2,
        frameon=False,
        borderpad=-0.3,
        borderaxespad=0,
        labelspacing=0.25,
    )

    fig.align_ylabels(axes[:, 0])
    save(fig, "query_sweep_panel", experiment_name, PLOT_FORMAT, subdir=SUBDIR_QUERY, dpi=PLOT_DPI)


# 4-way reliability split, in STATUS_CATEGORIES order
PERF_CATEGORY_LABELS: tuple[str, ...] = (
    "failure",
    "infeasible",
    "covered",
    "not-covered",
)
# blue, orange, green, red
PERF_CATEGORY_COLORS: tuple[str, ...] = ("#C44E52", "#DD8452", "#55A467", "#4C72B0")


def create_perf_plot(
    perf_record: dict[str, dict[str, object]],
    overlay_metrics: list[str] | None = None,
    savefig: bool = True,
    format: str = PLOT_FORMAT,
    hilight_ours: bool = DEFAULT_HILIGHT_OURS,
    experiment: str = "simulation",
    fname: str = "perf",
):
    """
    Per-method reliability and cost.

    Top (`bars`): 100%-stacked bars, the 4-way per-query split (mutually exclusive,
    in precedence order, summing to 100 by construction). Retired by default --
    PLOT_CONFIGS['*']['perf']['bars']; the split is still written to perf.pkl.
    Bottom: cost and stability -- wall-clock per query (log) and the across-seed
    SD of interval width, each on its own axis in its own units. With `bars` off
    this is the whole figure, both series intact.

    Overlaying both on the bars' percent axis was tried first and read as
    clutter: it forced the SD to be rescaled, giving the left axis two
    different meanings (PLAN 1).
    """
    overlay_metrics = list(overlay_metrics or [])
    try:
        cfg = _plot_config(experiment, fname)
        bars = bool(cfg.get("bars", True))
        # no function arguments here: the config keys or the defaults
        style = _style(cfg, legend=None, x_color="k", y_color="k", title=None, title_color="k")

        plt.rcParams.update(RC_PARAMS)

        methods = list(perf_record)
        labels = [TEX_MAPPER.get(m, m) for m in methods]
        positions = np.arange(len(methods))

        if bars and overlay_metrics:
            fig, (ax, ax_cost) = plt.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
        elif bars:
            fig, ax = plt.subplots()
            ax_cost = None
        elif overlay_metrics:
            fig, ax_cost = plt.subplots()  # the cost panel, promoted
            ax = None
        else:
            logger.warning("perf: bars off and no overlay metrics; nothing to draw.")
            return

        # ------------------------------------------------ reliability (top)
        if ax is not None:
            rates = np.array([perf_record[m]["rates"] for m in methods], dtype=float) * 100.0
            bottom = np.zeros(len(methods))
            bar_handles = []
            for index, (_category, color) in enumerate(zip(PERF_CATEGORY_LABELS, PERF_CATEGORY_COLORS, strict=False)):
                container = ax.bar(
                    positions,
                    rates[:, index],
                    bottom=bottom,
                    color=color,
                    edgecolor="black",
                    linewidth=0.4,
                    width=0.7,
                    zorder=2,
                )
                bottom += rates[:, index]
                bar_handles.append(container)

            ax.set_ylabel(r"test samples (\%)", fontsize=FS_LABEL)
            ax.set_ylim(0, 100)
            ax.tick_params(axis="y", labelsize=FS_TICK)
            if cfg.get("legend") is not False:
                ax.legend(
                    handles=bar_handles,
                    labels=list(PERF_CATEGORY_LABELS),
                    fontsize=FS_TICK * 0.75,
                    loc="lower left",
                    bbox_to_anchor=(0.0, 1.02),
                    ncol=4,
                    frameon=True,
                    edgecolor="black",
                    fancybox=False,
                    handlelength=1.4,
                    columnspacing=1.0,
                )

        # ------------------------------------------------- cost (bottom)
        axis_for_labels = ax
        if ax_cost is not None:
            axis_for_labels = ax_cost
            cost_handles = []

            if "wall_clock" in overlay_metrics:
                seconds = np.array([perf_record[m]["wall_clock"] for m in methods], dtype=float)
                handle = ax_cost.plot(
                    positions,
                    seconds,
                    marker="o",
                    linestyle="-",
                    color="#3b3b6d",
                    markersize=5,
                    label="wall clock / query",
                )[0]
                # x is categorical, so only the cost axis is overridable here.
                # No top-tail clip: on a per-method cost line the slowest method IS
                # the result, not a runaway to be cropped.
                finite = _finite(seconds)
                limits = (
                    (float(finite.min()), float(finite.max())) if len(finite) and finite.min() < finite.max() else None
                )
                limits = _apply_cfg_limits(limits, cfg.get("ylim"), "ylim")
                scale, kwargs = _resolve_scale("log", seconds, cfg, "y", limits, promote=False)
                ax_cost.set_yscale(scale, **kwargs)
                if limits:
                    ax_cost.set_ylim(_pad(ax_cost.yaxis, *limits))
                ax_cost.set_ylabel("s / query", fontsize=FS_TICK)
                ax_cost.tick_params(axis="y", labelsize=FS_TICK * 0.8)
                cost_handles.append(handle)

            if "seed_var" in overlay_metrics:
                seed_sd = np.array([perf_record[m]["seed_var"] for m in methods], dtype=float)
                twin = ax_cost.twinx()
                handle = twin.plot(
                    positions,
                    seed_sd,
                    marker="s",
                    linestyle="--",
                    color="black",
                    markersize=4,
                    label="width SD across seeds",
                )[0]
                twin.set_ylabel("width SD", fontsize=FS_TICK)
                twin.tick_params(axis="y", labelsize=FS_TICK * 0.8)
                cost_handles.append(handle)

            if cost_handles and cfg.get("legend") is not False:
                ax_cost.legend(
                    handles=cost_handles,
                    labels=[h.get_label() for h in cost_handles],
                    fontsize=FS_TICK * 0.7,
                    loc=cfg["legend"] if isinstance(cfg.get("legend"), (str, tuple)) else "upper left",
                    frameon=True,
                    edgecolor="black",
                    fancybox=False,
                    handlelength=1.6,
                    framealpha=0.92,
                )

        axis_for_labels.set_xticks(positions)
        axis_for_labels.set_xticklabels(
            _apply_tex_highlighting(labels, hilight_ours),
            fontsize=FS_TICK,
            rotation=20,
            ha="right",
            color=style["x_color"],
        )
        for axis in fig.axes:  # the bars, the cost axis and its twin
            axis.yaxis.label.set_color(style["y_color"])
            plt.setp(axis.get_yticklabels(), color=style["y_color"])
        if style["title"]:
            fig.axes[0].set_title(style["title"], fontsize=FS_LABEL, color=style["title_color"])

        if ax is not None and ax_cost is not None:
            fig.align_ylabels([ax, ax_cost])
        _label_major_ticks_only(*fig.axes)  # fig.axes includes the twin
        # x is categorical (one fixed tick per method), never re-located
        _at_least_two_major_ticks(*fig.axes, skip_x=tuple(fig.axes))
        fig.tight_layout()
        plt.show()

        if savefig:
            save(fig, fname, experiment, format, subdir=SUBDIR_PERF, dpi=PLOT_DPI)

    except Exception as e:
        logger.error(f"Failed to plot perf: {e}")
        import traceback

        logger.error(traceback.format_exc())


def create_digit_sweep_plot(
    exemplars: NDArray,
    y_results: dict[str, NDArray],
    labels: list[int],
    experiment: str = "do_mnist",
    fname: str = "digit_sweep",
    ylabel: str = r"$h({\bm{x}})$",
    ylim: tuple[float, float] = (-0.05, 1.05),
    legend_width: float = 0.24,
    thumbnail_zoom: float = 0.7,
    savefig: bool = True,
    format: str = PLOT_FORMAT,
    hilight_ours: bool = DEFAULT_HILIGHT_OURS,
):
    """
    Query sweep over frozen digit exemplars, with the images under the axis.

    Same reading as the radial sweeps: bands for intervals, lines for point
    estimates. The x-axis is an index over exemplars, so the connecting segments
    carry no interpolation claim -- they are there to make ten bounds legible.

    Args:
        exemplars: (n, 3, H, W) query images. Pass them at FULL resolution: the
            SEM's `subsample` exists for the models, not for the figure.
        y_results: method name -> predictions, as the other query plots take them
        labels: the digit each exemplar shows
        experiment: experiment name, for file organisation
        fname: output file stem
        ylabel: y-axis label
        ylim: y limits; the thumbnails hang off the lower one
        legend_width: figure fraction reserved for the legend column
        thumbnail_zoom: display pixels per image pixel
        savefig: whether to save the figure
        format: file format
        hilight_ours: whether to bold our methods in the legend
    """
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage

    plt.rcParams.update(RC_PARAMS)
    sns.set_palette("deep")
    colors = sns.color_palette()

    x = np.arange(len(labels), dtype=float)
    fig, ax = plt.subplots(figsize=(PAGE_WIDTH, 3.2))
    handles, all_labels = [], []

    for method_name, predictions in y_results.items():
        label = TEX_MAPPER.get(method_name, method_name)
        color = colors[COLOR_MAP[method_name]]

        if predictions.ndim == 3:  # interval estimate
            lower = predictions[:, :, 0].mean(axis=1)
            upper = predictions[:, :, 1].mean(axis=1)
            # An all-NaN method is INFEASIBLE everywhere and draws nothing, so its
            # legend entry is the only trace of it. Flagging that in the label
            # overflows the legend column; the caller logs it instead.
            handle = ax.fill_between(
                x,
                lower,
                upper,
                color=color,
                alpha=ALPHA_MAP.get(method_name, 0.2),
                zorder=-1,
            )
        else:  # point estimate
            mean_prediction = predictions.mean(axis=1)
            handle = ax.plot(
                x,
                mean_prediction,
                color="black" if method_name == "ATE" else color,
                linestyle=(POINT_ESTIMATE_STYLE if method_name in POINT_ESTIMATES else PARTIAL_IDENTIFICATION_STYLE),
                linewidth=2,
                solid_capstyle="round",
                zorder=1 if method_name == "ATE" else 0,
            )[0]

        handles.append(handle)
        all_labels.append(label)

    ax.set_ylabel(ylabel, fontsize=FS_LABEL)
    ax.set_ylim(*ylim)
    ax.set_xlim(x[0], x[-1])
    ax.set_xticks(x)
    ax.set_xticklabels([])  # the thumbnails ARE the ticks
    ax.tick_params(labelsize=FS_TICK)
    _label_major_ticks_only(ax)
    _at_least_two_major_ticks(ax, skip_x=(ax,))

    # thumbnails below the axis. The SEM renders RGB = [t,0,1-t]*grey, so the
    # background is exactly 0 and the ink mask doubles as the alpha channel --
    # without it every digit sits in a black box.
    for xi, image in zip(x, np.asarray(exemplars), strict=False):
        rgb = np.clip(np.transpose(image, (1, 2, 0)), 0.0, 1.0)
        rgba = np.dstack([rgb, np.clip(rgb.sum(-1), 0.0, 1.0)])
        ax.add_artist(
            AnnotationBbox(
                OffsetImage(rgba, zoom=thumbnail_zoom),
                (xi, ylim[0]),
                frameon=False,
                box_alignment=(0.5, 1.15),
                xycoords=("data", "data"),
                annotation_clip=False,
            )
        )

    fig.subplots_adjust(bottom=0.26, right=1.0 - legend_width)

    # single column to the right, unframed
    ax.legend(
        handles=handles,
        labels=_apply_tex_highlighting(all_labels, hilight_ours),
        fontsize=FS_TICK - 2,
        ncol=1,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        borderpad=0.4,
        handlelength=1.6,
        labelspacing=0.8,
    )

    plt.show()

    if savefig:
        save(fig, fname, experiment, format, subdir=SUBDIR_QUERY, dpi=PLOT_DPI)
