"""
Plotting utilities for experiment results.
"""

import warnings
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from loguru import logger
from numpy.typing import NDArray

from .constants import (
    ALPHA_MAP,
    COLOR_MAP,
    DEFAULT_HILIGHT_OURS,
    FS_LABEL,
    FS_TICK,
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
    SUBDIR_SCATTER,
    SUBDIR_SWEEP,
    TEX_MAPPER,
)
from .data_operations import bootstrap, save

PlotScale = Literal["linear", "log", "symlog", "asinh"]

# clip the top tail of the pooled means. Errors/widths: small is the signal, large is
# the runaway. A symmetric floor crops the TIGHTEST method, which is the result.
CLIP_PERCENTILE: float = 98.0
# asinh knee, as a fraction of the upper limit. PANEL_CONFIGS' own ylim/linear_width.
LINEAR_WIDTH_RATIO: float = 40.0
# promote linear -> log past this dynamic range. Fires on nothing today; a guard.
LOG_PROMOTE_RATIO: float = 100.0


def _plot_config(experiment: str, fname: str | None) -> dict[str, object]:
    """PLOT_CONFIGS['*'] under PLOT_CONFIGS[experiment], key by key."""
    if not fname:
        return {}
    return {**PLOT_CONFIGS.get("*", {}).get(fname, {}), **PLOT_CONFIGS.get(experiment, {}).get(fname, {})}


def _finite(*arrays) -> NDArray:
    """Pool the finite values. Empty is a valid answer; every caller handles it."""
    if not arrays:
        return np.array([], dtype=float)
    values = np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrays])
    return values[np.isfinite(values)]


def _limits(series: list[NDArray]) -> tuple[float, float] | None:
    """
    Clip the top tail of the pooled means, then guarantee no series goes blank.

    Point estimates only -- CI bands and SE crosshairs are deliberately excluded and
    left to clip against the frame.
    """
    pooled = _finite(*series)
    if not len(pooled):
        return None

    lo, hi = float(pooled.min()), float(np.percentile(pooled, CLIP_PERCENTILE))
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
    """Limits -> cfg -> scale -> pad -> set. Limits never depend on the scale."""
    x_limits = _apply_cfg_limits(_limits(x_series), cfg.get("xlim"), "xlim")
    y_limits = _apply_cfg_limits(_limits(y_series), cfg.get("ylim"), "ylim")

    xscale, x_kwargs = _resolve_scale(xscale, _finite(*x_series), cfg, "x", x_limits, promote=promote_x)
    yscale, y_kwargs = _resolve_scale(yscale, _finite(*y_series), cfg, "y", y_limits)
    ax.set_xscale(xscale, **x_kwargs)
    ax.set_yscale(yscale, **y_kwargs)

    if x_limits:
        ax.set_xlim(_pad(ax.xaxis, *x_limits) if pad_x else x_limits)
    if y_limits:
        ax.set_ylim(_pad(ax.yaxis, *y_limits))


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
            if label != TEX_MAPPER.get("DA+IVL-a", ""):
                # Apply bold formatting
                bold = label.replace(r"\alpha", r"{\boldsymbol{\alpha}}")
                bold = bold.replace(r"\Pi", r"{\boldsymbol{\Pi}}")
                label = rf"\textbf{{{bold}}}"
        highlighted.append(label)

    return highlighted


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
    vlines: tuple[float, ...] = (),
):
    """
    Create a parameter sweep plot showing method performance across parameter values.
    Aggressively robust to NaN/Inf values.

    `vlines` marks reference values on the x-axis (budget ratio 1, Prop. 2
    threshold, Thm. 1 threshold).

    Limits/scales come from PLOT_CONFIGS[experiment][fname], else automatically from
    the mean lines -- see _rescale.
    """
    try:
        # derived HERE, not inside `if savefig`, so the config id and the filename
        # cannot drift apart
        fname = fname or "".join(c for c in xlabel if c.isalnum())
        cfg = _plot_config(experiment, fname)

        # x can be MEASURED rather than a designed grid (trS plots the observed
        # rho tr(S)/k), so it is not guaranteed ascending. matplotlib draws
        # segments in array order, so an out-of-order x makes the line double
        # back on itself and read as jitter. Reorder (x, y) pairs together; a
        # no-op for the sweeps whose grid is already ascending.
        x_values = np.asarray(x_values, dtype=float)
        order = np.argsort(x_values, kind="stable")
        if not np.array_equal(order, np.arange(len(x_values))):
            x_values = x_values[order]
            y_results = {k: np.asarray(v)[order] for k, v in y_results.items()}

        if bootstrapped:
            y_results = bootstrap(y_results)

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
        plt.xlabel(xlabel, fontsize=FS_LABEL)
        plt.ylabel(ylabel, fontsize=FS_LABEL, color=y_color)
        plt.yticks(fontsize=FS_TICK, color=y_color)
        plt.xticks(fontsize=FS_TICK)

        # x keeps its exact [min, max]; padding it would visibly widen every sweep.
        # x is also never auto-promoted: PARAM_SPECS.xscale is an author's choice
        # (trS opts out to linear on purpose), not a default to be second-guessed.
        _rescale(plt.gca(), cfg, [x_values], all_means, xscale, yscale, pad_x=False, promote_x=False)

        # Reference thresholds (budget ratio 1, Prop. 2 / Thm. 1 thresholds).
        # gamma sweeps append the Thm. 1 ratio last (generic_runner.py).
        # AFTER _rescale: gated on the resolved xlim, so a narrowing override cannot
        # leave the label anchored off-frame. zorder=0 keeps these behind the data.
        x_lo, x_hi = plt.gca().get_xlim()
        for i, x in enumerate(vlines):
            if not (np.isfinite(x) and x_lo <= x <= x_hi):
                continue
            plt.axvline(x, color="0.4", linestyle=":", linewidth=1.0, zorder=0)
            if i > 0:  # the appended Thm. 1 threshold
                plt.text(
                    x,
                    0.5,
                    r"$\epsilon$-validity (Thm. 1)",
                    transform=plt.gca().get_xaxis_transform(),
                    rotation=90,
                    va="center",
                    ha="right",
                    fontsize=FS_TICK * 0.75,
                    color="0.4",
                )

        # Legend
        if cfg.get("legend") is False:
            hide_legend = True
        if isinstance(cfg.get("legend"), (str, tuple)):
            legend_loc = cfg["legend"]
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
            save(fig, f"{fname}_sweep", experiment, format, subdir=SUBDIR_SWEEP, dpi=PLOT_DPI)

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
):
    """
    Create a query sweep plot showing predictions across treatment values.

    Handles both point estimates and interval estimates (PI methods).

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
    """
    legend_items = [item for item in (legend_items or []) if item in y_results]

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

        # Update bounds
        max_mean = max(max_mean, upper_bound.max())
        min_mean = min(min_mean, lower_bound.min())

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
    plt.xlabel(xlabel, fontsize=FS_LABEL)
    plt.ylabel(ylabel, fontsize=FS_LABEL, color=y_color)
    plt.yticks(fontsize=FS_TICK, color=y_color)
    plt.xticks(fontsize=FS_TICK)
    plt.xlim([min(x_values), max(x_values)])

    padding = 0.05 * max_mean
    plt.ylim([min_mean - padding, max_mean + padding])
    plt.xscale(xscale)

    # Legend
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
        fname = "".join(c for c in xlabel if c.isalnum()) + "_sweep"
        save(fig, fname, experiment, format, subdir=SUBDIR_QUERY, dpi=PLOT_DPI)


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

    # === Legend ===
    ax_legend = axes[2, 1]
    ax_legend.axis("off")
    label_order = [TEX_MAPPER.get(n, n) for n in results_dict.keys()]
    handles = [legend_handles[l] for l in label_order if l in legend_handles]
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


def create_scatter_plot(
    results: dict[str, dict[str, NDArray]],
    param_values: NDArray,
    metric_x: str,
    metric_y: str,
    xlabel: str = "",
    ylabel: str = "",
    param_label: str = "",
    xscale: PlotScale = "linear",
    yscale: PlotScale = "linear",
    savefig: bool = True,
    format: str = PLOT_FORMAT,
    legend_loc: str | tuple[float, float] = "best",
    hilight_ours: bool = DEFAULT_HILIGHT_OURS,
    experiment: str = "simulation",
    fname: str | None = None,
    crosshairs: bool = True,
):
    """
    Trade-off scatter: one point per (method, sweep step) at the mean of two
    metrics, sized by the rank of the sweep parameter and annotated with its
    value. Crosshairs are +/- 1 SE across experiments.

    Args:
        results: method -> metric field -> (n_steps, n_experiments)
        param_values: the sweep grid, one entry per step
        metric_x, metric_y: metric FIELD names to place on each axis
        crosshairs: draw SE bars (caller gates on n_experiments >= 2)

    Limits/scales come from PLOT_CONFIGS[experiment][fname], else automatically from
    the mean points -- the SE crosshairs are excluded and clip.
    """
    try:
        fname = fname or "scatter"  # config id == filename; see create_sweep_plot
        cfg = _plot_config(experiment, fname)

        plt.rcParams.update(RC_PARAMS)
        sns.set_palette("deep")
        colors = sns.color_palette()
        fig = plt.figure()

        x_means, y_means = [], []
        n_steps = len(param_values)
        # bubble area by rank, so a log-spaced grid still reads evenly
        ranks = np.argsort(np.argsort(np.asarray(param_values, dtype=float)))
        sizes = np.linspace(28, 150, max(n_steps, 1))[ranks]

        handles, labels = [], []
        for order, (method_name, record) in enumerate(results.items()):
            if metric_x not in record or metric_y not in record:
                continue

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                mx = np.nanmean(record[metric_x], axis=1)
                my = np.nanmean(record[metric_y], axis=1)
                n_experiments = record[metric_x].shape[1]
                sx = np.nanstd(record[metric_x], axis=1) / np.sqrt(n_experiments)
                sy = np.nanstd(record[metric_y], axis=1) / np.sqrt(n_experiments)

            if np.all(np.isnan(mx)) or np.all(np.isnan(my)):
                continue

            x_means.append(mx)
            y_means.append(my)
            color = colors[COLOR_MAP[method_name]]

            if crosshairs:
                plt.errorbar(
                    mx, my, xerr=sx, yerr=sy, fmt="none", ecolor=color, elinewidth=0.8, capsize=2, alpha=0.6, zorder=1
                )

            plt.scatter(
                mx,
                my,
                s=sizes,
                color=color,
                alpha=ALPHA_MAP.get(method_name, 0.9),
                edgecolors="black",
                linewidths=0.4,
                zorder=2,
            )

            # Label the endpoints only: bubble size already encodes the ordering,
            # and consecutive steps often land close enough that per-point labels
            # overlap into an illegible pile.
            # stagger by method so labels stack instead of overprinting where
            # several methods land on the same point
            for index in {0, n_steps - 1} if n_steps else set():
                x, y, value = mx[index], my[index], param_values[index]
                if np.isfinite(x) and np.isfinite(y):
                    plt.annotate(
                        f"{value:g}",
                        (x, y),
                        textcoords="offset points",
                        xytext=(6, 5 + 9 * order),
                        fontsize=FS_TICK * 0.7,
                        color=color,
                    )

            handles.append(
                plt.Line2D(
                    [], [], marker="o", linestyle="none", color=color, markeredgecolor="black", markeredgewidth=0.4
                )
            )
            labels.append(TEX_MAPPER.get(method_name, method_name))

        plt.xlabel(xlabel, fontsize=FS_LABEL)
        plt.ylabel(ylabel, fontsize=FS_LABEL)
        plt.xticks(fontsize=FS_TICK)
        plt.yticks(fontsize=FS_TICK)
        if param_label:
            plt.title(f"annotated by {param_label}", fontsize=FS_TICK)

        _rescale(plt.gca(), cfg, x_means, y_means, xscale, yscale)

        if isinstance(cfg.get("legend"), (str, tuple)):
            legend_loc = cfg["legend"]
        if handles and cfg.get("legend") is not False:
            plt.legend(
                handles=handles,
                labels=_apply_tex_highlighting(labels, hilight_ours),
                fontsize=FS_TICK,
                loc=legend_loc,
                frameon=True,
                edgecolor="black",
                fancybox=False,
            )

        plt.tight_layout()
        plt.show()

        if savefig:
            save(fig, fname, experiment, format, subdir=SUBDIR_SCATTER, dpi=PLOT_DPI)

    except Exception as e:
        logger.error(f"Failed to plot scatter {fname}: {e}")
        import traceback

        logger.error(traceback.format_exc())


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
            for index, (category, color) in enumerate(zip(PERF_CATEGORY_LABELS, PERF_CATEGORY_COLORS)):
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
            _apply_tex_highlighting(labels, hilight_ours), fontsize=FS_TICK, rotation=20, ha="right"
        )

        if ax is not None and ax_cost is not None:
            fig.align_ylabels([ax, ax_cost])
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

    # thumbnails below the axis. The SEM renders RGB = [t,0,1-t]*grey, so the
    # background is exactly 0 and the ink mask doubles as the alpha channel --
    # without it every digit sits in a black box.
    for xi, image in zip(x, np.asarray(exemplars)):
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
