"""
Plotting utilities for experiment results.
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from typing import Dict, List, Tuple, Optional, Literal

from .constants import (
    DEFAULT_HILIGHT_OURS, POINT_ESTIMATES,
    FS_TICK, FS_LABEL, PLOT_DPI, PLOT_FORMAT,
    RC_PARAMS, TEX_MAPPER, COLOR_MAP, ALPHA_MAP,
    POINT_ESTIMATE_STYLE, PARTIAL_IDENTIFICATION_STYLE,
)
from .data_operations import bootstrap, save


def _get_method_color(method_name: str) -> str:
    """Get color for a method from the color palette."""
    palette = plt.rcParams['axes.prop_cycle'].by_key().get(
        'color', ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
    )
    color_index = COLOR_MAP.get(method_name, 0) % len(palette)
    return palette[color_index]


def _apply_tex_highlighting(labels: List[str], hilight_ours: bool) -> List[str]:
    """Apply bold formatting to our methods in labels."""
    if not hilight_ours:
        return labels
    
    highlighted = []
    for label in labels:
        if 'IVL' in label or 'average' in label:
            if label != TEX_MAPPER.get('DA+IVL-a', ''):
                # Apply bold formatting
                bold = label.replace(r'\alpha', r'{\boldsymbol{\alpha}}')
                bold = bold.replace(r'\Pi', r'{\boldsymbol{\Pi}}')
                label = fr'\textbf{{{bold}}}'
        highlighted.append(label)
    
    return highlighted


def create_param_sweep_plot(
    x_values: NDArray,
    y_results: Dict[str, NDArray],
    xlabel: str,
    ylabel: str = 'nCER',
    xscale: Literal['linear', 'log'] = 'linear',
    yscale: Literal['linear', 'log'] = 'linear',
    savefig: bool = True,
    format: str = PLOT_FORMAT,
    legend_items: Optional[List[str]] = None,
    legend_loc: str | Tuple[float, float] = 'best',
    y_color: str = 'k',
    hide_legend: bool = False,
    hilight_ours: bool = DEFAULT_HILIGHT_OURS,
    bootstrapped: bool = True,
):
    """
    Create a parameter sweep plot showing method performance across parameter values.
    
    Args:
        x_values: Parameter values for x-axis
        y_results: Dictionary mapping method names to error arrays
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        xscale: Scale for x-axis ('linear' or 'log')
        yscale: Scale for y-axis ('linear' or 'log')
        savefig: Whether to save the figure
        format: File format for saving
        legend_items: Specific methods to include in legend
        legend_loc: Location for legend
        y_color: Color for y-axis label and ticks
        hide_legend: Whether to hide the legend
        hilight_ours: Whether to highlight our methods
        bootstrapped: Whether to apply bootstrapping
    """
    if bootstrapped:
        y_results = bootstrap(y_results)
    
    legend_items = [item for item in (legend_items or []) if item in y_results]
    
    # Setup plot
    plt.rcParams.update(RC_PARAMS)
    sns.set_palette('deep')
    colors = sns.color_palette()
    fig = plt.figure()
    
    # Track bounds for ylim
    max_mean = 0.0
    min_mean = float('inf')
    all_labels = []
    plot_handles = []
    
    # Plot each method
    for method_name, errors in y_results.items():
        mean_error = errors.mean(axis=1)
        
        # Get display label
        label = TEX_MAPPER.get(method_name, method_name)
        all_labels.append(label)
        if method_name in legend_items:
            legend_items[legend_items.index(method_name)] = label
        
        # Choose line style
        linestyle = POINT_ESTIMATE_STYLE if method_name in POINT_ESTIMATES else PARTIAL_IDENTIFICATION_STYLE
        color = colors[COLOR_MAP[method_name]]
        
        # Plot mean
        handle = plt.plot(x_values, mean_error, color=color, label=label, linestyle=linestyle)[0]
        plot_handles.append(handle)
        
        # Update bounds
        max_mean = max(max_mean, max(mean_error))
        min_mean = min(min_mean, min(mean_error))
    
    # Add confidence intervals
    for method_name, errors in y_results.items():
        low = np.percentile(errors, 2.5, axis=1)
        high = np.percentile(errors, 97.5, axis=1)
        color = colors[COLOR_MAP[method_name]]
        plt.fill_between(x_values, low, high, color=color, alpha=0.2)
    
    # Formatting
    plt.xlabel(xlabel, fontsize=FS_LABEL)
    plt.ylabel(ylabel, fontsize=FS_LABEL, color=y_color)
    plt.yticks(fontsize=FS_TICK, color=y_color)
    plt.xticks(fontsize=FS_TICK)
    plt.xlim([min(x_values), max(x_values)])
    
    padding = 0.05 * (max_mean - min_mean)
    plt.ylim([min_mean - padding, max_mean + padding])
    plt.xscale(xscale)
    plt.yscale(yscale)
    
    # Legend
    if not hide_legend:
        labels = legend_items if legend_items else all_labels
        handles = [plot_handles[all_labels.index(item)] for item in labels]
        labels = _apply_tex_highlighting(labels, hilight_ours)
        
        plt.legend(
            handles=handles, labels=labels, fontsize=FS_TICK,
            loc=legend_loc, frameon=True, edgecolor='black', fancybox=False
        )
    
    plt.tight_layout()
    plt.show()
    
    if savefig:
        fname = ''.join(c for c in xlabel if c.isalnum()) + '_sweep'
        save(fig, fname, 'simulation', format, dpi=PLOT_DPI)


def create_query_sweep_plot(
    x_values: NDArray,
    y_results: Dict[str, NDArray],
    xlabel: str,
    ylabel: str = r'${\bm{h}}^\top {\bm{x}}$',
    xscale: Literal['linear', 'log'] = 'linear',
    savefig: bool = True,
    format: str = PLOT_FORMAT,
    legend_items: Optional[List[str]] = None,
    legend_loc: str | Tuple[float, float] = 'best',
    y_color: str = 'k',
    hide_legend: bool = False,
    hilight_ours: bool = DEFAULT_HILIGHT_OURS,
    experiment: str = 'simulation',
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
    sns.set_palette('deep')
    colors = sns.color_palette()
    fig = plt.figure()
    
    # Track bounds
    max_mean = float('-inf')
    min_mean = float('inf')
    all_labels = []
    plot_handles = []
    
    # Plot each method
    for method_name, predictions in y_results.items():
        # Handle interval estimates (PI methods) vs point estimates
        if 'PI' in method_name:
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
        if 'PI' in method_name:
            alpha = ALPHA_MAP.get(method_name, 0.2)
            handle = plt.fill_between(
                x_values, lower_bound, upper_bound,
                color=color, alpha=alpha
            )
        else:
            linestyle = POINT_ESTIMATE_STYLE if method_name in POINT_ESTIMATES else PARTIAL_IDENTIFICATION_STYLE
            line_color = 'black' if method_name == 'ATE' else color
            handle = plt.plot(
                x_values, mean_pred,
                color=line_color, label=label,
                linestyle=linestyle, linewidth=2,
                solid_capstyle='round'
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
            handles=handles, labels=labels, fontsize=FS_TICK,
            loc=legend_loc, frameon=True, edgecolor='black', fancybox=False
        )
    
    plt.tight_layout()
    plt.show()
    
    if savefig:
        fname = ''.join(c for c in xlabel if c.isalnum()) + '_sweep'
        save(fig, fname, experiment, format, dpi=PLOT_DPI)


def create_panel_plot(
    experiment_name: str,
    column_data: List[Tuple[Dict[str, NDArray], NDArray, NDArray]],
    histograms: Dict[str, Tuple[NDArray, NDArray]],
    legend_ncols: int = 2
):
    """
    Create a 4x3 panel plot showing multiple visualizations.
    
    Panel layout:
    - Row 0: Worst-case error
    - Row 1: Interval width
    - Row 2: Histograms
    - Row 3: Predictions
    
    Columns: PC1, Radial sweep, PC2
    
    Args:
        experiment_name: Name for saving
        column_data: List of (results_dict, ground_truth, x_grid) for each column
        histograms: Dictionary with 'pc1' and 'pc2' histogram data
        legend_ncols: Number of columns in legend
    """
    plt.rcParams.update(RC_PARAMS)
    
    # Column configuration
    column_titles = [
        r'principal direction 1' + '\n' + r'${\bm{x}} := t\cdot {\bm{u}}_1$',
        r'radial sweep' + '\n' + r'${\bm{x}} := {\sigma}_1\sin(\theta){\bm{u}}_1 + {\sigma}_2 \cos(\theta){\bm{u}}_2$',
        r'principal direction 2' + '\n' + r'${\bm{x}} := t\cdot {\bm{u}}_2$',
    ]
    x_labels = [r'$t$', r'$\theta$', r'$t$']
    
    fig, axes = plt.subplots(
        4, 3, figsize=(15, 8),
        sharex='col',
        gridspec_kw={'height_ratios': [0.2, 0.2, 0.2, 0.7]},
        constrained_layout=True,
    )
    
    # Get colors for original and augmented data
    orig_color = _get_method_color('ERM')
    aug_color = _get_method_color('DA+ERM')
    
    legend_handles = {}
    
    # Process each column
    for col_idx in range(3):
        results_dict, ground_truth, x_grid = column_data[col_idx]
        
        # === ROW 3: Predictions ===
        ax_pred = axes[3, col_idx]
        for method_name, predictions in results_dict.items():
            label = TEX_MAPPER.get(method_name, method_name)
            
            # Aggregate across experiments
            if predictions.ndim == 3:  # PI methods: (samples, experiments, 2)
                lower = predictions[:, :, 0].mean(axis=1)
                upper = predictions[:, :, 1].mean(axis=1)
                y_mean = None
            else:  # Point estimates: (samples, experiments)
                y_mean = predictions.mean(axis=1)
                lower = upper = None
            
            # Plot
            if 'PI' in method_name:
                alpha = ALPHA_MAP.get(method_name, 0.2)
                handle = ax_pred.fill_between(
                    x_grid, lower, upper, alpha=alpha,
                    color=_get_method_color(method_name),
                    zorder=-1,
                )
            else:
                linestyle = POINT_ESTIMATE_STYLE if method_name in POINT_ESTIMATES else PARTIAL_IDENTIFICATION_STYLE
                line_color = 'black' if method_name == 'ATE' else _get_method_color(method_name)
                zorder = 1 if method_name == 'ATE' else 0
                handle = ax_pred.plot(
                    x_grid, y_mean,
                    linestyle=linestyle, linewidth=2,
                    color=line_color,
                    zorder=zorder,
                )[0]
            
            if label not in legend_handles:
                legend_handles[label] = handle
        
        ax_pred.set_xlabel(x_labels[col_idx], fontsize=FS_LABEL)
        if col_idx == 0:
            ax_pred.set_ylabel(r'${\bm{h}}^\top {\bm{x}}$', fontsize=FS_LABEL)
        ax_pred.tick_params(labelsize=FS_TICK)
        ax_pred.set_xlim([x_grid.min(), x_grid.max()])
        
        # === ROW 1: Interval Width ===
        ax_width = axes[1, col_idx]
        for method_name, predictions in results_dict.items():
            if 'PI' in method_name:
                width = (predictions[:, :, 1] - predictions[:, :, 0]).mean(axis=1)
                alpha = ALPHA_MAP.get(method_name, 0.2)
                color = _get_method_color(method_name)
                ax_width.fill_between(x_grid, 0, width, alpha=alpha, color=color)
                ax_width.plot(x_grid, width, linewidth=0.5, color=color)
        
        if col_idx == 0:
            ax_width.set_ylabel('width', fontsize=FS_LABEL)
        ax_width.tick_params(labelsize=FS_TICK)
        ax_width.set_ylim(0, None)
        ax_width.margins(y=0)
        
        # === ROW 0: Worst-Case Error ===
        ax_worst = axes[0, col_idx]
        
        # Prepare ground truth for broadcasting
        gt_for_broadcast = ground_truth[:, None] if ground_truth.ndim == 1 else ground_truth
        
        for method_name, predictions in results_dict.items():
            if 'PI' in method_name:
                lower = predictions[:, :, 0]
                upper = predictions[:, :, 1]
                squared_errors = np.maximum(
                    (lower - gt_for_broadcast)**2,
                    (upper - gt_for_broadcast)**2
                )
                worst_err = squared_errors.max(axis=1)
                
                alpha = ALPHA_MAP.get(method_name, 0.2)
                color = _get_method_color(method_name)
                ax_worst.fill_between(x_grid, 0, worst_err, alpha=alpha, color=color)
                ax_worst.plot(x_grid, worst_err, linewidth=0.5, color=color)
        
        if col_idx == 0:
            ax_worst.set_ylabel(r'$E_{\mathrm{worst}}^{\operatorname{do}({\bm{x}})}$', fontsize=FS_LABEL)
        ax_worst.set_title(column_titles[col_idx], fontsize=FS_LABEL, pad=8)
        ax_worst.tick_params(labelsize=FS_TICK)
        ax_worst.set_ylim(0, None)
        ax_worst.margins(y=0)
        
        # === ROW 2: Histograms ===
        ax_hist = axes[2, col_idx]
        hist_key = 'pc1' if col_idx == 0 else ('pc2' if col_idx == 2 else None)
        
        if hist_key and hist_key in histograms:
            orig_proj, aug_proj = histograms[hist_key]
            ax_hist.hist(orig_proj, bins=50, density=True, alpha=0.45, color=orig_color)
            ax_hist.hist(aug_proj, bins=50, density=True, alpha=0.45, color=aug_color)
            if col_idx == 0:
                ax_hist.set_ylabel('density', fontsize=FS_LABEL)
        else:
            ax_hist.axis('off')
        
        ax_hist.tick_params(labelsize=FS_TICK)
        ax_hist.set_ylim(0, None)
        ax_hist.margins(y=0)
    
    # === Center Legend ===
    ax_legend = axes[2, 1]
    ax_legend.axis('off')
    
    label_order = [TEX_MAPPER.get(n, n) for n in results_dict.keys()]
    handles = [legend_handles[l] for l in label_order if l in legend_handles]
    
    leg = ax_legend.legend(
        handles=handles, labels=label_order,
        loc='center', ncol=legend_ncols,
        fontsize=FS_TICK + 2, frameon=False,
        borderpad=-0.3,
        borderaxespad=0,
        labelspacing=0.25,
    )
    
    fig.align_ylabels(axes[:, 0])
    save(fig, 'query_sweep_panel', experiment_name, PLOT_FORMAT, dpi=PLOT_DPI)