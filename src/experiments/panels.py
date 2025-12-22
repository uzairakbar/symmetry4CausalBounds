import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple
from src.experiments.utils import (
    RC_PARAMS, TEX_MAPPER, FS_LABEL, FS_TICK, color_map, 
    save, PLOT_FORMAT, PLOT_DPI, radial_sweep_pcs, sweep_along_pc
)


class PanelBuilder:
    """Builds 4x3 panel plots for any experiment"""
    
    def __init__(self, runner, experiment_name, use_augmented_geometry=True):
        """
        Args:
            runner: QuerySweepRunner instance with data already loaded
            experiment_name: Name for saving ('simulation', 'optical_device', etc)
            use_augmented_geometry: Use GX_raw (True) or X_raw (False) for geometry
        """
        self.runner = runner
        self.experiment_name = experiment_name
        self.use_augmented = use_augmented_geometry
        
    def build(self, sweep_samples):
        """Generate full 4x3 panel"""
        # 1. Define geometry
        # For PC calculations, use the geometry specified
        # But for histograms, ALWAYS compare X vs GX
        geometry_for_pcs = self.runner.GX_raw if self.use_augmented else self.runner.X_raw
        
        # Calculate PC sweep points on chosen geometry
        pc1_pts, _, mean, pc1_vec = sweep_along_pc(geometry_for_pcs, 0, sweep_samples, 3.0)
        pc2_pts, _, _, pc2_vec = sweep_along_pc(geometry_for_pcs, 1, sweep_samples, 3.0)
        rad_pts = radial_sweep_pcs(geometry_for_pcs, sweep_samples)
        
        # 2. Run models on these geometries
        res1, gt1 = self._run_sweep(pc1_pts)
        res2, gt2 = self._run_sweep(pc2_pts)
        resR, gtR = self._run_sweep(rad_pts)
        
        # 3. Build histogram data - ALWAYS compare X vs GX
        hist_data = self._build_histograms(
            self.runner.X_raw, self.runner.GX_raw, mean, pc1_vec, pc2_vec
        )
        
        # 4. Define plot grids using the SAME geometry as PC calculations
        std_pc1 = np.std((geometry_for_pcs - mean) @ pc1_vec)
        std_pc2 = np.std((geometry_for_pcs - mean) @ pc2_vec)
        
        t1 = np.linspace(-3 * std_pc1, 3 * std_pc1, sweep_samples)
        t2 = np.linspace(-3 * std_pc2, 3 * std_pc2, sweep_samples)
        theta = np.linspace(0, 2*np.pi, sweep_samples)
        
        cols = [(res1, gt1, t1), (resR, gtR, theta), (res2, gt2, t2)]
        
        # 5. Generate panel
        make_panel_4x3(self.experiment_name, cols, hist_data)
    
    def _run_sweep(self, raw_points):
        """Run models on specific geometry"""
        # Transform if needed
        if hasattr(self.runner, 'poly'):
            poly_points = self.runner.poly.fit_transform(raw_points)
        else:
            poly_points = raw_points
        
        # Temporarily override sweep values
        original_get = self.runner.get_sweep_values
        self.runner.get_sweep_values = lambda: poly_points
        _, results = self.runner.run(desc="Panel Sweep")
        self.runner.get_sweep_values = original_get
        
        # Get ground truth
        ate = poly_points @ self.runner.sem.solution
        return results, ate
    
    def _build_histograms(self, X0, GX0, mean, pc1_vec, pc2_vec):
        """Build histogram projection data"""
        proj_X_pc1 = (X0 - mean) @ pc1_vec
        proj_GX_pc1 = (GX0 - mean) @ pc1_vec
        proj_X_pc2 = (X0 - mean) @ pc2_vec
        proj_GX_pc2 = (GX0 - mean) @ pc2_vec
        
        return {
            'pc1': (proj_X_pc1, proj_GX_pc1),
            'pc2': (proj_X_pc2, proj_GX_pc2),
        }


def make_panel_4x3(
    experiment_name: str,
    column_data: List[Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]], 
    # List of (results_dict, ground_truth, x_axis_grid) for PC1, Radial, PC2
    histograms: Dict[str, Tuple[np.ndarray, np.ndarray]], 
    # {'pc1': (X_proj, G_proj), 'pc2': (X_proj, G_proj)}
    legend_ncols: int = 2
):
    """
    Generic plotter for the 4x3 panel figure.
    column_data: List containing data for [PC1, Radial, PC2]
    """
    plt.rcParams.update(RC_PARAMS)
    
    # Titles and labels configuration
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
        constrained_layout=True
    )

    # Helper colors
    def mcolor(name):
        palette = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0','C1','C2','C3','C4','C5'])
        return palette[color_map.get(name, 0) % len(palette)]
    
    orig_color = mcolor('ERM')
    aug_color = mcolor('DA+ERM')

    legend_handles = {}

    # Iterate over columns (PC1, Radial, PC2)
    for col in range(3):
        res, gt, xgrid = column_data[col]

        # --- Row 3: Predictions ---
        ax_pred = axes[3, col]
        for name, m in res.items():
            label = TEX_MAPPER.get(name, name)
            
            # Mean aggregation
            if m.ndim == 3: # PI (S, E, 2)
                low = m[:, :, 0].mean(axis=1)
                high = m[:, :, 1].mean(axis=1)
                y_mean = None
            else: # Point (S, E)
                y_mean = m.mean(axis=1)

            if 'PI' in name:
                alpha = 0.3 if 'INV' in name else 0.2
                h = ax_pred.fill_between(xgrid, low, high, alpha=alpha, edgecolor='none', facecolor=mcolor(name))
            else:
                style = '--' if name == 'ATE' else '-'
                col_line = 'black' if name == 'ATE' else mcolor(name)
                h = ax_pred.plot(xgrid, y_mean, linestyle=style, linewidth=2, color=col_line)[0]
            
            if label not in legend_handles:
                legend_handles[label] = h

        ax_pred.set_xlabel(x_labels[col], fontsize=FS_LABEL)
        if col == 0:
            ax_pred.set_ylabel(r'${\bm{h}}^\top {\bm{x}}$', fontsize=FS_LABEL)
        ax_pred.tick_params(labelsize=FS_TICK)
        ax_pred.set_xlim([xgrid.min(), xgrid.max()])

        # --- Row 1: Width (PI Only) ---
        ax_w = axes[1, col]
        has_pi = False
        for name, m in res.items():
            if 'PI' in name:
                has_pi = True
                width = (m[:, :, 1] - m[:, :, 0]).mean(axis=1)
                alpha = 0.3 if 'INV' in name else 0.2
                ax_w.fill_between(xgrid, 0, width, alpha=alpha, edgecolor='none', facecolor=mcolor(name))
                ax_w.plot(xgrid, width, linewidth=1.5, color=mcolor(name))
        
        if not has_pi: # placeholder
            ax_w.fill_between(xgrid, 0, 0, alpha=0.1, facecolor='0.8')

        if col == 0:
            ax_w.set_ylabel('width', fontsize=FS_LABEL)
        ax_w.tick_params(labelsize=FS_TICK)
        ax_w.set_ylim(0, None); ax_w.margins(y=0)

        # --- Row 0: Worst Case Error ---
        ax_ew = axes[0, col]
        has_ew = False
        
        # Prepare GT for broadcasting
        if gt.ndim == 1:
            gt_col = gt[:, None] 
        else:
            gt_col = gt

        for name, m in res.items():
            if 'PI' in name:
                has_ew = True
                lo, hi = m[:, :, 0], m[:, :, 1]
                se = np.maximum((lo - gt_col)**2, (hi - gt_col)**2)
                ew = se.max(axis=1) 

                alpha = 0.3 if 'INV' in name else 0.2
                ax_ew.fill_between(xgrid, 0, ew, alpha=alpha, edgecolor='none', facecolor=mcolor(name))
                ax_ew.plot(xgrid, ew, linewidth=1.5, color=mcolor(name))
        
        if not has_ew:
            ax_ew.fill_between(xgrid, 0, 0, alpha=0.1, facecolor='0.8')

        if col == 0:
            ax_ew.set_ylabel(r'$E_{\mathrm{worst}}^{\operatorname{do}({\bm{x}})}$', fontsize=FS_LABEL)
        ax_ew.set_title(column_titles[col], fontsize=FS_LABEL, pad=8)
        ax_ew.tick_params(labelsize=FS_TICK)
        ax_ew.set_ylim(0, None); ax_ew.margins(y=0)

        # --- Row 2: Histograms ---
        ax_hist = axes[2, col]
        key = 'pc1' if col == 0 else ('pc2' if col == 2 else None)
        if key and key in histograms:
            hx, hg = histograms[key]
            ax_hist.hist(hx, bins=40, density=True, alpha=0.45, color=orig_color)
            ax_hist.hist(hg, bins=40, density=True, alpha=0.45, color=aug_color)
            if col == 0: ax_hist.set_ylabel('density', fontsize=FS_LABEL)
        else:
            ax_hist.axis('off')
        
        ax_hist.tick_params(labelsize=FS_TICK)
        ax_hist.set_ylim(0, None); ax_hist.margins(y=0)

    # --- Legend (Center Cell [2,1]) ---
    ax_legend = axes[2, 1]
    ax_legend.axis('off') 
    
    label_order = [TEX_MAPPER.get(n, n) for n in res.keys()]
    handles = [legend_handles[l] for l in label_order if l in legend_handles]
    
    leg = ax_legend.legend(
        handles=handles, labels=label_order, loc='center', ncol=legend_ncols,
        fontsize=FS_TICK + 2, frameon=False, columnspacing=1.1
    )
    for h in leg.legendHandles: h.set_linewidth(2.0)
    
    fig.align_ylabels(axes[:, 0])
    save(fig, fname='panel_4x3', experiment=experiment_name, format=PLOT_FORMAT, dpi=PLOT_DPI)