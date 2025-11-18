import copy
import enlighten
import numpy as np
import scipy as sp
from typing import Tuple
import matplotlib.pyplot as plt
from loguru import logger
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from sklearn.preprocessing import PolynomialFeatures
from typing import Dict, Callable, Optional, List

from src.data_augmentors.real.optical_device import OpticalDeviceDA as DA

from src.sem.real.optical_device import OpticalDeviceSEM as SEM

from src.methods.abstract import pointIdentifier as Regressor
from src.methods.regression import LeastSquaresClosedForm as ERM

from src.methods.sensitivity_models import (
    PartialR2
)

from src.experiments.utils import (
    save,
    set_seed,
    box_plot,
    tex_table,
    fit_model,
    sweep_plot,
    estimation_error,
    ANNOTATE_BOX_PLOT,
    ANNOTATE_SWEEP_PLOT,
    radial_sweep_pcs,
    sweep_along_pc,
    RC_PARAMS,
    TEX_MAPPER,
    FS_LABEL,
    FS_TICK,
    color_map,
    PLOT_FORMAT,
    PLOT_DPI,
)


ModelBuilder = Callable[[Optional[float]], Regressor]

MANAGER = enlighten.get_manager()
EXPERIMENT: str='optical_device'
DEFAULT_CV_SAMPLES: int=10
DEFAULT_CV_FRAC: float=0.2
DEFAULT_CV_FOLDS: int=5
DEFAULT_CV_JOBS: int=1
GROUND_TRUTH: str='polynomial'
OPTICAL_DEVICE_DATASET: int=9


class SweepExperiment:
    def __init__(
            self,
            seed: int,
            n_samples: int,
            augmentation: str,
            sweep_samples: int,
            methods: Dict[str, Callable[[Optional[float]], Regressor]],
            hyperparameters: Optional[Dict[str, Dict[str, float]]]=None
        ):
        self.seed = seed
        self.n_samples = n_samples
        self.augmentation = augmentation
        self.sweep_samples = sweep_samples
        self.methods = methods
        self.hyperparameters = hyperparameters
    
    @staticmethod
    def fit(
            method_name: str,
            method: Callable[[Optional[str]], Regressor],
            X, y, G, GX,
            param: float=10.0,
            da: Optional[DA]=None,
            hyperparameters: Optional[Dict[str, Dict[str, float]]]=None
        ) -> Regressor:
        model = method()
        
        fit_model(
            model=model,
            name=method_name,
            X=X, y=y, G=G, GX=GX,
            hyperparameters=hyperparameters,
            da=da
        )
        
        return model
    
    def generate_dataset(self, sem: SEM, da: DA, param: float=10.0):
        X, y = sem(N = self.n_samples)
        GX, G = da(X)
        return X, y, G, GX

    def query_sweep(self, X):
        queries = radial_sweep_pcs(X, self.sweep_samples)
        return queries
        # queries = sweep_along_pc(X, n_steps=self.sweep_samples, pc_index=0)
        # return queries[0]

    def compute_result(self,
               method_name: str,
               method: Callable[[Optional[str]], Regressor],
               X, y, G, GX,
               query: np.ndarray,
               da: Optional[DA]=None) -> float:
        model = self.fit(
            method_name, method, X, y, G, GX, da=da,
            hyperparameters=self.hyperparameters
        )
        bounds = model.predict(query)
        return bounds

    def run_experiment(self):
        if self.seed >= 0:
            set_seed(self.seed)

        sem = SEM(
            experiment=OPTICAL_DEVICE_DATASET,
            ground_truth=GROUND_TRUTH
        )
        da = DA(self.augmentation)
        
        features = PolynomialFeatures(
            sem.poly_degree, include_bias=False
        )
        X, y, G, GX = self.generate_dataset(sem, da)
        
        query_values = self.query_sweep(GX)
        query_values = features.fit_transform(query_values)

        X = features.fit_transform(X)
        G = features.fit_transform(G)
        GX = features.fit_transform(GX)
        
        bounds_dim = (self.sweep_samples, 1, 2)
        results = {
            name: (np.zeros(bounds_dim[:-1]) if 'PI' not in name else np.zeros(bounds_dim))
            for name in self.methods
        }
        
        experiment_name = self.__class__.__name__
        pbar_experiment = MANAGER.counter(
            total=self.sweep_samples, desc=f'{experiment_name}', unit='params'
        )
        for i, query in enumerate(query_values[:, np.newaxis, :]):

            sem_solution = sem.solution
            
            pbar_methods = MANAGER.counter(
                total=len(self.methods), desc=f'SEM {0}', unit='methods', leave=False
            )
            for method_name, method in self.methods.items():
                if method_name == 'ATE':
                    results[method_name][i][0] = query @ sem_solution
                else:
                    results[method_name][i][0] = self.compute_result(
                        method_name, method, X, y, G, GX, query, da=da
                    )

                pbar_methods.update()
            pbar_methods.close()
            pbar_experiment.update()
        pbar_experiment.close()
        return np.linspace(0, 2*np.pi, len(query_values)), results


def make_panel_4x3(
    seed: int = 42,
    n_samples: int = 2_500,
    sweep_samples: int = 21,
    methods: List[str] = ('ATE','ERM','DA+ERM','PI','DA+PI'),
    augmentation: Optional[str] = None,
    hyperparameters: Optional[Dict[str, Dict[str, float]]] = None,
    experiment: str = EXPERIMENT,
    legend_ncols: int = 2,
):
    if seed >= 0:
        set_seed(seed)
    plt.rcParams.update(RC_PARAMS)

    # --- SEM / DA / features ---
    sem = SEM(experiment=OPTICAL_DEVICE_DATASET, ground_truth=GROUND_TRUTH)
    da = DA(augmentation)
    features = PolynomialFeatures(sem.poly_degree, include_bias=False)

    # Base data (raw feature space)
    X0, y0 = sem(N=n_samples)
    GX0, G0 = da(X0)

    # Sweeps in RAW space (use augmented geometry for consistency with DA)
    pc1_pts, t1, mean1, pc1_vec = sweep_along_pc(GX0, pc_index=0, n_steps=sweep_samples, std_range=3.0)
    pc2_pts, t2, mean2, pc2_vec = sweep_along_pc(GX0, pc_index=1, n_steps=sweep_samples, std_range=3.0)
    radial_pts = radial_sweep_pcs(GX0, n_points=sweep_samples)
    theta = np.linspace(0, 2*np.pi, sweep_samples, endpoint=False)

    # Transform for modeling (poly feature space)
    X = features.fit_transform(X0)
    G = features.fit_transform(G0)
    GX = features.fit_transform(GX0)

    def run_sweep(query_points_raw: np.ndarray) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        # transform queries to poly space once
        Q = features.fit_transform(query_points_raw)
        S, E = Q.shape[0], 1  # single dataset (real-world)
        all_methods: Dict[str, ModelBuilder] = {
            'ATE': lambda: None,
            'ERM': lambda: ERM(),
            'DA+ERM': lambda: ERM(),
            'PI': lambda: PartialR2(),
            'DA+PI': lambda: PartialR2(),
        }
        chosen = [m for m in methods if m in all_methods]
        builders: Dict[str, ModelBuilder] = {m: all_methods[m] for m in chosen}

        res: Dict[str, np.ndarray] = {name: (np.zeros((S, E, 2)) if 'PI' in name else np.zeros((S, E)))
                                      for name in builders}
        gt = np.zeros(S)
        sem_solution = sem.solution  # in poly space

        for i in range(S):
            q_poly = Q[i][np.newaxis, :]
            gt[i] = (q_poly @ sem_solution).item()
            for name, build in builders.items():
                if name == 'ATE':
                    res[name][i, 0] = gt[i]
                    continue
                model = build()
                # ERM uses X / DA+ERM uses GX; PI uses X / DA+PI uses GX internally via fit_model's 'name'
                fit_model(model=model, name=name, X=X, y=y0, G=G, GX=GX,
                          hyperparameters=hyperparameters, da=da)
                out = np.asarray(model.predict(q_poly)).squeeze()
                if 'PI' in name:
                    res[name][i, 0, 0] = out[0]
                    res[name][i, 0, 1] = out[1]
                else:
                    res[name][i, 0] = float(out)
        return res, gt

    res1, gt1 = run_sweep(pc1_pts)
    res2, gt2 = run_sweep(pc2_pts)
    resR, gtR = run_sweep(radial_pts)

    xs = [t1, t2, theta]
    packs = [(res1, gt1), (res2, gt2), (resR, gtR)]

    # helpers
    def mean_band(m):  return m[:, :, 0].mean(axis=1), m[:, :, 1].mean(axis=1)
    def mean_line(m):  return m.mean(axis=1)
    def width(m):      return (m[:, :, 1] - m[:, :, 0]).mean(axis=1) if m.ndim == 3 else np.zeros(m.shape[0])

    def e_worst_sq(m: np.ndarray, gt: np.ndarray, reduce: str = 'max') -> np.ndarray:
        """Worst-case squared error per sweep point."""
        gt = gt[:, None]
        if m.ndim == 2:
            se = (m - gt) ** 2
        elif m.ndim == 3 and m.shape[-1] == 2:
            lo, hi = m[:, :, 0], m[:, :, 1]
            se = np.maximum((lo - gt) ** 2, (hi - gt) ** 2)
        else:
            raise ValueError("Unexpected shape for m.")
        if reduce == 'max':
            return se.max(axis=1)
        elif isinstance(reduce, float) and 0.0 < reduce < 1.0:
            return np.quantile(se, q=reduce, axis=1)
        else:
            raise ValueError("reduce should be 'max' or a quantile in (0,1).")

    # Projections for density (RAW space; axis defined by augmented geometry)
    h_pc1_X = (X0 - mean1) @ pc1_vec
    h_pc1_G = (GX0 - mean1) @ pc1_vec
    h_pc2_X = (X0 - mean2) @ pc2_vec
    h_pc2_G = (GX0 - mean2) @ pc2_vec

    # figure
    fig, axes = plt.subplots(
        4, 3, figsize=(15, 8),
        sharex='col',
        gridspec_kw={'height_ratios': [0.2, 0.2, 0.2, 0.7]},
        constrained_layout=True
    )

    # colors
    def mcolor(name):
        palette = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0','C1','C2','C3','C4','C5'])
        return palette[color_map.get(name, 0) % len(palette)]

    palette = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0','C1','C2','C3','C4','C5'])
    orig_color = palette[color_map.get('ERM', 0) % len(palette)]
    aug_color  = palette[color_map.get('DA+ERM', 3) % len(palette)]

    legend_handles: Dict[str, any] = {}
    bottom_ylabs = [
        r'${\bm{h}}^\top {\bm{x}}_0$',
        r'${\bm{h}}^\top {\bm{x}}_1$',
        r'${\bm{h}}^\top {\bm{x}}(\theta)$'
    ]
    bottom_titles = ['PC1', 'PC2', 'Radial sweep']

    for col in range(3):
        (res, gt), xgrid = packs[col], xs[col]

        # Row 4: predictions
        ax_pred = axes[3, col]
        for name, m in res.items():
            label = TEX_MAPPER.get(name, name)
            if 'PI' in name:
                low, high = mean_band(m)
                h = ax_pred.fill_between(xgrid, low, high, alpha=0.2, edgecolor='none', facecolor=mcolor(name))
            else:
                y = mean_line(m)
                if name == 'ATE':
                    h = ax_pred.plot(xgrid, y, linestyle='--', linewidth=2, color='black')[0]
                else:
                    h = ax_pred.plot(xgrid, y, linewidth=2, color=mcolor(name))[0]
            if label not in legend_handles:
                legend_handles[label] = h

        ax_pred.set_xlabel(r'$t$' if col < 2 else r'$\theta$', fontsize=FS_LABEL)
        ax_pred.set_ylabel(bottom_ylabs[col], fontsize=FS_LABEL)
        ax_pred.tick_params(labelsize=FS_TICK)
        ax_pred.set_xlim([xgrid.min(), xgrid.max()])
        ax_pred.text(0.5, -0.25, bottom_titles[col], transform=ax_pred.transAxes,
                     ha='center', va='top', fontsize=FS_LABEL)

        # Row 3: Width (PI only)
        ax_w = axes[2, col]
        for name in ('PI', 'DA+PI'):
            if name in res:
                w = width(res[name])
                ax_w.fill_between(xgrid, 0.0, w, alpha=0.2, edgecolor='none', facecolor=mcolor(name))
                ax_w.plot(xgrid, w, linewidth=1.5, color=mcolor(name))
        if col == 0: ax_w.set_ylabel('Width', fontsize=FS_LABEL)
        ax_w.tick_params(labelsize=FS_TICK)
        ax_w.set_ylim(0, None); ax_w.margins(y=0)

        # Row 2: E_worst^2 (PI and DA+PI only)
        ax_ew = axes[1, col]
        for name in ('PI', 'DA+PI'):
            if name in res:
                ew2 = e_worst_sq(res[name], gt, reduce='max')
                ax_ew.fill_between(xgrid, 0.0, ew2, alpha=0.2, edgecolor='none', facecolor=mcolor(name))
                ax_ew.plot(xgrid, ew2, linewidth=1.5, color=mcolor(name))
        if col == 0: ax_ew.set_ylabel(r'$E_{\mathrm{worst}}$', fontsize=FS_LABEL)
        ax_ew.tick_params(labelsize=FS_TICK)
        ax_ew.set_ylim(0, None); ax_ew.margins(y=0)

        # Row 1: density (PC columns only; top-right is legend)
        ax_hist = axes[0, col]
        if col == 0:
            ax_hist.hist(h_pc1_X, bins=40, density=True, alpha=0.45, color=orig_color)
            ax_hist.hist(h_pc1_G, bins=40, density=True, alpha=0.45, color=aug_color)
            ax_hist.set_ylabel('Density', fontsize=FS_LABEL)
        elif col == 1:
            ax_hist.hist(h_pc2_X, bins=40, density=True, alpha=0.45, color=orig_color)
            ax_hist.hist(h_pc2_G, bins=40, density=True, alpha=0.45, color=aug_color)
        else:
            ax_hist.axis('off')
        ax_hist.tick_params(labelsize=FS_TICK)
        ax_hist.set_ylim(0, None); ax_hist.margins(y=0)

    # Legend in top-right
    ax_legend = axes[0, 2]
    ax_legend.axis('off')
    label_order = [TEX_MAPPER.get(n, n) for n in res1.keys()]  # keep method order
    handles = [legend_handles[lbl] for lbl in label_order if lbl in legend_handles]
    leg = ax_legend.legend(
        handles=handles, labels=label_order,
        loc='center', ncol=legend_ncols, fontsize=FS_TICK + 2,
        frameon=True, edgecolor='black', fancybox=False,
        borderpad=0.35, labelspacing=0.5, handlelength=2.2,
        handletextpad=0.6, columnspacing=1.1,
        bbox_to_anchor=(0.5, 0.5), bbox_transform=ax_legend.transAxes,
    )
    for h in leg.legendHandles:
        try: h.set_linewidth(2.0)
        except Exception: pass

    fig.align_ylabels(axes[:, 0])
    save(fig, fname='panel_4x3', experiment=experiment, format=PLOT_FORMAT, dpi=PLOT_DPI)


def run(
        seed: int,
        n_samples: int,
        sweep_samples: int,
        methods: List[str],
        augmentation: Optional[List[str]]=[None],
        hyperparameters: Optional[Dict[str, Dict[str, float]]]=None,
        plot_panel: bool=False,
        panel_only: bool=False,
    ):
    status = MANAGER.status_bar(
        status_format=u'Optical device{fill}Sweeping {sweep}{fill}{elapsed}',
        color='bold_underline_bright_white_on_lightslategray',
        justify=enlighten.Justify.CENTER, sweep='<parameter>',
        autorefresh=True, min_delta=0.5
    )

    cv = getattr(hyperparameters, 'cv', None)
    all_methods: Dict[str, ModelBuilder] = {
        'ATE': lambda: None,
        'ERM': lambda: ERM(),
        'DA+ERM': lambda: ERM(),
        'PI': lambda: PartialR2(),
        'DA+PI': lambda: PartialR2(),
    }
    methods: Dict[str, ModelBuilder] = {m: all_methods[m] for m in methods}
    sweep_methods: Dict[str, ModelBuilder] = {
        m: all_methods[m] for m in methods if m in (
            'ERM', 'DA+ERM', 'ATE',
            'PI', 'DA+PI',
        )
    }
    
    # sweep over treatment queries
    status.update(sweep='treatment query')
    logger.info('Sweeping over treatment queries.')
    treatment_values, outcome_values = SweepExperiment(
        seed=seed,
        n_samples=n_samples,
        methods=sweep_methods,
        augmentation=augmentation,
        sweep_samples=sweep_samples,
        hyperparameters=hyperparameters
    ).run_experiment()
    save(
        obj=treatment_values, fname='treatment_values', experiment=EXPERIMENT, format='pkl'
    )
    save(
        obj=outcome_values, fname='outcome_values', experiment=EXPERIMENT, format='pkl'
    )
    sweep_plot(
        treatment_values, outcome_values, **ANNOTATE_SWEEP_PLOT['pc12'], experiment=EXPERIMENT
    )

    if plot_panel:
        make_panel_4x3(seed=seed, n_samples=n_samples, sweep_samples=sweep_samples,
                       methods=methods, augmentation=augmentation[0] if isinstance(augmentation, list) else augmentation,
                       hyperparameters=hyperparameters, experiment=EXPERIMENT)


if __name__ == '__main__':
    CLI = ArgumentParser(description='Optical device experiment.')
    CLI.add_argument(
        '--seed', type=int, default=42, help='Random seed for the experiment. Negative is random.'
    )
    CLI.add_argument(
        '--n_samples', type=int, default=2_500, help='Number of samples per experiment.'
    )
    CLI.add_argument('--n_experiments', type=int, default=10, help='Number of experiments.')
    CLI.add_argument(
        '--sweep_samples', type=int, default=10, help='Sweep resolution across kappa, alpha and gamma.'
    )
    CLI.add_argument(
        '--methods',
        nargs="*",
        type=str,
        default=['ERM', 'DA+ERM', 'DA+IVL-CV', 'DA+IV'],
        help='Methods to use. Specify in space-separated format -- `ERM DA+ERM DA+IVL-CV DA+IV`.'
    )
    CLI.add_argument('--plot-panel', action='store_true')
    CLI.add_argument('--panel-only', action='store_true')
    args = CLI.parse_args()
    run(**vars(args))