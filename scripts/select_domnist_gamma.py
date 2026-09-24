"""Select the do-MNIST gamma by population coverage of h_* on split C.

The block's gamma is EXPLICIT: this script picks it, the user pastes it into
`config.yaml::do_mnist.gamma`. One gamma serves every method, and it is calibrated
on baseline PI alone. The script trains the block's replicate exactly as the run
does (the nets on split A, the mix-in, the B rows), draws `n_select` rows from
split C at `seed + 1` (never the run's evaluation population, which is MNIST test
at `pop_seed`), and for each of PI and DA+PI bisects log gamma to the smallest
value whose coverage of h_* reaches `target_coverage`. The shared value is PI's
(`shared_gamma`, `calibrated_on: PI`); DA+PI's own bisection is a diagnostic, and
its split-C coverage and width at the shared gamma are recorded beside it. The
script verifies the gaussian closed form when the link is gaussian, evaluates both
on a common grid and writes `artifacts/do_mnist/select/gamma_selection.json` and
`coverage.pdf`.

Run from a SCRATCH cwd holding the `config.yaml` (only its `defaults`,
`hyperparameters` and `do_mnist` entries are read; the other blocks are never
touched):

    cd ~/scratch/domnist_runs/port_full && PYTHONPATH=$REPO uv run --project $REPO \\
        python $REPO/scripts/select_domnist_gamma.py [--config PATH] [--n-select N] [--target T]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
from loguru import logger  # noqa: E402
from munch import munchify  # noqa: E402

from src.experiments.configs import DOMNIST_CONFIG, resolve_dataset_block  # noqa: E402
from src.experiments.do_mnist import EXPERIMENT_NAME, DoMNISTOrchestrator, population  # noqa: E402
from src.experiments.utils import fit_model, save, set_seed  # noqa: E402
from src.experiments.utils.coverage import bisect_gamma, coverage_at, grid_curves, required_gamma  # noqa: E402
from src.experiments.utils.diagnostics import centre_error_report  # noqa: E402
from src.experiments.utils.plotting import create_coverage_plot  # noqa: E402
from src.sem.do_mnist import SPLIT_DEFAULT  # noqa: E402

SUBDIR = "select"
SELECTED = ("PI", "DA+PI")
#: the method the one shared gamma is calibrated on; the other is a diagnostic
CALIBRATED_ON = "PI"


def load_block(path: str) -> tuple[dict, dict]:
    import yaml

    with open(path) as fh:
        config = yaml.safe_load(fh) or {}
    defaults = config.get("defaults", {}) or {}
    hyperparameters = config.get("hyperparameters", {}) or {}
    if "do_mnist" not in config:
        raise ValueError(f"{path} has no do_mnist block")
    # the selection never bootstraps: the CI toggle is pinned off here as every
    # yaml-reading gate pins it
    block = {**defaults, **config["do_mnist"], "im-ci": 0}
    block.pop("experiment", None)
    return block, hyperparameters


def select_gamma(block: dict, hyperparameters: dict, n_select: int, target: float, plot: bool = True) -> dict:
    block = resolve_dataset_block("do_mnist", {**block, "methods": list(SELECTED)})
    seed = int(block["seed"])
    set_seed(seed)
    orchestrator = DoMNISTOrchestrator(**block, hyperparameters=munchify(hyperparameters))
    # the run's own query runner: it trains the pair, mixes and fits the balls
    # exactly as the experiment does
    runner = orchestrator.get_query_runner_cls()(
        methods=orchestrator.methods, **{**orchestrator._get_clean_kwargs(), "n_experiments": 1}
    )
    context = runner.setup_data()
    models = {}
    for name in SELECTED:
        model = runner.methods[name]()
        fit_model(
            model=model,
            method_name=name,
            X=context.X,
            y=context.y,
            GX=context.GX,
            G=context.G,
            Z=context.Z,
            GX_inv=context.GX_inv,
            hyperparameters=runner.hyperparameters,
            da=context.da,
        )
        models[name] = model

    # the selection population: `n_select` draws from split C at seed + 1, the
    # SAME partition the replicate used
    parts = runner.sem.split(orchestrator.split or SPLIT_DEFAULT, orchestrator.split_seed)
    Xc, hc, hec, _ = population(parts["C"], n_select, seed + 1)
    idx = parts["C"].last_["idx"]
    for other in ("A", "B"):
        if np.isin(idx, parts[other].subset_).any():
            raise RuntimeError(f"the selection rows touch split {other}")
    logger.info(f"selection population: C n={len(hc):,}; target coverage {target}")

    lo, hi = DOMNIST_CONFIG.gamma_lo, DOMNIST_CONFIG.gamma_hi
    tol, max_iter = DOMNIST_CONFIG.gamma_tol, DOMNIST_CONFIG.max_iter
    record = dict(
        link=DOMNIST_CONFIG.link,
        calibrate_sigma=bool(orchestrator.calibrate_sigma),
        n_components=int(orchestrator.n_components),
        mix_in=float(orchestrator.mix_in),
        mix_in_n_A=float(runner.data_.mask_a.sum()),
        mix_in_n_B=float(runner.data_.mask_b.sum()),
        split=orchestrator.split,
        split_seed=int(orchestrator.split_seed),
        split_key=runner.data_.split_key,
        net=orchestrator.net,
        n_select=int(n_select),
        target_coverage=float(target),
        gamma_lo=lo,
        gamma_hi=hi,
        gamma_tol=tol,
        max_iter=max_iter,
        seed=seed,
        n=int(block["n_samples"]),
        n_pi=int(orchestrator.n_pi),
        augmentation=orchestrator.augmentation,
    )
    hc = np.asarray(hc).ravel()
    for name, model in models.items():
        logger.info(f"{name}: bisecting gamma on C")
        entry = bisect_gamma(model, Xc, hc, target, lo, hi, tol, max_iter)
        mu = np.asarray(model._mu(Xc)).ravel()
        centre = centre_error_report(mu, hc, np.asarray(hec).ravel())
        # the narrowest interval that could reach `target` with this centre
        centre["floor_at_target"] = 2.0 * float(np.quantile(np.abs(mu - hc), target))
        centre["leverage_tax"] = float(entry["width"]) / max(centre["floor_at_target"], 1e-12)
        entry["centre_error"] = centre
        logger.info(
            f"{name}: gamma={entry['gamma']:.5g} coverage={entry['coverage']:.4f} "
            f"width={entry['width']:.4f} ({entry['n_eval']} evaluations)"
        )
        if DOMNIST_CONFIG.link == "gaussian":
            # closed form: the order statistic of the required gamma, the same
            # smallest gamma with coverage >= target the bisection converges to
            closed = float(np.quantile(required_gamma(model, Xc, hc), target, method="inverted_cdf"))
            agree = bool(abs(np.log(entry["gamma"] / max(closed, 1e-300))) <= np.log1p(tol))
            entry["verify"] = dict(closed_form_gamma=closed, bisection_gamma=entry["gamma"], agree_within_tol=agree)
            (logger.info if agree else logger.warning)(
                f"{name} verify: closed form {closed:.5g} vs bisection {entry['gamma']:.5g} "
                f"({'within' if agree else 'OUTSIDE'} gamma_tol={tol})"
            )
        record[name] = entry

    # the one gamma of the run: PI's, fixed for every method
    shared = float(record[CALIBRATED_ON]["gamma"])
    record.update(shared_gamma=shared, calibrated_on=CALIBRATED_ON)
    for name, model in models.items():
        if name == CALIBRATED_ON:
            continue
        coverage, width = coverage_at(model, Xc, hc, shared)
        record[name]["at_shared_gamma"] = dict(gamma=shared, coverage=float(coverage), width=float(width))
        logger.info(f"{name} at the shared gamma {shared:.5g}: coverage {coverage:.4f} width {width:.4f} on C")

    # the bisections evaluate different gammas, so the figure gets ONE common grid
    grid = np.geomspace(lo, hi, DOMNIST_CONFIG.plot_points)
    sweep = {name: {"C": grid_curves(model, Xc, hc, grid)} for name, model in models.items()}
    record["grid"] = {
        "gamma": grid.tolist(),
        **{f"{m}_{k}": np.asarray(v["C"][k]).tolist() for m, v in sweep.items() for k in ("coverage", "width")},
    }
    save(record, "gamma_selection", EXPERIMENT_NAME, "json", subdir=SUBDIR)
    if plot:
        bias = DOMNIST_CONFIG.beta * (0.5 - DOMNIST_CONFIG.eta)  # |bias|; 2|b| is the optimal width
        marks = {f"{name} $\\gamma$": record[name]["gamma"] for name in models}
        create_coverage_plot(
            sweep, grid, marks=marks, ref_width=2 * bias, targets=(target,), experiment=EXPERIMENT_NAME, subdir=SUBDIR
        )
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--n-select", type=int, default=None, help=f"rows from split C (default {DOMNIST_CONFIG.n_select})"
    )
    parser.add_argument("--target", type=float, default=None, help="coverage target (default the block's)")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    block, hyperparameters = load_block(args.config)
    n_select = DOMNIST_CONFIG.n_select if args.n_select is None else args.n_select
    target = float(block.get("target_coverage", 0.99)) if args.target is None else args.target
    record = select_gamma(block, hyperparameters, n_select, target, plot=not args.no_plot)
    print(f"\n=== smallest gamma with coverage of h* >= {target} on split C (put PI's value into config) ===")
    for name in SELECTED:
        entry = record[name]
        print(f"{name}: gamma={entry['gamma']:.5g} coverage={entry['coverage']:.4f} width={entry['width']:.4f}")
        if "at_shared_gamma" in entry:
            at = entry["at_shared_gamma"]
            print(f"{name} at PI's gamma: coverage={at['coverage']:.4f} width={at['width']:.4f}")
    print(f"shared gamma (calibrated on {record['calibrated_on']}): {record['shared_gamma']:.17g}")


if __name__ == "__main__":
    main()
