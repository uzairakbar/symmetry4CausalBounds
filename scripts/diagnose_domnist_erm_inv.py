"""Diagnose the do-MNIST ERM+INV centre on the full draw (GPU nets, no bound solves).

Not a gate and not a selection: it picks no gamma and applies no accuracy rule. It
runs the run's own replicate (`draw_replicate`) and, through its `inv_fits` hook,
fits one `InvariantGradientDescentERM` per point of tau in {1e-3, 4e-4, 1e-4} (plus
the block's `erm_inv_tau`) x epochs in {1, 2} (plus the configured count) at the
exact point where the run fits its ERM+INV net: after the ERM, on the unmixed
split-A pairs, before the mix-in. The configured net is the one the replicate keeps,
so everything downstream is the run's. Per point it records

  - the achieved invariance error on the unmixed B pairs, mu clipped as the PI+INV
    constraint reads it (`E_inv_B`), and `constraint_met` (E_inv_B <= 1.5 tau).
    That flag reads the constraint alone: a net collapsed to a constant meets it,
    so f-accuracy is recorded beside it, as a diagnostic and never a criterion;
  - f-accuracy and RMSE to h_* on its own 1,000 rows from split C at `seed + 1`
    (never MNIST test, the evaluation population's pool);
  - the PI+INV constraint value at the centre (`con0`) and the floor on the ball at
    the block's gamma, both against eps^2;
  - the training time, the `state_sha1` and the AL trace.

The one thing it can change is the epoch count: if the configured tau's constraint
is not met at 1 epoch but is at 2, the `recommended` entry says 2 and
`DoMNISTConfig.erm_inv_epochs = 2` is pasted by hand. If it is met at neither, the
configured values stay and a WARNING is logged. The `recommended` entry's
`state_sha1` is what the run's `run.json` `erm_inv_state_sha1` must equal once the
epoch count is pasted. Writes `artifacts/do_mnist/select/erm_inv_diagnostics.json`
and `erm_inv_trace.pdf`.

Run from a SCRATCH cwd holding the `config.yaml` (its `defaults`, `hyperparameters`
and `do_mnist` entries are read, as by `select_domnist_gamma.py`):

    cd ~/scratch/domnist_runs/erm_inv_diag && PYTHONPATH=$REPO uv run --project $REPO \\
        python $REPO/scripts/diagnose_domnist_erm_inv.py [--config PATH] [--no-plot]
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
from loguru import logger  # noqa: E402
from munch import munchify  # noqa: E402
from select_domnist_gamma import load_block  # noqa: E402

from src.experiments.configs import DOMNIST_CONFIG, resolve_dataset_block  # noqa: E402
from src.experiments.do_mnist import (  # noqa: E402
    EXPERIMENT_NAME,
    DoMNISTOrchestrator,
    draw_replicate,
    e_inv,
    population,
    train_erm_inv,
)
from src.experiments.utils import fit_model, save, set_seed  # noqa: E402
from src.experiments.utils.plotting import create_al_trace_plot  # noqa: E402
from src.methods.regression import InvariantGradientDescentERM  # noqa: E402

SUBDIR = "select"
TAUS = (1e-3, 4e-4, 1e-4)
EPOCHS = (1, 2)
N_C = 1_000
#: the constraint counts as met when the achieved E_inv on the B pairs is within this
#: factor of its target
CONSTRAINT_MET = 1.5


def centre_record(net, data, Xc, hc, fc) -> dict:
    """Invariance on the B pairs and fit on the split-C rows of one centre net."""
    mu = np.asarray(net.predict_mean(Xc), dtype=float).ravel()
    return dict(
        E_inv_B=e_inv(net, data.X, data.GX_inv),
        f_accuracy_C=float(np.mean((mu > 0.5) == np.asarray(fc).ravel())),
        rmse_h_star_C=float(np.sqrt(np.mean((mu - np.asarray(hc).ravel()) ** 2))),
    )


def inv_floor(orchestrator, net, data, hyperparameters) -> dict:
    """PI+INV on this centre, built and fitted as the run does: `con0` (the
    centre's own constraint value), the floor at the block's gamma and eps^2."""
    nets = {**data.nets, "INV": net}
    model = orchestrator._build(
        ["PI+INV"],
        gamma=orchestrator.gamma,
        epsilon=orchestrator.epsilon,
        epsilon_iv=orchestrator.epsilon,
        outcome_models=nets,
        **orchestrator.toggles,
    )["PI+INV"]()
    fit_model(
        model=model,
        method_name="PI+INV",
        X=data.X,
        y=data.y,
        GX=data.GX,
        G=data.G,
        GX_inv=data.GX_inv,
        hyperparameters=hyperparameters,
    )
    start = time.perf_counter()
    floor = float(model.constraint_floor(model._radius(model.gamma)))
    return dict(
        con0=float(model._con_at_zero()),
        floor=floor,
        budget=float(model._budget()),
        floor_seconds=time.perf_counter() - start,
    )


def diagnose(block: dict, hyperparameters: dict, plot: bool = True) -> dict:
    # the diagnostic is about the `inv` centre whatever the scratch block says
    block = resolve_dataset_block("do_mnist", {**block, "methods": ["PI", "PI+INV"], "inv_recenter": "inv"})
    block.pop("experiment", None)
    seed = int(block["seed"])
    set_seed(seed)
    orchestrator = DoMNISTOrchestrator(**block, hyperparameters=munchify(hyperparameters))
    tau = orchestrator.erm_inv_tau
    base_epochs = int(hyperparameters.get("epochs", 1))
    epochs = DOMNIST_CONFIG.erm_inv_epochs or base_epochs
    taus = sorted(set(TAUS) | {tau}, reverse=True)
    epoch_grid = sorted(set(EPOCHS) | {epochs})
    logger.info(f"ERM+INV diagnostics: configured tau {tau:g} at {epochs} epoch(s); grid {taus} x {epoch_grid}")

    fits: dict[tuple[float, int], dict] = {}

    def inv_fits(X, GX, y, init_seed, train_kw):
        """Every grid point on the run's pairs and seed; returns the configured net."""
        configured = None
        for t in taus:
            for e in epoch_grid:
                start = time.perf_counter()
                net = train_erm_inv(X, GX, y, init_seed, t, train_kw, net=orchestrator.net, epochs=e)
                seconds = time.perf_counter() - start
                fits[(t, e)] = dict(
                    blob=net.state_blob(),
                    state_sha1=net.state_sha1(),
                    trace=[list(row) for row in net.al_trace_],
                    train_seconds=seconds,
                    n_batches=int(np.ceil(len(X) / int(train_kw.get("batch", 256)))),
                )
                logger.info(f"  tau {t:g} x {e} epoch(s): {seconds:.1f} s, last window c {net.al_trace_[-1][1]:.3g}")
                if (t, e) == (tau, epochs):
                    configured = net
                else:
                    del net
        return configured

    sem = orchestrator._sem_factory()
    data = draw_replicate(
        sem,
        orchestrator._sem_test_factory(),
        orchestrator._da_factory(),
        seed=seed,
        n_samples=int(block["n_samples"]),
        n_pi=orchestrator.n_pi,
        mix_in=orchestrator.mix_in,
        hyperparameters=hyperparameters,
        net=orchestrator.net,
        split=orchestrator.split,
        split_seed=orchestrator.split_seed,
        train_inv=True,
        erm_inv_tau=tau,
        inv_fits=inv_fits,
    )

    # its own draw from split C at seed + 1: the selection script's seed, not its rows
    parts = sem.split(orchestrator.split, orchestrator.split_seed)
    Xc, hc, _, fc = population(parts["C"], N_C, seed + 1)
    idx = parts["C"].last_["idx"]
    for other in ("A", "B"):
        if np.isin(idx, parts[other].subset_).any():
            raise RuntimeError(f"the diagnostic rows touch split {other}")

    points = []
    for (t, e), fit in fits.items():
        net = data.nets["INV"] if (t, e) == (tau, epochs) else InvariantGradientDescentERM.from_blob(fit["blob"])
        entry = dict(tau=t, epochs=e, state_sha1=fit["state_sha1"], train_seconds=fit["train_seconds"])
        entry.update(centre_record(net, data, Xc, hc, fc))
        entry["constraint_met"] = bool(entry["E_inv_B"] <= CONSTRAINT_MET * t)
        entry.update(inv_floor(orchestrator, net, data, hyperparameters))
        entry["trace"] = fit["trace"]
        points.append(entry)
        state = "constraint met" if entry["constraint_met"] else "constraint NOT met"
        logger.info(
            f"tau {t:g} x {e}: E_inv_B {entry['E_inv_B']:.4g} ({state}, f-acc {entry['f_accuracy_C']:.4f}) "
            f"RMSE {entry['rmse_h_star_C']:.4f} con0 {entry['con0']:.4g} "
            f"floor {entry['floor']:.4g} vs eps^2 {entry['budget']:.4g}"
        )
    reference = {
        name: centre_record(data.nets[key], data, Xc, hc, fc) for name, key in (("ERM", "X"), ("DA+ERM", "GX"))
    }

    def point(t, e):
        return next(p for p in points if p["tau"] == t and p["epochs"] == e)

    # the one rule: raise the epoch count to 2 only if the constraint is not met at 1
    # and is at 2. Accuracy is reported, never selected on
    chosen = epochs
    if not point(tau, epochs)["constraint_met"]:
        if epochs == 1 and 2 in epoch_grid and point(tau, 2)["constraint_met"]:
            chosen = 2
            logger.warning(
                f"ERM+INV: tau {tau:g} constraint not met at 1 epoch but met at 2 "
                f"(f-acc {point(tau, 2)['f_accuracy_C']:.4f}); set DoMNISTConfig.erm_inv_epochs = 2."
            )
        else:
            logger.warning(
                f"ERM+INV: tau {tau:g} constraint met at none of the epoch counts {epoch_grid}; the configured "
                "values stay, and the eps^2 check decides whether PI+INV is usable."
            )
    recommended = point(tau, chosen)
    met = "met" if recommended["constraint_met"] else "NOT met"
    logger.info(
        f"ERM+INV recommended: tau {tau:g} x {chosen}: constraint {met}, "
        f"f-acc {recommended['f_accuracy_C']:.4f} on split C (ERM {reference['ERM']['f_accuracy_C']:.4f})"
    )
    if recommended["con0"] > recommended["budget"] or recommended["floor"] > recommended["budget"]:
        logger.warning(
            f"ERM+INV: con0 {recommended['con0']:.4g} / floor {recommended['floor']:.4g} against eps^2 "
            f"{recommended['budget']:.4g}: PI+INV is likely infeasible at this epsilon."
        )

    record = dict(
        erm_inv_tau=tau,
        configured_epochs=epochs,
        constraint_met_factor=CONSTRAINT_MET,
        constraint_met_means="E_inv_B <= constraint_met_factor * tau; the constraint alone, not fit quality "
        "(a constant net meets it). f_accuracy_C beside it is a diagnostic, never a selection criterion",
        taus=taus,
        epochs=epoch_grid,
        n_c=N_C,
        n=int(block["n_samples"]),
        n_pi=int(orchestrator.n_pi),
        seed=seed,
        gamma=float(orchestrator.gamma),
        epsilon=float(orchestrator.epsilon),
        split=orchestrator.split,
        split_seed=int(orchestrator.split_seed),
        split_key=data.split_key,
        mix_in=float(orchestrator.mix_in),
        augmentation=orchestrator.augmentation,
        al=dict(
            mu0=DOMNIST_CONFIG.erm_inv_mu0,
            growth=DOMNIST_CONFIG.erm_inv_growth,
            updates_per_epoch=DOMNIST_CONFIG.erm_inv_updates_per_epoch,
            mu_max=DOMNIST_CONFIG.erm_inv_mu_max,
        ),
        train_seconds_X=float(data.diagnostics["train_seconds_X"]),
        train_seconds_GX=float(data.diagnostics["train_seconds_GX"]),
        points=points,
        reference=reference,
        recommended={k: v for k, v in recommended.items() if k != "trace"},
        epochs_changed=bool(chosen != epochs),
    )
    save(record, "erm_inv_diagnostics", EXPERIMENT_NAME, "json", subdir=SUBDIR)
    if plot:
        n_batches = next(iter(fits.values()))["n_batches"]
        create_al_trace_plot(
            {(p["tau"], p["epochs"]): p["trace"] for p in points},
            n_batches,
            experiment=EXPERIMENT_NAME,
            subdir=SUBDIR,
        )
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    block, hyperparameters = load_block(args.config)
    record = diagnose(block, hyperparameters, plot=not args.no_plot)
    print("\n=== ERM+INV diagnostics (E_inv on the B pairs; f-accuracy and RMSE on split C) ===")
    for p in record["points"]:
        print(
            f"tau {p['tau']:<7g} epochs {p['epochs']}: E_inv_B {p['E_inv_B']:.4g} "
            f"{'constraint met' if p['constraint_met'] else 'constraint NOT met':<18} f-acc {p['f_accuracy_C']:.4f} "
            f"RMSE {p['rmse_h_star_C']:.4f} con0 {p['con0']:.4g} floor {p['floor']:.4g} (eps^2 {p['budget']:.4g})"
        )
    for name, r in record["reference"].items():
        print(f"{name:<7}: E_inv_B {r['E_inv_B']:.4g} f-acc {r['f_accuracy_C']:.4f} RMSE {r['rmse_h_star_C']:.4f}")
    rec = record["recommended"]
    print(f"recommended: tau {rec['tau']:g} at {rec['epochs']} epoch(s), state_sha1 {rec['state_sha1']}")
    if record["epochs_changed"]:
        print("put erm_inv_epochs = 2 into DoMNISTConfig")


if __name__ == "__main__":
    main()
