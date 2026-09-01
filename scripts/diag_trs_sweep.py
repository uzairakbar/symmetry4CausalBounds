"""Diagnose the simulation trS sweep: is the x-axis monotone in the knob?

`ExpansionStrategy` plots the MEASURED expansion rho*tr(S)/k, whose two factors move
in OPPOSITE directions with the knob. If the product folds back, the steps get
reordered along x and a monotonically-narrowing family renders as "width grows as x
shrinks". The docstring asserts monotonicity; nothing checks it.

    python scripts/diag_trs_sweep.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiments.base import SweepData  # noqa: E402
from src.experiments.simulation import SimulationOrchestrator  # noqa: E402
from src.experiments.utils import set_seed  # noqa: E402
from src.experiments.utils.metrics import (
    trace_S_over_k,  # noqa: E402
    rho_hat,
    evaluate_queries,
)
from src.experiments.configs import SIMULATION_CONFIG  # noqa: E402

N_STEPS = 12
METHODS = ["PI", "DA+PI", "DA+PI_IV"]


def main():
    set_seed(42)
    orchestrator = SimulationOrchestrator(
        seed=42,
        n_samples=2048,
        n_experiments=1,
        sweep_samples=N_STEPS,
        kernel_dim=0,
        methods=METHODS,
        hyperparameters={},
        n_jobs=8,
        calibrate=False,
        pad=False,
        clipy=True,
    )
    runner = orchestrator.get_sweep_runner_cls("trS")(
        methods=orchestrator.methods,
        method_factory=orchestrator.build_methods,
        **{k: v for k, v in orchestrator.kwargs.items() if k != "methods"},
    )

    knobs = runner.get_param_range()
    print(f"grid: {N_STEPS} steps, {knobs[0]:.4g} .. {knobs[-1]:.4g}\n")

    header = (
        f"{'knob':>9} {'rho':>8} {'trS/k@1.0':>10} {'trS/k@.999':>11} "
        f"{'x=rho*trS':>10} " + " ".join(f"{m:>10}" for m in METHODS) + f" {'eps_iv':>9}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for knob in knobs:
        data = SweepData.coerce(runner.generate_data(0, knob))
        models = runner.build_models(0, 0, data)

        rho = rho_hat(data.X, data.GX, data.y)
        trs_full = trace_S_over_k(data.X, data.GX, keep=1.0)
        trs_trunc = trace_S_over_k(data.X, data.GX, keep=0.999)
        x = rho * trs_full

        widths = {}
        for name in METHODS:
            estimate = models[name].predict(data.X_test, **runner.get_predict_kwargs(knob, 0))
            record = evaluate_queries(data.estimand, estimate, getattr(models[name], "query_status", None), 0.0)
            widths[name] = record.interval_width

        eps_iv = runner.fit_epsilon_iv(0)
        rows.append((knob, rho, trs_full, trs_trunc, x, widths, eps_iv))
        print(
            f"{knob:9.4g} {rho:8.4f} {trs_full:10.5f} {trs_trunc:11.5f} "
            f"{x:10.5f} " + " ".join(f"{widths[m]:10.5f}" for m in METHODS) + f" {eps_iv:9.6f}"
        )

    # --------------------------------------------------------------- verdicts
    knob_values = np.array([r[0] for r in rows])
    x_values = np.array([r[4] for r in rows])
    trs_full = np.array([r[2] for r in rows])
    trs_trunc = np.array([r[3] for r in rows])
    rhos = np.array([r[1] for r in rows])
    eps_ivs = np.array([r[6] for r in rows])

    print("\n--- monotonicity ---")
    for label, values in (("rho vs knob", rhos), ("tr(S)/k vs knob", trs_full), ("x vs knob", x_values)):
        d = np.diff(values)
        mono = bool((d > 0).all() or (d < 0).all())
        direction = "increasing" if (d > 0).all() else "decreasing" if (d < 0).all() else "NON-MONOTONE"
        print(
            f"  {label:20s} {direction:14s} "
            + (
                ""
                if mono
                else f"sign flips at knob "
                f"{[f'{knob_values[i + 1]:.3g}' for i in range(len(d) - 1) if d[i] * d[i + 1] < 0]}"
            )
        )

    order = np.argsort(x_values)
    reordered = not np.array_equal(order, np.arange(len(x_values)))
    print(f"\n  plotting against x REORDERS the steps: {reordered}")
    if reordered:
        print(f"    knob order along x: {[f'{knob_values[i]:.3g}' for i in order]}")

    def shape(values):
        """Constant counts as monotone; NaN means the method did not solve."""
        if np.isnan(values).any():
            return f"{int(np.isnan(values).sum())}/{len(values)} NaN"
        d = np.diff(values)
        if np.allclose(d, 0.0):
            return "constant"
        return "monotone" if (d >= 0).all() or (d <= 0).all() else "NON-MONOTONE"

    print("\n--- widths: monotone in the knob, or in x? ---")
    for name in METHODS:
        w = np.array([r[5][name] for r in rows])
        print(f"  {name:10s} vs knob: {shape(w):16s} vs plotted x: {shape(w[order])}")

    print("\n--- is the IV budget constant across the sweep? ---")
    print(f"  eps_iv range: {eps_ivs.min():.9f} .. {eps_ivs.max():.9f}  (spread {np.ptp(eps_ivs):.3g})")
    w_da = np.array([r[5]["DA+PI"] for r in rows])
    w_iv = np.array([r[5]["DA+PI_IV"] for r in rows])
    print(f"  max|width(DA+PI_IV) - width(DA+PI)| = {np.nanmax(np.abs(w_iv - w_da)):.3g}")

    print("\n--- does the trace truncation change the axis? ---")
    rel = np.abs(trs_trunc - trs_full) / np.maximum(np.abs(trs_full), 1e-12)
    print(f"  max relative |trS@0.999 - trS@1.0| / trS@1.0 = {rel.max():.3%}")


if __name__ == "__main__":
    main()
