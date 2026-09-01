"""A25: the constraint floor and the budget guard.

Three properties, in order of how badly a regression would hurt:
  1. the closed-form floor equals a cvxpy reference solve;
  2. the guard is a NO-OP wherever the oracle budget is already feasible;
  3. it rescues the steps that are all-INFEASIBLE without it.

    python scripts/a25_floor_guard.py
"""

import os
import sys

import cvxpy as cp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiments.base import SweepData  # noqa: E402
from src.experiments.configs import EPS_TOL, FLOOR_GUARD_R  # noqa: E402
from src.experiments.optical_device import OpticalOrchestrator  # noqa: E402
from src.experiments.simulation import SimulationOrchestrator  # noqa: E402
from src.experiments.utils import set_seed  # noqa: E402
from src.experiments.utils.metrics import evaluate_queries  # noqa: E402
from src.methods.regression import LeastSquaresClosedForm as OLS  # noqa: E402
from src.methods.sensitivity_models import (
    constraint_floor,  # noqa: E402
    inv_constraint_terms,
    iv_constraint_terms,
)

METHODS = ["PI", "DA+PI", "PI_INV", "DA+PI_IV"]
FAIL = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAIL.append(name)


def cvxpy_floor(design, y, gamma, kind, GX=None, Z=None, calibrate=False):
    design = np.asarray(design)
    N, M = design.shape
    h_erm = OLS().fit(design, y).solution.flatten()
    resid = np.asarray(y).flatten() - design @ h_erm
    scale = float(np.sqrt(np.mean(resid**2))) if calibrate else 1.0
    delta = np.sqrt(N) * scale * np.sqrt(gamma)
    A, b = inv_constraint_terms(design, GX) if kind == "inv" else iv_constraint_terms(design, y, Z)
    _, R = np.linalg.qr(design)
    h = cp.Variable(M)
    problem = cp.Problem(
        cp.Minimize(cp.norm(cp.Constant(A) @ h - cp.Constant(b), 2)),
        [cp.norm(cp.Constant(R) @ (h - cp.Constant(h_erm)), 2) <= delta],
    )
    for solver in (cp.CLARABEL, cp.ECOS):
        try:
            problem.solve(solver=solver, verbose=False)
        except Exception:  # noqa: S112 - solver fallback chain, try the next one
            continue
        if problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            return float((problem.value / np.sqrt(N)) ** 2)
    return np.nan


def sim_runner(steps=12):
    set_seed(42)
    orch = SimulationOrchestrator(
        seed=42,
        n_samples=2048,
        n_experiments=1,
        sweep_samples=steps,
        kernel_dim=0,
        methods=METHODS,
        hyperparameters={},
        n_jobs=8,
        calibrate=False,
        pad=False,
        clipy=True,
    )
    return orch.get_sweep_runner_cls("trS")(
        methods=orch.methods,
        method_factory=orch.build_methods,
        **{k: v for k, v in orch.kwargs.items() if k != "methods"},
    )


# ------------------------------------------------------- 1. closed form == solver


def a25_closed_form():
    runner = sim_runner(steps=4)
    worst = 0.0
    for knob in runner.get_param_range():
        data = SweepData.coerce(runner.generate_data(0, knob))
        for gamma in (0.05, 0.505):
            for kind, kw in (("inv", dict(GX=data.GX)), ("iv", dict(Z=data.G))):
                design = data.X if kind == "inv" else data.GX
                got = constraint_floor(design, data.y, gamma, kind=kind, **kw)
                want = cvxpy_floor(design, data.y, gamma, kind, **kw)
                # floors are SQUARED, so 1e-12 here is a 1e-6 budget, i.e. zero;
                # a relative test there compares solver noise with solver noise
                if abs(got - want) < 1e-11:
                    continue
                worst = max(worst, abs(got - want) / max(abs(want), 1e-300))
    check("A25 closed-form floor == cvxpy", worst < 1e-4, f"worst rel {worst:.2e}")


# ------------------------------------------------- 2. no-op when already feasible


def a25_noop_when_feasible():
    """The guard must not touch a budget that already clears its floor -- an
    unconditional `budget^2 >= r*floor` would, and would loosen a live constraint."""
    set_seed(69)
    orch = OpticalOrchestrator(
        seed=69,
        n_samples=200,
        n_experiments=2,
        sweep_samples=3,
        methods=METHODS,
        hyperparameters={},
        n_jobs=1,
        calibrate=False,
        pad=False,
        clipy=True,
        augmentation="rotation > hflip > vflip > gaussian-noise",
    )
    runner = orch.get_sweep_runner_cls("gamma")(
        methods=orch.methods,
        method_factory=orch.build_methods,
        **{k: v for k, v in orch.kwargs.items() if k != "methods"},
    )

    seen_feasible = False
    for e in range(2):
        data = SweepData.coerce(runner.generate_data(e, runner.get_param_range()[0]))
        gamma = runner.fit_gamma(e)
        oracle = runner.get_oracle(e)
        for kind, attr, fit in (
            ("inv", "epsilon_star", runner.fit_epsilon),
            ("iv", "eps_iv_star", runner.fit_epsilon_iv),
        ):
            raw = getattr(oracle, attr, None)
            if raw is None or not np.isfinite(raw):
                continue
            raw = float(raw) + EPS_TOL
            kw = dict(GX=data.GX) if kind == "inv" else dict(Z=data.G)
            design = data.X if kind == "inv" else data.GX
            floor = constraint_floor(design, data.y, gamma, kind=kind, calibrate=runner.calibrate, **kw)
            got = fit(e, 0, data)
            if raw**2 >= floor:
                seen_feasible = True
                check(
                    f"A25 no-op, exp {e} {kind} (oracle {raw:.5f}, sqrt(floor) {np.sqrt(floor):.5f})",
                    got == raw,
                    f"got {got:.6f}",
                )
            else:
                check(
                    f"A25 rescue, exp {e} {kind}", abs(got - np.sqrt(FLOOR_GUARD_R * floor)) < 1e-12, f"got {got:.6f}"
                )
    check("A25 the fixture exercised the no-op path", seen_feasible)


# --------------------------------------------------------- 3. rescues NaN steps


def a25_rescues_infeasible():
    runner = sim_runner()
    rescued = nan_steps = 0
    for index, knob in enumerate(runner.get_param_range()):
        data = SweepData.coerce(runner.generate_data(0, index and knob or knob))
        gamma = runner.fit_gamma(0)
        floor = constraint_floor(data.GX, data.y, gamma, kind="iv", Z=data.G, calibrate=runner.calibrate)
        oracle_iv = float(getattr(runner.get_oracle(0), "eps_iv_star", 0.0)) + EPS_TOL
        guarded = runner.fit_epsilon_iv(0, index, data)

        if oracle_iv**2 < floor:
            nan_steps += 1
            models = runner.build_models(0, index, data)
            record = evaluate_queries(
                data.estimand,
                models["DA+PI_IV"].predict(data.X_test, **runner.get_predict_kwargs(knob, 0)),
                getattr(models["DA+PI_IV"], "query_status", None),
                0.0,
            )
            solved = np.isfinite(record.interval_width) and record.interval_width > 0
            rescued += bool(solved)
            check(
                f"A25 knob {knob:.4g} rescued (budget {oracle_iv:.5f} -> {guarded:.5f})",
                solved,
                f"W {record.interval_width:.4f} C {record.coverage:.3f}",
            )
            check(
                f"A25 knob {knob:.4g} covers at the guarded budget", record.coverage >= 0.95, f"C {record.coverage:.3f}"
            )
        else:
            check(f"A25 knob {knob:.4g} untouched", guarded == oracle_iv, f"{guarded:.6f}")

    check("A25 the sweep had infeasible steps to rescue", nan_steps > 0, f"{nan_steps} steps")
    check("A25 every infeasible step was rescued", rescued == nan_steps, f"{rescued}/{nan_steps}")


if __name__ == "__main__":
    a25_closed_form()
    a25_noop_when_feasible()
    a25_rescues_infeasible()
    print(f"\n{'A25 ALL PASS' if not FAIL else 'A25 FAILURES: ' + ', '.join(FAIL)}")
    sys.exit(bool(FAIL))
