"""A34: the query sweeps take their knife-edge tolerance from the dataset configs.

`EPS_TOL` (2**-5) stays the tolerance of every param sweep and of the floor
guards; the query sweeps read `eps_tol` (2**-8) from `SimulationConfig` and
`OpticalDeviceConfig`. Three legs:

  (i)   the configured values: both `eps_tol` fields are 2**-8, `EPS_TOL` is 2**-5;
  (ii)  sim, through the orchestrator's own query runner class (1 experiment,
        n = 256): `runner.eps_tol` is the config's, and `runner.epsilon_iv` is
        eps_iv* + 2**-8 when that clears the IV floor (else the guarded value; the
        leg says which branch ran); the PARAM sweep runner on the same orchestrator
        keeps EPS_TOL, so when both are feasible their budgets differ by exactly
        2**-5 - 2**-8;
  (iii) optical: `_epsilon_budget(None, tol=eps_tol)` sits 2**-8 over the measured
        eps*, the default call 2**-5 over it, and the query runner's
        `default_epsilon` is the former.

Both query runners also carry the raw declared gamma (`raw_gamma`): (i) pins the
declared radii, sqrt(SimulationConfig.gamma) == 1.0 and sqrt(OpticalDeviceConfig.gamma)
== 0.5; (ii) and (iii) check `runner.default_gamma == config gamma / sigma-hat^2` of
the draw and that the fitted PI ball's radius `scale * sqrt(budget)` is sqrt(config
gamma), which is what the rescale exists for.

    python scripts/a34_eps_tol.py
"""

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from src.experiments.base import SweepData  # noqa: E402
from src.experiments.configs import EPS_TOL, FLOOR_GUARD_R, OPTICAL_CONFIG, SIMULATION_CONFIG  # noqa: E402
from src.experiments.optical_device import OpticalOrchestrator  # noqa: E402
from src.experiments.simulation import SimulationOrchestrator  # noqa: E402
from src.experiments.utils import set_seed  # noqa: E402
from src.experiments.utils.metrics import sigma_sq_hat  # noqa: E402
from src.experiments.utils.model_fitting import fit_model  # noqa: E402
from src.methods.sensitivity_models import constraint_floor  # noqa: E402

QUERY_TOL = 2**-8
FAIL = []


def check(tag, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {tag} {detail}")
    if not ok:
        FAIL.append(tag)


def common(**extra):
    return dict(
        seed=42,
        n_experiments=1,
        sweep_samples=4,
        methods=["PI", "DA+PI"],
        hyperparameters={},
        n_jobs=1,
        recalibrate=True,
        pad=False,
        clipy=True,
        mean_match=True,
        **extra,
    )


def query_runner(orch):
    return orch.get_query_runner_cls()(methods=orch.methods, **{**orch._get_clean_kwargs(), "n_experiments": 1})


def check_raw_gamma(leg, runner, declared):
    """The declared gamma is a raw squared radius: rescaled by 1/sigma-hat^2 of the
    draw, so the PI ball's radius comes back to sqrt(declared)."""
    sigma_sq = sigma_sq_hat(runner.X, runner.y, intercept=runner.mean_match)
    check(
        f"{leg} runner.default_gamma == declared gamma / sigma-hat^2",
        runner.raw_gamma and np.isclose(runner.default_gamma, declared / sigma_sq, rtol=1e-12),
        f"{runner.default_gamma:.6g} vs {declared:.6g} / {sigma_sq:.6g}",
    )
    context = runner.setup_data()
    model = runner.methods["PI"]()
    fit_model(model=model, method_name="PI", X=context.X, y=context.y, GX=context.GX, G=context.G, da=context.da)
    radius = model.scale * np.sqrt(model.budget(model.gamma))
    check(
        f"{leg} fitted PI radius == sqrt(declared gamma)",
        np.isclose(radius, np.sqrt(declared), rtol=1e-6),
        f"{radius:.6g} vs {np.sqrt(declared):.6g} (scale {model.scale:.6g}, budget {model.budget(model.gamma):.6g})",
    )


def leg_i():
    print("(i) configured values")
    check("(i) SimulationConfig.eps_tol == 2**-8", SIMULATION_CONFIG.eps_tol == QUERY_TOL)
    check("(i) OpticalDeviceConfig.eps_tol == 2**-8", OPTICAL_CONFIG.eps_tol == QUERY_TOL)
    check("(i) EPS_TOL == 2**-5", EPS_TOL == 2**-5)
    check(
        "(i) sqrt(SimulationConfig.gamma) == 1.0", np.sqrt(SIMULATION_CONFIG.gamma) == 1.0, f"{SIMULATION_CONFIG.gamma}"
    )
    check("(i) sqrt(OpticalDeviceConfig.gamma) == 0.5", np.sqrt(OPTICAL_CONFIG.gamma) == 0.5, f"{OPTICAL_CONFIG.gamma}")


def leg_ii():
    print("(ii) sim query runner vs param sweep runner")
    set_seed(42)
    orch = SimulationOrchestrator(n_samples=256, kernel_dim=0, treatment_dim=32, **common())
    runner = query_runner(orch)
    check("(ii) runner.eps_tol == SIMULATION_CONFIG.eps_tol", runner.eps_tol == SIMULATION_CONFIG.eps_tol)
    check_raw_gamma("(ii)", runner, SIMULATION_CONFIG.gamma)

    eps_iv_star = float(runner.oracle.eps_iv_star)
    floor = constraint_floor(
        runner.GX,
        runner.y,
        runner.default_gamma,
        kind="iv",
        Z=runner.G,
        mean_match=runner.mean_match,
        rho=runner.fit_rho(),
        recalibrate=runner.recalibrate,
    )
    query_budget = eps_iv_star + QUERY_TOL
    query_feasible = query_budget**2 >= floor
    print(f"      eps_iv* {eps_iv_star:.4g}, floor {floor:.4g}, query budget {query_budget:.6g}")
    if query_feasible:
        print("      branch: query budget feasible, exact eps_iv* + 2**-8 expected")
        check(
            "(ii) runner.epsilon_iv == eps_iv* + 2**-8",
            np.isclose(runner.epsilon_iv, query_budget, rtol=1e-9),
            f"{runner.epsilon_iv:.6g} vs {query_budget:.6g}",
        )
    else:
        guarded = float(np.sqrt(FLOOR_GUARD_R * floor))
        print("      branch: query budget under the floor, guarded value expected")
        check(
            "(ii) runner.epsilon_iv == sqrt(FLOOR_GUARD_R * floor)",
            np.isclose(runner.epsilon_iv, guarded, rtol=1e-9),
            f"{runner.epsilon_iv:.6g} vs {guarded:.6g}",
        )

    sweep = orch.get_sweep_runner_cls("gamma")(
        methods=orch.methods, method_factory=orch.build_methods, **orch._get_clean_kwargs()
    )
    knob = sweep.get_param_range()[0]
    data = SweepData.coerce(sweep.generate_data(0, knob))
    sweep_budget = sweep.fit_epsilon_iv(0, 0, data)
    sweep_star = float(sweep.get_oracle(0).eps_iv_star)
    sweep_feasible = (sweep_star + EPS_TOL) ** 2 >= floor
    print(f"      sweep eps_iv* {sweep_star:.4g}, sweep budget {sweep_budget:.6g}")
    check("(ii) sweep runner budget == eps_iv* + EPS_TOL", np.isclose(sweep_budget, sweep_star + EPS_TOL, rtol=1e-9))
    if query_feasible and sweep_feasible:
        check(
            "(ii) sweep - query == 2**-5 - 2**-8",
            np.isclose(sweep_budget - runner.epsilon_iv, EPS_TOL - QUERY_TOL, rtol=1e-9),
            f"{sweep_budget - runner.epsilon_iv:.6g} vs {EPS_TOL - QUERY_TOL:.6g}",
        )
    else:
        print("      (ii) difference check skipped: a budget was floor-guarded")


def leg_iii():
    print("(iii) optical budgets")
    set_seed(42)
    orch = OpticalOrchestrator(n_samples=1000, augmentation="rotation > hflip > vflip > random-permutation", **common())
    measured = orch.measured_epsilon_star()
    with_tol = orch._epsilon_budget(None, tol=OPTICAL_CONFIG.eps_tol)
    default = orch._epsilon_budget(None)
    print(f"      eps* {measured:.6g}; +eps_tol {with_tol:.6g}; +EPS_TOL {default:.6g}")
    check(
        "(iii) _epsilon_budget(None, tol=eps_tol) - eps* == 2**-8",
        np.isclose(with_tol - measured, QUERY_TOL, rtol=1e-9),
    )
    check("(iii) _epsilon_budget(None) - eps* == 2**-5", np.isclose(default - measured, EPS_TOL, rtol=1e-9))
    runner = query_runner(orch)
    check("(iii) query runner default_epsilon == eps* + 2**-8", np.isclose(runner.default_epsilon, with_tol, rtol=1e-9))
    check("(iii) query runner eps_tol == OPTICAL_CONFIG.eps_tol", runner.eps_tol == OPTICAL_CONFIG.eps_tol)
    check_raw_gamma("(iii)", runner, OPTICAL_CONFIG.gamma)


if __name__ == "__main__":
    os.chdir(REPO)  # the optical loader reads data/ relative to the cwd; nothing is written
    leg_i()
    leg_ii()
    leg_iii()
    print(f"\n{'A34 ALL PASS' if not FAIL else 'A34 FAILURES: ' + ', '.join(FAIL)}")
    sys.exit(bool(FAIL))
