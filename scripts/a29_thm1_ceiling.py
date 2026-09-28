"""A29: the Thm. 1 epsilon ceiling and the gamma threshold it inverts.

Three legs:
  1. the closed forms, on random oracles: eps+ is TIGHT at gamma_min, gamma_min
     never exceeds gamma* nor the old (merely sufficient) threshold, and
     eps_valid <= eps+ exactly on gamma >= gamma_min;
  2. the changed line, IN DATA: on the simulation fixture the post-DA bias is
     measurable, so the smallest budget whose FITTED DA+PI ball still contains
     h_* is known -- `thm1_gamma_min` must reproduce it exactly;
  3. a report on optical, where the DA is not T-invariant, so Thm. 1's premise
     fails and the threshold is a reference rather than a prediction.

Every budget is in the paper's units (gamma* = bias^2/sigma^2, radius
sigma sqrt(gamma)); the DA+PI ball at the INHERITED gamma is the one Thm. 1
speaks about.

    python scripts/a29_thm1_ceiling.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiments.base import SweepData  # noqa: E402
from src.experiments.optical_device import OpticalOrchestrator  # noqa: E402
from src.experiments.simulation import SimulationOrchestrator  # noqa: E402
from src.experiments.utils import set_seed  # noqa: E402
from src.methods.regression import LeastSquaresClosedForm as OLS  # noqa: E402
from src.oracle import (  # noqa: E402
    OracleParameters,
    epsilon_star,
    thm1_eps_ceiling,
    thm1_eps_valid,
    thm1_gamma_min,
)

N_RANDOM = 20_000
METHODS = ["PI", "DA+PI"]
FAIL = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAIL.append(name)


def old_gamma_min(oracle):
    """The pre-2026-09 threshold: inversion of the SUFFICIENT ceiling
    sigma^2 (rho - 1)^2 / rho. Kept here as the reference the new one must
    dominate -- it is the only place this formula still exists."""
    rho = oracle.rho
    return float(max(0.0, np.sqrt(oracle.gamma_star) - (rho - 1.0) / np.sqrt(rho)) ** 2)


def oracle_of(bias_sq, sigma_sq, rho):
    """An OracleParameters carrying gamma* = bias^2/sigma^2."""
    return OracleParameters(
        gamma_star=float(bias_sq / sigma_sq),
        epsilon_star=0.0,
        gamma_z_star=None,
        bias_sq=float(bias_sq),
        sigma_sq=float(sigma_sq),
        rho=float(rho),
    )


# ------------------------------------------------------------ 1. closed forms


def a29_closed_forms():
    rng = np.random.default_rng(0)
    bias_sq = rng.uniform(0.0, 4.0, N_RANDOM)
    sigma_sq = rng.uniform(0.05, 4.0, N_RANDOM)
    rho = 1.0 + rng.exponential(0.5, N_RANDOM)

    worst_tight = worst_bracket = 0.0
    over_star = over_old = 0
    for b, s, r in zip(bias_sq, sigma_sq, rho, strict=True):
        o = oracle_of(b, s, r)
        g_min = thm1_gamma_min(o)
        over_star += g_min > o.gamma_star + 1e-12
        over_old += g_min > old_gamma_min(o) + 1e-12
        if g_min <= 0.0:
            continue
        # tight: at gamma_min the hypothesis of Thm. 1 holds with equality.
        # Scaled by the natural size of a squared bias, NOT by the ceiling
        # itself: eps+ legitimately goes to 0 as rho -> 1, and a relative
        # test there compares cancellation noise with cancellation noise.
        valid, ceiling = thm1_eps_valid(o, g_min), thm1_eps_ceiling(o, g_min)
        worst_tight = max(worst_tight, abs(valid - ceiling) / max(1.0, b))
        # and it is a THRESHOLD: eps_valid <= eps+ iff gamma >= gamma_min
        for factor, want in ((0.5, False), (2.0, True)):
            g = g_min * factor
            holds = thm1_eps_valid(o, g) <= thm1_eps_ceiling(o, g) + 1e-15
            worst_bracket += float(holds != want)

    check("A29: eps_valid == eps+ at gamma_min", worst_tight < 1e-12, f"worst {worst_tight:.2e}")
    check("A29: gamma_min <= gamma*", over_star == 0, f"{over_star} violations")
    check("A29: gamma_min <= old threshold", over_old == 0, f"{over_old} violations")
    check("A29: the bracket brackets", worst_bracket == 0, f"{int(worst_bracket)} violations")

    # rho unavailable: eps_valid does not need it, the other two degrade gracefully
    blind = OracleParameters(gamma_star=0.8, epsilon_star=0.0, gamma_z_star=None, bias_sq=0.4, sigma_sq=0.5, rho=None)
    r_base = np.sqrt(blind.sigma_sq * 0.3)
    want = max(0.0, np.sqrt(blind.bias_sq) - r_base) ** 2
    check("A29: eps_valid is defined without rho", abs(thm1_eps_valid(blind, 0.3) - want) < 1e-15)
    check("A29: eps+ is nan without rho", np.isnan(thm1_eps_ceiling(blind, 0.3)))
    check("A29: gamma_min falls back to gamma* without rho", thm1_gamma_min(blind) == blind.gamma_star)

    # rho = 1: the DA gives up nothing, so it buys back nothing
    o = oracle_of(0.7, 0.9, 1.0)
    check("A29: rho = 1 => gamma_min == gamma*", abs(thm1_gamma_min(o) - o.gamma_star) < 1e-12)


# ------------------------------------------- 2. the changed line, in the data


def sim_runner(n_experiments=8, sweep_samples=8):
    set_seed(42)
    orch = SimulationOrchestrator(
        seed=42,
        n_samples=2048,
        n_experiments=n_experiments,
        sweep_samples=sweep_samples,
        kernel_dim=0,
        treatment_dim=32,
        methods=METHODS,
        hyperparameters={},
        n_jobs=1,
        recalibrate=True,
        pad=False,
        clipy=True,
    )
    return orch.get_sweep_runner_cls("gamma")(
        methods=orch.methods,
        method_factory=orch.build_methods,
        **{k: v for k, v in orch.kwargs.items() if k != "methods"},
    )


def sample_quantities(data, sem, mean_match=True):
    """(A^2, B^2, s^2, s~^2) on one draw: OLS on X and on GX, against `sem.f`.

    Measured in the class the SOLVER searches, which under `mean_match` carries a
    free intercept (Lem. 2, Asm. 1's base clause) -- so `OLS(fit_intercept=True)`
    and the truth read off `sem.f`, not off `sem.solution`. The optical ground
    truth's intercept is not decoration: taking `solution` alone reports
    A^2 = 0.5024 where the Lem. 2 geometry gives 0.4025 = `sem.bias_sq` exactly,
    a 17% error in a number printed beside the pinned vline.
    """
    out = []
    for design in (data.X, data.GX):
        h_erm = OLS(fit_intercept=mean_match).fit(design, data.y)
        residual = data.y.flatten() - np.asarray(h_erm.predict(design)).flatten()
        bias = np.asarray(h_erm.predict(design)).flatten() - np.asarray(sem.f(design)).flatten()
        out.append((float(np.mean(bias**2)), float(np.mean(residual**2))))
    (a_sq, s_sq), (b_sq, s_da_sq) = out
    return a_sq, b_sq, s_sq, s_da_sq


def a29_in_data():
    """The set-membership transition is OBSERVABLE on sim: the DA is exactly
    invariant, so B^2 = A^2 - C^2 holds in-sample and the smallest ratio whose
    fitted DA+PI ball contains h_* is r_emp = B^2 / (s~^2 gamma*_sample)."""
    runner = sim_runner()
    knob = runner.get_param_range()[0]
    rows = []
    for j in range(runner.n_experiments):
        data = SweepData.coerce(runner.generate_data(j, knob))
        a_sq, b_sq, s_sq, s_da_sq = sample_quantities(data, runner.sems[j], runner.mean_match)

        # precondition: Lem. 4's identity, exact here (OLS Pythagoras + GX W = X W)
        drift = abs(b_sq - (a_sq - (s_da_sq - s_sq)))
        check(f"A29 sim exp {j}: Lem. 4 identity in sample", drift <= 1e-10 * s_da_sq, f"|d| {drift:.2e}")

        gamma_star_sample = a_sq / s_sq
        r_emp = b_sq / (s_da_sq * gamma_star_sample)
        o = oracle_of(a_sq, s_sq, s_da_sq / s_sq)

        got = thm1_gamma_min(o) / o.gamma_star
        check(
            f"A29 sim exp {j}: gamma_min == the fitted transition",
            abs(got - r_emp) <= 1e-10,
            f"{got:.6f} vs {r_emp:.6f} (old formula {old_gamma_min(o) / o.gamma_star:.6f})",
        )

        population = runner.get_oracle(j)
        rows.append(
            (
                r_emp,
                thm1_gamma_min(population) / population.gamma_star,
                old_gamma_min(population) / population.gamma_star,
            )
        )

    rows = np.array(rows)
    print(
        f"  report: sim r_emp {rows[:, 0].mean():.4f} | population threshold new {rows[:, 1].mean():.4f} "
        f"old {rows[:, 2].mean():.4f} (per-experiment spread +-{rows[:, 0].std():.3f}; the population "
        "line is not a discriminating statistic)"
    )


# --------------------------------------------------- 3. optical: premise fails


def a29_optical_report(n_experiments=8):
    """Optical runs the config's own fixture, so the threshold reported here is
    the one the published configuration implies."""
    set_seed(42)
    orch = OpticalOrchestrator(
        seed=42,
        n_samples=1000,
        n_experiments=n_experiments,
        sweep_samples=8,
        methods=METHODS,
        hyperparameters={},
        n_jobs=1,
        recalibrate=True,
        pad=False,
        clipy=True,
        augmentation="rotation > gaussian-noise",
    )
    runner = orch.get_sweep_runner_cls("gamma")(
        methods=orch.methods,
        method_factory=orch.build_methods,
        **{k: v for k, v in orch.kwargs.items() if k != "methods"},
    )
    knob = runner.get_param_range()[0]
    new, old = [], []
    for j in range(n_experiments):
        population = runner.get_oracle(j)
        new.append(thm1_gamma_min(population) / population.gamma_star)
        old.append(old_gamma_min(population) / population.gamma_star)
        if j >= 2:  # the sample-fit report is illustrative; two draws make the point
            continue
        data = SweepData.coerce(runner.generate_data(j, knob))
        a_sq, b_sq, s_sq, s_da_sq = sample_quantities(data, runner.sems[j], runner.mean_match)
        r_emp = b_sq / (s_da_sq * (a_sq / s_sq))
        o = oracle_of(a_sq, s_sq, s_da_sq / s_sq)
        eps = epsilon_star(runner.sems[j], runner.das[j], features=runner._features)
        print(
            f"  report: optical exp {j} SAMPLE-FIT (not the population threshold): r_emp {r_emp:.4f} vs "
            f"new {thm1_gamma_min(o) / o.gamma_star:.4f} old {old_gamma_min(o) / o.gamma_star:.4f} "
            f"| rho {s_da_sq / s_sq:.4f} eps* {eps:.4f}"
        )
    print(
        f"  report: optical population threshold new {np.mean(new):.4f} old {np.mean(old):.4f} "
        f"(gamma* {runner.get_oracle(0).gamma_star:.4f}, rho {runner.get_oracle(0).rho:.4f})"
    )
    print(
        "  the optical DA is NOT T-invariant (eps* >> 0), so Lem. 4's A^2 = B^2 + C^2 fails and\n"
        "  Thm. 1's premise does not hold there: BOTH lines are references, not predictions, and\n"
        "  Thm. 3.A's eps-padding is what carries validity. Do not 'fix' the formula against them."
    )


if __name__ == "__main__":
    a29_closed_forms()
    a29_in_data()
    a29_optical_report()
    print(f"\n{'A29 ALL PASS' if not FAIL else 'A29 FAILURES: ' + ', '.join(FAIL)}")
    sys.exit(bool(FAIL))
