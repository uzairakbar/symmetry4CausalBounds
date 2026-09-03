"""A29: the Thm. 1 epsilon ceiling and the gamma threshold it inverts.

Three legs:
  1. the closed forms, on random oracles: eps+ is TIGHT at gamma_min, gamma_min
     never exceeds gamma* nor the old (merely sufficient) threshold, and
     eps_valid <= eps+ exactly on gamma >= gamma_min;
  2. the changed line, IN DATA: on the simulation fixture the post-DA bias is
     measurable, so the smallest budget whose FITTED DA+PI ball still contains
     h_* is known -- `thm1_gamma_min` must reproduce it exactly;
  3. a report on optical, where the DA is not T-invariant, so Thm. 1's premise
     fails and the threshold is a reference rather than a prediction;
  4. the two PLOTTED vlines, pinned: nothing else in the gate suite digests them
     (a10 stores values/results/statuses only), so a change to the published
     figures' annotation would otherwise pass every gate unnoticed.

    python scripts/a29_thm1_ceiling.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiments.base import SweepData  # noqa: E402
from src.experiments.configs import OPTICAL_CONFIG  # noqa: E402
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
# The gamma-sweep vline, mean over the config's 8 replicates at `calibrate: true`:
# thm1_gamma_min / gamma* on the POPULATION oracle -- i.e. the number the figure
# actually draws. Deterministic given the seeds. DELIBERATE geometry changes move
# these; update the constant in the same commit and say so in the message.
VLINE_SIM, VLINE_OPTICAL = 0.4493541624, 0.2784036752
VLINE_RTOL = 1e-6
# The optical pin belongs to the PUBLISHED optical configuration. Trying another
# ground truth or dataset index legitimately moves the line (measured: a linear
# ground truth sends it to 0, i.e. Thm. 1 certifies validity at every budget), so
# the check reports instead of failing when `OpticalDeviceConfig` has moved.
VLINE_OPTICAL_CONFIG = ("polynomial", 8)
FAIL = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAIL.append(name)


def old_gamma_min(oracle, calibrate):
    """The pre-2026-09 threshold: inversion of the SUFFICIENT ceiling
    sigma^2 (rho - 1)^2 / rho. Kept here as the reference the new one must
    dominate -- it is the only place this formula still exists."""
    rho = oracle.rho
    if calibrate:
        return float(max(0.0, np.sqrt(oracle.gamma_star) - (rho - 1.0) / np.sqrt(rho)) ** 2)
    return float(max(0.0, oracle.bias_sq - oracle.sigma_sq * (rho - 1.0)))


def oracle_of(bias_sq, sigma_sq, rho, calibrate):
    """An OracleParameters carrying gamma* in the units `calibrate` implies."""
    return OracleParameters(
        gamma_star=float(bias_sq / sigma_sq if calibrate else bias_sq),
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

    for calibrate in (True, False):
        worst_tight = worst_bracket = 0.0
        over_star = over_old = 0
        for b, s, r in zip(bias_sq, sigma_sq, rho, strict=True):
            o = oracle_of(b, s, r, calibrate)
            g_min = thm1_gamma_min(o, calibrate)
            over_star += g_min > o.gamma_star + 1e-12
            over_old += g_min > old_gamma_min(o, calibrate) + 1e-12
            if g_min <= 0.0:
                continue
            # tight: at gamma_min the hypothesis of Thm. 1 holds with equality.
            # Scaled by the natural size of a squared bias, NOT by the ceiling
            # itself: eps+ legitimately goes to 0 as rho -> 1, and a relative
            # test there compares cancellation noise with cancellation noise.
            valid, ceiling = thm1_eps_valid(o, g_min, calibrate), thm1_eps_ceiling(o, g_min, calibrate)
            worst_tight = max(worst_tight, abs(valid - ceiling) / max(1.0, b))
            # and it is a THRESHOLD: eps_valid <= eps+ iff gamma >= gamma_min
            for factor, want in ((0.5, False), (2.0, True)):
                g = g_min * factor
                holds = thm1_eps_valid(o, g, calibrate) <= thm1_eps_ceiling(o, g, calibrate) + 1e-15
                worst_bracket += float(holds != want)

        label = "calibrated" if calibrate else "raw"
        check(f"A29 {label}: eps_valid == eps+ at gamma_min", worst_tight < 1e-12, f"worst {worst_tight:.2e}")
        check(f"A29 {label}: gamma_min <= gamma*", over_star == 0, f"{over_star} violations")
        check(f"A29 {label}: gamma_min <= old threshold", over_old == 0, f"{over_old} violations")
        check(f"A29 {label}: the bracket brackets", worst_bracket == 0, f"{int(worst_bracket)} violations")

    # rho unavailable: eps_valid does not need it, the other two degrade gracefully
    blind = OracleParameters(gamma_star=0.8, epsilon_star=0.0, gamma_z_star=None, bias_sq=0.4, sigma_sq=0.5, rho=None)
    for calibrate in (True, False):
        label = "calibrated" if calibrate else "raw"
        r_base = np.sqrt(blind.sigma_sq * 0.3) if calibrate else np.sqrt(0.3)
        want = max(0.0, np.sqrt(blind.bias_sq) - r_base) ** 2
        check(
            f"A29 {label}: eps_valid is defined without rho", abs(thm1_eps_valid(blind, 0.3, calibrate) - want) < 1e-15
        )
        check(f"A29 {label}: eps+ is nan without rho", np.isnan(thm1_eps_ceiling(blind, 0.3, calibrate)))
        check(
            f"A29 {label}: gamma_min falls back to gamma* without rho",
            thm1_gamma_min(blind, calibrate) == blind.gamma_star,
        )

    # rho = 1: the DA gives up nothing, so it buys back nothing
    for calibrate in (True, False):
        o = oracle_of(0.7, 0.9, 1.0, calibrate)
        check(
            f"A29 {'calibrated' if calibrate else 'raw'}: rho = 1 => gamma_min == gamma*",
            abs(thm1_gamma_min(o, calibrate) - o.gamma_star) < 1e-12,
        )


# ------------------------------------------- 2. the changed line, in the data


def sim_runner(n_experiments=8, sweep_samples=8):
    set_seed(42)
    orch = SimulationOrchestrator(
        seed=42,
        n_samples=2048,
        n_experiments=n_experiments,
        sweep_samples=sweep_samples,
        kernel_dim=0,
        methods=METHODS,
        hyperparameters={},
        n_jobs=1,
        calibrate=True,
        pad=False,
        clipy=True,
    )
    return orch.get_sweep_runner_cls("gamma")(
        methods=orch.methods,
        method_factory=orch.build_methods,
        **{k: v for k, v in orch.kwargs.items() if k != "methods"},
    )


def sample_quantities(data, W):
    """(A^2, B^2, s^2, s~^2) on one draw: OLS on X and on GX, against the truth W."""
    W = np.asarray(W).flatten()
    out = []
    for design in (data.X, data.GX):
        h_erm = OLS().fit(design, data.y).solution.flatten()
        residual = data.y.flatten() - design @ h_erm
        bias = design @ (h_erm - W)
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
        a_sq, b_sq, s_sq, s_da_sq = sample_quantities(data, runner.sems[j].solution)

        # precondition: Lem. 4's identity, exact here (OLS Pythagoras + GX W = X W)
        drift = abs(b_sq - (a_sq - (s_da_sq - s_sq)))
        check(f"A29 sim exp {j}: Lem. 4 identity in sample", drift <= 1e-10 * s_da_sq, f"|d| {drift:.2e}")

        gamma_star_sample = a_sq / s_sq
        r_emp = b_sq / (s_da_sq * gamma_star_sample)
        o = oracle_of(a_sq, s_sq, s_da_sq / s_sq, calibrate=True)

        got = thm1_gamma_min(o, True) / o.gamma_star
        check(
            f"A29 sim exp {j}: calibrated gamma_min == the fitted transition",
            abs(got - r_emp) <= 1e-10,
            f"{got:.6f} vs {r_emp:.6f} (old formula {old_gamma_min(o, True) / o.gamma_star:.6f})",
        )
        got_raw = thm1_gamma_min(oracle_of(a_sq, s_sq, s_da_sq / s_sq, calibrate=False), False)
        check(f"A29 sim exp {j}: raw gamma_min == B^2", abs(got_raw - b_sq) <= 1e-10, f"{got_raw:.6g} vs {b_sq:.6g}")

        population = runner.get_oracle(j)
        rows.append(
            (
                r_emp,
                thm1_gamma_min(population, True) / population.gamma_star,
                old_gamma_min(population, True) / population.gamma_star,
            )
        )

    rows = np.array(rows)
    print(
        f"  report: sim r_emp {rows[:, 0].mean():.4f} | population vline new {rows[:, 1].mean():.4f} "
        f"old {rows[:, 2].mean():.4f} (per-experiment spread +-{rows[:, 0].std():.3f}; the population "
        "line is not a discriminating statistic)"
    )
    return float(rows[:, 1].mean()), float(rows[:, 2].mean())


# --------------------------------------------------- 3. optical: premise fails


def a29_optical_report(n_experiments=8):
    """Optical runs the config's own fixture, so the vline reported here is the
    one the published figure draws."""
    set_seed(42)
    orch = OpticalOrchestrator(
        seed=42,
        n_samples=1000,
        n_experiments=n_experiments,
        sweep_samples=8,
        methods=METHODS,
        hyperparameters={},
        n_jobs=1,
        calibrate=True,
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
        new.append(thm1_gamma_min(population, True) / population.gamma_star)
        old.append(old_gamma_min(population, True) / population.gamma_star)
        if j >= 2:  # the sample-fit report is illustrative; two draws make the point
            continue
        data = SweepData.coerce(runner.generate_data(j, knob))
        a_sq, b_sq, s_sq, s_da_sq = sample_quantities(data, runner.sems[j].solution)
        r_emp = b_sq / (s_da_sq * (a_sq / s_sq))
        o = oracle_of(a_sq, s_sq, s_da_sq / s_sq, calibrate=True)
        eps = epsilon_star(runner.sems[j], runner.das[j], features=runner._features)
        print(
            f"  report: optical exp {j} SAMPLE-FIT (not the plotted vline): r_emp {r_emp:.4f} vs "
            f"new {thm1_gamma_min(o, True) / o.gamma_star:.4f} old {old_gamma_min(o, True) / o.gamma_star:.4f} "
            f"| rho {s_da_sq / s_sq:.4f} eps* {eps:.4f}"
        )
    print(
        f"  report: optical PLOTTED vline new {np.mean(new):.4f} old {np.mean(old):.4f} "
        f"(gamma* {runner.get_oracle(0).gamma_star:.4f}, rho {runner.get_oracle(0).rho:.4f})"
    )
    print(
        "  the optical DA is NOT T-invariant (eps* >> 0), so Lem. 4's A^2 = B^2 + C^2 fails and\n"
        "  Thm. 1's premise does not hold there: BOTH lines are references, not predictions, and\n"
        "  Thm. 3.A's eps-padding is what carries validity. Do not 'fix' the formula against them."
    )
    return float(np.mean(new)), float(np.mean(old))


def a29_vline_pin(sim, optical):
    """The published annotation, pinned. a10 digests bounds and statuses, never
    vlines, so without this leg the one number this formula moves is ungated.

    A DELIBERATE geometry change moves these; update the constants in the same
    commit that moves them and say so in the message."""
    optical_config = (OPTICAL_CONFIG.ground_truth_model, OPTICAL_CONFIG.dataset_index)
    published = optical_config == VLINE_OPTICAL_CONFIG
    if not published:
        print(
            f"  report: OpticalDeviceConfig is {optical_config}, not the published "
            f"{VLINE_OPTICAL_CONFIG}; its vline pin is reported, not gated."
        )

    for name, (new, old), reference, hard in (
        ("sim", sim, VLINE_SIM, True),
        ("optical", optical, VLINE_OPTICAL, published),
    ):
        ok = abs(new - reference) <= VLINE_RTOL * reference
        tag = "PASS" if ok else ("FAIL" if hard else "WARN")
        print(f"[{tag}] A29 {name}: plotted vline == the recorded value {new:.6f} vs {reference:.6f}")
        if hard and not ok:
            FAIL.append(f"A29 {name}: plotted vline == the recorded value")
        check(f"A29 {name}: the new line is at most the old one", new <= old, f"{new:.6f} <= {old:.6f}")


if __name__ == "__main__":
    a29_closed_forms()
    sim_vline = a29_in_data()
    optical_vline = a29_optical_report()
    a29_vline_pin(sim_vline, optical_vline)
    print(f"\n{'A29 ALL PASS' if not FAIL else 'A29 FAILURES: ' + ', '.join(FAIL)}")
    sys.exit(bool(FAIL))
