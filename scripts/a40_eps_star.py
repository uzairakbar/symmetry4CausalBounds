"""A40: the robustness sweep's true eps* is a rule, the same on every dataset.

The epsilon sweep (and only it) retunes a strength-knob DA to eps* (the sweeps'
q0.95 of |W|) = `ROBUSTNESS_EPSILON_RADII` x sigma sqrt(gamma*), the confounding
radius R of that SEM, swapping in the chain of `ROBUSTNESS_AUGMENTATION` where the
configured one has no knob (optical). It replaced the per-dataset constants 3.0 /
5.0 / 0.5 (4.2, 7.9 and 1.1 R), so the legs that read those are restated on the
rule. Legs:

  (i)   plumbing: constructing the epsilon sweep runner through each orchestrator
        (`get_sweep_runner_cls("epsilon")`, the production path) carries the
        multiplier, pinned at 1.0, its per-SEM target is that multiple of the
        runner's own oracle sigma sqrt(gamma*), and the tuned DA's oracle eps*
        lands on it to 1e-6 on both: the tuner bisects the shared q0.95 reading
        of |W| (`epsilon_star_q95`), the very reading the sweep oracle takes
        (before it, optical's tuner solved on one draw and its pooled oracle read
        4% under it). Catches: a
        multiplier that no longer flows, a target off the oracle, a target the
        tuner cannot reach. Misses: a sweep that never pads with it (a38/a29).
  (ii)  mechanism: the sim target IS the post-DA ball's radius in outcome units,
        sigma sqrt(gamma*) = sqrt(bias^2) (sigma~ sqrt(gamma*/rho) under
        `recalibrate: true`), read off the runner's own oracle, at the RECORDED R.
        The old leg required a constant ABOVE the radius so DA+PI would leave it;
        at 1 R the lowest reading is read off the IV lines instead, (iii). Catches:
        a multiplier or a ruler that moved.
  (iii) the lowest reading, in data, on the lines the figure draws: the simulation block of
        recipes/robustnessFig11.yaml (its d, iv, n and methods), two experiments,
        5 steps, `runner.run` under config.yaml's toggles. The old leg read plain
        DA+PI on config.yaml's block, which the recipe does not draw and which no
        longer dips at 1 R. Now: r = 1 is the grid's midpoint (the grid is centred
        on it); every DA+ line is 1.0 at r = 1; the number of
        experiments each line is feasible on, per step, is RECORDED; and so is
        the lowest reading. Stated as it is: every DA IV row reads the swept
        eps = r eps* + EPS_TOL through gamma~_z(eps), in the q0.95 norm the INV
        budget reads, so a line is INFEASIBLE only where that budget falls under
        its floor: on the four-octave grid the (T,Z) lines at r = 2^-4 and 2^-2,
        every line feasible at r >= 1, and none dips on this fixture (the lowest
        reading is 1.0). Before the App. D radii the T family's RMS-type budget
        fell under its floor as r shrank (INFEASIBLE at 0.5, one of two
        experiments at 0.71, a 0.574 dip there). The 0.7 floor binds every line
        at every all-feasible step. Catches: a target or a radius
        that moves the feasible counts or the dip, a curve that does not recover
        at eps*. Misses: the 10-experiment band.
  (v)   isolation of the appended component: the optical epsilon runner's DA (read
        off the runner, `das[0]`) is config.yaml's chain plus the component of
        `ROBUSTNESS_AUGMENTATION`, so it carries gaussian-noise; every other
        strategy's runner (gamma, omega, n, m, recalibrate), the query runner and
        the orchestrator's own budget DA are config.yaml's chain and carry no
        gaussian-noise. Likewise the norm of W eps* is: every strategy runner's
        SEMs carry `OpticalDeviceConfig.epsilon_quantile` and budget at that eps*
        (+ EPS_TOL), the query runner's SEM `query_epsilon_quantile` (the RMS) and
        its budget that eps* (+ EPS_TOL). Catches: a component set to None (no
        knob, so no dip either), a leak into any other sweep or the panel, a
        quantile that misses a sweep or leaks into the query. Misses: a chain in
        config.yaml that already names gaussian-noise, where the sweep and the
        rest legitimately share it.
  (vi)  optical, mirroring (iii): three experiments, 5 steps, the configured
        toggles, all three pinned to device 8 (`sweep_devices`, patched here), the
        device the RECORDED numbers were read on; every DA+ line is feasible on
        the RECORDED experiments per step (all three but DA+PI+IV's one at
        r = 2^-4), 1.0 at r = 1 and above 0.7 wherever all three are feasible,
        and the lowest DA+ coverage under r = 1 is
        the RECORDED 1.0. The device caps the
        dip: no multiple of R from 1 to 8 moved it off 1.000 (the comment at
        `ROBUSTNESS_EPSILON_RADII`), and the old 5.0 was already flat on working5.
        Catches: the component gone or the chain changed (the widths move and a
        line leaves 1.0). Misses: a dip, which this device does not show.
  (vii) the optical target is the RECORDED 1 R, and eps* / std(h*) on the pool
        stays under 10: the round-5 value 8 was 11 std of h* and was rejected as
        an artefact of a destroyed image. Catches: a ruler or multiplier that
        moved, or one that pushes the target past the artefact regime (1 R is
        0.88 std of h*).
  (viii) the norm of W on the other two datasets, as (v) reads it on optical:
        every strategy runner's SEMs carry the dataset config's `epsilon_quantile`
        (q0.95) and the query runner's SEM its `query_epsilon_quantile` (the
        RMS), on the sim orchestrator and on a cigarette one (the robustness
        recipe's plasmode block, one experiment); the cigarette budgets are that
        measured eps* (+ EPS_TOL, the one tolerance), and the sim epsilon runner's
        tuned oracle eps* is the shared q0.95 reading of its W (`epsilon_star_q95`),
        not the RMS. Catches: a quantile that
        misses a dataset or a sweep, or leaks into a query panel. Misses: do-MNIST,
        whose epsilon is declared.
  The old (iv), the pin on the inert optical 2^-1, is folded into (vii).

Writes only into a fresh directory under `~/scratch/tmp/a40/`, removed when it
passes. Nothing here touches do-MNIST (its sweeps stop at `m`).

    MPLBACKEND=Agg python scripts/a40_eps_star.py [--seed 42]
"""

import argparse
import os
import shutil
import sys
import tempfile
import warnings
from dataclasses import replace

import numpy as np
import yaml
from loguru import logger

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from src.data_augmentors.optical_device import P  # noqa: E402
from src.experiments import generic_runner, optical_device  # noqa: E402
from src.experiments.cigarettes import CigaretteOrchestrator  # noqa: E402
from src.experiments.configs import (  # noqa: E402
    CIGARETTE_CONFIG,
    DATASET_DEFAULTS,
    EPS_TOL,
    OPTICAL_CONFIG,
    ROBUSTNESS_AUGMENTATION,
    ROBUSTNESS_EPSILON_RADII,
    SIMULATION_CONFIG,
)
from src.experiments.generic_runner import STRATEGIES  # noqa: E402
from src.experiments.optical_device import OpticalOrchestrator  # noqa: E402
from src.experiments.simulation import SimulationOrchestrator  # noqa: E402
from src.experiments.utils import set_seed  # noqa: E402
from src.oracle import _draw, _invariance_signal, epsilon_star_q95, preserve_rng  # noqa: E402

METHODS = ["PI", "DA+PI"]
METHODS_OPTICAL = ["PI", "DA+PI", "DA+PI+IV"]
N_STEPS = 5
N_EXPERIMENTS = 2
N_EXPERIMENTS_OPTICAL = 3
N_JOBS = 4
COVERAGE_FLOOR = 0.7
EPSILON_RADII = 1.0
# RECORDED at seed 42: the confounding radius sigma sqrt(gamma*) of the sim and
# optical robustness SEMs (the rule's targets; optical under the sweeps' chain
# `rotation > hflip > vflip > translate` at p 0.25)
SIM_RADIUS = 0.7106
OPTICAL_RADIUS = 0.6345
RADIUS_ATOL = 1e-3
# RECORDED at seed 42 on (iii)'s fixture (2 experiments, r 2^-4 .. 2^4 in 5 steps):
# the experiments each DA+ line is feasible on per step, and its one dip (coverage,
# line, r, feasible experiments behind it). Re-recorded 2026-10-02 on the
# four-octave grid: the (T,Z) lines are INFEASIBLE on both experiments at r = 2^-4
# and 2^-2, where the joint row's gamma~_z(r eps*) falls under its floor, and every
# line is feasible at r >= 1; none dips. On the one-octave grid (r 0.5 .. 2) every
# line was feasible at every step (2026-09-29, under the sweeps' App. D T radius)
SIM_FEASIBLE = {
    "DA+PI+IV(Z)": [2, 2, 2, 2, 2],
    "DA+PI+IV(T,Z)": [0, 0, 2, 2, 2],
    "PI&DA+PI+IV(Z)": [2, 2, 2, 2, 2],
    "PI&DA+PI+IV(T,Z)": [0, 0, 2, 2, 2],
}
# RECORDED at seed 42 on (vi)'s fixture (3 experiments on device 8, r 2^-4 .. 2^4 in
# 5 steps), 2026-10-02 on the four-octave grid: DA+PI+IV is INFEASIBLE on one of the
# three at r = 2^-4; every line is feasible on all three elsewhere, as on the
# one-octave grid at every step
OPTICAL_FEASIBLE = {"DA+PI": [3, 3, 3, 3, 3], "DA+PI+IV": [2, 3, 3, 3, 3]}
SIM_DIP = (1.0, "", 0, 0)  # at coverage 1.0 the (line, r, count) are not compared
RECIPE = os.path.join(REPO, "recipes", "robustnessFig11.yaml")
STD_RATIO_BOUND = 10.0
SHIPPED_CHAIN = "rotation > hflip > vflip > random-permutation"
TMPROOT = os.path.expanduser("~/scratch/tmp/a40")
FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAIL.append(name)


def configured():
    """The toggles, the sim draw (`n_samples`, `treatment_dim`) and the optical
    chain of config.yaml, falling back to DatasetDefaults / the shipped chain as
    main.py does."""
    with open(os.path.join(REPO, "config.yaml")) as handle:
        config = yaml.safe_load(handle) or {}
    defaults = config.get("defaults") or {}
    sim = config.get("simulation") or {}
    opt = config.get("optical_device") or {}
    fallback = DATASET_DEFAULTS["simulation"]
    toggles = dict(
        recalibrate=bool(defaults.get("recalibrate", True)),
        pad=bool(defaults.get("pad", False)),
        clipy=bool(defaults.get("clipy", True)),
        mean_match=bool(defaults.get("mean_match", True)),
    )
    draw = dict(
        n_samples=int(sim.get("n_samples", fallback.n_samples)),
        treatment_dim=int(sim.get("treatment_dim", fallback.treatment_dim)),
    )
    return toggles, draw, str(opt.get("augmentation", SHIPPED_CHAIN))


def shipped_p():
    """config.yaml's optical rotation / flip probability, the DA's `P` when it names none."""
    with open(os.path.join(REPO, "config.yaml")) as handle:
        return float(((yaml.safe_load(handle) or {}).get("optical_device") or {}).get("augmentation_p", P))


def build(experiment, draw, chain, seed, **toggles):
    set_seed(seed)
    common = dict(seed=seed, sweep_samples=N_STEPS, hyperparameters={}, n_jobs=N_JOBS, **toggles)
    if experiment == "simulation":
        return SimulationOrchestrator(kernel_dim=0, n_experiments=N_EXPERIMENTS, methods=METHODS, **draw, **common)
    return OpticalOrchestrator(
        n_samples=1000,
        augmentation=chain,
        augmentation_p=shipped_p(),
        n_experiments=N_EXPERIMENTS_OPTICAL,
        methods=METHODS_OPTICAL,
        **common,
    )


def epsilon_runner(orch):
    """The production epsilon runner, with the `epsilon_true` it was handed captured
    at the base class boundary (the strategy sets it before calling up)."""
    handed = {}
    original = generic_runner.GenericParamSweep.__init__

    def spy(self, **kwargs):
        handed["epsilon_true"] = kwargs.get("epsilon_true")
        original(self, **kwargs)

    generic_runner.GenericParamSweep.__init__ = spy
    try:
        runner = orch.get_sweep_runner_cls("epsilon")(
            methods=orch.methods, method_factory=orch.build_methods, **orch._get_clean_kwargs()
        )
    finally:
        generic_runner.GenericParamSweep.__init__ = original
    return runner, handed.get("epsilon_true")


def sweep_runner(orch, param):
    return orch.get_sweep_runner_cls(param)(
        methods=orch.methods, method_factory=orch.build_methods, **orch._get_clean_kwargs()
    )


def query_runner(orch):
    return orch.get_query_runner_cls()(methods=orch.methods, **{**orch._get_clean_kwargs(), "n_experiments": 1})


def chain_of(da) -> list[str]:
    return [component.augmentation for component in da._augmentations]


def parse_chain(chain: str) -> list[str]:
    return chain.replace(" ", "").split(">")


def leg_i(runners):
    print("(i) the rule's multiplier reaches each dataset's runner and its tuned DA lands on the target")
    check(
        "(i) ROBUSTNESS_EPSILON_RADII is 1.0", ROBUSTNESS_EPSILON_RADII == EPSILON_RADII, f"{ROBUSTNESS_EPSILON_RADII}"
    )
    for name, (runner, handed) in runners.items():
        check(f"(i) {name}: no constant handed down as epsilon_true", handed is None, f"{handed}")
        check(f"(i) {name}: runner reports experiment_name", runner.experiment_name == name)
        check(f"(i) {name}: runner carries the multiplier", runner.epsilon_radii == ROBUSTNESS_EPSILON_RADII)
        oracle, sem = runner.get_oracle(0), runner.sems[0]
        want = ROBUSTNESS_EPSILON_RADII * float(np.sqrt(oracle.sigma_sq * oracle.gamma_star))
        target = runner.robustness_target(sem)
        check(
            f"(i) {name}: target == multiplier x the oracle's sigma sqrt(gamma*)",
            np.isclose(target, want),
            f"{target:.6g} vs {want:.6g}",
        )
        achieved = oracle.epsilon_star
        strength = runner.das[0].strength
        # the tuner and the sweep oracle read the same shared q0.95 of |W|
        check(
            f"(i) {name}: tuned DA reaches the target",
            strength is not None and strength > 0 and np.isclose(achieved, target, rtol=1e-6),
            f"eps* {achieved:.6g} vs target {target:.6g} at strength {strength!r:.6}",
        )


def leg_ii(sim):
    print("(ii) the sim target is the post-DA ball's radius")
    oracle = sim.get_oracle(0)
    radius_outcome = float(np.sqrt(oracle.sigma_sq * oracle.gamma_star))
    target = sim.robustness_target(sim.sems[0])
    print(
        f"      gamma* {oracle.gamma_star:.4g}, bias^2 {oracle.bias_sq:.4g}, sigma^2 {oracle.sigma_sq:.4g}; "
        f"sigma sqrt(gamma*) {radius_outcome:.4g}; target {target:.4g}"
    )
    check("(ii) sigma sqrt(gamma*) == sqrt(bias^2)", np.isclose(radius_outcome, np.sqrt(oracle.bias_sq)))
    check("(ii) target == 1 x sigma sqrt(gamma*)", np.isclose(target, radius_outcome), f"{target:.6g}")
    check("(ii) at the RECORDED R", np.isclose(radius_outcome, SIM_RADIUS, atol=RADIUS_ATOL), f"{radius_outcome:.4g}")


def dip(runner, tag, feasible_recorded, lowest_recorded):
    """`runner.run`; every DA+ line (standalone or intersected) recovers to 1 at
    r = 1; its per-step count of feasible experiments equals `feasible_recorded`;
    it stays above the floor on every step where all experiments are feasible;
    and the lowest DA+ mean coverage under r = 1 (over the experiments a line is
    feasible on) is `lowest_recorded` = (coverage, line, r, feasible count)."""
    x, results, _ = runner.run(f"a40 {tag} epsilon sweep")
    x = np.asarray(x, dtype=float)
    lines = [m for m in results if "DA+" in m]
    base = "PI" if "PI" in results else "PI+IV"
    raw = {m: np.asarray(results[m]["coverage"], dtype=float) for m in lines}
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        cov = {m: np.nanmean(raw[m], axis=1) for m in lines}
        w_da = {m: np.nanmean(np.asarray(results[m]["interval_width"], dtype=float), axis=1) for m in lines}
    w_base = np.asarray(results[base]["interval_width"], dtype=float).mean(axis=1)
    print("      r:        " + " ".join(f"{v:.4g}" for v in x))
    for m in lines:
        print(
            f"      {m:18s} "
            + " ".join(f"{v:.3f}" for v in cov[m])
            + f" | width/{base} at r = 1 {w_da[m][len(x) // 2] / w_base[len(x) // 2]:.3f}"
        )
    mid = len(x) // 2
    check(f"{tag} r = 1 is the grid's midpoint", len(x) % 2 == 1 and np.isclose(x[mid], 1.0), f"{x}")
    counts = {m: np.isfinite(raw[m]).sum(axis=1).astype(int).tolist() for m in lines}
    check(f"{tag} feasible experiments per step == RECORDED", counts == feasible_recorded, f"{counts}")
    under = [(cov[m][i], m, i) for m in lines for i in range(mid) if np.isfinite(cov[m][i])]
    value, line, step = min(under)
    want, want_line, want_r, want_count = lowest_recorded
    check(
        f"{tag} lowest DA+ coverage under r = 1 == RECORDED {want:g} ({want_line}, r {want_r:g}, on {want_count})",
        np.isclose(value, want, atol=1e-3)
        and (want == 1.0 or (line.endswith(want_line) and np.isclose(x[step], want_r, atol=1e-3)))
        and (want == 1.0 or counts[line][step] == want_count),
        f"{value:.4f} ({line}, r {x[step]:.4g}, on {counts[line][step]})",
    )
    n_exp = raw[lines[0]].shape[1]
    for m in lines:
        check(f"{tag} {m} coverage == 1 at r = 1", np.isclose(cov[m][mid], 1.0), f"{cov[m][mid]:.4f}")
        full = np.asarray(counts[m]) == n_exp
        check(
            f"{tag} {m} above the floor on every all-feasible step",
            bool(full.any() and np.min(cov[m][full]) > COVERAGE_FLOOR),
            f"at r {np.round(x[full], 3).tolist()}: {np.round(cov[m][full], 4).tolist()}",
        )
    ratio = w_da[lines[0]] / w_base
    return x, cov[lines[0]], ratio


def recipe_sim(seed, **toggles):
    """The simulation block of the robustness recipe: its draw and its methods."""
    with open(RECIPE) as handle:
        block = yaml.safe_load(handle)["simulation"]
    set_seed(seed)
    return SimulationOrchestrator(
        kernel_dim=int(block.get("kernel_dim", 0)),
        treatment_dim=int(block["treatment_dim"]),
        iv=int(block.get("iv", 0)),
        n_samples=int(block["n_samples"]),
        n_experiments=N_EXPERIMENTS,
        sweep_samples=N_STEPS,
        methods=list(block["methods"]),
        seed=seed,
        hyperparameters={},
        n_jobs=N_JOBS,
        **toggles,
    )


def leg_iii(seed, **toggles):
    orch = recipe_sim(seed, **toggles)
    print(
        f"(iii) the lowest reading on {N_EXPERIMENTS} sim experiments of the robustness recipe,"
        f" methods {orch.kwargs['methods']}"
    )
    runner, _ = epsilon_runner(orch)
    return dip(runner, "(iii)", SIM_FEASIBLE, SIM_DIP)


def leg_v(orch, opt_eps, chain):
    print("(v) the appended component reaches the optical epsilon runner and nothing else")
    component = ROBUSTNESS_AUGMENTATION["optical_device"]
    check(
        "(v) ROBUSTNESS_AUGMENTATION names gaussian-noise for optical", component == "gaussian-noise", repr(component)
    )
    configured_chain = parse_chain(chain)
    want = configured_chain + ["gaussian-noise"] if "gaussian-noise" not in configured_chain else configured_chain
    eps_chain = chain_of(opt_eps.das[0])
    check("(v) epsilon runner DA == configured chain + gaussian-noise", eps_chain == want, f"{eps_chain}")
    check("(v) epsilon runner DA carries gaussian-noise", "gaussian-noise" in eps_chain)
    built = {p: sweep_runner(orch, p) for p in STRATEGIES if p != "epsilon"}
    query = query_runner(orch)
    others = {f"{p} runner": chain_of(runner.das[0]) for p, runner in built.items()}
    others["query runner"] = chain_of(query.da)
    others["orchestrator budget"] = chain_of(orch._oracle_pieces()[1])
    for name, got in others.items():
        check(f"(v) {name} DA == configured chain", got == configured_chain, f"{got}")
        check(f"(v) {name} DA has no gaussian-noise", "gaussian-noise" not in got)

    # the norm of W eps* is: the sweeps' quantile on every strategy, the query's own
    sweep_q, query_q = OPTICAL_CONFIG.epsilon_quantile, OPTICAL_CONFIG.query_epsilon_quantile
    sweep_budget = orch.measured_epsilon_star(sweep_q) + EPS_TOL
    for name, runner in {**{f"{p} runner": r for p, r in built.items()}, "epsilon runner": opt_eps}.items():
        got = {getattr(sem, "epsilon_quantile", "unset") for sem in runner.sems}
        check(f"(v) {name} SEMs carry epsilon_quantile {sweep_q}", got == {sweep_q}, f"{got}")
        check(
            f"(v) {name} budgets at that eps*",
            runner.default_epsilon == sweep_budget,
            f"{runner.default_epsilon:.6f} vs {sweep_budget:.6f}",
        )
    got = getattr(query.sem, "epsilon_quantile", "unset")
    check(f"(v) query runner SEM carries query_epsilon_quantile {query_q}", got == query_q, f"{got}")
    query_budget = orch.measured_epsilon_star(query_q) + EPS_TOL
    check(
        "(v) query runner budgets at that eps*",
        query.default_epsilon == query_budget,
        f"{query.default_epsilon:.6f} vs {query_budget:.6f}",
    )


def leg_vi(opt):
    print(f"(vi) the optical curve on {N_EXPERIMENTS_OPTICAL} experiments")
    return dip(opt, "(vi)", OPTICAL_FEASIBLE, (1.0, "", 0, 0))


def leg_vii(opt):
    print("(vii) the optical target and its size against h*")
    sem = opt.sems[0]
    target = opt.robustness_target(sem)
    check(
        "(vii) optical target at the RECORDED 1 R",
        np.isclose(target, OPTICAL_RADIUS, atol=RADIUS_ATOL),
        f"{target:.4g}",
    )
    h = np.asarray(sem.f(opt._features(sem.X)), dtype=float).ravel()
    ratio = target / float(np.std(h))
    check(
        f"(vii) eps* / std(h*) < {STD_RATIO_BOUND:g}",
        ratio < STD_RATIO_BOUND,
        f"std(h*) {np.std(h):.4f}, ratio {ratio:.3f}",
    )


def leg_viii(sim_orch, sim_eps, seed, **toggles):
    print("(viii) the sweeps' q0.95 and the query's RMS on simulation and cigarettes")
    set_seed(seed)
    with open(RECIPE) as handle:
        block = yaml.safe_load(handle)["cigarettes"]
    cig_orch = CigaretteOrchestrator(
        target=block["target"],
        spec=block["spec"],
        anchor=block["anchor"],
        da_amplitude=float(block["da_amplitude"]),
        iv=list(block["iv"]),
        gamma_z=float(block["gamma_z"]),
        seed=seed,
        n_samples=int(block["n_samples"]),
        n_experiments=1,
        sweep_samples=N_STEPS,
        hyperparameters={},
        n_jobs=N_JOBS,
        methods=METHODS,
        **toggles,
    )
    for name, orch, config in (
        ("simulation", sim_orch, SIMULATION_CONFIG),
        ("cigarettes", cig_orch, CIGARETTE_CONFIG),
    ):
        sweep_q, query_q = config.epsilon_quantile, config.query_epsilon_quantile
        check(f"(viii) {name}: the sweeps take q0.95, the query the RMS", (sweep_q, query_q) == (0.95, None))
        for param in STRATEGIES:
            runner = sweep_runner(orch, param)
            got = {getattr(sem, "epsilon_quantile", "unset") for sem in runner.sems}
            check(f"(viii) {name}: {param} runner SEMs carry {sweep_q}", got == {sweep_q}, f"{got}")
        query = query_runner(orch)
        got = getattr(query.sem, "epsilon_quantile", "unset")
        check(f"(viii) {name}: query runner SEM carries {query_q}", got == query_q, f"{got}")
    budgets = (
        (
            "gamma runner",
            sweep_runner(cig_orch, "gamma").default_epsilon,
            cig_orch.measured_epsilon_star(0.95) + EPS_TOL,
        ),
        (
            "query runner",
            query_runner(cig_orch).default_epsilon,
            cig_orch.measured_epsilon_star(None) + EPS_TOL,
        ),
    )
    for name, got, want in budgets:
        check(f"(viii) cigarettes: {name} budgets at its eps*", got == want, f"{got:.6g} vs {want:.6g}")
    sem, da = sim_eps.sems[0], sim_eps.das[0]
    with preserve_rng():
        X, _, _ = _draw(sem, 2048)
        w = _invariance_signal(sem, da, X)[0]
    rms = float(np.sqrt(np.mean(w**2)))
    reading = epsilon_star_q95(sem, da)
    achieved = sim_eps.get_oracle(0).epsilon_star
    check(
        "(viii) simulation: the tuned eps* is the shared q0.95 reading of W, not its RMS",
        achieved == reading and not np.isclose(achieved, rms, rtol=0.2),
        f"oracle {achieved:.4f}, the q0.95 reading {reading:.4f}, RMS {rms:.4f}",
    )


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    os.chdir(REPO)
    os.makedirs(TMPROOT, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="a40_", dir=TMPROOT)
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    seed = parser.parse_args().seed
    toggles, draw, chain = configured()
    # the optical legs read device 8's numbers; the sweeps would otherwise span devices
    optical_device.OPTICAL_CONFIG = replace(
        optical_device.OPTICAL_CONFIG,
        sweep_devices=(optical_device.OPTICAL_CONFIG.dataset_index,) * N_EXPERIMENTS_OPTICAL,
    )
    orchs = {name: build(name, draw, chain, seed, **toggles) for name in ("simulation", "optical_device")}
    runners = {name: epsilon_runner(orch) for name, orch in orchs.items()}
    leg_i(runners)
    sim, _ = runners["simulation"]
    opt, _ = runners["optical_device"]
    leg_ii(sim)
    x, cov, ratio = leg_iii(seed, **toggles)
    leg_v(orchs["optical_device"], opt, chain)
    x_opt, cov_opt, ratio_opt = leg_vi(opt)
    leg_vii(opt)
    leg_viii(orchs["simulation"], sim, seed, **toggles)
    with open(os.path.join(tmp, "dip.txt"), "w") as handle:
        for name, xs, cs, ws in (("simulation", x, cov, ratio), ("optical_device", x_opt, cov_opt, ratio_opt)):
            for r, c, w in zip(xs, cs, ws, strict=True):
                handle.write(f"{name} {r:.6g} {c:.4f} {w:.4f}\n")
    if not FAIL:
        shutil.rmtree(tmp, ignore_errors=True)
        print("A40 PASS")
    else:
        print(f"A40 FAIL: {FAIL} (log in {tmp})")
        sys.exit(1)
