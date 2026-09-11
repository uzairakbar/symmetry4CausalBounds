"""A40: the robustness sweep's true eps* is per dataset, and both datasets dip.

`ROBUSTNESS_EPSILON_TRUE` is a dict keyed by EXPERIMENT_NAME; the epsilon sweep
(and only it) retunes a strength-knob DA to it, swapping in the chain of
`ROBUSTNESS_AUGMENTATION` where the configured one has no knob (optical). Legs:

  (i)   plumbing: constructing the epsilon sweep runner through each orchestrator
        (`get_sweep_runner_cls("epsilon")`, the production path) hands the dataset's
        own constant down as `epsilon_true`, for both datasets, read through the
        runner's kwargs rather than a fixture, and the tuned DA's oracle eps* equals
        the constant (the knob reached it) on both; on optical within 5%, since the
        tuner solves on one draw and the pooled oracle averages eight. Catches: a
        lookup pinned to one key, a constant that no longer flows, a target the
        tuner cannot reach.
        Misses: a sweep that reads the constant but never pads with it (a38/a29).
  (ii)  mechanism: the sim constant exceeds the post-DA ball's radius, both as
        sqrt(gamma*) and in outcome units sigma sqrt(gamma*) = sqrt(bias^2), read
        off the runner's own oracle. This is the reason a dip exists at all.
        Catches: a constant put back under the radius (the old 2^-1). Misses:
        whether the excess is enough to move the centre out; that is (iii).
  (iii) the dip, in data: two sim experiments, 5 steps, `runner.run` under the
        toggles, `n_samples` and `treatment_dim` of config.yaml (DatasetDefaults
        when omitted, so a moved dimension is tested at the figure's own); the
        mean DA+PI coverage at the smallest r is < 1, at r = 1 it is 1.0, and its
        minimum over the grid stays above 0.7. Catches: a flat line (constant too
        small), a cliff (too large), a curve that does not recover at eps*.
        Misses: the 8-experiment band and the other DA+ lines, which the probe
        log holds (see the comment at the constant).
  (v)   isolation of the chain swap: the optical epsilon runner's DA (read off the
        runner, `das[0]`) is exactly the chain of `ROBUSTNESS_AUGMENTATION` and
        carries gaussian-noise; the optical gamma runner's DA, the query runner's
        DA and the orchestrator's own budget DA are the configured chain of
        config.yaml and carry no gaussian-noise. Catches: an override set to None
        (no knob, so no dip either), an override that leaks into another sweep or
        the panel. Misses: a leak into trS/n/m, which build their DA through the
        same factory as gamma.
  (vi)  the optical dip, mirroring (iii): two optical experiments, 5 steps, the
        configured toggles; mean DA+PI coverage < 1 at the smallest r, 1.0 at
        r = 1, minimum above 0.7. Catches: the constant back at 2^-1 (flat), the
        override gone (flat). Misses: how faint the dip is; the constant's comment
        says the device caps it near 0.96.
  (vii) the optical constant is the chosen 5.0, and eps* / std(h*) on the pool
        stays under 10: the round-5 value 8 was 11 std of h* and was rejected as
        an artefact of a destroyed image, and the comment at the constant shows 6
        already sits on the device's coverage floor. Catches: an edit that moves
        the constant, or one that pushes it past the artefact regime. Misses:
        that 5 is itself 7 std of h*; the number is measured, not principled.
  The old (iv), the pin on the inert optical 2^-1, is folded into (vii).

Writes only into a fresh directory under `~/scratch/tmp/a40/`, removed when it
passes. Nothing here touches do-MNIST (its sweeps stop at `m`).

    MPLBACKEND=Agg python scripts/a40_eps_star.py
"""

import os
import shutil
import sys
import tempfile

import numpy as np
import yaml
from loguru import logger

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from src.experiments import generic_runner  # noqa: E402
from src.experiments.configs import (  # noqa: E402
    DATASET_DEFAULTS,
    ROBUSTNESS_AUGMENTATION,
    ROBUSTNESS_EPSILON_TRUE,
)
from src.experiments.optical_device import OpticalOrchestrator  # noqa: E402
from src.experiments.simulation import SimulationOrchestrator  # noqa: E402
from src.experiments.utils import set_seed  # noqa: E402

METHODS = ["PI", "DA+PI"]
N_STEPS = 5
N_EXPERIMENTS = 2
N_JOBS = 4
COVERAGE_FLOOR = 0.7
OPTICAL_EPS_STAR = 5.0
STD_RATIO_BOUND = 10.0
POOLED_ORACLE_RTOL = 0.05
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


def build(experiment, draw, chain, **toggles):
    set_seed(42)
    common = dict(
        seed=42,
        n_experiments=N_EXPERIMENTS,
        sweep_samples=N_STEPS,
        methods=METHODS,
        hyperparameters={},
        n_jobs=N_JOBS,
        **toggles,
    )
    if experiment == "simulation":
        return SimulationOrchestrator(kernel_dim=0, **draw, **common)
    return OpticalOrchestrator(n_samples=1000, augmentation=chain, **common)


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
    print("(i) the configured constant reaches each dataset's runner and its tuned DA lands on it")
    for name, (runner, handed) in runners.items():
        want = ROBUSTNESS_EPSILON_TRUE[name]
        check(f"(i) {name}: epsilon_true handed == constant", handed == want, f"{handed} vs {want}")
        check(f"(i) {name}: runner reports experiment_name", runner.experiment_name == name)
        achieved = runner.get_oracle(0).epsilon_star
        strength = runner.das[0].strength
        # the tuner solves on one frozen draw; a SEM with a fixed pool (optical)
        # then reports the oracle pooled over ORACLE_POOL_DRAWS draws, and the
        # RMS of a noise DA moves a few percent between draws (measured 2.4%)
        tolerance = POOLED_ORACLE_RTOL if getattr(runner.sems[0], "pool", None) is not None else 1e-6
        check(
            f"(i) {name}: tuned DA reaches the constant",
            strength is not None and np.isclose(achieved, want, rtol=tolerance),
            f"eps* {achieved:.6g} at strength {strength!r:.6}, rtol {tolerance:g}",
        )


def leg_ii(sim):
    print("(ii) the sim constant clears the post-DA ball radius")
    oracle = sim.get_oracle(0)
    radius_paper = float(np.sqrt(oracle.gamma_star))
    radius_outcome = float(np.sqrt(oracle.sigma_sq * oracle.gamma_star))
    eps = ROBUSTNESS_EPSILON_TRUE["simulation"]
    print(
        f"      gamma* {oracle.gamma_star:.4g}, bias^2 {oracle.bias_sq:.4g}, sigma^2 {oracle.sigma_sq:.4g}; "
        f"sqrt(gamma*) {radius_paper:.4g}, sigma sqrt(gamma*) {radius_outcome:.4g}; eps* {eps:.4g}"
    )
    check("(ii) eps* > sqrt(gamma*)", eps > radius_paper, f"{eps:.4g} vs {radius_paper:.4g}")
    check("(ii) eps* > sigma sqrt(gamma*)", eps > radius_outcome, f"{eps:.4g} vs {radius_outcome:.4g}")


def dip(runner, tag):
    """`runner.run`; the DA+PI coverage curve must dip at the smallest r, recover at
    r = 1 and never go under the floor."""
    x, results, _ = runner.run(f"a40 {tag} epsilon sweep")
    x = np.asarray(x, dtype=float)
    per_exp = np.asarray(results["DA+PI"]["coverage"], dtype=float)
    cov = per_exp.mean(axis=1)
    w_da = np.asarray(results["DA+PI"]["interval_width"], dtype=float).mean(axis=1)
    w_pi = np.asarray(results["PI"]["interval_width"], dtype=float).mean(axis=1)
    print("      r:        " + " ".join(f"{v:.4g}" for v in x))
    print("      DA+PI cov " + " ".join(f"{v:.3f}" for v in cov))
    print("      per exp at the smallest r: " + " ".join(f"{v:.3f}" for v in per_exp[0]))
    print("      DA+PI/PI  " + " ".join(f"{v:.3f}" for v in w_da / w_pi))
    check(f"{tag} r = 1 is the last grid point", np.isclose(x[-1], 1.0), f"{x[-1]:.6g}")
    check(f"{tag} DA+PI coverage < 1 at the smallest r", cov[0] < 1.0, f"{cov[0]:.4f} at r {x[0]:.4g}")
    check(f"{tag} DA+PI coverage == 1 at r = 1", np.isclose(cov[-1], 1.0), f"{cov[-1]:.4f}")
    check(f"{tag} DA+PI min coverage above the floor", np.nanmin(cov) > COVERAGE_FLOOR, f"{np.nanmin(cov):.4f}")
    return x, cov, w_da / w_pi


def leg_iii(sim, draw):
    print(f"(iii) the dip on {N_EXPERIMENTS} sim experiments, n {draw['n_samples']}, d {draw['treatment_dim']}")
    return dip(sim, "(iii)")


def leg_v(orch, opt_eps, chain):
    print("(v) the chain swap reaches the optical epsilon runner and nothing else")
    override = ROBUSTNESS_AUGMENTATION["optical_device"]
    check("(v) ROBUSTNESS_AUGMENTATION is set for optical", override is not None, f"{override!r}")
    want = parse_chain(override) if override else []
    configured_chain = parse_chain(chain)
    eps_chain = chain_of(opt_eps.das[0])
    check("(v) epsilon runner DA == ROBUSTNESS_AUGMENTATION", eps_chain == want, f"{eps_chain}")
    check("(v) epsilon runner DA carries gaussian-noise", "gaussian-noise" in eps_chain)
    prefix = eps_chain[: len(configured_chain)] == configured_chain
    check("(v) epsilon runner DA keeps the configured chain as prefix", prefix)
    others = {
        "gamma runner": chain_of(sweep_runner(orch, "gamma").das[0]),
        "query runner": chain_of(query_runner(orch).da),
        "orchestrator budget": chain_of(orch._oracle_pieces()[1]),
    }
    for name, got in others.items():
        check(f"(v) {name} DA == configured chain", got == configured_chain, f"{got}")
        check(f"(v) {name} DA has no gaussian-noise", "gaussian-noise" not in got)


def leg_vi(opt):
    print(f"(vi) the dip on {N_EXPERIMENTS} optical experiments")
    return dip(opt, "(vi)")


def leg_vii(opt):
    print("(vii) the optical constant and its size against h*")
    eps = ROBUSTNESS_EPSILON_TRUE["optical_device"]
    check("(vii) optical == 5.0", eps == OPTICAL_EPS_STAR, f"{eps}")
    sem = opt.sems[0]
    h = np.asarray(sem.f(opt._features(sem.X)), dtype=float).ravel()
    ratio = eps / float(np.std(h))
    check(
        f"(vii) eps* / std(h*) < {STD_RATIO_BOUND:g}",
        ratio < STD_RATIO_BOUND,
        f"std(h*) {np.std(h):.4f}, ratio {ratio:.3f}",
    )


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    os.chdir(REPO)
    os.makedirs(TMPROOT, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="a40_", dir=TMPROOT)
    toggles, draw, chain = configured()
    orchs = {name: build(name, draw, chain, **toggles) for name in ("simulation", "optical_device")}
    runners = {name: epsilon_runner(orch) for name, orch in orchs.items()}
    leg_i(runners)
    sim, _ = runners["simulation"]
    opt, _ = runners["optical_device"]
    leg_ii(sim)
    x, cov, ratio = leg_iii(sim, draw)
    leg_v(orchs["optical_device"], opt, chain)
    x_opt, cov_opt, ratio_opt = leg_vi(opt)
    leg_vii(opt)
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
