"""A40: the robustness sweep's true eps* is per dataset, and the sim value dips.

`ROBUSTNESS_EPSILON_TRUE` is a dict keyed by EXPERIMENT_NAME; the epsilon sweep
(and only it) retunes a strength-knob DA to it. Four legs:

  (i)   plumbing: constructing the epsilon sweep runner through each orchestrator
        (`get_sweep_runner_cls("epsilon")`, the production path) hands the dataset's
        own constant down as `epsilon_true`, for both datasets, read through the
        runner's kwargs rather than a fixture; on sim the tuned DA's oracle eps*
        equals the constant (the knob reached it), on optical the shipped chain has
        no knob so the runner drops the target and keeps the chain's own eps*.
        Catches: a lookup pinned to one key, a constant that no longer flows, a
        target the tuner cannot reach. Misses: a sweep that reads the constant but
        never pads with it (the pad is a38/a29 territory).
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
  (iv)  optical stays at 2^-1. Catches: an edit that moves it. Misses: nothing
        else; the optical curve is flat by mechanism, not by this number.

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
from src.experiments.configs import DATASET_DEFAULTS, ROBUSTNESS_EPSILON_TRUE  # noqa: E402
from src.experiments.optical_device import OpticalOrchestrator  # noqa: E402
from src.experiments.simulation import SimulationOrchestrator  # noqa: E402
from src.experiments.utils import set_seed  # noqa: E402

METHODS = ["PI", "DA+PI"]
N_STEPS = 5
N_EXPERIMENTS = 2
N_JOBS = 4
COVERAGE_FLOOR = 0.7
TMPROOT = os.path.expanduser("~/scratch/tmp/a40")
FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAIL.append(name)


def configured():
    """The toggles and the sim draw (`n_samples`, `treatment_dim`) of config.yaml,
    the latter falling back to DatasetDefaults as main.py does."""
    with open(os.path.join(REPO, "config.yaml")) as handle:
        config = yaml.safe_load(handle) or {}
    defaults = config.get("defaults") or {}
    sim = config.get("simulation") or {}
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
    return toggles, draw


def build(experiment, draw, **toggles):
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
    return OpticalOrchestrator(n_samples=1000, augmentation="rotation > hflip > vflip > random-permutation", **common)


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


def leg_i(runners):
    print("(i) the configured constant reaches each dataset's runner")
    for name, (runner, handed) in runners.items():
        want = ROBUSTNESS_EPSILON_TRUE[name]
        check(f"(i) {name}: epsilon_true handed == constant", handed == want, f"{handed} vs {want}")
        check(f"(i) {name}: runner reports experiment_name", runner.experiment_name == name)
    sim, _ = runners["simulation"]
    achieved = sim.get_oracle(0).epsilon_star
    strength = sim.das[0].strength
    check(
        "(i) simulation: tuned DA reaches the constant",
        np.isclose(achieved, ROBUSTNESS_EPSILON_TRUE["simulation"], rtol=1e-6),
        f"eps* {achieved:.6g} at strength {strength:.4g}",
    )
    opt, _ = runners["optical_device"]
    check(
        "(i) optical: shipped chain has no knob, target dropped",
        opt.das[0].strength is None and opt.epsilon_true is None,
    )
    print(f"      optical eps* stays at the chain's own {opt.get_oracle(0).epsilon_star:.4g}")


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


def leg_iii(sim, draw):
    print(f"(iii) the dip on {N_EXPERIMENTS} sim experiments, n {draw['n_samples']}, d {draw['treatment_dim']}")
    x, results, _ = sim.run("a40 epsilon sweep")
    x = np.asarray(x, dtype=float)
    per_exp = np.asarray(results["DA+PI"]["coverage"], dtype=float)
    cov = per_exp.mean(axis=1)
    w_da = np.asarray(results["DA+PI"]["interval_width"], dtype=float).mean(axis=1)
    w_pi = np.asarray(results["PI"]["interval_width"], dtype=float).mean(axis=1)
    print("      r:        " + " ".join(f"{v:.4g}" for v in x))
    print("      DA+PI cov " + " ".join(f"{v:.3f}" for v in cov))
    print("      per exp at the smallest r: " + " ".join(f"{v:.3f}" for v in per_exp[0]))
    print("      DA+PI/PI  " + " ".join(f"{v:.3f}" for v in w_da / w_pi))
    check("(iii) r = 1 is the last grid point", np.isclose(x[-1], 1.0), f"{x[-1]:.6g}")
    check("(iii) DA+PI coverage < 1 at the smallest r", cov[0] < 1.0, f"{cov[0]:.4f} at r {x[0]:.4g}")
    check("(iii) DA+PI coverage == 1 at r = 1", np.isclose(cov[-1], 1.0), f"{cov[-1]:.4f}")
    check("(iii) DA+PI min coverage above the floor", np.nanmin(cov) > COVERAGE_FLOOR, f"{np.nanmin(cov):.4f}")
    return x, cov, w_da / w_pi


def leg_iv():
    print("(iv) optical constant unchanged")
    check("(iv) optical == 2**-1", ROBUSTNESS_EPSILON_TRUE["optical_device"] == 2**-1)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    os.chdir(REPO)
    os.makedirs(TMPROOT, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="a40_", dir=TMPROOT)
    toggles, draw = configured()
    runners = {name: epsilon_runner(build(name, draw, **toggles)) for name in ("simulation", "optical_device")}
    leg_i(runners)
    sim, _ = runners["simulation"]
    leg_ii(sim)
    x, cov, ratio = leg_iii(sim, draw)
    leg_iv()
    with open(os.path.join(tmp, "dip.txt"), "w") as handle:
        for r, c, w in zip(x, cov, ratio, strict=True):
            handle.write(f"{r:.6g} {c:.4f} {w:.4f}\n")
    if not FAIL:
        shutil.rmtree(tmp, ignore_errors=True)
        print("A40 PASS")
    else:
        print(f"A40 FAIL: {FAIL} (log in {tmp})")
        sys.exit(1)
