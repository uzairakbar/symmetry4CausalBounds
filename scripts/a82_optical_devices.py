"""A82: one optical device per sweep experiment (setup only but for (iii); a few
minutes on 16 cores, most of it the ten DA tunings and oracles of (v)).

The optical sweeps run experiment j on `OpticalOrchestrator.sweep_devices()[j]`,
each with the polynomial feature map of its own selected degree; the two odd
recordings (`OpticalDeviceConfig.excluded_devices`) are never swept, and every
one-experiment runner stays on `dataset_index`. Legs:

  (i)   the default order is `dataset_index` (8) first, then every other device
        not excluded, ascending: (8, 0, 1, 2, 3, 4, 5, 7, 9, 10); the indices are
        pinned to their files (6 the brfactor-0 recording, 11 the pure-confounding
        one, 8 exp_no_78), so a reordered data directory fails here. A gamma sweep
        at ten experiments builds ten SEMs on those devices once each (read off a
        tagging SEM and the SEMs' own pools), `polys[j]` has the degree of
        `sems[j]` (both 1 and 2 present), and the features of experiment j fit
        `sems[j].f`; an eleventh experiment raises.
        Catches: a hook that falls back to `sem_factory`, one shared feature map
        (a degree-1 device would crash `sem.f`), a device reused, an odd device
        swept.
  (ii)  one-experiment runners stay on device 8: the perf-shaped epsilon runner
        (`n_experiments: 1`, as `_run_perf` builds it) and the query runner are
        built (setup only; perf is never run) and their SEM is device 8.
        Catches: a default order that does not start at `dataset_index`, which
        would move Fig. 6 and the perf record silently.
  (iii) `sweep_devices = (8, 8)` reproduces the pre-change sweep bit for bit: a
        two-experiment, two-step gamma sweep against the same runner with
        `make_sem` / `poly_for_sem` reset to the base class (one factory SEM per
        experiment, the shared transform), every metric array but the wall clock
        equal.
        Catches: a per-j accessor that changes the draw, the split or the
        features on the single-device path.
  (iv)  simulation and cigarettes keep one SEM class per experiment and the
        shared transform: their runners' `polys[j]` is the shared `poly` (None) at
        every j. Simulation: every experiment is a freshly drawn SEM (`W_XY`
        pairwise distinct). Cigarettes (the robustness recipe's plasmode block):
        the covariate panel is identical across experiments, the confounder and
        the synthetic outcome are redrawn (pairwise distinct), and experiment j
        splits with `random_state = seed + j`.
  (v)   setup on all ten devices: the gamma runner's oracles have finite gamma*;
        every strategy runner's SEMs carry `epsilon_quantile` (q0.95); the
        epsilon runner's DA, retuned per device in the shared q0.95 reading of
        |W| (`epsilon_star_q95`, which its oracle reads too), lands its oracle eps* on
        the rule's target ROBUSTNESS_EPSILON_RADII x sigma sqrt(gamma*) of that
        device within POOLED_ORACLE_RTOL wherever its zero-strength eps* is below
        the target, and is clamped at zero strength where it is not (a NOTE); the
        n ladder's smallest cell keeps more train rows than the fit's k on every
        device. Reported per device: degree, gamma*, k, the floor, the target,
        the tuned strength and eps*.

    uv run python scripts/a82_optical_devices.py
"""

import os
import sys
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
    N_PERCENT_RANGE,
    OPTICAL_CONFIG,
    ROBUSTNESS_EPSILON_RADII,
    percent_of,
)
from src.experiments.generic_runner import STRATEGIES, GenericParamSweep  # noqa: E402
from src.experiments.optical_device import OpticalOrchestrator  # noqa: E402
from src.experiments.simulation import SimulationOrchestrator  # noqa: E402
from src.experiments.utils import set_seed  # noqa: E402
from src.oracle import STRENGTH_BRACKET, epsilon_star_q95, gamma_star  # noqa: E402

SEED = 42
N_LOADED = 12
EXCLUDED = (6, 11)
DEVICES = (8, 0, 1, 2, 3, 4, 5, 7, 9, 10)
# index -> file, as `OpticalDeviceSEM.load_dataset` orders them
FILE_TAGS = {6: ("exp_no_75", "brfactor_0"), 11: ("exp_no_82", "pure_confounding"), 8: ("exp_no_78",)}
METHODS = ["PI", "DA+PI"]
N_JOBS = 16
N_EXPERIMENTS_OTHER = 3
# the tuner solves on one frozen draw of the device's rows; the oracle pools
# ORACLE_POOL_DRAWS draws, which reads off the one: measured at seed 42 from -8.3%
# (device 8, as a40 records) to +8.4% (device 0) on the devices the tuner varies
POOLED_ORACLE_RTOL = 0.10
SHIPPED_CHAIN = "rotation > hflip > vflip > random-permutation"
RECIPE = os.path.join(REPO, "recipes", "robustnessFig11.yaml")
FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAIL.append(name)


class TaggedSEM(optical_device.SEM):
    """The optical SEM, remembering which device it was built on."""

    def __init__(self, experiment=0, **kwargs):
        super().__init__(experiment=experiment, **kwargs)
        self.device = int(experiment)


def shipped():
    """config.yaml, parsed once."""
    with open(os.path.join(REPO, "config.yaml")) as handle:
        return yaml.safe_load(handle) or {}


def chain():
    """config.yaml's optical chain, the shipped one when it names none."""
    return str((shipped().get("optical_device") or {}).get("augmentation", SHIPPED_CHAIN))


def augmentation_p():
    """config.yaml's optical rotation / flip probability, the DA's `P` when it names none."""
    return float((shipped().get("optical_device") or {}).get("augmentation_p", P))


def orchestrator(n_experiments, sweep_samples=2):
    set_seed(SEED)
    return OpticalOrchestrator(
        seed=SEED,
        n_samples=1000,
        n_experiments=n_experiments,
        sweep_samples=sweep_samples,
        methods=METHODS,
        hyperparameters={},
        n_jobs=N_JOBS,
        augmentation=chain(),
        augmentation_p=augmentation_p(),
    )


def sweep(orch, param, cls=None, **overrides):
    cls = cls or orch.get_sweep_runner_cls(param)
    return cls(methods=orch.methods, method_factory=orch.build_methods, **{**orch._get_clean_kwargs(), **overrides})


def leg_i():
    n = len(DEVICES)
    print(f"(i) {n} experiments, {n} devices, {n} feature maps")
    files = sorted(f for f in os.listdir("data/optical_device") if "confounder" in f and "random" not in f)
    check(f"(i) {N_LOADED} recordings loaded", len(files) == N_LOADED, f"{len(files)}")
    for index, tags in FILE_TAGS.items():
        name = files[index] if index < len(files) else ""
        check(f"(i) device {index} is {' '.join(tags)}", all(tag in name for tag in tags), name)
    check(
        "(i) excluded_devices", tuple(OPTICAL_CONFIG.excluded_devices) == EXCLUDED, f"{OPTICAL_CONFIG.excluded_devices}"
    )
    check("(i) dataset_index is 8", OPTICAL_CONFIG.dataset_index == 8)
    order = OpticalOrchestrator.sweep_devices()
    check("(i) default order: dataset_index first, then ascending, 6 and 11 left out", order == DEVICES, f"{order}")
    runner = sweep(orchestrator(n), "gamma")
    devices = [sem.device for sem in runner.sems]
    check("(i) sems[j] on devices[j]", devices == list(order), f"{devices}")
    check(f"(i) {n} distinct devices, none excluded", len(set(devices)) == n and not set(devices) & set(EXCLUDED))
    pools = [sem.X for sem in runner.sems]
    distinct = all(not np.array_equal(pools[a], pools[b]) for a in range(n) for b in range(a))
    check("(i) the SEMs' pools pairwise distinct", distinct)
    degrees = [sem.poly_degree for sem in runner.sems]
    check(
        "(i) polys[j].degree == sems[j].poly_degree",
        [poly.degree for poly in runner.polys] == degrees,
        f"degrees {degrees}",
    )
    check("(i) both degrees present", set(degrees) == {1, 2}, f"{sorted(set(degrees))}")
    fits = []
    for j, sem in enumerate(runner.sems):
        h = np.asarray(sem.f(runner._features_at(j)(sem.X)), dtype=float)
        fits.append(h.shape == (len(sem.X), 1) and bool(np.isfinite(h).all()))
    check("(i) sems[j].f on experiment j's features", all(fits), f"{fits}")
    try:
        sweep(orchestrator(n + 1), "gamma")
        raised = False
    except ValueError:
        raised = True
    check(f"(i) {n + 1} experiments raise", raised)


def leg_ii():
    print("(ii) one-experiment runners stay on device 8 (setup only)")
    orch = orchestrator(len(DEVICES))
    perf = orch.get_sweep_runner_cls("epsilon")(
        methods=orch.methods, method_factory=orch.build_methods, **{**orch._get_clean_kwargs(), "n_experiments": 1}
    )
    check("(ii) perf-shaped epsilon runner: one SEM, device 8", [s.device for s in perf.sems] == [8])
    query = orch.get_query_runner_cls()(methods=orch.methods, **{**orch._get_clean_kwargs(), "n_experiments": 1})
    check("(ii) query runner: device 8", query.sem.device == 8, f"{query.sem.device}")


def leg_iii():
    print("(iii) sweep_devices (8, 8) reproduces the single-device sweep bit for bit")
    original = optical_device.OPTICAL_CONFIG
    optical_device.OPTICAL_CONFIG = replace(original, sweep_devices=(8, 8))
    try:
        orch = orchestrator(2)
        pinned = sweep(orch, "gamma")
        check("(iii) pinned runner on devices (8, 8)", [s.device for s in pinned.sems] == [8, 8])
        x_new, new, _ = pinned.run("a82 per-device")
        base = orch.get_sweep_runner_cls("gamma")

        class Legacy(base):
            make_sem = GenericParamSweep.make_sem
            poly_for_sem = GenericParamSweep.poly_for_sem

        orch = orchestrator(2)
        legacy = sweep(orch, "gamma", cls=Legacy)
        x_old, old, _ = legacy.run("a82 legacy")
    finally:
        optical_device.OPTICAL_CONFIG = original
    check("(iii) same x", np.array_equal(np.asarray(x_new), np.asarray(x_old)))
    same = {
        (m, metric): np.array_equal(np.asarray(new[m][metric]), np.asarray(old[m][metric]), equal_nan=True)
        for m in old
        for metric in old[m]
        if metric != "wall_clock"
    }
    bad = [key for key, ok in same.items() if not ok]
    check(f"(iii) all {len(same)} (method, metric) arrays equal", not bad, f"{bad}")


def leg_iv():
    n = N_EXPERIMENTS_OTHER
    print(f"(iv) simulation and cigarettes: {n} experiments, shared transform")
    common = dict(seed=SEED, n_experiments=n, sweep_samples=2, methods=METHODS, hyperparameters={}, n_jobs=N_JOBS)
    set_seed(SEED)
    sim = sweep(SimulationOrchestrator(n_samples=256, kernel_dim=0, treatment_dim=4, **common), "gamma")
    with open(RECIPE) as handle:
        block = yaml.safe_load(handle)["cigarettes"]
    set_seed(SEED)
    cig = sweep(
        CigaretteOrchestrator(
            target=block["target"],
            spec=block["spec"],
            anchor=block["anchor"],
            da_amplitude=float(block["da_amplitude"]),
            iv=list(block["iv"]),
            gamma_z=float(block["gamma_z"]),
            n_samples=int(block["n_samples"]),
            **common,
        ),
        "gamma",
    )
    for name, runner in (("simulation", sim), ("cigarettes", cig)):
        check(
            f"(iv) {name}: polys[j] is poly at every j",
            len(runner.polys) == runner.n_experiments and all(p is runner.poly for p in runner.polys),
            f"poly {runner.poly!r}",
        )
    pairs = [(a, b) for a in range(n) for b in range(a)]
    check(
        "(iv) simulation: a fresh SEM per experiment (W_XY pairwise distinct)",
        all(not np.allclose(sim.sems[a].W_XY, sim.sems[b].W_XY) for a, b in pairs),
    )
    check("(iv) cigarettes: plasmode", block["target"] == "plasmode", f"{block['target']}")
    check(
        "(iv) cigarettes: one covariate panel",
        all(np.array_equal(cig.sems[a].X, cig.sems[b].X) for a, b in pairs),
    )
    check(
        "(iv) cigarettes: the synthetic outcome redrawn per experiment",
        all(not np.allclose(cig.sems[a].y, cig.sems[b].y) for a, b in pairs),
    )
    states, original = [], generic_runner.train_test_split

    def spy(*args, **kwargs):
        states.append(kwargs.get("random_state"))
        return original(*args, **kwargs)

    generic_runner.train_test_split = spy
    try:
        for j in range(n):
            cig._draw_base(j)
    finally:
        generic_runner.train_test_split = original
    check("(iv) cigarettes: experiment j splits at seed + j", states == [SEED + j for j in range(n)], f"{states}")


def leg_v():
    n = len(DEVICES)
    print(f"(v) setup on all {n} devices")
    orch = orchestrator(n)
    runners = {param: sweep(orch, param) for param in STRATEGIES}
    gamma, eps = runners["gamma"], runners["epsilon"]
    quantile = OPTICAL_CONFIG.epsilon_quantile
    for param, runner in runners.items():
        got = {getattr(sem, "epsilon_quantile", "unset") for sem in runner.sems}
        check(f"(v) {param} runner SEMs carry epsilon_quantile {quantile}", got == {quantile}, f"{got}")
    rows = percent_of(1000, N_PERCENT_RANGE[0])
    n_train = int(round((1.0 - OPTICAL_CONFIG.test_fraction) * rows))
    print(f"      smallest n cell: {rows} rows, {n_train} train rows")
    print("       j device degree    gamma*     k n_train-k     floor    target  strength  pooled eps*  /target")
    finite, tuned, clamped, headroom = [], [], [], []
    for j, sem in enumerate(gamma.sems):
        oracle, achieved = gamma.get_oracle(j), eps.get_oracle(j).epsilon_star
        features = gamma._features_at(j)
        k = features(sem.X[:1]).shape[1] + int(gamma.mean_match)
        esem = eps.sems[j]
        target = ROBUSTNESS_EPSILON_RADII * float(np.sqrt(esem.sigma_sq * gamma_star(esem)))
        da, strength = eps.das[j], eps.das[j].strength
        da.strength = STRENGTH_BRACKET[0]
        floor = epsilon_star_q95(esem, da, X=esem.pool[0], features=eps._features_at(j))
        da.strength = strength
        finite.append(bool(np.isfinite(oracle.gamma_star)))
        if floor >= target:
            # the tuner's documented clamp: the device's defect already exceeds
            # the target at zero strength, so the sweep cannot vary its DA
            clamped.append(sem.device)
            tuned.append(strength == STRENGTH_BRACKET[0])
        else:
            tuned.append(bool(strength > STRENGTH_BRACKET[0] and np.isclose(achieved, target, rtol=POOLED_ORACLE_RTOL)))
        headroom.append(n_train - k)
        print(
            f"      {j:>2} {sem.device:>6} {sem.poly_degree:>6} {oracle.gamma_star:>9.4g} {k:>5} {n_train - k:>9} "
            f"{floor:>9.4g} {target:>9.4g} {strength:>9.4g} {achieved:>12.5g} {achieved / target:>8.3f}"
        )
    check("(v) gamma* finite on every device", all(finite), f"{finite}")
    check(
        f"(v) the tuned DA's pooled eps* lands on the target within {POOLED_ORACLE_RTOL:.0%}, "
        "or clamps where the floor exceeds it",
        all(tuned),
        f"{tuned}",
    )
    print(f"  [NOTE] (v) devices whose zero-strength eps* already exceeds the target (DA not varied): {clamped}")
    check("(v) the smallest n cell keeps n_train > k on every device", min(headroom) > 0, f"min {min(headroom)}")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    os.chdir(REPO)
    optical_device.SEM = TaggedSEM
    leg_i()
    leg_ii()
    leg_iii()
    leg_iv()
    leg_v()
    print(f"\n{'A82 ALL PASS' if not FAIL else 'A82 FAILURES: ' + ', '.join(FAIL)}")
    sys.exit(bool(FAIL))
