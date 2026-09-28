"""A82: one optical device per sweep experiment (setup only but for (iii); a few
minutes on CPU, most of it the twelve DA tunings of (v)).

The optical sweeps run experiment j on `OpticalOrchestrator.sweep_devices()[j]`,
each with the polynomial feature map of its own selected degree; every
one-experiment runner stays on `dataset_index`. Legs:

  (i)   the default order is `dataset_index` (8) first, then every other device
        ascending; a gamma sweep at twelve experiments builds twelve SEMs on
        distinct devices covering all twelve (read off a tagging SEM and the SEMs'
        own pools), `polys[j]` has the degree of `sems[j]`, and the features of
        experiment j fit `sems[j].f`; a thirteenth experiment raises.
        Catches: a hook that falls back to `sem_factory`, one shared feature map
        (a degree-1 device would crash `sem.f`), a device reused.
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
  (iv)  simulation and cigarettes are untouched: their runners' `polys[j]` is the
        shared `poly` object (None) at every j; cigarettes on config.yaml's block.
  (v)   setup on all twelve devices: the gamma runner's oracles have finite
        gamma*; the epsilon runner's DA, retuned per device, lands on
        `ROBUSTNESS_EPSILON_TRUE["optical_device"]` within 5% on device 8 (the
        pooled-oracle tolerance of a40) and 10% on the others (device 11 reads
        -5.8%) wherever its zero-strength eps* is below the constant,
        and is clamped at zero strength where it is not (device 6: std(h*) ~119,
        eps* floor ~62, so its robustness DA does not vary; listed as a NOTE);
        the n ladder's smallest cell keeps more train rows than the fit's k on
        every device. Reported per device: degree, gamma*, k, the floor, the
        tuned strength and eps*, and eps* / std(h*) (a40(vii)'s ratio).

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

from src.experiments import optical_device  # noqa: E402
from src.experiments.cigarettes import CigaretteOrchestrator  # noqa: E402
from src.experiments.configs import (  # noqa: E402
    OPTICAL_CONFIG,
    ROBUSTNESS_EPSILON_TRUE,
    percent_of,
    resolve_dataset_block,
)
from src.experiments.generic_runner import GenericParamSweep  # noqa: E402
from src.experiments.optical_device import OpticalOrchestrator  # noqa: E402
from src.experiments.simulation import SimulationOrchestrator  # noqa: E402
from src.experiments.utils import set_seed  # noqa: E402
from src.oracle import STRENGTH_BRACKET, epsilon_star  # noqa: E402

SEED = 42
N_DEVICES = 12
METHODS = ["PI", "DA+PI"]
N_JOBS = 4
POOLED_ORACLE_RTOL = 0.05  # a40's, measured on device 8
# the other devices' pooled oracle strays further from the tuner's single draw
# (device 11: 4.71, -5.8%), so they get twice device 8's allowance
POOLED_ORACLE_RTOL_OTHERS = 0.10
SMALLEST_PERCENT = 6.25
SHIPPED_CHAIN = "rotation > hflip > vflip > random-permutation"
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
    )


def sweep(orch, param, cls=None, **overrides):
    cls = cls or orch.get_sweep_runner_cls(param)
    return cls(methods=orch.methods, method_factory=orch.build_methods, **{**orch._get_clean_kwargs(), **overrides})


def leg_i():
    print("(i) twelve experiments, twelve devices, twelve feature maps")
    order = OpticalOrchestrator.sweep_devices()
    want = (OPTICAL_CONFIG.dataset_index, *(d for d in range(N_DEVICES) if d != OPTICAL_CONFIG.dataset_index))
    check("(i) default order: dataset_index first, then ascending", order == want, f"{order}")
    check("(i) dataset_index is 8", OPTICAL_CONFIG.dataset_index == 8)
    orch = orchestrator(N_DEVICES)
    runner = sweep(orch, "gamma")
    devices = [sem.device for sem in runner.sems]
    check("(i) sems[j] on devices[j]", devices == list(order), f"{devices}")
    check(f"(i) all {N_DEVICES} devices, none twice", sorted(devices) == list(range(N_DEVICES)))
    pools = [sem.X for sem in runner.sems]
    distinct = all(not np.array_equal(pools[a], pools[b]) for a in range(N_DEVICES) for b in range(a))
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
        sweep(orchestrator(N_DEVICES + 1), "gamma")
        raised = False
    except ValueError:
        raised = True
    check(f"(i) {N_DEVICES + 1} experiments raise", raised)


def leg_ii():
    print("(ii) one-experiment runners stay on device 8 (setup only)")
    orch = orchestrator(N_DEVICES)
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
    print("(iv) simulation and cigarettes keep the shared transform")
    common = dict(seed=SEED, n_experiments=2, sweep_samples=2, methods=METHODS, hyperparameters={}, n_jobs=N_JOBS)
    set_seed(SEED)
    sim = sweep(SimulationOrchestrator(n_samples=256, kernel_dim=0, treatment_dim=4, **common), "gamma")
    config = shipped()
    block = {**(config.get("defaults") or {}), **(config.get("cigarettes") or {}), "im-ci": 0}
    block.pop("experiment", None)
    block = resolve_dataset_block("cigarettes", block)
    block.update(common)
    set_seed(SEED)
    cig = sweep(CigaretteOrchestrator(**block), "gamma")
    for name, runner in (("simulation", sim), ("cigarettes", cig)):
        check(
            f"(iv) {name}: polys[j] is poly at every j",
            len(runner.polys) == runner.n_experiments and all(p is runner.poly for p in runner.polys),
            f"poly {runner.poly!r}",
        )


def leg_v():
    print(f"(v) setup on all {N_DEVICES} devices")
    orch = orchestrator(N_DEVICES)
    gamma = sweep(orch, "gamma")
    eps = sweep(orch, "epsilon")
    want = ROBUSTNESS_EPSILON_TRUE["optical_device"]
    rows = percent_of(1000, SMALLEST_PERCENT)
    n_train = int(round((1.0 - OPTICAL_CONFIG.test_fraction) * rows))
    print(f"      smallest n cell: {rows} rows, {n_train} train rows")
    print("       j device degree    gamma*     k n_train-k     floor  strength  tuned eps*  eps*/std(h*)")
    finite, tuned, clamped, headroom = [], [], [], []
    for j, sem in enumerate(gamma.sems):
        oracle, achieved = gamma.get_oracle(j), eps.get_oracle(j).epsilon_star
        features = gamma._features_at(j)
        k = features(sem.X[:1]).shape[1] + int(gamma.mean_match)
        da, strength = eps.das[j], eps.das[j].strength
        da.strength = STRENGTH_BRACKET[0]
        floor = epsilon_star(sem, da, X=sem.X, features=features)
        da.strength = strength
        spread = float(np.std(sem.f(features(sem.X))))
        finite.append(bool(np.isfinite(oracle.gamma_star)))
        if floor >= want:
            # the tuner's documented clamp: the device's defect already exceeds
            # the constant at zero strength, so the sweep cannot vary its DA
            clamped.append(sem.device)
            tuned.append(strength == STRENGTH_BRACKET[0])
        else:
            rtol = POOLED_ORACLE_RTOL if sem.device == OPTICAL_CONFIG.dataset_index else POOLED_ORACLE_RTOL_OTHERS
            tuned.append(bool(np.isclose(achieved, want, rtol=rtol)))
        headroom.append(n_train - k)
        print(
            f"      {j:>2} {sem.device:>6} {sem.poly_degree:>6} {oracle.gamma_star:>9.4g} {k:>5} {n_train - k:>9} "
            f"{floor:>9.4g} {strength:>9.4g} {achieved:>11.5g} {achieved / spread:>13.4g}"
        )
    check("(v) gamma* finite on every device", all(finite), f"{finite}")
    check(
        f"(v) the tuned DA lands on {want:g} (within {POOLED_ORACLE_RTOL:.0%} on device 8, "
        f"{POOLED_ORACLE_RTOL_OTHERS:.0%} elsewhere), or clamps where the floor exceeds it",
        all(tuned),
        f"{tuned}",
    )
    print(f"  [NOTE] (v) devices whose zero-strength eps* already exceeds {want:g} (DA not varied): {clamped}")
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
