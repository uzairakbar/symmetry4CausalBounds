"""A53: the dataset hooks are hooks, and the shipped two experiments do not move.

refactor4 is the only branch that edits shared files. Everything it adds is either a
new dict key (`cigarettes` in the two ROBUSTNESS tables and the two grid functions),
a defaulted argument (`fname`, `subdir`, `extent`) or a predicate rewritten to an
equivalent one. So the gate is half regression and half interface. Legs:

  (i)   regression, in bytes. The sim and optical query sweeps (`sweep_samples=8`,
        methods PI and DA+PI, one experiment) and their gamma sweeps at a fixed
        two-point grid are hashed and compared against digests RECORDED ON THE
        PARENT COMMIT 5b7965a with this same code. Catches: any change of numbers
        on either shipped experiment, including one hiding behind a defaulted
        argument. The gamma sweep is in the harness because it is the half that
        goes through `finite_pool`, `_sweep_data` and the two metrics -- a query
        sweep touches none of them. Misses: a change confined to a method or a
        toggle these two runs do not exercise.
  (ii)  the rewritten `finite_pool` predicate agrees with the old class-name test
        on both shipped SEMs. Catches: a pool attribute appearing on a generator
        SEM. Misses: a third SEM neither predicate was written for.
  (iii) the trS and n grids: sim and optical elementwise unchanged, cigarettes at
        [0.125, 8] and [245, 2450]. Catches: a branch that swallows another
        dataset's grid.
  (iv)  the two ROBUSTNESS tables carry exactly three keys, with the shipped two
        values untouched. Catches: an edited sim or optical constant.
  (v)   the cigarette epsilon sweep runs: the tuned DA reaches the configured
        eps* within 5%, DA+PI and DA+PI+IV dip under the PI baseline at the
        smallest ratio, climb back to it by r = 1 and never go under 0.7. The
        baseline is PI's own coverage (0.963 measured), not 1.0: the sweep fits
        bootstrap replicates against a pool oracle, so nothing covers 1.000 here.
        Catches: a constant that is too small (a flat line) or too large (a cliff).
  (vi)  `create_query_sweep_plot` writes the same filename as before when `fname`
        is omitted, and `fname` is keyword-with-default. Catches: a positional
        argument, a changed derived name.
  (vii) the target interface degenerates. `extent` is all-zero on both shipped
        SEMs, and on real draws `coverage` and `approximation_error` return
        EXACTLY what the pre-refactor4 expressions did, both at `extent=0.0` and
        at the default. The draws are padded into a third case with points on
        both sides of the interval, so neither branch of either formula is
        untested. Catches: a non-zero default, a mis-signed padding, an
        approximation branch written as a min over padded endpoints.
  (ix)  the extent array TRAVELS. Leg (vii) proves the two metrics honour a
        positive `extent` and that the interface is inert at zero; it says nothing
        about whether the array ever reaches them. So: the sim SEM with one method
        swapped for a constant half-width, through the real `GammaRatioStrategy`,
        against the same draw without it. `SweepData.extent` must arrive filled,
        and coverage must strictly fall while the approximation error strictly
        rises. Catches: a severed fill -- `_extent` returning zeros, the
        `SweepData` field dropped at a construction site, `metric_extent` pinned
        at 0.0 -- every one of which leaves the sliver silently invisible in every
        sweep figure while legs (i)-(viii) stay green. Misses: the query path,
        which has no metrics; a55 is where that lands.
  (viii) the do-MNIST SEM inherits `extent` from the base class and returns zeros
        on a 3-row array. CLASS ONLY: the SEM is never constructed and no dataset
        is touched. Catches: `extent` defined on a subclass instead of the base.

    MPLBACKEND=Agg python scripts/a53_dataset_plumbing.py [--seed 42] [--micro]
    MPLBACKEND=Agg python scripts/a53_dataset_plumbing.py --record

`--record` prints leg (i)'s digests and exits; it is what was run on the parent
commit to produce PARENT_DIGESTS, and it imports nothing that refactor4 adds.
`--micro` shrinks leg (v) to a 500-row fit sample; leg (i) is never shrunk, since
its digests are pinned.
"""

import argparse
import hashlib
import inspect
import os
import shutil
import sys
import tempfile

import numpy as np
import yaml
from loguru import logger

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from src.experiments.base import SweepData  # noqa: E402
from src.experiments.configs import (  # noqa: E402
    DATASET_DEFAULTS,
    PARAM_SPECS,
    ROBUSTNESS_AUGMENTATION,
    ROBUSTNESS_EPSILON_TRUE,
)
from src.experiments.generic_runner import STRATEGIES  # noqa: E402
from src.experiments.optical_device import OpticalOrchestrator  # noqa: E402
from src.experiments.simulation import SimulationOrchestrator  # noqa: E402
from src.experiments.utils import set_seed  # noqa: E402
from src.experiments.utils.constants import SUBDIR_QUERY  # noqa: E402
from src.experiments.utils.metrics import approximation_error, coverage, evaluate_queries  # noqa: E402
from src.experiments.utils.plotting import create_query_sweep_plot  # noqa: E402
from src.sem.abstract import StructuralEquationModel  # noqa: E402
from src.sem.optical_device import OpticalDeviceSEM  # noqa: E402
from src.sem.simulation import LinearSimulationSEM  # noqa: E402

METHODS = ["PI", "DA+PI"]
SWEEP_SAMPLES = 8
GAMMA_GRID = (2.0**-6, 1.0)
N_JOBS = 4
# hashed metrics: everything `METRIC_FIELDS` records except `wall_clock`, which is
# a timer and differs between two runs of identical code
HASHED_METRICS = ("approximation_error", "coverage", "interval_width", "worst_error")
SHIPPED_CHAIN = "rotation > hflip > vflip > random-permutation"
OPTICAL_POOL = 1000

# leg (i) digests, RECORDED ON THE PARENT COMMIT 5b7965a with this file's own
# `regression_digests` (run there as `--record`, twice, agreeing bit for bit).
# Caveat: these are float bit patterns. A rerun on the SAME node reproduces them
# exactly; a different node, BLAS build or thread count may drift by a few ULP and
# turn the leg red without anything being wrong. Re-record with `--record` on the
# parent before believing a lone (i) failure.
PARENT_DIGESTS: dict[str, str] = {
    "optical_device/gamma": "74a4205ff747ab3a",
    "optical_device/query": "f2acb071110c9bb8",
    "simulation/gamma": "7b21bf701c0cd4c9",
    "simulation/query": "97a1e953dad97e47",
}

# leg (iii)
CIGARETTE_TRS = (0.125, 8.0)
CIGARETTE_N = (245, 2450)
# leg (iv)
DATASET_KEYS = {"simulation", "optical_device", "cigarettes"}
SHIPPED_ROBUSTNESS = {"simulation": (3.0, None), "optical_device": (5.0, "gaussian-noise")}
# leg (v)
METHODS_IV = ["PI", "DA+PI", "DA+PI+IV"]
CIGARETTE_SAMPLES = 2450
EPSILON_GRID = (2.0**-6, 2.0**-3, 1.0)
COVERAGE_FLOOR = 0.7
EPS_STAR_RTOL = 0.05
MICRO_SAMPLES = 500
# leg (ix): a half-width in the sim's outcome units. Wider than the HALF-widths the
# sim's own intervals have at gamma* (PI 3.88, DA+PI 2.38), so the set falls out of
# both and every method's coverage has to move, not just the tighter one.
STUB_EXTENT = 5.0
# leg (vi): the name `create_query_sweep_plot` has always derived from the xlabel
DERIVED_XLABEL = r"$\vartheta$"
DERIVED_FNAME = "vartheta_sweep"

# read once, through getattr: a break that MOVES `extent` off the base class must
# make legs (vii) and (viii) print FAIL, not raise on the way to them
BASE_EXTENT = getattr(StructuralEquationModel, "extent", None)

TMPROOT = os.path.expanduser("~/scratch/tmp/a53")
FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAIL.append(name)


def configured():
    """The toggles, the sim draw and the optical chain of config.yaml, falling back
    to DatasetDefaults / the shipped chain as main.py does (the a40 pattern)."""
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


def build(experiment, draw, chain, seed, **toggles):
    common = dict(seed=seed, sweep_samples=SWEEP_SAMPLES, hyperparameters={}, n_jobs=N_JOBS, methods=METHODS, **toggles)
    if experiment == "simulation":
        return SimulationOrchestrator(kernel_dim=0, n_experiments=1, **draw, **common)
    return OpticalOrchestrator(n_samples=OPTICAL_POOL, augmentation=chain, n_experiments=1, **common)


# =============================================================================
# LEG (i): REGRESSION DIGESTS
# =============================================================================


def _digest(arrays) -> str:
    """sha256 over the shapes and the raw float bytes, in the order given."""
    handle = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(np.asarray(array, dtype=float))
        handle.update(repr(array.shape).encode())
        handle.update(array.tobytes())
    return handle.hexdigest()[:16]


def regression_digests(draw, chain, seed, **toggles) -> dict[str, str]:
    """The shipped bounds, hashed. Two runs per dataset:

    query   `GenericQuerySweep`, `sweep_samples` queries, the PI and DA+PI bounds.
    gamma   `GammaRatioStrategy` on a two-point grid, every metric of every method
            except the wall clock, plus the plotted x. This is the run that goes through `finite_pool`,
            `_sweep_data` and `evaluate_queries`, so it is where a plumbing edit
            would show; the query sweep touches none of the three.

    The seed is reset before each construction and again before each run: the sim
    SEM draws from the global stream, so a shared process state would make the
    second run depend on the first.
    """
    digests = {}
    for name in ("simulation", "optical_device"):
        set_seed(seed)
        orch = build(name, draw, chain, seed, **toggles)

        set_seed(seed)
        query = orch.get_query_runner_cls()(methods=orch.methods, **{**orch._get_clean_kwargs(), "n_experiments": 1})
        set_seed(seed)
        queries, results = query.run(f"a53 {name} query")
        digests[f"{name}/query"] = _digest([queries] + [results[m] for m in sorted(results)])

        set_seed(seed)
        sweep = orch.get_sweep_runner_cls("gamma")(
            methods=orch.methods,
            method_factory=orch.build_methods,
            param_grid_override=np.asarray(GAMMA_GRID),
            **orch._get_clean_kwargs(),
        )
        set_seed(seed)
        x, metrics, _ = sweep.run(f"a53 {name} gamma sweep")
        digests[f"{name}/gamma"] = _digest([x] + [metrics[m][k] for m in sorted(metrics) for k in HASHED_METRICS])
    return digests


def leg_i(digests):
    print("(i) the sim and optical bounds are bit-identical to the parent commit")
    for key in sorted(PARENT_DIGESTS):
        want, got = PARENT_DIGESTS[key], digests.get(key, "")
        check(f"(i) {key} digest", bool(want) and got == want, f"{got} vs {want}")


# =============================================================================
# LEG (ii): THE finite_pool PREDICATE
# =============================================================================


def old_finite_pool(sem) -> bool:
    """The predicate as it stood at 5b7965a (`generic_runner.py:320`)."""
    return "OpticalDeviceSEM" in str(type(sem))


def new_finite_pool(sem) -> bool:
    """The predicate refactor4 ships. Read off the runner, not retyped, so a leg
    that passes here is a leg about the code and not about this file."""
    from src.experiments.generic_runner import GenericParamSweep

    class Probe:
        sems = [sem]

    return GenericParamSweep.finite_pool.fget(Probe())


def leg_ii(orchs):
    print("(ii) the rewritten predicate agrees with the class-name test on both shipped SEMs")
    for name, orch in orchs.items():
        sem = orch._sem_factory()
        old, new = old_finite_pool(sem), new_finite_pool(sem)
        check(f"(ii) {name}: predicates agree", old == new, f"old {old}, new {new}")
        recorded = name == "optical_device"
        check(f"(ii) {name}: pool is {'set' if recorded else 'None'}", new == recorded)


# =============================================================================
# LEG (iii): THE SWEEP GRIDS
# =============================================================================


def leg_iii():
    print("(iii) the trS and n grids: two unchanged, one added")
    trS, n = PARAM_SPECS["trS"].grid_fn, PARAM_SPECS["n"].grid_fn
    steps = 7
    expected_trS = {
        "simulation": np.logspace(-1.5, 1.0, num=steps),
        "optical_device": np.linspace(0.2, 0.99, num=steps),
    }
    expected_n = {
        "simulation": np.linspace(128, 1024, 16, dtype=int),
        "optical_device": np.linspace(128, 1000, 16, dtype=int),
    }
    for name in ("simulation", "optical_device"):
        check(f"(iii) {name}: trS grid unchanged", np.array_equal(trS(name, steps), expected_trS[name]))
        check(f"(iii) {name}: n grid unchanged", np.array_equal(n(name, steps), expected_n[name]))

    cig_trS = np.asarray(trS("cigarettes", steps), dtype=float)
    cig_n = np.asarray(n("cigarettes", steps), dtype=int)
    check(
        "(iii) cigarettes: trS grid ends",
        np.allclose([cig_trS[0], cig_trS[-1]], CIGARETTE_TRS) and len(cig_trS) == steps,
        f"[{cig_trS[0]:.4g}, {cig_trS[-1]:.4g}] over {len(cig_trS)} steps",
    )
    check("(iii) cigarettes: trS grid is increasing", np.all(np.diff(cig_trS) > 0))
    check(
        "(iii) cigarettes: n grid ends",
        (cig_n[0], cig_n[-1]) == CIGARETTE_N and len(cig_n) == 16,
        f"[{cig_n[0]}, {cig_n[-1]}] over {len(cig_n)} steps",
    )
    # the branches must not collide: the added grid is nobody else's
    for name in ("simulation", "optical_device"):
        check(
            f"(iii) cigarettes trS differs from {name}",
            not np.allclose(cig_trS, np.asarray(trS(name, steps), dtype=float)),
        )
        check(f"(iii) cigarettes n differs from {name}", not np.array_equal(cig_n, np.asarray(n(name, 16), dtype=int)))


# =============================================================================
# LEG (iv): THE ROBUSTNESS TABLES
# =============================================================================


def leg_iv():
    print("(iv) the robustness tables gain a key and keep their two values")
    for label, table in (("epsilon_true", ROBUSTNESS_EPSILON_TRUE), ("augmentation", ROBUSTNESS_AUGMENTATION)):
        check(f"(iv) {label}: exactly three keys", set(table) == DATASET_KEYS, f"{sorted(table)}")
    for name, (eps, component) in SHIPPED_ROBUSTNESS.items():
        check(f"(iv) {name}: eps* constant unchanged", ROBUSTNESS_EPSILON_TRUE.get(name) == eps, f"{eps}")
        check(f"(iv) {name}: appended component unchanged", ROBUSTNESS_AUGMENTATION.get(name) == component)
    check("(iv) cigarettes: eps* constant is 0.5", ROBUSTNESS_EPSILON_TRUE.get("cigarettes") == 0.5)
    check("(iv) cigarettes: no appended component", ROBUSTNESS_AUGMENTATION.get("cigarettes") is None)


# =============================================================================
# LEG (v): THE CIGARETTE ROBUSTNESS SWEEP
# =============================================================================


def cigarette_epsilon_runner(seed, n_samples, **toggles):
    """The epsilon strategy at `experiment_name='cigarettes'`, so the two
    ROBUSTNESS tables are read through the production lookup. The orchestrator
    that would supply these factories is refactor5's; a54 runs the same strategy
    through it."""
    from src.data_augmentors.cigarettes import ScaleTranslation
    from src.sem.cigarettes import CigaretteSEM, V

    def sem_factory():
        return CigaretteSEM(spec="t3", target="iv", anchor="own-tax", bootstrap=True)

    amplitude = float(np.std(sem_factory().X @ (V / np.linalg.norm(V))))

    def da_factory(sem=None, append=None):
        return ScaleTranslation(V, std=amplitude)

    def method_factory(gamma, epsilon, epsilon_iv=None, rho=1.0, epsilon_iv_z=0.0):
        # the runner hands every factory `epsilon_iv_z` (the non-DA +IV term) since
        # e4fb1a5; inert here, the SEM carries no real instrument under this design
        from src.experiments.configs import MethodRegistry

        return MethodRegistry.build_methods(
            METHODS_IV,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_iv=epsilon_iv,
            epsilon_iv_z=epsilon_iv_z,
            rho=rho,
            n_jobs=N_JOBS,
            **toggles,
        )

    return STRATEGIES["epsilon"](
        sem_factory=sem_factory,
        da_factory=da_factory,
        method_factory=method_factory,
        experiment_name="cigarettes",
        test_fraction=0.1,
        param_grid_override=np.asarray(EPSILON_GRID),
        seed=seed,
        n_samples=n_samples,
        n_experiments=1,
        sweep_samples=len(EPSILON_GRID),
        methods=METHODS_IV,
        hyperparameters={},
        n_jobs=N_JOBS,
        **toggles,
    )


def leg_v(seed, micro, **toggles):
    print("(v) the cigarette robustness sweep reaches its constant and dips")
    runner = cigarette_epsilon_runner(seed, MICRO_SAMPLES if micro else CIGARETTE_SAMPLES, **toggles)
    want = ROBUSTNESS_EPSILON_TRUE["cigarettes"]
    achieved = runner.get_oracle(0).epsilon_star
    strength = runner.das[0].strength
    check(
        "(v) the tuned DA reaches the constant",
        strength is not None and np.isclose(achieved, want, rtol=EPS_STAR_RTOL),
        f"eps* {achieved:.6g} vs {want:g} at strength {strength!r}",
    )

    x, results, _ = runner.run("a53 cigarettes epsilon sweep")
    x = np.asarray(x, dtype=float)
    lines = [m for m in results if m.startswith("DA+")]
    cov = {m: np.asarray(results[m]["coverage"], dtype=float).mean(axis=1) for m in results}
    print("      r:        " + " ".join(f"{v:.4g}" for v in x))
    for m in results:
        print(f"      {m:9s} " + " ".join(f"{v:.3f}" for v in cov[m]))
    # The BASELINE, not 1.0: the sweep fits state-cluster bootstrap replicates while
    # the oracle reads the whole pool (SS6), so gamma* is the population value and
    # PI itself covers 0.96, not 1.00. What a working robustness axis has to show is
    # a dip under that baseline at a misstated epsilon and a full recovery at eps*.
    baseline = cov["PI"]
    check("(v) the PI baseline is flat in r", np.allclose(baseline, baseline[0]), f"{baseline[0]:.4f}")
    for m in lines:
        check(f"(v) {m} dips under the baseline at the smallest r", cov[m][0] < baseline[0], f"{cov[m][0]:.4f}")
        check(f"(v) {m} is non-decreasing in r", np.all(np.diff(cov[m]) >= -1e-12), f"{cov[m]}")
        check(
            f"(v) {m} recovers to the baseline at r = 1",
            cov[m][-1] >= baseline[-1],
            f"{cov[m][-1]:.4f} vs {baseline[-1]:.4f}",
        )
        check(f"(v) {m} stays above the floor", np.nanmin(cov[m]) > COVERAGE_FLOOR, f"{np.nanmin(cov[m]):.4f}")


# =============================================================================
# LEG (vi): THE PLOT FILENAME
# =============================================================================


def leg_vi(tmp):
    print("(vi) the query plot keeps its derived filename when `fname` is omitted")
    parameters = inspect.signature(create_query_sweep_plot).parameters
    fname = parameters.get("fname")
    check("(vi) `fname` exists", fname is not None)
    if fname is not None:
        check(
            "(vi) `fname` defaults to None",
            fname.default is None,
            f"default {fname.default!r}, kind {fname.kind}",
        )
    # the positional order the orchestrators call with must not have moved
    positional = [n for n, p in parameters.items() if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD]
    check("(vi) the first three arguments are unchanged", positional[:3] == ["x_values", "y_results", "xlabel"])

    here = os.getcwd()
    os.chdir(tmp)
    try:
        x = np.linspace(0.0, 1.0, 4)
        y = {"PI": np.stack([np.stack([-np.ones(4), np.ones(4)], axis=1)], axis=1)}
        create_query_sweep_plot(x, y, xlabel=DERIVED_XLABEL, experiment="a53", hide_legend=True)
        written = sorted(os.listdir(os.path.join(tmp, "artifacts", "a53", SUBDIR_QUERY)))
    finally:
        os.chdir(here)
    check("(vi) the derived filename is unchanged", written == [f"{DERIVED_FNAME}.pdf"], f"{written}")


# =============================================================================
# LEG (vii): THE TARGET INTERFACE
# =============================================================================


def shipped_coverage(estimand, estimate):
    """`metrics.coverage` as it stood at 5b7965a, verbatim."""
    estimate = _as_interval(estimate)
    flat = estimand.squeeze()
    covered = (flat >= estimate[:, 0]) & (flat <= estimate[:, 1])
    return float(np.mean(np.where(np.isnan(estimate).any(axis=1), False, covered)))


def shipped_approximation_error(estimand, estimate):
    """`metrics.approximation_error` as it stood at 5b7965a, verbatim (unnormalised,
    which is `DEFAULT_NORMALIZE_ERROR`)."""
    estimate = _as_interval(estimate)
    lower, upper = estimate[:, 0], estimate[:, 1]
    flat = estimand.squeeze()
    inside = (flat >= lower) & (flat <= upper)
    distance = np.minimum((lower - flat) ** 2, (upper - flat) ** 2)
    return float(np.nanmean(np.where(inside, 0, distance)[:, None]))


def _as_interval(estimate):
    return np.repeat(estimate, 2, axis=1) if estimate.shape[-1] == 1 else estimate


def _draws(orchs, seed):
    """(estimand, interval) from a real draw of each shipped experiment, plus a
    SHRUNK copy of each so that points sit on both sides of the interval. Without
    the shrunk case every query is covered and neither formula's outside branch is
    exercised, so the leg could not fail."""
    cases = []
    for name, orch in orchs.items():
        set_seed(seed)
        runner = orch.get_query_runner_cls()(methods=orch.methods, **{**orch._get_clean_kwargs(), "n_experiments": 1})
        set_seed(seed)
        queries, results = runner.run(f"a53 {name} metric draw")
        estimand = np.asarray(runner.sem.f(queries), dtype=float).reshape(-1, 1)
        interval = np.asarray(results["PI"], dtype=float)[:, 0, :]
        cases.append((name, estimand, interval))
        centre = interval.mean(axis=1, keepdims=True)
        cases.append((f"{name} shrunk", estimand, centre + 0.05 * (interval - centre)))
    return cases


def leg_vii(orchs, seed):
    print("(vii) the target interface degenerates to today's point target")
    for name, orch in orchs.items():
        sem = orch._sem_factory()
        X = np.asarray(sem.X if hasattr(sem, "X") else sem(N=4)[0], dtype=float)[:4]
        extent = sem.extent(X)
        check(
            f"(vii) {name}: extent is all-zero of length len(X)",
            extent.shape == (len(X),) and not np.any(extent),
            f"shape {extent.shape}, max {np.max(np.abs(extent)) if extent.size else 0}",
        )
    for cls in (LinearSimulationSEM, OpticalDeviceSEM):
        check(
            f"(vii) {cls.__name__} inherits extent from the base class",
            BASE_EXTENT is not None and getattr(cls, "extent", None) is BASE_EXTENT,
        )

    outside = []
    for name, estimand, interval in _draws(orchs, seed):
        flat = estimand.squeeze()
        inside = int(np.sum((flat >= interval[:, 0]) & (flat <= interval[:, 1])))
        outside.append(len(flat) - inside)
        print(f"      {name}: {inside} of {len(flat)} queries inside the PI interval")
        want_cov = shipped_coverage(estimand, interval)
        want_err = shipped_approximation_error(estimand, interval)
        for label, kwargs in (("extent=0", {"extent": 0.0}), ("default", {})):
            got_cov = coverage(estimand, interval, **kwargs)
            got_err = approximation_error(estimand, interval, **kwargs)
            check(f"(vii) {name}: coverage ({label}) is the shipped value", got_cov == want_cov, f"{got_cov!r}")
            check(
                f"(vii) {name}: approximation_error ({label}) is the shipped value",
                got_err == want_err,
                f"{got_err!r} vs {want_err!r}",
            )
        record = evaluate_queries(estimand, interval)
        check(
            f"(vii) {name}: evaluate_queries agrees",
            record.coverage == want_cov and record.approximation_error == want_err,
        )
        # a non-degenerate extent must actually move both, or the argument is inert
        half = 0.5 * float(np.mean(interval[:, 1] - interval[:, 0]))
        moved_cov = coverage(estimand, interval, extent=half)
        moved_err = approximation_error(estimand, interval, extent=half)
        check(
            f"(vii) {name}: a positive extent moves both metrics",
            moved_cov <= want_cov and moved_err >= want_err and (moved_cov < want_cov or moved_err > want_err),
            f"coverage {want_cov:.4f} -> {moved_cov:.4f}, error {want_err:.6g} -> {moved_err:.6g}",
        )
    # a leg that only ever sees covered queries tests one branch of each formula
    check(
        "(vii) the cases cover both branches",
        min(outside) == 0 and max(outside) > 0,
        f"queries outside per case: {outside}",
    )


# =============================================================================
# LEG (ix): THE FILL
# =============================================================================


class SetTargetSEM(LinearSimulationSEM):
    """The simulation SEM with a SET target: same centre, a constant half-width.

    Only a method is added, no state, so a SEM can be promoted to this class after
    it is built and its draw is bit-identical. Any metric that then moves moved
    because the extent array travelled.
    """

    def extent(self, X) -> np.ndarray:
        return np.full(len(X), STUB_EXTENT)


def leg_ix(draw, chain, seed, **toggles):
    print("(ix) the extent array travels SEM -> SweepData -> evaluate_queries")
    orch = build("simulation", draw, chain, seed, **toggles)
    point_factory = orch._sem_factory

    def set_factory():
        sem = point_factory()
        sem.__class__ = SetTargetSEM
        return sem

    def sweep(factory):
        """One gamma step at the oracle budget, through the production strategy."""
        orch._sem_factory = factory
        set_seed(seed)
        runner = orch.get_sweep_runner_cls("gamma")(
            methods=orch.methods,
            method_factory=orch.build_methods,
            param_grid_override=np.asarray([1.0]),
            **orch._get_clean_kwargs(),
        )
        set_seed(seed)
        data = runner.generate_data(0, 1.0)
        set_seed(seed)
        _, results, _ = runner.run("a53 extent fill")
        return data, results

    try:
        point_data, point = sweep(point_factory)
        set_data, wide = sweep(set_factory)
    finally:
        orch._sem_factory = point_factory

    n_queries = len(point_data.X_test)
    check(
        "(ix) a point target fills extent with zeros",
        point_data.extent is not None and np.array_equal(point_data.extent, np.zeros(n_queries)),
        f"{np.shape(point_data.extent)}",
    )
    check(
        "(ix) and reaches the metrics as zero",
        not np.any(point_data.metric_extent),
        f"max {float(np.max(np.abs(point_data.metric_extent))):g}",
    )
    # X is needed since the Z carrier: `__post_init__` spells a None Z as (n, 0)
    check("(ix) an unfilled field still reads 0.0", SweepData(np.zeros((2, 1)), *([None] * 5)).metric_extent == 0.0)
    arrived = set_data.extent
    check(
        "(ix) a set target fills extent with its own half-width",
        arrived is not None and np.array_equal(arrived, np.full(n_queries, STUB_EXTENT)),
        f"{None if arrived is None else (np.shape(arrived), float(np.min(arrived)), float(np.max(arrived)))}",
    )
    check(
        "(ix) and reaches the metrics unchanged",
        np.array_equal(np.asarray(set_data.metric_extent), np.full(n_queries, STUB_EXTENT)),
        f"{np.asarray(set_data.metric_extent).ravel()[:2]}",
    )

    for name in point:
        scalar = {m: (float(point[name][m][0, 0]), float(wide[name][m][0, 0])) for m in HASHED_METRICS}
        print(f"      {name:7s} " + "  ".join(f"{m} {a:.4f}->{b:.4f}" for m, (a, b) in scalar.items()))
        cov_point, cov_set = scalar["coverage"]
        err_point, err_set = scalar["approximation_error"]
        check(f"(ix) {name}: coverage strictly falls", cov_set < cov_point, f"{cov_point:.4f} -> {cov_set:.4f}")
        check(
            f"(ix) {name}: approximation error strictly rises",
            err_set > err_point,
            f"{err_point:.4g} -> {err_set:.4g}",
        )
        # `worst_error` is a sup over the ESTIMATE's endpoints against the target
        # CENTRE, so it reads `f` whatever the extent is (SS5)
        worst_point, worst_set = scalar["worst_error"]
        check(f"(ix) {name}: worst_error still reads the centre", worst_set == worst_point, f"{worst_point:.6g}")


# =============================================================================
# LEG (viii): THE do-MNIST CLASS
# =============================================================================


def leg_viii():
    print("(viii) the do-MNIST SEM inherits extent -- class only, no dataset, no construction")
    from src.sem.do_mnist import DoMNISTSEM

    inherited = getattr(DoMNISTSEM, "extent", None)
    check("(viii) extent comes from the base class", inherited is not None and inherited is BASE_EXTENT)
    check(
        "(viii) the signature is (self, X)",
        BASE_EXTENT is not None and list(inspect.signature(BASE_EXTENT).parameters) == ["self", "X"],
    )
    if inherited is None:
        check("(viii) it returns zeros of the query count", False, "no `extent` on the class")
        return
    stub = object.__new__(DoMNISTSEM)  # no __init__: nothing is downloaded or read
    out = inherited(stub, np.zeros((3, 2)))
    check("(viii) it returns zeros of the query count", out.shape == (3,) and not np.any(out), f"{out!r}")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    os.chdir(REPO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--micro", action="store_true", help="shrink leg (v)'s fit sample")
    parser.add_argument("--record", action="store_true", help="print leg (i)'s digests and exit")
    args = parser.parse_args()
    toggles, draw, chain = configured()

    if args.record:
        for key, value in sorted(regression_digests(draw, chain, args.seed, **toggles).items()):
            print(f'    "{key}": "{value}",')
        sys.exit(0)

    os.makedirs(TMPROOT, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="a53_", dir=TMPROOT)
    leg_i(regression_digests(draw, chain, args.seed, **toggles))
    orchs = {name: build(name, draw, chain, args.seed, **toggles) for name in ("simulation", "optical_device")}
    leg_ii(orchs)
    leg_iii()
    leg_iv()
    leg_v(args.seed, args.micro, **toggles)
    leg_vi(tmp)
    leg_vii(orchs, args.seed)
    leg_viii()
    leg_ix(draw, chain, args.seed, **toggles)

    if not FAIL:
        shutil.rmtree(tmp, ignore_errors=True)
        print("A53 PASS")
    else:
        print(f"A53 FAIL: {FAIL} (log in {tmp})")
        sys.exit(1)
