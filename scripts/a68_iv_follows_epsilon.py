"""A68: the T-as-IV budget follows the swept ratio, on a grid centred on r = 1.

refactor11 makes two changes the epsilon figure is about. `fit_epsilon_iv` gains a
`ratio`, and `EpsilonRatioStrategy.get_predict_kwargs` passes `r eps_iv* + EPS_TOL`
beside the epsilon it already passed, unguarded, so a DA+ method with a T
constraint re-solves per grid point instead of only padding. And the `epsilon`
spec gets its own grid function, log-symmetric about 1 with 1 exactly ON it, so
both halves of the robustness axis are on the figure. Legs:

  (i)   r = 1 in BOTH directions: where the fit-time guard does not fire the
        predict kwarg equals `fit_epsilon_iv(0, 0, data)` bit for bit, and where it
        DOES fire they differ, the predict one staying raw. Catches: the ratio
        applied twice, `data` leaking into the predict-time call.
  (ii)  r != 1 moves the T budget and ONLY it: the per-step values are
        `r eps_iv* + EPS_TOL` exactly and `fit_epsilon_iv_z` and `gamma_z` are
        identical at every r; on the plasmode cigarette recipe a fitted
        `DA+PI+IV(T)` reads different widths at r = 0.5 and r = 2 while
        `DA+PI+IV(Z)` moves only by the padding. Catches: the kwarg dropped.
  (iii) `solves_on_epsilon` per mode on FITTED models. Catches: the property
        keyed on `_has_iv` rather than `_has_t`.
  (iv)  perf cumulation: `_prepare` call counts per method on a 4-point grid handed
        in directly. Catches: the class attribute put back (the T rows read 3).
  (v)   do-MNIST, STATIC ONLY: signatures and source text, no net, no data, no run.
  (vi)  the grid: odd length, exactly one 1.0 at the midpoint, log-symmetric, 0.5
        to 2.0, strictly increasing, and `gamma` still on `_RATIO_GRID`.
  (vii) R5.2, the ordering: no `+IV` family more infeasible than `PI+INV` at the
        same grid point, against RECORDED integers so a family that empties at a
        NEW point fails even when the ordering still holds.
  (viii) ruff and ASCII on the touched files.
  (ix)  the omega sweep refits the T budget per step, not only the epsilon sweep.
        On OPTICAL, the one omega dataset whose h_* is not exactly invariant: the
        returned budget moves across knobs, is not the frozen setup one, and is
        `ratio * raw + EPS_TOL` with the tolerance unscaled. Then on simulation,
        which carries an observed instrument: r_Z does not follow the knob, and
        the recorded fact that bounds the fix -- eps_iv* is 0 to machine
        precision at every knob there, so that panel cannot move. Catches:
        `ExpansionStrategy.fit_epsilon_iv` missing, so the budget falls back to
        the setup-time oracle.

    MPLBACKEND=Agg python scripts/a68_iv_follows_epsilon.py [--only LEG]
"""

import argparse
import inspect
import os
import subprocess
import sys

import numpy as np
from loguru import logger

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import a25_floor_guard as a25  # noqa: E402
import a59_sim_iv as a59  # noqa: E402
import a60_cigarettes_iv as a60  # noqa: E402
import a64_perf_aggregate as a64  # noqa: E402
import digest_leg  # noqa: E402
from munch import munchify  # noqa: E402

import src.experiments.perf as perf  # noqa: E402
from src.experiments.configs import (  # noqa: E402
    _EPSILON_RATIO_GRID,
    _RATIO_GRID,
    EPS_TOL,
    FLOOR_GUARD_R,
    PARAM_SPECS,
    resolve_dataset_block,
)
from src.experiments.generic_runner import ExpansionStrategy  # noqa: E402
from src.experiments.simulation import SimulationOrchestrator  # noqa: E402
from src.experiments.utils import set_seed  # noqa: E402
from src.experiments.utils.constants import IV_MODE_METHODS, REAL_Z_METHODS  # noqa: E402
from src.experiments.utils.metrics import STATUS_CATEGORIES  # noqa: E402
from src.methods.sensitivity_models import constraint_floor  # noqa: E402
from src.oracle import eps_iv_star  # noqa: E402

# the recorded infeasible-query counts per (method, grid point). Measured on this
# tree at the fixtures leg (vii) names; a family that empties at a NEW point fails
# (a) even where the ordering of (b) still holds
ORDERING = {
    # sim `iv: 0`, 51 queries: PI+INV is refuted over the whole under-budget half
    # and the T family enters one grid point earlier
    "simulation iv: 0": {
        "PI": [0] * 9,
        "PI+INV": [51, 51, 51, 51, 0, 0, 0, 0, 0],
        "DA+PI+IV": [51, 51, 51, 0, 0, 0, 0, 0, 0],
    },
    # the plasmode recipe, 245 queries: every family that carries a constraint is
    # refuted at r = 0.5 alone, where PI+INV is too, and feasible above it. The
    # boundary ratio is per-draw, so a production run can move it
    "cigarettes plasmode": {
        "PI": [0] * 9,
        "PI+INV": [245, 0, 0, 0, 0, 0, 0, 0, 0],
        "PI+IV": [0] * 9,
        "DA+PI+IV": [245, 0, 0, 0, 0, 0, 0, 0, 0],
        "DA+PI+IV(T)": [245, 0, 0, 0, 0, 0, 0, 0, 0],
        "DA+PI+IV(Z)": [0] * 9,
    },
}
FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAIL.append(name)


def sim_runner(methods=None, **overrides):
    """The simulation `iv: 4` recipe on the epsilon strategy, cut to gate scale."""
    block = resolve_dataset_block("simulation", a59.recipe_block())
    reduced = {**block, "n_experiments": 1, "n_samples": 512, "sweep_samples": 8, "n_jobs": 1, **overrides}
    if methods is not None:
        reduced["methods"] = list(methods)
    set_seed(reduced["seed"])
    orch = SimulationOrchestrator(**reduced, hyperparameters=munchify(digest_leg.HYPERPARAMETERS))
    return orch.get_sweep_runner_cls("epsilon")(
        methods={k: v for k, v in orch.methods.items() if k != "ATE"},
        method_factory=orch.build_methods,
        **orch._get_clean_kwargs(),
    )


def plasmode_runner(**overrides):
    """The NEW plasmode recipe on the epsilon strategy, cut to gate scale."""
    block = a60.recipe_block(n_experiments=1, sweep_samples=8, n_jobs=1, **overrides)
    orch = a60.orchestrator(block)
    return a60.sweep_runner(orch, "epsilon")


def plasmode_block(**overrides):
    import yaml

    with open(os.path.join(REPO, "recipes", "robustnessFig11.yaml")) as handle:
        config = yaml.safe_load(handle)
    defaults = config.pop("defaults", {}) or {}
    block = {**defaults, **config["cigarettes"], **overrides}
    block.pop("experiment", None)
    return resolve_dataset_block("cigarettes", block)


def plasmode_sweep(**overrides):
    block = plasmode_block(n_experiments=1, sweep_samples=8, n_jobs=1, **overrides)
    orch = a60.orchestrator(block)
    return a60.sweep_runner(orch, "epsilon")


def infeasible_counts(runner, label):
    """Infeasible queries per method per grid point, through the REAL runner."""
    x, _, statuses = runner.run(label)
    index = list(STATUS_CATEGORIES).index("infeasible")
    return x, {name: statuses[name][:, 0, index].astype(int).tolist() for name in statuses}


# ------------------------------------------------------------------ legs


def leg_i():
    print("(i) r = 1, in both directions")
    runner = sim_runner()
    data = runner.generate_data(0, 1.0)
    runner.build_models(0, 0, data)

    def raw_star():
        return float(runner.get_oracle(0).eps_iv_star)

    raw = raw_star() + EPS_TOL
    predict = runner.get_predict_kwargs(1.0, 0)["epsilon_iv"]
    check("(i) the predict kwarg at r = 1 is eps_iv_star + EPS_TOL bit for bit", predict == raw, f"{predict!r}")

    floor = constraint_floor(
        data.GX,
        data.y,
        runner.fit_gamma(0),
        kind="iv",
        Z=np.asarray(data.G).reshape(len(data.GX), -1),
        mean_match=runner.mean_match,
        rho=runner.fit_rho(0, data),
        recalibrate=runner.recalibrate,
    )
    print(f"      RECORDED: r_T^2 {raw**2:.6g} against the T floor {floor:.6g}")
    check("(i) on this fixture the fit-time guard does NOT fire", raw**2 >= floor, f"{raw**2:.4g} >= {floor:.4g}")
    check("(i) so the fitted budget equals it too", runner.fit_epsilon_iv(0, 0, data) == raw)

    # the second case, with the guard FIRED. The epsilon strategy retunes the DA,
    # so eps_iv_star is large and the r = 1 budget clears the floor; what does NOT
    # clear it is the same pipeline at a small ratio, which is the sweep's left
    # half. There the two calls must disagree: guarded with data, raw without
    guarded_ratio = 0.5
    under = guarded_ratio * raw_star() + EPS_TOL
    check("(i) at r = 0.5 the assumed budget is under the T floor", under**2 < floor, f"{under**2:.4g} < {floor:.4g}")
    with_data = runner.fit_epsilon_iv(0, 0, data, ratio=guarded_ratio)
    without = runner.fit_epsilon_iv(0, ratio=guarded_ratio)
    guarded = float(np.sqrt(FLOOR_GUARD_R * max(floor, 0.0)))
    check("(i) with data the guard raises it to sqrt(9 floor)", abs(with_data - guarded) < 1e-12, f"{with_data:.6g}")
    check("(i) without data it is the raw r * eps_iv_star + EPS_TOL", without == under, f"{without!r}")
    check("(i) so the two disagree, which is the rule", with_data != without, f"{with_data:.6g} vs {without:.6g}")
    step = runner.get_predict_kwargs(guarded_ratio, 0)["epsilon_iv"]
    check("(i) and the predict kwarg is the raw one", step == under, f"{step!r}")


def leg_ii():
    print("(ii) r != 1 moves the T budget and only it")
    runner = sim_runner()
    data = runner.generate_data(0, 1.0)
    runner.build_models(0, 0, data)
    star = float(runner.get_oracle(0).eps_iv_star)
    # any class carrying the OBSERVED instrument's constraint declares gamma_z: the
    # (Z)-only spellings, plus the bare IV_MODE_METHODS defaults, whose mode is (T,Z)
    # and so carries Z beside T. The recipe decides which of them run, so the gate
    # takes the first it lists rather than pinning PI+IV
    z_constrained = REAL_Z_METHODS | frozenset(IV_MODE_METHODS)
    witness = next((name for name in runner.methods if name in z_constrained), None)
    check("(ii) the block lists a +IV class carrying gamma_z", witness is not None, f"{sorted(runner.methods)}")
    if witness is None:
        return
    z_budget, gamma_z = runner.fit_epsilon_iv_z(0, data), runner.methods[witness]().gamma_z
    grid = np.asarray(runner.get_param_range(), dtype=float)
    exact = all(runner.get_predict_kwargs(r, 0)["epsilon_iv"] == float(r) * star + EPS_TOL for r in grid)
    check("(ii) every per-step T budget is r * eps_iv_star + EPS_TOL exactly", exact)
    # the real claim is that the swept kwargs carry the T budget and NOTHING else:
    # a Z budget or a gamma_z among them would move the observed instrument's radius
    keys = {frozenset(runner.get_predict_kwargs(r, 0)) for r in grid}
    check(
        "(ii) every step passes exactly {epsilon, epsilon_iv}",
        keys == {frozenset({"epsilon", "epsilon_iv"})},
        f"{keys}",
    )
    check(
        "(ii) and the Z budget and gamma_z are what they were",
        runner.fit_epsilon_iv_z(0, data) == z_budget and runner.methods[witness]().gamma_z == gamma_z,
    )
    epsilons = [runner.get_predict_kwargs(r, 0)["epsilon"] for r in grid]
    check("(ii) the epsilon beside it still moves", len(set(epsilons)) == len(grid))

    # on the plasmode recipe, through the real runner: a T-mode method moves with
    # the ratio and a (Z)-mode one does not re-solve at all
    cig = plasmode_sweep(methods=["PI", "DA+PI+IV(T)", "DA+PI+IV(Z)"])
    x, results, _ = cig.run("a68 (ii)")
    x = np.asarray(x, dtype=float)
    widths = {name: np.asarray(results[name]["interval_width"]).ravel() for name in results}
    print(f"      RECORDED plasmode widths over r in [{x[0]:.3f}, {x[-1]:.3f}]:")
    for name, v in widths.items():
        print(f"        {name:14s} {np.round(v, 4).tolist()}")
    eps_star = float(cig.get_oracle(0).epsilon_star)
    middle = int(np.argmin(np.abs(x - 1.0)))
    # a method with no T constraint does not re-solve: every width it reports is
    # its r = 1 width plus the padding 2 (r - 1) eps*, exactly
    padding = 2.0 * (x - x[middle]) * eps_star
    z_row = widths["DA+PI+IV(Z)"]
    z_gap = float(np.nanmax(np.abs(z_row - z_row[middle] - padding)))
    check("(ii) DA+PI+IV(Z) moves by the padding and nothing else", z_gap < 1e-3, f"max |residual| {z_gap:.2e}")
    t_row = widths["DA+PI+IV(T)"]
    t_gap = float(np.nanmax(np.abs(t_row - t_row[middle] - padding)))
    check("(ii) DA+PI+IV(T) does NOT: it re-solves at the swept T budget", t_gap > 0.1, f"max |residual| {t_gap:.4f}")
    monotone = bool(np.all(np.diff(t_row[~np.isnan(t_row)]) > 0))
    check("(ii) and it is monotone in the ratio", monotone, f"{t_row[-1]:.4f}")
    check("(ii) PI is flat", float(np.nanmax(widths["PI"]) - np.nanmin(widths["PI"])) < 1e-9)


def leg_iii():
    print("(iii) solves_on_epsilon per mode, on fitted models")
    import a67_iv_decoupled as a67

    _, X, y, GX, G, Z = a67.fixture(42)
    wants = {
        "PI": False,
        "PI+IV": False,
        "PI+INV": True,
        "PI+INV+IV": True,
        "DA+PI+IV": True,
        "DA+PI+IV(T)": True,
        "DA+PI+IV(Z)": False,
        "PI&DA+PI+IV": True,
        "PI&DA+PI+IV(Z)": False,
    }
    for tag, instrument in (("a real Z", Z), ("an empty Z", None)):
        models = a67.built(list(wants), X, y, GX, G, instrument)
        for name, want in wants.items():
            got = bool(models[name].solves_on_epsilon)
            check(f"(iii) {tag}: {name} solves_on_epsilon {want}", got == want, f"{got}")


def leg_iv():
    print("(iv) perf cumulation: _prepare calls per method on a 4-point grid")
    expected = {
        ("PI",): 3,
        ("PI+INV",): 12,
        ("DA+PI+IV(T)",): 12,
        ("DA+PI+IV(Z)",): 3,
        ("PI&DA+PI+IV(T)",): 24,
        tuple(a64.TEN): 102,
    }
    for methods, want in expected.items():
        runner = a64.perf_runner(
            a64.sim_block(methods, n_samples=512, treatment_dim=32, pad=True, clipy=False, n_jobs=-1)
        )
        # the grid is handed in directly, so the spec's own length does not enter
        x = np.array([0.5, 0.8, 1.0, 2.0])
        data = runner.generate_data(0, x[0])
        with a64.prepare_spy() as spy:
            results, _ = perf.wall_clock(runner, data, x, repeats=3, seconds_per_solve=1.0)
        label = f"{list(methods)}" if len(methods) < 5 else f"the {len(methods)}-method list"
        check(f"(iv) {label}: {want} _prepare calls", spy.count == want, f"{spy.count}")
        monotone = all(np.all(np.diff(v[:, 0]) >= -1e-12) for v in results.values())
        check(f"(iv) {label}: cumulative arrays non-decreasing", monotone)
        if "DA+PI+IV(T)" in methods:
            v = results["DA+PI+IV(T)"][:, 0]
            check("(iv) DA+PI+IV(T)'s last > its first", v[-1] > v[0], f"{v[0]:.3f} -> {v[-1]:.3f} s")


def leg_v():
    print("(v) do-MNIST, STATIC ONLY: no net, no data, no run")
    import src.experiments.do_mnist as do_mnist
    import src.methods.partial_r2_net as nets

    params = inspect.signature(do_mnist.DoMNISTMixin.fit_epsilon_iv).parameters
    check(
        "(v) DoMNISTMixin.fit_epsilon_iv takes ratio, default 1.0", "ratio" in params and params["ratio"].default == 1.0
    )
    precompute = inspect.signature(nets.IVConstrainedPartialR2Net._precompute).parameters
    check("(v) IVConstrainedPartialR2Net._precompute takes T", "T" in precompute)
    check("(v) and still accepts Z", "Z" in precompute)
    source = inspect.getsource(nets.IntersectedIVPartialR2Net._fit_branches)
    check("(v) IntersectedIVPartialR2Net fits its DA branch with T=G", "T=G" in source and "Z=G" not in source)


def leg_vi():
    print("(vi) the recentred grid")
    check("(vi) the epsilon spec is on _EPSILON_RATIO_GRID", PARAM_SPECS["epsilon"].grid_fn is _EPSILON_RATIO_GRID)
    check("(vi) and gamma is still on _RATIO_GRID", PARAM_SPECS["gamma"].grid_fn is _RATIO_GRID)
    for n in (1, 2, 3, 4, 8, 9, 16, 32, 128):
        grid = _EPSILON_RATIO_GRID("simulation", n)
        middle = len(grid) // 2
        ok = (
            len(grid) == (n | 1)
            and int(np.count_nonzero(grid == 1.0)) == 1
            and grid[middle] == 1.0
            and np.allclose(grid * grid[::-1], 1.0, atol=1e-12)
            and bool(np.all(np.diff(grid) > 0))
            # n = 1 is the degenerate grid [1.0]: symmetric and centred, no edges
            and (len(grid) == 1 or (grid[0] == 0.5 and grid[-1] == 2.0))
        )
        check(f"(vi) n = {n}: length {n | 1}, one exact 1.0 at the midpoint, symmetric on [0.5, 2]", ok, f"{len(grid)}")
    check("(vi) the vline is still at 1.0 and inside the range", PARAM_SPECS["epsilon"].vlines == (1.0,))
    gamma = _RATIO_GRID("simulation", 8)
    want = np.geomspace(2**-6, 1.0, num=8)
    check("(vi) the gamma grid is unchanged at n = 8", np.array_equal(gamma, want), f"{np.round(gamma, 5).tolist()}")


def leg_vii():
    print("(vii) R5.2: no +IV family more infeasible than PI+INV at the same ratio")
    fixtures = (
        ("simulation iv: 0", lambda: sim_runner(methods=["PI", "PI+INV", "DA+PI+IV"], iv=0)),
        (
            "cigarettes plasmode",
            lambda: plasmode_sweep(methods=["PI", "PI+INV", "PI+IV", "DA+PI+IV", "DA+PI+IV(T)", "DA+PI+IV(Z)"]),
        ),
    )
    for label, make in fixtures:
        runner = make()
        x, counts = infeasible_counts(runner, f"a68 (vii) {label}")
        print(f"      RECORDED {label}, grid {np.round(np.asarray(x, dtype=float), 3).tolist()}:")
        for name, row in counts.items():
            print(f"        {name:14s} {row}")
        recorded = ORDERING.get(label)
        if recorded is None:
            check(f"(vii) {label}: no recorded table yet", False, "record the printed rows in ORDERING")
            continue
        check(f"(vii) {label}: every count equals the recorded integer", counts == recorded, f"{counts}")
        baseline = counts.get("PI+INV")
        ordered = all(
            all(a <= b for a, b in zip(row, baseline, strict=True))
            for name, row in counts.items()
            if "IV" in name and name != "PI+INV"
        )
        check(f"(vii) {label}: no +IV family more infeasible than PI+INV", ordered)
        check(f"(vii) {label}: PI+INV has a feasible point, so that is not vacuous", min(baseline) == 0, f"{baseline}")


def leg_viii():
    print("(viii) ruff and ASCII on the touched files")
    ruff = os.path.join(os.path.dirname(sys.executable), "ruff")
    for command in (["check", "."], ["format", "--check", "."]):
        done = subprocess.run([ruff] + command, cwd=REPO, capture_output=True, text=True)
        check(
            f"(viii) ruff {' '.join(command)} clean", done.returncode == 0, str(done.stdout.strip().splitlines()[-1:])
        )
    added = subprocess.run(
        ["git", "diff", "--unified=0", "c566b76", "--", "src", "config.yaml", "recipes", "scripts"],
        cwd=REPO,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    bad = [
        line[:70]
        for line in added
        if line.startswith("+") and not line.startswith("+++") and (not line.isascii() or "\u2014" in line)
    ]
    check("(viii) the added lines are ASCII, no em dash", not bad, str(bad[:2]))


def leg_ix():
    print("(ix) the omega sweep refits the T budget per step")

    def budgets(dataset):
        runner = a25.omega_recipe_runner(dataset, steps=4)
        raw, budget, z_budget, eps, expect = [], [], [], [], []
        for index, knob in enumerate(runner.get_param_range()):
            data = runner.generate_data(0, knob)
            raw.append(float(runner._step_epsilon_iv[0]))
            eps.append(float(runner._step_epsilon[0]) - EPS_TOL)
            # no data: `_floor_guard` is a no-op, so this is the budget itself
            budget.append(float(runner.fit_epsilon_iv(0, index)))
            z_budget.append(float(runner.fit_epsilon_iv_z(0, data)))
            # the budget restated from the runner's own state, independent of
            # what `generate_data` chose to pass. `preserve_rng` makes the draw
            # reproducible, so this is an equality and not a tolerance
            expect.append(
                float(
                    eps_iv_star(
                        runner.sems[0],
                        runner.das[0],
                        X=runner._base_data(0)[0],
                        features=runner._features,
                        mean_match=runner.mean_match,
                        **runner.augment_kwargs_fn(knob),
                    )[0]
                )
            )
        check(
            f"(ix) {dataset}: the cache is eps_iv* at the step's X, kwargs and mean_match",
            all(r == e for r, e in zip(raw, expect, strict=True)),
            f"{np.round(raw, 8).tolist()} vs {np.round(expect, 8).tolist()}",
        )
        # ||E[W#|T]|| <= ||W#|| <= ||W|| = eps*, and only if both budgets read the
        # same draw of the same X -- which is the whole point of the shared local
        check(
            f"(ix) {dataset}: 0 <= eps_iv* <= eps* at every knob",
            all(0.0 <= r <= e for r, e in zip(raw, eps, strict=True)),
            f"{np.round(raw, 8).tolist()} vs {np.round(eps, 8).tolist()}",
        )
        return runner, raw, budget, z_budget, eps

    runner, raw, budget, z_budget, eps = budgets("optical_device")
    frozen = float(runner.get_oracle(0).eps_iv_star)
    print(f"      RECORDED optical frozen eps_iv* {frozen:.6g}, per-step {np.round(raw, 6).tolist()}")
    check("(ix) the omega runner IS an ExpansionStrategy", isinstance(runner, ExpansionStrategy))

    # MOVES: EPS_TOL (2^-5) is far bigger than the optical budget itself, so the
    # threshold is absolute and well clear of solver noise, which is ~1e-12 here
    spread = float(np.max(budget) - np.min(budget))
    check(f"(ix) the refit budget moves across knobs (spread {spread:.4g}, frozen {frozen:.4g})", spread > 1e-3)
    moved = float(np.max(np.abs(np.asarray(budget) - (frozen + EPS_TOL))))
    check(f"(ix) and it is not the frozen setup budget (max gap {moved:.4g})", moved > 1e-3)
    check(
        "(ix) each step is its own raw budget + EPS_TOL, the tolerance unscaled",
        all(b == r + EPS_TOL for b, r in zip(budget, raw, strict=True)),
        f"{np.round(budget, 6).tolist()}",
    )
    ratio = 0.5
    # only the LAST step's cache survives the loop, so compare against it
    scaled = float(runner.fit_epsilon_iv(0, len(raw) - 1, ratio=ratio))
    check(
        "(ix) ratio scales the budget and not the tolerance",
        scaled == ratio * raw[-1] + EPS_TOL,
        f"{scaled!r} vs {ratio * raw[-1] + EPS_TOL!r}",
    )
    # edit (a): `eps_iv_star` must FORWARD its augment kwargs. No fixture in the
    # repo makes that observable -- optical's DA persists `p` and `eps_iv*` is 0
    # on the other two -- so it is pinned at the call itself, with a DA that
    # records what it was handed
    seen = {}

    def recording_da(X, **kwargs):
        seen.update(kwargs)
        return X + 1.0, np.ones((len(X), 1))

    class _StubSEM:
        def f(self, X):
            return np.asarray(X).sum(axis=1, keepdims=True)

    eps_iv_star(_StubSEM(), recording_da, X=np.eye(8), mean_match=False, scale=0.25)
    check("(ix) eps_iv_star forwards its augment kwargs to the DA", seen == {"scale": 0.25}, f"{seen}")

    resolved = type(runner).fit_epsilon_iv
    source = inspect.getsource(resolved)
    check(
        "(ix) the override is the one in force, and it guards on the declared path",
        resolved is ExpansionStrategy.fit_epsilon_iv and "declared=self.declared_iv" in source,
        f"{resolved.__qualname__}",
    )

    # RECORDED, because it bounds what this fix can do: on the simulation SEM
    # h_* is exactly invariant under `translate`, so eps_iv* is 0 to machine
    # precision at EVERY knob and the T budget is pure EPS_TOL before and after
    # the refit. The sim and cigarettes omega panels therefore do NOT move; only
    # optical does. Whatever bends those two panels, it is not a frozen budget.
    _, sim_raw, sim_budget, sim_z, _ = budgets("simulation")
    print(f"      RECORDED simulation per-step eps_iv* {sim_raw}")
    check(
        "(ix) on simulation h_* is exactly invariant, so the T budget is EPS_TOL at every knob",
        float(np.max(np.abs(sim_raw))) < 1e-9 and max(abs(b - EPS_TOL) for b in sim_budget) < 1e-12,
        f"{sim_budget}",
    )
    # simulation is the one of the two that carries an observed instrument, so
    # the decoupling is testable here and vacuous on optical (r_Z is 0 there)
    check(
        "(ix) the observed instrument's radius is non-zero and untouched by the knob",
        min(sim_z) > 0.0 and max(sim_z) - min(sim_z) == 0.0,
        f"{np.round(sim_z, 8).tolist()}",
    )


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    os.chdir(REPO)
    os.environ.setdefault("MPLBACKEND", "Agg")
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=None)
    args = parser.parse_args()
    legs = [
        ("i", leg_i),
        ("ii", leg_ii),
        ("iii", leg_iii),
        ("iv", leg_iv),
        ("v", leg_v),
        ("vi", leg_vi),
        ("vii", leg_vii),
        ("viii", leg_viii),
        ("ix", leg_ix),
    ]
    if args.only:
        legs = [(tag, leg) for tag, leg in legs if tag.lower() == args.only.lower()]
        if not legs:
            sys.exit(f"unknown leg {args.only!r}")
    for tag, leg in legs:
        try:
            leg()
        except Exception as error:  # a raise is a FAIL line, and the later legs still report
            check(f"({tag}) ran without raising", False, f"{type(error).__name__}: {error}")
    if not FAIL:
        print("A68 PASS")
    else:
        print(f"A68 FAIL: {FAIL}")
        sys.exit(1)
