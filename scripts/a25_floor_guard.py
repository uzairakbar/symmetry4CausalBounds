"""A25: the constraint floor, and no budget is ever raised to it.

The file keeps its name (a28 imports `cvxpy_floor` from it; a68 imports
`omega_recipe_runner`), but there is no floor guard any more: a budget under the
constraint's own floor is left as is and every query reads INFEASIBLE, the rule
PI+INV has always followed on the epsilon sweep (PLAN v16 SS2.2, SS5.9). Four legs:
  1. the closed-form floor equals a cvxpy reference solve;
  2. never raised: on the simulation m fixture and the optical gamma fixture, the
     fitted budgets WITH data equal the raw `oracle + EPS_TOL` bit for bit on
     feasible and infeasible cells alike, and an infeasible cell logs exactly one
     INFO BELOW line naming its floor (a feasible one logs none);
  3. empty reads INFEASIBLE: on the omega recipe fixture every knob with
     r_T^2 < floor gives all-INFEASIBLE statuses and a NaN width at RECORDED knob
     indices, its coverage is whatever `evaluate_queries` gives an all-NaN interval
     (NaN under the empty-cell rule, 0 without it), and the rendered width line has
     a gap there;
  4. completeness: `git grep` finds no floor guard in `src` or `scripts` beyond
     the two sentences allowed to name it.

    python scripts/a25_floor_guard.py [--only LEG]
"""

import argparse
import os
import subprocess
import sys
import warnings

import cvxpy as cp
import numpy as np
import yaml
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import src.experiments.utils.plotting as plotting  # noqa: E402
from src.experiments.base import SweepData  # noqa: E402
from src.experiments.configs import EPS_TOL, resolve_dataset_block  # noqa: E402
from src.experiments.optical_device import OpticalOrchestrator  # noqa: E402
from src.experiments.simulation import SimulationOrchestrator  # noqa: E402
from src.experiments.utils import set_seed  # noqa: E402
from src.experiments.utils.metrics import evaluate_queries  # noqa: E402
from src.methods.regression import LeastSquaresClosedForm as OLS  # noqa: E402
from src.methods.sensitivity_models import (  # noqa: E402
    SolveStatus,
    constraint_floor,
    inv_constraint_terms,
    iv_constraint_terms,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# the omega figure's recipe, named ONCE: a68 leg (ix) fits the same fixture
OMEGA_RECIPE = "sharpnessInformativenessFig10.yaml"
METHODS = ["PI", "DA+PI", "PI+INV", "DA+PI+IV"]
# leg 3, MEASURED on this fixture (simulation block, n 512, 8 knobs, one
# experiment): the knob indices where the per-step r_T^2 sits under the T floor.
# The floor guard used to raise the budget there; now those knobs read INFEASIBLE
EMPTY_KNOBS = [0, 1, 2]
# the two sentences allowed to name the floor guard (leg 4): do-MNIST's query
# comment, left alone, and the docstring that says it is gone
GUARD_ALLOWED = ("src/experiments/do_mnist.py", "there is no floor guard any more")
FAIL = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAIL.append(name)


def cvxpy_floor(design, y, gamma, kind, GX=None, Z=None, mean_match=True):
    """Independent reference for `constraint_floor`, solved by cvxpy. `gamma` is
    the budget the ball is solved at (sigma-hat sqrt(gamma) radius); pass the
    recalibrated one for a DA ball.

    `mean_match=True` states Lem. 2 EXPLICITLY where production eliminates it: the
    hypothesis carries a free intercept coordinate and the slice
    E_n[h(X)] = E_n[Y] is an equality constraint, rather than being folded into
    centred coordinates. Same set, different parameterisation and different
    solver -- which is what makes it a reference and not a re-run.
    """
    design = np.asarray(design)
    y = np.asarray(y)
    N, M = design.shape

    if not mean_match:
        h_erm = OLS().fit(design, y).solution.flatten()
        resid = y.flatten() - design @ h_erm
        scale = float(np.sqrt(np.mean(resid**2)))
        delta = np.sqrt(N) * scale * np.sqrt(gamma)
        A, b = inv_constraint_terms(design, GX) if kind == "inv" else iv_constraint_terms(design, y, Z)
        _, R = np.linalg.qr(design)
        h = cp.Variable(M)
        constraints = [cp.norm(cp.Constant(R) @ (h - cp.Constant(h_erm)), 2) <= delta]
    else:
        mu, ybar = design.mean(axis=0), float(np.mean(y))
        centred, y_centred = design - mu, y - ybar
        # the constraint terms are the production ones on the slice; only the
        # GEOMETRY (ball + equality) is restated in the explicit parameterisation
        if kind == "inv":
            A, b = inv_constraint_terms(centred, np.asarray(GX) - mu)
        else:
            A, b = iv_constraint_terms(centred, y_centred, Z)

        augmented = np.hstack([design, np.ones((N, 1))])
        h1_erm = np.linalg.lstsq(augmented, y.flatten(), rcond=None)[0]
        resid = y.flatten() - augmented @ h1_erm
        scale = float(np.sqrt(np.mean(resid**2)))
        delta = np.sqrt(N) * scale * np.sqrt(gamma)
        _, R = np.linalg.qr(augmented)
        h1 = cp.Variable(M + 1)
        h = h1[:M]
        constraints = [
            cp.norm(cp.Constant(R) @ (h1 - cp.Constant(h1_erm)), 2) <= delta,
            cp.Constant(augmented.mean(axis=0)) @ h1 == ybar,
        ]

    problem = cp.Problem(cp.Minimize(cp.norm(cp.Constant(A) @ h - cp.Constant(b), 2)), constraints)
    for solver in (cp.CLARABEL, cp.ECOS):
        try:
            problem.solve(solver=solver, verbose=False)
        except Exception:  # noqa: S112 - solver fallback chain, try the next one
            continue
        if problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            return float((problem.value / np.sqrt(N)) ** 2)
    return np.nan


def sim_runner(steps=12):
    set_seed(42)
    orch = SimulationOrchestrator(
        seed=42,
        n_samples=2048,
        n_experiments=1,
        sweep_samples=steps,
        kernel_dim=0,
        treatment_dim=32,
        methods=METHODS,
        hyperparameters={},
        n_jobs=8,
        recalibrate=True,
        pad=False,
        clipy=True,
    )
    return orch.get_sweep_runner_cls("omega")(
        methods=orch.methods,
        method_factory=orch.build_methods,
        **{k: v for k, v in orch.kwargs.items() if k != "methods"},
    )


def recipe_runner(dataset, param, n_experiments=1, steps=8, methods=None, **overrides):
    """A `param` runner on the omega figure's own `dataset` block, cut to gate scale:
    the n and m recipes are Fig10's blocks with another `param`, so one file serves
    every fixture here."""
    path = os.path.join(REPO, "recipes", OMEGA_RECIPE)
    if not os.path.exists(path):
        available = sorted(f for f in os.listdir(os.path.join(REPO, "recipes")) if f.endswith(".yaml"))
        raise FileNotFoundError(
            f"recipes/{OMEGA_RECIPE} is gone; a25 legs 2 and 3 and a68 leg (ix) fit their "
            f"fixtures on it. recipes/ carries {available}. Re-point OMEGA_RECIPE."
        )
    with open(path) as handle:
        config = yaml.safe_load(handle)
    defaults = config.pop("defaults", {}) or {}
    if dataset not in config:
        raise KeyError(f"recipes/{OMEGA_RECIPE} carries no `{dataset}:` block, only {sorted(config)}")
    block = {**defaults, **config[dataset]}
    block.pop("experiment", None)
    block.update(n_experiments=n_experiments, sweep_samples=steps, n_jobs=1, **overrides)
    if methods is not None:
        block["methods"] = list(methods)
    block = resolve_dataset_block(dataset, block)
    set_seed(block["seed"])
    Orchestrator = SimulationOrchestrator if dataset == "simulation" else OpticalOrchestrator
    orch = Orchestrator(**block, hyperparameters={})
    return orch.get_sweep_runner_cls(param)(
        methods={k: v for k, v in orch.methods.items() if k != "ATE"},
        method_factory=orch.build_methods,
        **orch._get_clean_kwargs(),
    )


def omega_recipe_runner(dataset="simulation", steps=8, methods=None, **overrides):
    """An omega runner on the omega figure's own block, cut to gate scale (n 512,
    one experiment).

    The one definition of that fixture: a68 leg (ix) imports this rather than
    keeping a second copy, so `OMEGA_RECIPE` is the only place the file name
    appears. A rename must land as a named FAIL, never as a traceback out of a
    gate that then looks merely broken.

    `sim_runner` above is a hand-rolled orchestrator with no instrument, no
    padding and no mean matching, and on it the T floor sits four orders of
    magnitude under the budget at every knob. The recipe block does go under
    the floor over part of the grid, which is what leg 3 reads.
    """
    return recipe_runner(dataset, "omega", steps=steps, methods=methods, **{"n_samples": 512, **overrides})


def cell_floor(runner, e, data, kind):
    """The floor `_floor_report` measures, restated: 'inv' on (X, GX) at the plain
    ball, 'iv' on the T constraint alone (GX, G) at the recalibrated one."""
    kw = dict(GX=data.GX) if kind == "inv" else dict(Z=data.G)
    design = data.X if kind == "inv" else data.GX
    if kind == "iv":  # the DA ball is recalibrated, as in `_floor_report`
        kw.update(rho=runner.fit_rho(e, data), recalibrate=runner.recalibrate)
    return constraint_floor(design, data.y, runner.fit_gamma(e), kind=kind, mean_match=runner.mean_match, **kw)


def printed_floor(message):
    """The floor an INFO BELOW line prints, `... < floor X); ...`; NaN if none."""
    if "< floor " not in message:
        return np.nan
    return float(message.split("< floor ")[1].split(")")[0])


# ------------------------------------------------------- 1. closed form == solver


def a25_closed_form():
    runner = sim_runner(steps=4)
    for mean_match in (True, False):
        worst = 0.0
        for knob in runner.get_param_range():
            data = SweepData.coerce(runner.generate_data(0, knob))
            for gamma in (0.05, 0.505):
                for kind, kw in (("inv", dict(GX=data.GX)), ("iv", dict(Z=data.G))):
                    design = data.X if kind == "inv" else data.GX
                    got = constraint_floor(design, data.y, gamma, kind=kind, mean_match=mean_match, **kw)
                    want = cvxpy_floor(design, data.y, gamma, kind, mean_match=mean_match, **kw)
                    # floors are SQUARED, so 1e-12 here is a 1e-6 budget, i.e. zero;
                    # a relative test there compares solver noise with solver noise
                    if abs(got - want) < 1e-11:
                        continue
                    worst = max(worst, abs(got - want) / max(abs(want), 1e-300))
        label = "on the Lem. 2 slice" if mean_match else "uncentred"
        check(f"A25 closed-form floor == cvxpy, {label}", worst < 1e-4, f"worst rel {worst:.2e}")


# -------------------------------------------------------------- 2. never raised


def optical_gamma_runner():
    set_seed(69)
    orch = OpticalOrchestrator(
        seed=69,
        n_samples=200,
        n_experiments=2,
        sweep_samples=3,
        methods=METHODS,
        hyperparameters={},
        n_jobs=1,
        recalibrate=True,
        pad=False,
        clipy=True,
        augmentation="rotation > hflip > vflip > gaussian-noise",
    )
    return orch.get_sweep_runner_cls("gamma")(
        methods=orch.methods,
        method_factory=orch.build_methods,
        **{k: v for k, v in orch.kwargs.items() if k != "methods"},
    )


def never_raised(label, runner, steps):
    """Every (experiment, step, kind) cell of `runner` at the step indices `steps`:
    the budget WITH data is the raw oracle + EPS_TOL bit for bit, and exactly the
    infeasible cells log one INFO BELOW line naming their floor. Returns the
    (feasible, infeasible) cell counts."""
    grid = runner.get_param_range()
    records = []
    sink = logger.add(lambda message: records.append(message.record), level="DEBUG")
    feasible = infeasible = 0
    moved, unlogged, logged_feasible = [], [], []
    try:
        for e in range(runner.n_experiments):
            oracle = runner.get_oracle(e)
            for i in steps:
                data = SweepData.coerce(runner.generate_data(e, grid[i]))
                for kind, attr, fit in (
                    ("inv", "epsilon_star", runner.fit_epsilon),
                    ("iv", "eps_iv_star", runner.fit_epsilon_iv),
                ):
                    raw = getattr(oracle, attr, None)
                    if raw is None or not np.isfinite(raw):
                        continue
                    raw = float(raw) + EPS_TOL
                    floor = cell_floor(runner, e, data, kind)
                    records.clear()
                    got = fit(e, i, data)
                    lines = [r for r in records if "BELOW" in r["message"]]
                    if got != raw:
                        moved.append(f"exp {e} step {i} {kind}: {raw!r} -> {got!r}")
                    if raw**2 >= floor:
                        feasible += 1
                        if lines:
                            logged_feasible.append(f"exp {e} step {i} {kind}")
                        continue
                    infeasible += 1
                    named = (
                        len(lines) == 1
                        and lines[0]["level"].name == "INFO"
                        and "never raised" in lines[0]["message"]
                        and abs(printed_floor(lines[0]["message"]) - floor) <= 6e-4 * max(floor, 1e-12)
                    )
                    if not named:
                        unlogged.append(f"exp {e} step {i} {kind}: {len(lines)} lines")
    finally:
        logger.remove(sink)
    print(f"      {label}: {feasible} feasible, {infeasible} infeasible cells")
    check(f"A25 {label}: every budget is the raw oracle + EPS_TOL, bit for bit", not moved, f"{moved[:3]}")
    check(
        f"A25 {label}: every infeasible cell logs one INFO BELOW line naming its floor", not unlogged, f"{unlogged[:3]}"
    )
    check(f"A25 {label}: no feasible cell logs a BELOW line", not logged_feasible, f"{logged_feasible[:3]}")
    return feasible, infeasible


def a25_never_raised():
    """No budget moves, feasible or not. The sim m fixture is the grid where the
    oracle INV budget sits under its floor in most cells (PLAN v16 SS2.2); the
    optical gamma one is where both budgets clear it."""
    try:
        m_runner = recipe_runner("simulation", "m", n_experiments=2, steps=16)
    except (FileNotFoundError, KeyError) as error:  # a renamed recipe is a FAIL, not a traceback
        check("A25 the recipe fixture is present", False, str(error))
        return
    last = len(m_runner.get_param_range()) - 1
    sim = never_raised("sim m", m_runner, (0, last // 2, last))
    optical = never_raised("optical gamma", optical_gamma_runner(), (0,))
    check("A25 the fixtures exercised an infeasible cell", sim[1] + optical[1] > 0, f"{sim[1] + optical[1]}")
    check("A25 the fixtures exercised a feasible cell", sim[0] + optical[0] > 0, f"{sim[0] + optical[0]}")


# ---------------------------------------------------- 3. empty reads INFEASIBLE


def a25_empty_reads_infeasible():
    try:
        runner = omega_recipe_runner("simulation", methods=METHODS)
    except (FileNotFoundError, KeyError) as error:  # a renamed recipe is a FAIL, not a traceback
        check("A25 the omega fixture recipe is present", False, str(error))
        return
    grid = runner.get_param_range()
    widths, under = np.full((len(grid), 1), np.nan), []
    for index, knob in enumerate(grid):
        data = SweepData.coerce(runner.generate_data(0, knob))
        floor = cell_floor(runner, 0, data, "iv")
        budget = runner.fit_epsilon_iv(0, index, data)
        models = runner.build_models(0, index, data)
        model = models["DA+PI+IV"]
        estimate = model.predict(data.X_test, **runner.get_predict_kwargs(knob, 0))
        status = np.asarray(getattr(model, "query_status", None))
        record = evaluate_queries(data.estimand, estimate, status, 0.0, extent=data.metric_extent)
        widths[index, 0] = record.interval_width
        print(
            f"      knob {index} ({knob:.4g}): r_T^2 {budget**2:.4g} vs floor {floor:.4g}, "
            f"W {record.interval_width:.4f} C {record.coverage:.3f}"
        )
        if budget**2 >= floor:
            continue
        under.append(index)
        # what `evaluate_queries` makes of an all-empty cell, whichever rule is in force
        empty = evaluate_queries(
            data.estimand,
            np.full((len(data.X_test), 2), np.nan),
            np.full(len(data.X_test), SolveStatus.INFEASIBLE, dtype=int),
            0.0,
            extent=data.metric_extent,
        ).coverage
        same = (np.isnan(record.coverage) and np.isnan(empty)) or record.coverage == empty
        check(
            f"A25 knob {index} under the floor: every query INFEASIBLE",
            bool(np.all(status == SolveStatus.INFEASIBLE)),
            f"{np.bincount(status, minlength=len(SolveStatus)).tolist()}",
        )
        check(f"A25 knob {index}: the width is NaN", np.isnan(record.interval_width), f"{record.interval_width}")
        check(f"A25 knob {index}: coverage reads as an all-NaN cell does", same, f"{record.coverage} vs {empty}")

    print(f"      MEASURED knobs under the T floor: {under}")
    check("A25 the knobs under the floor are the RECORDED ones", under == EMPTY_KNOBS, f"{under} vs {EMPTY_KNOBS}")
    check("A25 the fixture has a knob under the floor and one above it", 0 < len(under) < len(grid), f"{under}")
    fig, ax = plt.subplots()
    try:
        handles, _ = plotting._draw_series(ax, np.asarray(grid, dtype=float), {"DA+PI+IV": widths})
        line = handles.get("DA+PI+IV")
        check("A25 the width line is drawn", line is not None)
        if line is not None:
            ydata = np.asarray(line.get_ydata(), dtype=float)
            gaps = [int(i) for i in np.flatnonzero(np.isnan(ydata))]
            check("A25 the width line has a gap at exactly those knobs", gaps == under, f"{gaps}")
    finally:
        plt.close(fig)


# -------------------------------------------------------------- 4. completeness


def a25_no_guard_left():
    """`git grep -i` for the guard's names over `src` and `scripts`, any case,
    spaced or hyphenated. This file is excluded (it names them to look for them)
    and so is its own module name, `a25_floor_guard`, which a28 and a68 import."""
    done = subprocess.run(
        [
            "git",
            "grep",
            "-n",
            "-i",
            "-e",
            "FLOOR_GUARD_R",
            "-e",
            "_floor_guard",
            "-e",
            "floor guard",
            "-e",
            "floor-guard",
            "--",
            "src",
            "scripts",
            ":!scripts/a25_floor_guard.py",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    check("A25 git grep ran", done.returncode in (0, 1), done.stderr.strip())
    hits = [line for line in done.stdout.splitlines() if line]
    names = ("floor_guard_r", "_floor_guard", "floor guard", "floor-guard")
    left = [
        hit
        for hit in hits
        if any(name in hit.lower().replace("a25_floor_guard", "") for name in names)
        and not any(allowed in hit for allowed in GUARD_ALLOWED)
    ]
    allowed = [hit for hit in hits if any(allowed in hit for allowed in GUARD_ALLOWED)]
    for hit in allowed:
        print(f"      allowed: {hit[:110]}")
    check("A25 no floor guard left in src or scripts", not left, f"{left[:3]}")
    check("A25 both allowed sentences are still there", len(allowed) == len(GUARD_ALLOWED), f"{len(allowed)}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", message="Mean of empty slice")  # every metric of an empty cell
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=None, help="one leg: 1, 2, 3 or 4")
    args = parser.parse_args()
    legs = [
        ("1", a25_closed_form),
        ("2", a25_never_raised),
        ("3", a25_empty_reads_infeasible),
        ("4", a25_no_guard_left),
    ]
    if args.only:
        legs = [(tag, leg) for tag, leg in legs if tag == args.only]
        if not legs:
            sys.exit(f"unknown leg {args.only!r}")
    for _, leg in legs:
        leg()
    print(f"\n{'A25 ALL PASS' if not FAIL else 'A25 FAILURES: ' + ', '.join(FAIL)}")
    sys.exit(bool(FAIL))
