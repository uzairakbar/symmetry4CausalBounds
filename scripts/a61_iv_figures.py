"""A61: the neighbour-price figures and tables, and the normalised sweep figures.

refactor6 touches `src/experiments/cigarettes.py` (F1, F2, the T1 benchmark column
and the T2 benchmark table, all under a configured instrument set only, so the
shipped path writes nothing new), `configs.py` (the two figure annotations, the
`normalize` toggle in TOGGLE_KEYS), `plotting.py` (`normalize_sweep`, the `normalize`
keyword on `create_sweep_plot`, `vlines` on `create_query_sweep_plot`),
`constants.py` (`DEFAULT_NORMALIZE_SWEEP`, the suffix and baseline tables, the
`_coverage` rejection in `validate_plot_keys`), `base.py` (`_run_sweeps` passes the
toggle) and both recipes (`normalize: true`). Legs:

  (D)   the digest leg (scripts/digest_leg.py), as a56 to a60. Catches: a new
        artifact on the shipped path (leg (D) fails on one that appears), a moved
        number. Misses: every path under a configured set, which (i) to (v) cover.
  (i)   both recipes run end to end at reduced scale (1 experiment, 4 sweep samples,
        the gamma sweep) through the production path, and every SS10 file exists
        and is non-empty: F1 and F2 with their pkls, T1 carrying the benchmark
        column, T2, and the four gamma sweep figures; F1's own grid
        (`beta_pn_gamma_values.pkl`) spans exactly `GAMMA_RANGE` and its outcomes
        carry no NaN cell, so (ii)'s "shipped grid" is the figure's; `_run_sweeps` handed
        `normalize=True` to every sweep figure of both runs. `create_sweep_plot`
        swallows exceptions and only logs them, so existence is the check that
        bites. Catches: a figure that raises inside the plotter, the toggle not
        reaching `_run_sweeps`. Misses: what the figures show, which (ii), (iii)
        and (vi) read off the same models and pkls.
  (ii)  no INFEASIBLE cell and no OPTIMAL_INACCURATE solve for PI, PI+IV, PI+INV+IV
        and DA+PI+IV on F1's grid (`GAMMA_RANGE`, 0.005 steps) on beta_pn, read
        off the raw cvxpy status of both problems after every solve, since
        `_solve_single` reads OPTIMAL_INACCURATE as OK. Catches: the grid extended
        under a method's feasibility floor (DA+PI+IV's is above 0.13 at every
        bound in play), a solve the plotter would draw from an inaccurate point.
        Misses: the other coefficients, which T1 solves at one budget only.
  (iii) the lower bounds of PI+INV+IV and DA+PI+IV on beta_pn vary by less than
        1e-6 on the sub-grid gamma >= 0.190, on F1's own models (the query path)
        and at the SS4.1 convention (INV epsilon and r_T at EPS_TOL, no pad, the
        seed-0 DA draw); the PI+INV+IV binding point at that convention is 0.176 to
        within the 0.005 grid (MEASURED p8); the DA+PI+IV binding point and flat
        value are RECORDED on both paths (the review: between 0.170 and 0.175,
        1.052921 at the joint bound 0.069877), never pinned. Catches: flatness
        pinned on the full grid (the spread at 0.150 is 3e-2), a lower bound that
        keeps moving with the budget. Misses: the upper end, which does move.
  (iv)  the SS6.2 benchmark table: 0.3121 (lag q), 0.1502 (tax differential) and
        0.0680 (own excise) to 1e-4, and gamma_OVB == gamma_CH to 1e-10 on every
        row, controls re-residualised inside each covariate's sample. Catches: the
        OVB formula replaced (the identity breaks), a full-sample FWL on the lag
        row (5e-3 apart). Misses: the proxies' exact values, recorded beside p8's.
  (v)   the SS4.1 width table AT THE SHIPPED BUDGET RULE, on the seed-0 DA draw and
        the SS4.1 convention: PI to 1e-3 of p5, PI+IV and PI+INV+IV lower bounds to
        1e-6 of p8 and upper bounds to 1e-3 of p5 at the five gammas, and the
        DA+PI+IV row RECORDED at its joint bound 0.069877 (r_T EPS_TOL and r_Z
        0.0625 in root sum square) beside the review's numbers. The same table on
        F1's query-path models is recorded too. Catches: a moved IV or INV
        arithmetic, a wrong instrument matrix. Misses: a solver bump under 1e-6.
  (vi)  the normalise rule (SS10.1): on a synthetic sweep the baseline reads
        exactly 1.0 where its mean is positive and NaN, never inf and never a
        floor, where it is exactly 0.0, every other method divided by the same
        number; PI+IV is the baseline when PI is absent, nothing is divided
        without either (one warning), a `_coverage` and an `_approx_error` id
        are left alone (the rollback itself is a63-iii's), and
        `validate_plot_keys` raises at import on `normalize` under a `_coverage`
        id; the y-label carries the baseline's name. On the recipe run's own
        gamma sweep the PI series of the width and worst_error figures after the
        bootstrap reads 1.0 to 1e-12 at every positive step and NaN at every
        zero step. With the toggle on or off the sweep pkls of the shipped
        cigarette block equal leg (D)'s: the toggle is plot-only. Catches: a
        baseline other than the rule's (DA+PI is not 1.0), the `base > 0` guard
        dropped (inf or 0/0 where NaN is pinned), a toggle that leaks into a
        solver. Misses: figure bytes, which carry a timestamp.

    MPLBACKEND=Agg python scripts/a61_iv_figures.py [--reference JSON] [--skip-digest]

`--skip-digest` drops leg (D) and the two pkl comparisons of (vi) and exists for the
break-it runs of the other legs; the committed state is always gated with it on.
"""

import argparse
import json
import os
import pickle
import shutil
import sys
from functools import partial

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import yaml
from loguru import logger

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import digest_leg  # noqa: E402
from munch import munchify  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

import src.experiments.base as base_module  # noqa: E402
import src.experiments.cigarettes as cig_module  # noqa: E402
from src.data_augmentors.cigarettes import ScaleTranslation  # noqa: E402
from src.experiments.cigarettes import (  # noqa: E402
    GAMMA_RANGE,
    HEADLINE_METHODS,
    PN,
    CigaretteOrchestrator,
    benchmark_covariates,
    benchmark_gamma,
)
from src.experiments.configs import (  # noqa: E402
    EPS_TOL,
    GAMMA_Z_DEFAULT,
    MethodRegistry,
    parse_experiment_plan,
    resolve_dataset_block,
)
from src.experiments.utils import PanelBuilder, bootstrap, set_seed  # noqa: E402
from src.experiments.utils.constants import (  # noqa: E402
    ARTIFACTS_DIRECTORY,
    SUBDIR_QUERY,
    SUBDIR_SWEEP,
    TEX_MAPPER,
    plot_keys_for,
    validate_plot_keys,
)
from src.experiments.utils.metrics import rho_hat  # noqa: E402
from src.experiments.utils.model_fitting import fit_model  # noqa: E402
from src.experiments.utils.plotting import create_sweep_plot, normalize_sweep  # noqa: E402
from src.main import ORCHESTRATORS  # noqa: E402
from src.methods.sensitivity_models import SolveStatus  # noqa: E402
from src.sem.cigarettes import CigaretteSEM, V, build_design, instrument_set  # noqa: E402

PHASE_B = ("tax_s", "y", "cpi")
GRID_STEP = 0.005
FLAT_FROM = 0.190  # the flat-floored regime of SS4.1
FLAT_TOL = 1e-6
BINDING_POINT = 0.176  # p8, PI+INV+IV at the SS4.1 convention
TABLE_GAMMAS = (0.150, 0.250, 0.311, 0.37052, 0.450)  # p5's columns
# p8 B1 lower bounds (6 decimals) on beta_pn, raw, at TABLE_GAMMAS' first three and
# 0.371 / 0.450 (p8 ran 0.3710 where p5 ran 0.37052; the flat rows do not care)
P8_LOWER = {
    "PI+IV": (0.672691, 0.374940, 0.248499, None, 0.018921),
    "PI+INV+IV": (0.984557, 0.951673, 0.951673, 0.951673, 0.951673),
}
# p5 (4 decimals): PI interval and every upper bound
P5_PI = ((-0.1787, 1.4202), (-0.4114, 1.6528), (-0.5304, 1.7718), (-0.6357, 1.8772), (-0.7640, 2.0054))
P5_UPPER = {
    "PI+IV": (1.3468, 1.6446, 1.7710, 1.8772, 2.0054),
    "PI+INV+IV": (1.3425, 1.6446, 1.7710, 1.8772, 2.0054),
}
REVIEW_DA = {"flat": 1.052921, "binding": (0.170, 0.175), "floor": 0.1343, "width_at_0.25": 0.6004}
BENCHMARK_PINS = {"lag_q": 0.3121, "tax_diff": 0.1502, "log_tax_s": 0.0680}
P8_BENCHMARKS = {
    "lag_q": 0.31214,
    "tax_diff": 0.15018,
    "log_tax_s": 0.06798,
    "log_tax_ratio": 0.01005,
    "log_pop": 0.00451,
    "log_tax_f": 0.00348,
    "log_tax_sn": 0.00234,
    "log_pop_n": 0.00154,
    "log_pop_ratio": 0.00016,
}
TOGGLES = dict(recalibrate=True, clipy=False, mean_match=True, n_jobs=1)
FAIL = []
SPY = {}  # (dataset, fname) -> the `normalize` kwarg `_run_sweeps` passed


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAIL.append(name)


# ------------------------------------------------------------------ helpers


def recipe(name):
    fname = {"cigarettes": "neighbour-price_fig12", "simulation": "iv_fig13"}[name]
    with open(os.path.join(REPO, "recipes", f"{fname}.yaml")) as handle:
        config = yaml.safe_load(handle)
    defaults = config.pop("defaults", {}) or {}
    return {**defaults, **config[name]}


def reduced_block(name, **overrides):
    block = recipe(name)
    block.pop("experiment", None)
    return resolve_dataset_block(name, {**block, "n_experiments": 1, "sweep_samples": 4, "n_jobs": 1, **overrides})


def run_reduced(name):
    """The recipe through the production path, query panel and the gamma sweep,
    with `create_sweep_plot` spied on for the `normalize` kwarg."""
    plan = parse_experiment_plan(
        {"query": True, "sweep": {"param": ["gamma"], "metric": recipe(name)["experiment"]["sweep"]["metric"]}}
    )
    block = reduced_block(name)
    folder = os.path.join(ARTIFACTS_DIRECTORY, name)
    shutil.rmtree(folder, ignore_errors=True)
    original = base_module.create_sweep_plot

    def spy(*args, **kwargs):
        SPY[(name, kwargs.get("fname"))] = kwargs.get("normalize")
        return original(*args, **kwargs)

    base_module.create_sweep_plot = spy
    try:
        set_seed(block["seed"])
        ORCHESTRATORS[name](**block, hyperparameters=munchify(digest_leg.HYPERPARAMETERS)).run(plan)
    finally:
        base_module.create_sweep_plot = original
    return folder


def orchestrator(block):
    set_seed(block["seed"])
    return CigaretteOrchestrator(**block, hyperparameters=munchify(digest_leg.HYPERPARAMETERS))


def panel_models(orch):
    """F1's own models: the query panel's, fitted on the full panel and the real Z."""
    runner = orch.get_query_runner_cls()(methods=orch.methods, **{**orch._get_clean_kwargs(), "n_experiments": 1})
    panel = PanelBuilder(runner, "cigarettes", False)
    panel._fit_all_models()
    return runner, {name: panel.fitted_models[name] for name in HEADLINE_METHODS}


def convention_models(design, Z):
    """The SS4.1 convention on the seed-0 DA draw: INV epsilon and r_T at EPS_TOL,
    no pad, r_Z declared at 2^-8, rho_hat of the draw; through the registry and
    `fit_model` as the runners fit."""
    np.random.seed(0)
    da = ScaleTranslation(V, std=float(np.std(design.X @ (V / np.linalg.norm(V)))))
    GX, G = da(design.X)
    rho = float(rho_hat(design.X, GX, design.y, intercept=True))
    builders = MethodRegistry.build_methods(
        list(HEADLINE_METHODS),
        gamma=0.25,
        epsilon=EPS_TOL,
        epsilon_iv=EPS_TOL,
        epsilon_iv_z=0.0,
        gamma_z=GAMMA_Z_DEFAULT,
        rho=rho,
        pad=False,
        **TOGGLES,
    )
    models = {}
    for name in HEADLINE_METHODS:
        model = builders[name]()
        fit_model(model=model, method_name=name, X=design.X, y=design.y, GX=GX, G=G, Z=Z, da=da)
        models[name] = model
    return models


def grid():
    return np.round(np.arange(GAMMA_RANGE[0], GAMMA_RANGE[1] + GRID_STEP / 2, GRID_STEP), 6)


def beta_pn(models, gammas, scale):
    """{name: (n_grid, 2)} raw intervals on beta_pn, as F1 reads them."""
    query = np.eye(4)[PN][None, :]
    return {
        name: np.array([scale * model.predict(query, gamma=float(g))[0] for g in gammas])
        for name, model in models.items()
    }


def raw_statuses(model, gamma):
    """The cvxpy statuses of both problems at beta_pn, past `_solve_single`'s
    OPTIMAL_INACCURATE-as-OK reading."""
    payloads = model._prepare(np.eye(4)[PN][None, :], gamma)
    if payloads is None:
        return ["EMPTY"]
    with threadpool_limits(limits=1):
        model._begin_chunk()
        for x in payloads:
            model._solve_single(x)
    return [model.min_problem.status, model.max_problem.status]


def flatness(gammas, lower):
    """(spread on gamma >= FLAT_FROM, flat value, binding bracket): the bracket is
    the last grid point still off the flat value by more than FLAT_TOL and the
    next one; (None, None) when the whole grid is flat."""
    flat_mask = gammas >= FLAT_FROM
    flat = float(lower[flat_mask][-1])
    spread = float(np.nanmax(np.abs(lower[flat_mask] - flat)))
    moving = np.flatnonzero(np.abs(lower - flat) > FLAT_TOL)
    if len(moving) == 0:
        return spread, flat, (None, None)
    last = int(moving[-1])
    return spread, flat, (float(gammas[last]), float(gammas[min(last + 1, len(gammas) - 1)]))


def nonempty(path):
    return os.path.isfile(path) and os.path.getsize(path) > 0


# ------------------------------------------------------------------ legs


def leg_d(reference):
    print("(D) the shipped configuration does not move")
    for label, ok, detail in digest_leg.compare(REPO, reference):
        check(f"(D) {label}", ok, detail)


def leg_i():
    print("(i) both recipes run at reduced scale and every SS10 file exists")
    folder = run_reduced("cigarettes")
    query, sweep = os.path.join(folder, SUBDIR_QUERY), os.path.join(folder, SUBDIR_SWEEP)
    for fname in (
        "beta_pn_gamma_sweep.pdf",
        "beta_pn_gamma_values.pkl",
        "beta_pn_gamma_outcomes.pkl",
        "beta_pn_gamma_vlines.pkl",
        "beta_pn_budget_sweep.pdf",
        "beta_pn_budget_values.pkl",
        "beta_pn_budget_outcomes.pkl",
        "beta_pn_budget_vlines.pkl",
        "coefficients.tex",
        "benchmarks.tex",
        "ladder.tex",
    ):
        check(f"(i) cigarettes/query/{fname} exists and is non-empty", nonempty(os.path.join(query, fname)))
    for metric in ("width", "worst_error", "approx_error", "coverage"):
        check(
            f"(i) cigarettes/sweep/gamma_{metric}_sweep.pdf exists",
            nonempty(os.path.join(sweep, f"gamma_{metric}_sweep.pdf")),
        )
    with open(os.path.join(query, "coefficients.tex")) as handle:
        table = handle.read()
    check("(i) T1 carries the benchmark-gamma column", r"\gamma_{\mathrm{lag}}" in table and "0.312" in table)
    check("(i) T1 names the instrument set and the two benchmarks", "instrument set" in table and "0.1502" in table)
    with open(os.path.join(query, "beta_pn_gamma_outcomes.pkl"), "rb") as handle:
        outcomes = pickle.load(handle)  # noqa: S301 - our own artifact
    check(
        "(i) F1 carries one band per headline method",
        tuple(outcomes) == HEADLINE_METHODS and all(v.shape == (4, 1, 2) for v in outcomes.values()),
    )
    # the figure's OWN grid and cells, not the module constant the gate's (ii) grid
    # is built from: a grid started under a feasibility floor writes NaN cells
    with open(os.path.join(query, "beta_pn_gamma_values.pkl"), "rb") as handle:
        f1_grid = pickle.load(handle)  # noqa: S301
    spans = (
        len(f1_grid) == 4
        and f1_grid[0] == GAMMA_RANGE[0]
        and f1_grid[-1] == GAMMA_RANGE[1]
        and np.all(np.diff(f1_grid) > 0)
    )
    check("(i) F1's grid spans exactly GAMMA_RANGE", spans, f"{f1_grid}")
    check("(i) F1's outcomes carry no NaN cell", all(np.all(np.isfinite(v)) for v in outcomes.values()))
    with open(os.path.join(query, "beta_pn_gamma_vlines.pkl"), "rb") as handle:
        vlines = pickle.load(handle)  # noqa: S301
    print(
        f"      RECORDED F1 marks: floor {vlines[0]:.4f}, tax-diff {vlines[1]:.4f}, lag-q {vlines[2]:.4f}, "
        f"gamma*(b) {vlines[3]:.4f}, 3x {vlines[4]:.4f}"
    )
    check(
        "(i) F1's marks are the plan's (0.111, 0.150, 0.312, 0.371, 0.450) to 1e-3",
        np.abs(vlines - (0.1107, 0.1502, 0.3121, 0.3705, 0.4505)).max() < 1e-3,
    )
    with open(os.path.join(query, "beta_pn_budget_vlines.pkl"), "rb") as handle:
        marks = pickle.load(handle)  # noqa: S301
    print(
        f"      RECORDED F2 marks: cluster-bootstrap moment median {marks[0]:.4f}, p95 {marks[1]:.4f} "
        "(p5: 0.1039, 0.2256)"
    )
    check(
        "(i) F2's marks are p5's 0.104 and 0.226 to 1e-3",
        abs(marks[0] - 0.1039) < 1e-3 and abs(marks[1] - 0.2256) < 1e-3,
    )
    for metric in ("width", "worst_error", "approx_error", "coverage"):
        check(
            f"(i) cigarettes: _run_sweeps handed normalize=True to gamma_{metric}",
            SPY.get(("cigarettes", f"gamma_{metric}")) is True,
        )
    with open(os.path.join(sweep, "gamma_results.pkl"), "rb") as handle:
        leg_i.results = pickle.load(handle)  # noqa: S301

    folder = run_reduced("simulation")
    sweep = os.path.join(folder, SUBDIR_SWEEP)
    for metric in ("width", "worst_error", "approx_error", "coverage"):
        check(
            f"(i) simulation/sweep/gamma_{metric}_sweep.pdf exists",
            nonempty(os.path.join(sweep, f"gamma_{metric}_sweep.pdf")),
        )
        check(
            f"(i) simulation: _run_sweeps handed normalize=True to gamma_{metric}",
            SPY.get(("simulation", f"gamma_{metric}")) is True,
        )
    check("(i) simulation/query panel exists", nonempty(os.path.join(folder, SUBDIR_QUERY, "outcome_values.pkl")))


def leg_ii_iii():
    print("(ii) feasibility on F1's grid, and (iii) the flat lower bound")
    orch = orchestrator(reduced_block("cigarettes"))
    runner, models = panel_models(orch)
    design = runner.sem.design
    gammas = grid()
    check(
        f"(ii) the grid is {GAMMA_RANGE} in {GRID_STEP} steps",
        gammas[0] == GAMMA_RANGE[0] and gammas[-1] == GAMMA_RANGE[1] and len(gammas) == 61,
    )
    for name, model in models.items():
        bad = []
        for gamma in gammas:
            statuses = raw_statuses(model, float(gamma))
            if any(status != cp.OPTIMAL for status in statuses):
                bad.append((float(gamma), statuses))
        check(
            f"(ii) {name}: every solve OPTIMAL on the grid, none INFEASIBLE or OPTIMAL_INACCURATE",
            not bad,
            f"{bad[:3]}",
        )
        model.predict(np.eye(4)[PN][None, :], gamma=float(gammas[0]))
        check(
            f"(ii) {name}: query_status OK at the left edge", np.all(np.asarray(model.query_status) == SolveStatus.OK)
        )

    intervals = beta_pn(models, gammas, design.sigma)
    print(
        "      RECORDED (iii) on F1's models (query path: tolerance 2^-8 on the INV cone and r_T, pad as configured):"
    )
    for name in ("PI+INV+IV", "DA+PI+IV"):
        spread, flat, bracket = flatness(gammas, intervals[name][:, 0])
        print(
            f"        {name}: flat lower bound {flat:.6f} from gamma >= {FLAT_FROM}, spread {spread:.2e}, "
            f"binding bracket {bracket}"
        )
        check(f"(iii) F1's {name} lower bound varies by < 1e-6 on gamma >= 0.190", spread < FLAT_TOL, f"{spread:.2e}")
    leg_ii_iii.query_path = intervals

    Z = instrument_set(design, PHASE_B)
    conv = convention_models(design, Z)
    conv_intervals = beta_pn({n: conv[n] for n in ("PI+INV+IV", "DA+PI+IV")}, gammas, design.sigma)
    print("      RECORDED (iii) at the SS4.1 convention (seed-0 DA draw, epsilon and r_T at EPS_TOL, no pad):")
    for name in ("PI+INV+IV", "DA+PI+IV"):
        spread, flat, bracket = flatness(gammas, conv_intervals[name][:, 0])
        print(
            f"        {name}: flat lower bound {flat:.6f} from gamma >= {FLAT_FROM}, spread {spread:.2e}, "
            f"binding bracket {bracket}"
        )
        check(
            f"(iii) convention: {name} lower bound varies by < 1e-6 on gamma >= 0.190",
            spread < FLAT_TOL,
            f"{spread:.2e}",
        )
        if name == "PI+INV+IV":
            inside = bracket[0] is not None and bracket[0] <= BINDING_POINT <= bracket[1] + 1e-9
            check(
                "(iii) convention: the PI+INV+IV binding point is 0.176 within the 0.005 grid",
                inside,
                f"bracket {bracket}",
            )
            check(
                "(iii) convention: the PI+INV+IV flat value is 0.951673 to 1e-6",
                abs(flat - 0.951673) < 1e-6,
                f"{flat:.6f}",
            )
        else:
            print(
                f"        (review: DA+PI+IV flat {REVIEW_DA['flat']}, binding between {REVIEW_DA['binding']}, "
                f"at the joint bound {conv[name].iv_bound:.6f})"
            )
    leg_ii_iii.convention = conv


def leg_iv():
    print("(iv) the SS6.2 benchmark table")
    panel = CigaretteSEM.panel()
    rows = {key: benchmark_gamma(panel, w, "t3") for key, w in benchmark_covariates(panel).items()}
    for key, (n, r2y, r2w, gamma_ovb, gamma_ch) in rows.items():
        print(
            f"      {key:14s} n {n:4d} r2y {r2y:.4f} r2w {r2w:.4f} gamma_OVB {gamma_ovb:.5f} "
            f"gamma_CH {gamma_ch:.5f} [p8 {P8_BENCHMARKS[key]:.5f}]"
        )
        check(
            f"(iv) {key}: gamma_OVB == gamma_CH to 1e-10",
            abs(gamma_ovb - gamma_ch) < 1e-10,
            f"{abs(gamma_ovb - gamma_ch):.2e}",
        )
    for key, pin in BENCHMARK_PINS.items():
        check(f"(iv) {key}: gamma = {pin} to 1e-4", abs(rows[key][3] - pin) < 1e-4, f"{rows[key][3]:.5f}")
    check(
        "(iv) the lag row uses 2401 rows, the excise rows 2450",
        rows["lag_q"][0] == 2401 and rows["tax_diff"][0] == 2450,
    )
    check(
        "(iv) every proxy sits below the PI+IV floor 0.111",
        all(rows[k][3] < 0.111 for k in rows if k not in ("lag_q", "tax_diff")),
    )
    check("(iv) the primary benchmark is above the tax differential", rows["lag_q"][3] > rows["tax_diff"][3])


def leg_v():
    print("(v) the SS4.1 width table at the shipped budget rule")
    design = build_design(CigaretteSEM.panel(), spec="t3", anchor="own-tax")
    conv = getattr(leg_ii_iii, "convention", None) or convention_models(design, instrument_set(design, PHASE_B))
    gammas = np.array(TABLE_GAMMAS)
    table = beta_pn(conv, gammas, design.sigma)
    print(
        "      RECORDED, raw beta_pn [lower, upper] width (ratio to PI), at gamma "
        + ", ".join(f"{g:.4g}" for g in gammas)
    )
    for name in HEADLINE_METHODS:
        cells = []
        for i in range(len(gammas)):
            lo, hi = table[name][i]
            cells.append(
                f"[{lo:+.4f}, {hi:+.4f}] {hi - lo:.4f} ({(hi - lo) / (table['PI'][i, 1] - table['PI'][i, 0]):.3f})"
            )
        print(f"        {name:10s} " + " | ".join(cells))
    for i, gamma in enumerate(gammas):
        lo, hi = table["PI"][i]
        check(
            f"(v) PI at {gamma:.4g}: p5's interval to 1e-3",
            abs(lo - P5_PI[i][0]) < 1e-3 and abs(hi - P5_PI[i][1]) < 1e-3,
        )
        for name in ("PI+IV", "PI+INV+IV"):
            lo, hi = table[name][i]
            want = P8_LOWER[name][i]
            if want is not None:
                check(f"(v) {name} at {gamma:.4g}: p8's lower bound {want} to 1e-6", abs(lo - want) < 1e-6, f"{lo:.6f}")
            check(
                f"(v) {name} at {gamma:.4g}: p5's upper bound to 1e-3", abs(hi - P5_UPPER[name][i]) < 1e-3, f"{hi:.4f}"
            )
    da = table["DA+PI+IV"]
    width = da[1, 1] - da[1, 0]
    print(
        f"      RECORDED DA+PI+IV at the joint bound {conv['DA+PI+IV'].iv_bound:.6f}: width at 0.25 {width:.4f} "
        f"(review {REVIEW_DA['width_at_0.25']}), lower bounds {np.round(da[:, 0], 6).tolist()}"
    )
    check(
        "(v) DA+PI+IV is the tightest of the three IV methods at 0.25 at the convention",
        width < table["PI+INV+IV"][1, 1] - table["PI+INV+IV"][1, 0],
    )
    query_path = getattr(leg_ii_iii, "query_path", None)
    if query_path is not None:
        g = grid()
        at = {name: query_path[name][np.argmin(np.abs(g - 0.25))] for name in HEADLINE_METHODS}
        print(
            "      RECORDED the same rows on F1's query-path models at gamma 0.25: "
            + ", ".join(f"{n} [{lo:+.4f}, {hi:+.4f}]" for n, (lo, hi) in at.items())
        )


def leg_vi(reference):
    print("(vi) the normalise rule")
    y = {
        "PI": np.array([[1.0, 1.0], [0.0, 0.0], [2.0, 2.0], [np.nan, np.nan]]),
        # a POSITIVE miss on the zero-baseline step: x/0 is inf without the guard
        "DA+PI": np.array([[0.5, 0.5], [0.3, 0.0], [1.0, 3.0], [1.0, 1.0]]),
    }
    out, baseline = normalize_sweep(y, "gamma_width")
    check("(vi) the baseline is PI", baseline == "PI")
    check("(vi) PI reads exactly 1.0 where its mean is positive", np.array_equal(out["PI"][[0, 2]], np.ones((2, 2))))
    check(
        "(vi) DA+PI is divided by the same per-step number", np.allclose(out["DA+PI"][[0, 2]], [[0.5, 0.5], [0.5, 1.5]])
    )
    check(
        "(vi) a zero baseline step reads NaN for every method, never inf (x/0 and 0/0 alike)",
        np.all(np.isnan(out["PI"][1])) and np.all(np.isnan(out["DA+PI"][1])),
        f"{out['DA+PI'][1]}",
    )
    check("(vi) an all-NaN baseline step reads NaN too", np.all(np.isnan(out["DA+PI"][3])))
    check("(vi) the input is left untouched", y["PI"][2, 0] == 2.0)
    out, baseline = normalize_sweep({"PI+IV": y["PI"], "DA+PI": y["DA+PI"]}, "gamma_worst_error")
    check("(vi) PI+IV is the baseline when PI is absent", baseline == "PI+IV" and out["PI+IV"][0, 0] == 1.0)
    same, baseline = normalize_sweep({"DA+PI": y["DA+PI"]}, "gamma_worst_error")
    check("(vi) without PI or PI+IV nothing is divided", baseline is None and same["DA+PI"] is y["DA+PI"])
    same, baseline = normalize_sweep(y, "gamma_coverage")
    check("(vi) a _coverage id is left alone", baseline is None and same is y)
    same, baseline = normalize_sweep(y, "gamma_approx_error")
    check("(vi) an _approx_error id is left alone", baseline is None and same is y)
    try:
        validate_plot_keys("x", {"gamma_coverage": {"normalize": True}}, plot_keys_for)
        check("(vi) validate_plot_keys raises on normalize under a _coverage id", False)
    except ValueError as error:
        check("(vi) validate_plot_keys raises on normalize under a _coverage id", "normalize" in str(error))
    validate_plot_keys("x", {"gamma_width": {"normalize": True}}, plot_keys_for)
    check("(vi) validate_plot_keys accepts normalize under a _width id", True)
    plt.close("all")
    create_sweep_plot(
        np.array([1.0, 2.0, 3.0, 4.0]),
        y,
        xlabel="x",
        ylabel="average interval width",
        fname="gamma_width",
        savefig=False,
        normalize=True,
    )
    label = plt.gca().get_ylabel()
    check(
        "(vi) the y-label carries the baseline's name", label == rf"average interval width / {TEX_MAPPER['PI']}", label
    )
    plt.close("all")

    results = getattr(leg_i, "results", None)
    if results is None:
        check("(vi) the recipe run's gamma sweep is available", False, "leg (i) did not run")
    else:
        for metric, fname in (
            ("interval_width", "gamma_width"),
            ("worst_error", "gamma_worst_error"),
        ):
            series = bootstrap({name: record[metric] for name, record in results.items()})
            base = np.nanmean(series["PI"], axis=1)
            out, baseline = normalize_sweep(series, fname)
            ratio = np.nanmean(out["PI"], axis=1)
            positive, zero = base > 0, base == 0
            ok_one = np.all(np.abs(ratio[positive] - 1.0) < 1e-12)
            ok_nan = np.all(np.isnan(out["PI"][zero])) and all(np.all(np.isnan(out[name][zero])) for name in out)
            print(f"      RECORDED {fname}: {int(positive.sum())} positive PI steps, {int(zero.sum())} zero steps")
            check(f"(vi) {fname}: PI reads 1.0 to 1e-12 at every positive step", baseline == "PI" and ok_one)
            check(f"(vi) {fname}: every method reads NaN at every zero step", ok_nan)

    if reference is None:
        print("      (vi) pkl comparisons SKIPPED by --skip-digest")
        return
    with open(reference) as handle:
        want = json.load(handle)["digests"]["cigarettes"]
    shipped = digest_leg.BLOCKS["cigarettes"]
    digest_leg._enter(REPO)
    for toggle in (False, True):
        digest_leg.BLOCKS["cigarettes"] = {**shipped, "normalize": toggle}
        try:
            with threadpool_limits(limits=1):
                got = digest_leg.run_dataset("cigarettes", "sweep")
        finally:
            digest_leg.BLOCKS["cigarettes"] = shipped
        for artifact, digest in got.items():
            check(
                f"(vi) normalize: {toggle}: cigarettes/{artifact} equals leg (D)'s",
                want.get(artifact) == digest,
                f"{want.get(artifact)} vs {digest}",
            )


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    os.chdir(REPO)
    os.environ.setdefault("MPLBACKEND", "Agg")
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", default=digest_leg.DEFAULT_REFERENCE)
    parser.add_argument("--skip-digest", action="store_true")
    args = parser.parse_args()
    print(f"tree: {cig_module.__file__}")
    legs = [
        ("(i)", leg_i),
        ("(ii)/(iii)", leg_ii_iii),
        ("(iv)", leg_iv),
        ("(v)", leg_v),
        ("(vi)", partial(leg_vi, None if args.skip_digest else args.reference)),
    ]
    if not args.skip_digest:
        legs.append(("(D)", partial(leg_d, args.reference)))
    for tag, leg in legs:
        try:
            leg()
        except Exception as error:  # a raise is a FAIL line, and the later legs still report
            check(f"{tag} ran without raising", False, f"{type(error).__name__}: {error}")
    if args.skip_digest:
        print("(D) SKIPPED by --skip-digest: a break-it run, not the committed state")
    if not FAIL:
        print("A61 PASS")
    else:
        print(f"A61 FAIL: {FAIL}")
        sys.exit(1)
