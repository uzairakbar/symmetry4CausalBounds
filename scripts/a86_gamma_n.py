"""A86: the finite-sample budget gamma_n (no experiment is run; seconds).

`finite_sample_budget(d, g, n, a)` is gamma_n(d; g) = F^-1_{chi2_d(n g)}(1 - a) / n,
the per-unit (1 - a) quantile of a noncentral chi-square on d degrees of freedom
with noncentrality n g; a = 0 is the raw (population) budget g. Legs:

  (i)   against scipy's chi2 / ncx2 quantiles on a grid of (d, g, n, a); a = 0
        returns g exactly; g <= 0 is the central chi-square; increasing in d, g
        and a's complement (1 - a), decreasing in n towards g; and the reference
        values at the fixed split a = alpha / 3: sim k 33, n 1843, gamma 1 ->
        1.1194; optical k 55, n 900, gamma 0.66 -> 0.8428; cigarettes k 5, gamma
        0.25 at n 2205 -> 0.2993 and at n 49 -> 0.7493; the mean row (1 + gamma)
        gamma_n(1; 0) at n 1843, gamma 1 -> 0.0062 (each within 1e-4).
        Catches: a quantile at the wrong level, a dropped 1/n, a noncentrality
        not scaled by n, a raw mode that pads.
  (ii)  the ERM ball and the mean row off the cvx parameters, through
        `MethodRegistry.build_methods` and `fit_model` on a synthetic fixture,
        for PI, PI+INV, PI+IV (empty Z), DA+PI and PI&DA+PI (both branches):
        the ball s sqrt(gamma_n(k; gamma~)) at a = alpha / 3 with k = d + 1 (the
        slice's intercept) and n = n_eff, the mean row s sqrt((1 + gamma~)
        gamma_n(1; 0)); n_eff the original samples on an m = 4 tiling (`X_base`)
        and `unit_cap` when set; raw (alpha 0) the population ball s sqrt(gamma~)
        bit for bit and no mean row; the cigarette orchestrator's `unit_cap` is
        the panel's 49 states on `target: iv` and None on `plasmode`, and the
        simulation and optical toggles carry none.
        The IV rows on a fixture with an observed Z and T, gamma_z 2^-8: PI+IV,
        PI+INV+IV and the intersection's baseline carry the Z row at the declared
        gamma_z; DA+PI+IV(Z) / (T) the one row at gamma~_z(eps) = (eps / s~ +
        sqrt(gamma_z / rho))^2; DA+PI+IV(T,Z) and the intersection's DA branch
        the joint row alone (`IV_LAYOUT` ("tz",), d = d_T + d_Z), each at
        sqrt(s^2 (1 + gamma~) gamma_n(d; g / (1 + gamma~))) at alpha / 3 (one row),
        raw s sqrt(g) to 1e-12; a DA row moves with a predict-time epsilon and a
        non-DA one does not.
        Catches: a ball at the wrong level, dof or n, the m sweep counting its
        copies, a cap leaking onto the plasmode sweeps, a raw run that pads, an IV
        row at the wrong leak, level or layout, a DA row frozen at fit.
  (iii) the mean row: at a query at the design mean the padded PI interval is
        ybar +- the mean radius exactly (delta binds; raw: the point ybar); on
        20 queries the padded interval contains the raw one for PI and DA+PI;
        the closed form (`CLOSED_FORM_SOLUTION`) equals cvxpy within 1e-6
        relative to the width, padded and raw. The IV rows see delta: padded
        PI+IV on an instrument off its mean (Q' 1 != 0), at every bound on 20
        queries the solved (h, delta) keeps the block's data residual
        || Q'(y_c - X_c h - delta 1) || (with the jitter rows) under its
        threshold, and the same program with delta dropped from the IV row
        breaks that at some query.
        Catches: a delta outside the ball, a cost that drops delta at the
        design mean, a closed form off the program, an IV row blind to delta.
  (iv)  validity on a small `LinearSimulationSEM` (d 8, iv 1, n 400, gamma =
        gamma* = 0.25, gamma_z 2^-8, padded): (a) the PI+IV rows' statistics at
        h* over `VALIDITY_ROWS_DRAWS` draws (the ball with delta, the mean row,
        the Z row, each against its padded radius): every row holds on at least
        1 - alpha/3 - 3 SE of the draws and all three jointly on 0.95 - 3 SE;
        (b) `VALIDITY_SOLVE_DRAWS` solved draws, the simultaneous coverage of h*
        at 20 fixed queries of PI, PI+IV, DA+PI and DA+PI+IV(T,Z) at least
        0.95 - 3 SE, and of PI&DA+PI+IV(T,Z) at least 0.90 - 3 SE.
        Catches: a pad too small for its level, a mis-split alpha, an IV row
        that excludes h*.
  (v)   nesting on one simulation draw (iv 1), raw and padded: PI+IV inside PI,
        PI+INV+IV inside PI+INV, every DA+PI+IV mode inside DA+PI and
        PI&DA+PI+IV inside PI&DA+PI (to 1e-6 of the width).
        Catches: a row that widens an interval (a split that is not fixed).
  (vi)  config: `gamma_n` 95 -> `gamma_n_alpha` 0.05, absent / 0 / false -> 0.0
        with one INFO line when absent; `true`, 100, -5 and "95" raise; every
        non-do-MNIST recipe and config.yaml resolve to `RECIPE_ALPHA`; the
        retired `im-ci` key raises; do-MNIST accepts `gamma_n` and carries no
        `gamma_n_alpha`. The one tolerance: `EPS_TOL` is 2^-8 and no dataset config
        carries a tolerance or pad budget of its own; the optical
        `_epsilon_budget(None)` sits exactly `EPS_TOL` over the measured eps* in
        both norms (the query's RMS, the sweeps' q0.95 reading
        `epsilon_star_q95`), and a built DA+ model's pad is its epsilon; the
        optical and cigarette query budgets are bit-identical to c29af19's
        (`QUERY_EPSILON_C29AF19`) and the simulation query's is its oracle RMS
        eps* + 2^-8 (c29af19 declared 2^-8; within 1e-12 of it); the query
        runners rescale the raw declared gamma by 1 / sigma-hat^2, so the fitted
        PI radius is sqrt(gamma) (1.0 sim, 0.5 optical); the do-MNIST builders
        hand the copsens T-as-IV cone the block epsilon.
        Catches: a level read as alpha, a bool taken as a level, a silent pad.
  (vii) one record: `_run_sweeps` writes `<param>_values`, `_results`,
        `_statuses` and `_axis` and nothing else, the aggregate's sweep grid reads
        `_values` and `_results`, the runner takes no CI level, `BoundedSA` has no
        pad tolerance, `src.experiments.utils` has no CI module; and no symbol of
        the retired Imbens-Manski CI is left in src/, scripts/, recipes/,
        config.yaml, README.md or scripts/README.md (this script and a77's
        unknown-key check aside, which name it to reject it).
        Catches: a second sweep record, a leftover of the bootstrap path.
  (ix)  the purge: a grep over src/, scripts/, recipes/, config.yaml, README.md
        and scripts/README.md for every retired symbol (`RETIRED`, whole words,
        case-sensitive; this script and a77's unknown-key lines aside),
        `py_compile` on every script, `ruff check --select F` on src/ and
        scripts/, and every script with a command line (argparse) exits 0 on
        `--help`.
        Catches: a leftover of the old machinery, a script that no longer loads.
  (x)   grids: the m grid off `sweep_samples` (1..16 at 16, 1..8 at 8, 32 points
        at 32), the n grid `geomspace(10, 100, count)` at every count; the sim n
        strategy takes its 8 row counts (205 .. 2048 of 2048) at sweep_samples 8,
        the m strategy 8 steps at 6.25% of 2048 = 128 rows; no 16 / 17 literal in
        either `grid_fn`. Catches: a grid that ignores the knob.
  (xi)  the Slurm launcher (`sbatch_sweeps.py --dry-run`), no submission: one yaml
        per (dataset, param) and per perf block, each resolving; no two tasks share
        a (dataset, subdir, stem); only the directives asked for; the task dirs'
        links; a do_mnist block never fanned out; no NEW machine-specific value in
        src/, config.yaml, the recipes or the launcher outside its `EPILOG` /
        docstring.
  (xii) no exact-zero budget: on every non-do-MNIST recipe block and
        config.yaml (one experiment, the block's sweep params' runners and its
        query runner), every built interval model's epsilon, ball budget, IV leak
        g and IV radius is > 0; the simulation's gamma_z resolves to 2^-8 under
        `iv: 1`, the cigarettes' to 0.0177; optical has no Z, so its gamma_z 0 is
        exempt and its DA T rows are positive through eps.
        Catches: a budget that silently degenerates to an exact constraint.

    uv run python scripts/a86_gamma_n.py [--only LEG]

Writes only under $A86_TMPROOT (default ~/scratch/tmp/a86), removed on PASS.
"""

import argparse
import ast
import contextlib
import glob
import importlib.util
import inspect
import os
import py_compile
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import yaml
from loguru import logger
from scipy.stats import chi2, ncx2

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

ALPHA = 0.05
SPLIT = ALPHA / 3
# (label, d, g, n, a, gamma_n) at the fixed split, within 1e-4
REFERENCES = (
    ("sim ball", 33, 1.0, 1843, SPLIT, 1.1194),
    ("optical ball", 55, 0.66, 900, SPLIT, 0.8428),
    ("cigarettes ball n 2205", 5, 0.25, 2205, SPLIT, 0.2993),
    ("cigarettes ball n 49", 5, 0.25, 49, SPLIT, 0.7493),
)
MEAN_ROW = (1843, 1.0, 0.0062)  # n, gamma, (1 + gamma) gamma_n(1; 0)
TMPROOT = os.environ.get("A86_TMPROOT", os.path.expanduser("~/scratch/tmp/a86"))
DATASETS = ("simulation", "optical_device", "cigarettes")
# what config.yaml and every non-do-MNIST recipe resolve `gamma_n` to
RECIPE_ALPHA = 0.0
# the cigarette panel's states, the pads' units on `target: iv`
CIGARETTE_STATES = 49
GAMMA_Z_SIM = 2**-8
# the query panels' budgets at c29af19 (`_epsilon_budget(query_epsilon, tol=eps,
# quantile=None)` on opticalDeviceFig6's and cigarettesFig7's blocks)
QUERY_EPSILON_C29AF19 = {"optical_device": 0.3335209199891572, "cigarettes": 0.003906250000000246}
GAMMA_Z_CIGARETTES = 0.0177
VALIDITY_ROWS_DRAWS = 400
VALIDITY_SOLVE_DRAWS = 250
VALIDITY_GAMMA = 0.25
VALIDITY_SEED = 20_000
# every retired symbol of the IM-CI, the pad tolerance and the old IV budgets
RETIRED = (
    "im-ci", "im_ci", "IM_CI", "imbens", "Imbens", "bootstrap_bounds", "results_raw", "_raw_record", "RAW_MTIME_SLACK",
    "sweep_record_for", "pad_tolerance", "iv_recalibrate", "leak_t", "leak_tz", "epsilon_iv_z", "eps_iv_star",
    "eps_iv_z_star", "z_moment_star", "eps_rms", "_declared_allowance", "_z_allowance", "_recalibrated", "t_bound",
    "tz_bound", "z_bound", "declared_iv", "fit_iv_leaks", "fit_epsilon_iv", "fit_epsilon_iv_z",
    "_baseline_epsilon_iv_z", "_step_epsilon_iv", "epsilon_star_pointwise", "invariance_error",
    "eps_tol", "sweep_eps_tol", "PAD_QUANTILE", "epsilon_pad_star", "measured_epsilon_pad", "_pad_budget",
    "pad_epsilon", "epsilon_q95",
)  # fmt: skip
# the retired Imbens-Manski CI's symbols (the yaml key, its parsed name, the
# constants, the helpers, the raw record and its reader, the pad tolerance)
IM_CI_SYMBOLS = re.compile(
    r"im-ci|im_ci|IM_CI|imbens|bootstrap_bounds|results_raw|_raw_record|RAW_MTIME_SLACK|sweep_record_for"
    r"|pad_tolerance",
    re.IGNORECASE,
)
# the lines that name the retired key to reject it
IM_CI_ALLOWED = (("scripts/a77_domnist_block.py", "retired"),)
SWEEP_STEMS = {"values", "results", "statuses", "axis"}
# the machine-specific values no file but the launcher's labelled example may carry
MACHINE = re.compile(r"coc-cpu|oms-csp|coc-ice|ice-cpu|module load|scratch|pace\.gatech|atl1-", re.IGNORECASE)
# pre-existing hits, allowed: the cigarette data fallback, the do-MNIST data dir, a
# log path quoted in a configs.py comment
MACHINE_ALLOWED = (
    ("src/sem/cigarettes.py", 'DATA_ROOT: str = "~/scratch/data"'),
    ("src/sem/do_mnist.py", "~/scratch/data/mnist"),
    ("src/experiments/configs.py", "~/scratch/tmp/impl_v7/logs"),
)
FAIL = []
MADE = []  # the directories this run created under TMPROOT, removed on PASS


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAIL.append(name)


def leg_i():
    from src.methods.sensitivity_models import finite_sample_budget as fsb

    print("(i) finite_sample_budget against scipy")
    worst = 0.0
    for d in (1, 3, 12, 33):
        for g in (0.0, 2**-8, 0.25, 1.0):
            for n in (49, 400, 2205):
                for a in (SPLIT, ALPHA / 6, ALPHA / 9):
                    want = (chi2.ppf(1 - a, d) if g <= 0 else ncx2.ppf(1 - a, d, n * g)) / n
                    worst = max(worst, abs(fsb(d, g, n, a) - want) / want)
    check("(i) == scipy chi2 / ncx2 quantile / n", worst < 1e-12, f"(worst relative {worst:.1e})")
    raw = [(g, fsb(d, g, 400, 0.0)) for d in (1, 33) for g in (0.0, 2**-8, 1.0)]
    check("(i) a = 0 returns g", all(got == g for g, got in raw), f"{raw}")
    check("(i) g < 0 is the central chi-square", fsb(3, -1.0, 400, SPLIT) == fsb(3, 0.0, 400, SPLIT))
    check("(i) increasing in d", fsb(1, 0.25, 400, SPLIT) < fsb(3, 0.25, 400, SPLIT) < fsb(12, 0.25, 400, SPLIT))
    check("(i) increasing in g", fsb(3, 0.0, 400, SPLIT) < fsb(3, 0.1, 400, SPLIT) < fsb(3, 0.25, 400, SPLIT))
    check(
        "(i) increasing as a falls",
        fsb(3, 0.25, 400, SPLIT) < fsb(3, 0.25, 400, ALPHA / 6) < fsb(3, 0.25, 400, ALPHA / 9),
    )
    ladder = [fsb(5, 0.25, n, SPLIT) for n in (49, 220, 2205, 10**6)]
    check(
        "(i) decreasing in n towards g",
        bool(np.all(np.diff(ladder) < 0)) and 0.25 < ladder[-1] < 0.253,
        f"{np.round(ladder, 4).tolist()}",
    )
    for label, d, g, n, a, want in REFERENCES:
        got = fsb(d, g, n, a)
        check(f"(i) reference {label}", abs(got - want) < 1e-4, f"{got:.6f} (want {want})")
    n, gamma, want = MEAN_ROW
    got = (1 + gamma) * fsb(1, 0.0, n, SPLIT)
    check("(i) reference mean row n 1843", abs(got - want) < 1e-4, f"{got:.6f} (want {want})")


def read(path):
    with open(path) as handle:
        return handle.read()


@contextlib.contextmanager
def workdir(prefix):
    """A fresh directory under TMPROOT as cwd, with the repo's `data/` linked in."""
    os.makedirs(TMPROOT, exist_ok=True)
    path = tempfile.mkdtemp(prefix=prefix, dir=TMPROOT)
    MADE.append(path)
    if os.path.isdir(os.path.join(REPO, "data")):
        os.symlink(os.path.join(REPO, "data"), os.path.join(path, "data"))
    os.chdir(path)
    try:
        yield path
    finally:
        os.chdir(REPO)


def fixture(seed=0, n=400, d=8, m=1):
    """A synthetic linear draw: X, y, a perturbed GX and one translation column,
    tiled m-fold with the untiled rows beside it, as the m sweep fits."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = X @ rng.normal(size=d) + rng.normal(size=n)
    G = rng.normal(size=(n * m, 1))
    GX = np.tile(X, (m, 1)) + 0.1 * rng.normal(size=(n * m, d))
    arrays = dict(X=np.tile(X, (m, 1)), y=np.tile(y, m), GX=GX, G=G)
    if m > 1:
        arrays.update(X_base=X, y_base=y, Z_base=np.zeros((n, 0)))
    return arrays, rng.normal(size=(20, d))


def balls(model):
    """The fitted PartialR2 balls of a model: itself, or an intersection's branches."""
    return [model.baseline, model.augmented] if hasattr(model, "baseline") else [model]


def leg_ii():
    from src.experiments.configs import MethodRegistry
    from src.experiments.utils import fit_model
    from src.methods.sensitivity_models import finite_sample_budget as fsb

    print("(ii) the ERM ball and the mean row off the cvx parameters")
    names = ["PI", "PI+INV", "PI+IV", "DA+PI", "PI&DA+PI"]
    gamma, d, n = 0.5, 8, 400
    for alpha, m, cap in ((0.0, 1, None), (ALPHA, 1, None), (ALPHA, 4, None), (ALPHA, 1, 100), (0.0, 4, 100)):
        arrays, queries = fixture(n=n, d=d, m=m)
        builders = MethodRegistry.build_methods(
            names, gamma=gamma, epsilon=1.0, rho=1.3, pad=True, gamma_n_alpha=alpha, unit_cap=cap
        )
        n_eff = n if cap is None else min(n, cap)
        tag = f"alpha {alpha:g}, m {m}, cap {cap}"
        for name in names:
            model = builders[name]()
            fit_model(model=model, method_name=name, **arrays)
            model.predict(queries)
            for ball in balls(model):
                g = ball.budget(gamma)
                want = ball.scale * np.sqrt(fsb(d + 1, g, n_eff, alpha / 3))
                got = ball.radius_param.value
                exact = got == ball.scale * np.sqrt(g) if alpha == 0.0 else abs(got - want) <= 1e-12 * want
                check(f"(ii) {tag}: {name} ball", ball.n_eff == n_eff and exact, f"{got:.6g} (want {want:.6g})")
                if alpha:
                    mean = ball.scale * np.sqrt((1 + g) * fsb(1, 0.0, n_eff, alpha / 3))
                    got = ball.mean_param.value
                    check(f"(ii) {tag}: {name} mean row", abs(got - mean) <= 1e-12 * mean, f"{got:.6g}")
                else:
                    check(f"(ii) {tag}: {name} has no mean row", ball.delta_var is None and not ball.mean_row)
    from src.experiments.cigarettes import CigaretteOrchestrator
    from src.experiments.configs import resolve_dataset_block
    from src.experiments.simulation import SimulationOrchestrator

    with open(os.path.join(REPO, "config.yaml")) as handle:
        config = yaml.safe_load(handle) or {}
    block = {**(config.get("defaults") or {}), **(config.get("cigarettes") or {})}
    block.pop("experiment", None)
    for target, want in (("iv", CIGARETTE_STATES), ("plasmode", None)):
        resolved = resolve_dataset_block("cigarettes", {**block, "target": target})
        orch = CigaretteOrchestrator(**{**resolved, "n_jobs": 1}, hyperparameters={})
        check(
            f"(ii) cigarettes target {target}: unit_cap {want}",
            orch.toggles["unit_cap"] == want and orch.kwargs["unit_cap"] == want,
            f"{orch.toggles['unit_cap']}",
        )
    sim = SimulationOrchestrator(seed=42, kernel_dim=0, treatment_dim=32, methods=["PI"], gamma_n_alpha=ALPHA)
    check(
        "(ii) the simulation toggles carry gamma_n_alpha and no unit_cap",
        sim.toggles["gamma_n_alpha"] == ALPHA and "unit_cap" not in sim.toggles,
    )
    iv_rows()


def iv_rows():
    import src.methods.sensitivity_models as sm
    from src.experiments.configs import MethodRegistry
    from src.experiments.utils import fit_model
    from src.methods.sensitivity_models import finite_sample_budget as fsb

    check("(ii) IV_LAYOUT is the joint row alone", sm.IV_LAYOUT == ("tz",), f"{sm.IV_LAYOUT}")
    arrays, queries = fixture(seed=3)
    rng = np.random.default_rng(4)
    arrays["Z"] = arrays["X"][:, :1] + 0.5 * rng.normal(size=(len(arrays["X"]), 1)) + 1.0
    gamma, epsilon, n = 0.5, 0.1, len(arrays["X"])
    names = ["PI+IV", "PI+INV+IV", "DA+PI+IV(Z)", "DA+PI+IV(T)", "DA+PI+IV(T,Z)", "PI&DA+PI+IV(T,Z)"]
    want_rows = {"PI+IV": ("z",), "PI+INV+IV": ("z",), "DA+PI+IV(Z)": ("z",), "DA+PI+IV(T)": ("t",)}
    want_rows["DA+PI+IV(T,Z)"] = ("tz",)
    for alpha in (0.0, ALPHA):
        builders = MethodRegistry.build_methods(
            names, gamma=gamma, epsilon=epsilon, rho=1.3, pad=True, gamma_z=GAMMA_Z_SIM, gamma_n_alpha=alpha
        )
        for name in names:
            model = builders[name]()
            fit_model(model=model, method_name=name, **arrays)
            model.predict(queries)
            parts = balls(model)
            if name.startswith("PI&"):
                labels = [(f"{name} baseline", parts[0], ("z",)), (f"{name} DA branch", parts[1], ("tz",))]
            else:
                labels = [(name, parts[0], want_rows[name])]
            for label, ball, rows in labels:
                tag = f"alpha {alpha:g}: {label}"
                check(f"(ii) {tag} rows {rows}", ball.rows == rows, f"{ball.rows}")
                g_tilde = ball.budget(gamma)
                da = ball._da_fit
                leak = (epsilon / ball.scale + np.sqrt(GAMMA_Z_SIM / ball.rho)) ** 2 if da else GAMMA_Z_SIM
                for row in ball.rows:
                    dof = {"z": 1, "t": 1, "tz": 2}[row]
                    if alpha:
                        want = np.sqrt(ball.sigma_sq * (1 + g_tilde) * fsb(dof, leak / (1 + g_tilde), n, alpha / 3))
                    else:
                        want = ball.scale * np.sqrt(leak)
                    got = ball.iv_threshold_params[row].value / np.sqrt(ball.N_samples)
                    check(
                        f"(ii) {tag} row {row} radius", abs(got - want) <= 1e-12 * want, f"{got:.6g} (want {want:.6g})"
                    )
                if alpha and ball.rows:
                    row = ball.rows[0]
                    before = ball.iv_threshold_params[row].value
                    ball.predict(queries, epsilon=2 * epsilon)
                    moved = ball.iv_threshold_params[row].value != before
                    ball.predict(queries, epsilon=epsilon)
                    check(f"(ii) {tag}: the row {'moves' if da else 'stays'} with epsilon", moved == da)


def simulation_draw(seed, n=400, d=8, sem=None):
    """One draw of the small iv = 1 simulation: X, y, Z, GX, G, with the SEM."""
    from src.data_augmentors.simulation import NullSpaceTranslation
    from src.oracle import preserve_rng
    from src.sem.simulation import LinearSimulationSEM

    with preserve_rng():
        np.random.seed(VALIDITY_SEED)
        sem = LinearSimulationSEM(treatment_dimension=d, gamma=VALIDITY_GAMMA, iv_dim=1) if sem is None else sem
        da = NullSpaceTranslation(sem.W_XY, kernel_dim=-1)
        np.random.seed(seed)
        X, y = sem.sample(N=n)
        X, Z = sem.split_instruments(X)
        GX, G = da(X)
    return sem, dict(X=X, y=np.asarray(y).ravel(), Z=Z, GX=GX, G=G)


def row_statistics(seed):
    """The padded PI+IV rows at h* on one draw: (ball, mean, Z) held."""
    from src.experiments.configs import MethodRegistry
    from src.experiments.utils import fit_model

    sem, arrays = simulation_draw(seed)
    model = MethodRegistry.build_methods(
        ["PI+IV"], gamma=VALIDITY_GAMMA, epsilon=0.0, gamma_z=GAMMA_Z_SIM, gamma_n_alpha=ALPHA
    )["PI+IV"]()
    fit_model(model=model, method_name="PI+IV", **arrays)
    w = sem.W_XY.ravel()
    X, y, Z = arrays["X"], arrays["y"], arrays["Z"]
    delta = float(model.mu_ @ w - model.y_offset_)
    ball = float(np.mean(((X - model.mu_) @ (w - model.h_erm)) ** 2) + delta**2)
    Q, _ = np.linalg.qr(Z)
    moment = float(np.sum((Q.T @ (y - X @ w)) ** 2) / len(X))
    gamma = model.gamma
    return (
        ball <= model.ball_radius(gamma) ** 2,
        delta**2 <= model.mean_radius(gamma) ** 2,
        moment <= model.iv_radius("z", gamma) ** 2,
    )


def coverage_draw(seed, queries):
    """Simultaneous coverage of h* at `queries` per method on one solved draw."""
    from src.experiments.configs import EPS_TOL, MethodRegistry
    from src.experiments.utils import fit_model
    from src.experiments.utils.metrics import rho_hat

    sem, arrays = simulation_draw(seed)
    names = ["PI", "PI+IV", "DA+PI", "DA+PI+IV(T,Z)", "PI&DA+PI+IV(T,Z)"]
    rho = float(rho_hat(arrays["X"], arrays["GX"], arrays["y"], intercept=True))
    builders = MethodRegistry.build_methods(
        names, gamma=VALIDITY_GAMMA, epsilon=EPS_TOL, rho=rho, pad=True, gamma_z=GAMMA_Z_SIM, gamma_n_alpha=ALPHA
    )
    truth = queries @ sem.W_XY.ravel()
    covered = {}
    for name in names:
        model = builders[name]()
        fit_model(model=model, method_name=name, **arrays)
        bounds = model.predict(queries)
        covered[name] = bool(np.all((bounds[:, 0] <= truth + 1e-9) & (truth <= bounds[:, 1] + 1e-9)))
    return covered


def leg_iv():
    from joblib import Parallel, delayed

    print("(iv) validity on the small iv = 1 simulation")
    held = np.array(Parallel(n_jobs=-1)(delayed(row_statistics)(seed) for seed in range(VALIDITY_ROWS_DRAWS)))
    level = 1 - ALPHA / 3
    se = np.sqrt(level * (1 - level) / len(held))
    for k, row in enumerate(("ball", "mean row", "Z row")):
        rate = float(held[:, k].mean())
        check(f"(iv)(a) {row} holds at h* on >= {level:.4f} - 3 SE", rate >= level - 3 * se, f"{rate:.4f}")
    joint = float(np.all(held, axis=1).mean())
    se = np.sqrt(0.95 * 0.05 / len(held))
    check("(iv)(a) all three jointly on >= 0.95 - 3 SE", joint >= 0.95 - 3 * se, f"{joint:.4f}")
    sem, _ = simulation_draw(0)
    queries = np.random.default_rng(5).normal(size=(20, sem.W_XY.shape[0]))
    draws = Parallel(n_jobs=-1)(delayed(coverage_draw)(10_000 + i, queries) for i in range(VALIDITY_SOLVE_DRAWS))
    for name in draws[0]:
        rate = float(np.mean([draw[name] for draw in draws]))
        nominal = 0.90 if name.startswith("PI&") else 0.95
        se = np.sqrt(nominal * (1 - nominal) / len(draws))
        check(f"(iv)(b) {name}: simultaneous coverage >= {nominal} - 3 SE", rate >= nominal - 3 * se, f"{rate:.3f}")


def leg_v():
    from src.experiments.configs import MethodRegistry
    from src.experiments.utils import fit_model

    print("(v) nesting: every +IV inside its non-IV counterpart")
    sem, arrays = simulation_draw(77)
    queries = np.random.default_rng(6).normal(size=(20, sem.W_XY.shape[0]))
    pairs = [
        ("PI+IV", "PI"),
        ("PI+INV+IV", "PI+INV"),
        ("DA+PI+IV(T,Z)", "DA+PI"),
        ("DA+PI+IV(Z)", "DA+PI"),
        ("DA+PI+IV(T)", "DA+PI"),
        ("PI&DA+PI+IV(T,Z)", "PI&DA+PI"),
    ]
    names = sorted({name for pair in pairs for name in pair})
    for alpha in (0.0, ALPHA):
        builders = MethodRegistry.build_methods(
            names, gamma=VALIDITY_GAMMA, epsilon=0.5, rho=1.2, pad=True, gamma_z=GAMMA_Z_SIM, gamma_n_alpha=alpha
        )
        bounds = {}
        for name in names:
            model = builders[name]()
            fit_model(model=model, method_name=name, **arrays)
            bounds[name] = model.predict(queries)
        for inner, outer in pairs:
            a, b = bounds[inner], bounds[outer]
            tol = 1e-6 * float(np.nanmax(b[:, 1] - b[:, 0]))
            finite = np.all(np.isfinite(a), axis=1)
            inside = np.all(a[finite, 0] >= b[finite, 0] - tol) and np.all(a[finite, 1] <= b[finite, 1] + tol)
            check(
                f"(v) alpha {alpha:g}: {inner} inside {outer}", bool(inside) and finite.any(), f"{finite.sum()} finite"
            )


def leg_iii():
    import src.methods.sensitivity_models as sm
    from src.experiments.configs import MethodRegistry
    from src.experiments.utils import fit_model

    print("(iii) the mean row")
    arrays, queries = fixture()
    queries[0] = arrays["X"].mean(axis=0)
    gamma, ybar = 0.5, float(np.mean(arrays["y"]))
    bounds = {}
    for alpha in (0.0, ALPHA):
        for closed in (False, True):
            sm.CLOSED_FORM_SOLUTION = closed
            try:
                builders = MethodRegistry.build_methods(
                    ["PI", "DA+PI"], gamma=gamma, epsilon=0.0, rho=1.3, pad=False, gamma_n_alpha=alpha
                )
                for name in ("PI", "DA+PI"):
                    model = builders[name]()
                    fit_model(model=model, method_name=name, **arrays)
                    bounds[alpha, closed, name] = (model, model.predict(queries))
            finally:
                sm.CLOSED_FORM_SOLUTION = False
    model, padded = bounds[ALPHA, False, "PI"]
    radius = model.mean_radius(gamma)
    check(
        "(iii) at the design mean the padded PI interval is ybar +- the mean radius",
        np.allclose(padded[0], [ybar - radius, ybar + radius], rtol=0, atol=1e-6 * radius),
        f"{padded[0]} vs {ybar:.6f} +- {radius:.6f}",
    )
    check("(iii) ... where delta binds", abs(abs(float(model.delta_var.value[0])) - radius) <= 1e-6 * radius)
    raw = bounds[0.0, False, "PI"][1]
    check("(iii) the raw PI interval at the design mean is the point ybar", np.allclose(raw[0], ybar, atol=1e-9))
    for name in ("PI", "DA+PI"):
        raw, padded = bounds[0.0, False, name][1], bounds[ALPHA, False, name][1]
        inside = np.all(padded[:, 0] <= raw[:, 0] + 1e-9) and np.all(padded[:, 1] >= raw[:, 1] - 1e-9)
        check(f"(iii) {name}: padded contains raw on 20 queries", inside)
        for alpha in (0.0, ALPHA):
            cvx, closed = bounds[alpha, False, name][1], bounds[alpha, True, name][1]
            gap = float(np.max(np.abs(cvx - closed)) / np.max(cvx[:, 1] - cvx[:, 0]))
            check(f"(iii) {name} alpha {alpha:g}: the closed form == cvxpy", gap < 1e-6, f"(relative {gap:.1e})")
    worst = iv_sees_delta()
    check(
        "(iii) PI+IV: the solved (h, delta) keeps the IV data residual under its threshold",
        worst[False] < 1e-6,
        f"(worst excess {worst[False]:.2e})",
    )
    check(
        "(iii) ... and fails it with delta dropped from the IV row",
        worst[True] > 1e-3,
        f"(worst excess {worst[True]:.2e})",
    )


def iv_violation(model, X, y, Z):
    """The excess of the Z block's data residual over its threshold at the (h, delta)
    of the problem just solved."""
    Q, _ = np.linalg.qr(Z)
    A = model.iv_terms_["z"][0]  # (Q' X_c; sqrt(jitter) I)
    jitter = A[Q.shape[1] :]
    h, delta = model.h_var.value, float(model.delta_var.value[0])
    residual = np.concatenate([Q.T @ (y - model.y_offset_ - (X - model.mu_) @ h - delta), -jitter @ h])
    return float(np.linalg.norm(residual) - model.iv_threshold_params["z"].value)


def iv_sees_delta():
    from src.experiments.configs import MethodRegistry
    from src.experiments.utils import fit_model

    arrays, queries = fixture(seed=1)
    queries[0] = arrays["X"].mean(axis=0)
    rng = np.random.default_rng(2)
    X, y = arrays["X"], arrays["y"]
    # an instrument off its mean, correlated with X: Q' 1 is far from zero
    Z = (X[:, :1] + 0.5 * rng.normal(size=(len(X), 1))) + 3.0
    worst = {}
    for dropped in (False, True):
        builders = MethodRegistry.build_methods(["PI+IV"], gamma=0.5, epsilon=0.0, gamma_z=2**-6, gamma_n_alpha=ALPHA)
        model = builders["PI+IV"]()
        fit_model(model=model, method_name="PI+IV", X=X, y=y, Z=Z)
        if dropped:
            model.iv_terms_ = {k: (A, b, np.zeros_like(q), d) for k, (A, b, q, d) in model.iv_terms_.items()}
            model._setup_cvx_problems()
        model._set_solver_parameters(model.gamma)
        excess = []
        for query in queries - model.mu_:
            # as `_solve_single` normalises: a query at the design mean keeps delta
            norm = np.linalg.norm(query)
            norm = 1.0 if norm < 1e-9 else norm
            model.x_param.value = query / norm
            model.weight_param.value = 1.0 / norm
            for problem in (model.min_problem, model.max_problem):
                problem.solve(solver="CLARABEL")
                excess.append(iv_violation(model, X, y, Z))
        worst[dropped] = max(excess)
    return worst


def leg_vi():
    from src.experiments.configs import resolve_dataset_block

    print("(vi) config")
    minimal = {"seed": 42, "kernel_dim": 0}
    for level, want in ((95, 0.05), (0, 0.0), (False, 0.0), (90.0, 0.1)):
        got = resolve_dataset_block("simulation", {**minimal, "gamma_n": level})["gamma_n_alpha"]
        check(f"(vi) gamma_n {level!r} -> {want}", got == want, f"{got!r}")
    messages = []
    sink = logger.add(lambda message: messages.append(str(message)), level="INFO")
    try:
        got = resolve_dataset_block("simulation", dict(minimal))["gamma_n_alpha"]
    finally:
        logger.remove(sink)
    absent = [m for m in messages if "gamma_n absent" in m]
    check("(vi) absent -> 0.0 with one INFO line", got == 0.0 and len(absent) == 1, f"{absent}")
    for level in (True, 100, -5, "95"):
        try:
            resolve_dataset_block("simulation", {**minimal, "gamma_n": level})
            raised = False
        except ValueError:
            raised = True
        check(f"(vi) gamma_n {level!r} raises", raised)
    for retired in ({"im-ci": 95}, {"gamma_n_alpha": 0.05}):
        try:
            resolve_dataset_block("simulation", {**minimal, **retired})
            raised = False
        except ValueError:
            raised = True
        check(f"(vi) {sorted(retired)[0]} in the yaml raises (unknown)", raised)
    sources = ["config.yaml", *sorted(glob.glob(os.path.join(REPO, "recipes", "*.yaml")))]
    for source in sources:
        with open(os.path.join(REPO, source)) as handle:
            config = yaml.safe_load(handle) or {}
        defaults = config.get("defaults") or {}
        for dataset in DATASETS:
            if dataset not in config:
                continue
            block = {**defaults, **config[dataset]}
            block.pop("experiment", None)
            got = resolve_dataset_block(dataset, block)["gamma_n_alpha"]
            check(f"(vi) {os.path.basename(source)} {dataset}: alpha {RECIPE_ALPHA:g}", got == RECIPE_ALPHA, f"{got}")
    domnist = {
        "seed": 42,
        "augmentation": "translate",
        "gamma": 0.085,
        "epsilon": 0.04,
        "methods": ["PI"],
        "gamma_n": 95,
    }
    resolved = resolve_dataset_block("do_mnist", domnist)
    check("(vi) do-MNIST accepts gamma_n and carries no gamma_n_alpha", "gamma_n_alpha" not in resolved)
    one_tolerance()


def raw_gamma(tag, runner, declared):
    """The declared gamma is a raw squared radius: rescaled by 1/sigma-hat^2 of the
    draw, so the PI ball's radius comes back to sqrt(declared)."""
    from src.experiments.utils import fit_model
    from src.experiments.utils.metrics import sigma_sq_hat

    sigma_sq = sigma_sq_hat(runner.X, runner.y, intercept=runner.mean_match)
    check(
        f"(vi) {tag}: runner.default_gamma == declared gamma / sigma-hat^2",
        runner.raw_gamma and np.isclose(runner.default_gamma, declared / sigma_sq, rtol=1e-12),
        f"{runner.default_gamma:.6g} vs {declared:.6g} / {sigma_sq:.6g}",
    )
    model = runner.methods["PI"]()
    fit_model(model=model, method_name="PI", X=runner.X, y=runner.y, GX=runner.GX, G=runner.G)
    radius = model.scale * np.sqrt(model.budget(model.gamma))
    check(f"(vi) {tag}: fitted PI radius == sqrt(declared gamma)", np.isclose(radius, np.sqrt(declared), rtol=1e-6))


def recipe_block(recipe, dataset):
    from src.experiments.configs import resolve_dataset_block

    with open(os.path.join(REPO, "recipes", f"{recipe}.yaml")) as handle:
        config = yaml.safe_load(handle)
    block = {**config["defaults"], **config[dataset], "n_jobs": 1, "n_experiments": 1}
    block.pop("experiment", None)
    return resolve_dataset_block(dataset, block)


def one_tolerance():
    import dataclasses

    from src.experiments import configs
    from src.experiments.cigarettes import CigaretteOrchestrator
    from src.experiments.configs import EPS_TOL, OPTICAL_CONFIG, SIMULATION_CONFIG
    from src.experiments.do_mnist import DoMNISTOrchestrator, DoMNISTQuerySweep
    from src.experiments.optical_device import OpticalOrchestrator
    from src.experiments.simulation import SimulationOrchestrator
    from src.experiments.utils import fit_model, set_seed
    from src.oracle import epsilon_star_q95

    check("(vi) EPS_TOL is 2^-8", EPS_TOL == 2**-8, f"{EPS_TOL!r}")
    owned = [
        f"{config.__name__}.{field.name}"
        for config in (configs.SimulationConfig, configs.OpticalDeviceConfig, configs.CigaretteConfig)
        for field in dataclasses.fields(config)
        if "tol" in field.name or field.name.startswith("pad")
    ]
    check("(vi) no dataset config carries a tolerance or a pad budget of its own", not owned, f"{owned}")
    set_seed(42)
    optical = OpticalOrchestrator(**recipe_block("opticalDeviceFig6", "optical_device"), hyperparameters={})
    rms = optical.measured_epsilon_star(None)
    check("(vi) optical _epsilon_budget(None) == RMS eps* + EPS_TOL", optical._epsilon_budget(None) == rms + EPS_TOL)
    sem, da, features = optical._oracle_pieces(0.95)
    q95 = epsilon_star_q95(sem, da, X=sem.X, features=features)
    check(
        "(vi) optical _epsilon_budget(None, q 0.95) == the q0.95 reading + EPS_TOL",
        optical._epsilon_budget(None, quantile=0.95) == q95 + EPS_TOL,
        f"{q95:.6f} + 2^-8",
    )
    got = optical._epsilon_budget(OPTICAL_CONFIG.query_epsilon, quantile=OPTICAL_CONFIG.query_epsilon_quantile)
    check(
        "(vi) optical query budget bit-identical to c29af19", got == QUERY_EPSILON_C29AF19["optical_device"], f"{got!r}"
    )
    runner = optical.get_query_runner_cls()(methods=optical.methods, **optical._get_clean_kwargs())
    raw_gamma("optical query", runner, OPTICAL_CONFIG.gamma)
    model = runner.methods["DA+PI"]()
    fit_model(model=model, method_name="DA+PI", X=runner.X, y=runner.y, GX=runner.GX, G=runner.G)
    check("(vi) optical DA+PI: the pad is its epsilon", model.pad and model.pad_amount == float(model.epsilon))
    cigarettes = CigaretteOrchestrator(**recipe_block("cigarettesFig7", "cigarettes"), hyperparameters={})
    got = cigarettes._epsilon_budget(None, quantile=None)
    check(
        "(vi) cigarette query budget bit-identical to c29af19", got == QUERY_EPSILON_C29AF19["cigarettes"], f"{got!r}"
    )
    set_seed(42)
    block = {**recipe_block("ivSimulationFig5", "simulation"), "methods": ["PI", "PI+IV", "DA+PI+IV(T,Z)"]}
    sim = SimulationOrchestrator(**block, hyperparameters={})
    runner = sim.get_query_runner_cls()(methods=sim.methods, **sim._get_clean_kwargs())
    want = float(runner.oracle.epsilon_star) + EPS_TOL
    check(
        "(vi) simulation query eps == its oracle RMS eps* + 2^-8 (c29af19's declared 2^-8 to 1e-12)",
        SIMULATION_CONFIG.epsilon is None and runner.default_epsilon == want and abs(want - 2**-8) < 1e-12,
        f"{runner.default_epsilon!r}",
    )
    raw_gamma("simulation query", runner, SIMULATION_CONFIG.gamma)
    source = inspect.getsource(DoMNISTOrchestrator.build_methods) + inspect.getsource(DoMNISTQuerySweep.__init__)
    check(
        "(vi) do-MNIST: the copsens T-as-IV cone takes the block epsilon",
        "epsilon_iv=self.epsilon if epsilon_iv is None else epsilon_iv" in source
        and "epsilon_iv=self.default_epsilon" in source,
    )


def leg_vii():
    from src.aggregate import sweep_grid
    from src.experiments.base import BaseExperimentRunner, ExperimentOrchestrator
    from src.methods.sensitivity_models import BoundedSA

    print("(vii) one record, no Imbens-Manski CI left")
    stems = set(re.findall(r'f"\{param\}_(\w+)"', inspect.getsource(ExperimentOrchestrator._run_sweeps)))
    check("(vii) _run_sweeps writes values, results, statuses and axis only", stems == SWEEP_STEMS, f"{sorted(stems)}")
    read_back = set(re.findall(r"\{param\}_(\w+)\.pkl", inspect.getsource(sweep_grid)))
    check("(vii) the aggregate's sweep grid reads values, results and axis", read_back == {"values", "results", "axis"})
    check(
        "(vii) the runner takes no CI level",
        not any("ci" in name.split("_") for name in inspect.signature(BaseExperimentRunner.__init__).parameters),
    )
    check("(vii) BoundedSA has no pad tolerance", not any("tolerance" in name for name in vars(BoundedSA)))
    check(
        "(vii) no CI module under src.experiments.utils",
        importlib.util.find_spec("src.experiments.utils.im_ci") is None,
    )
    paths = glob.glob(os.path.join(REPO, "src", "**", "*.py"), recursive=True)
    paths += glob.glob(os.path.join(REPO, "scripts", "*.py")) + glob.glob(os.path.join(REPO, "recipes", "*.yaml"))
    paths += [os.path.join(REPO, name) for name in ("config.yaml", "README.md", os.path.join("scripts", "README.md"))]
    hits = []
    for path in paths:
        rel = os.path.relpath(path, REPO)
        if rel == os.path.join("scripts", os.path.basename(__file__)):
            continue
        for number, line in enumerate(read(path).splitlines(), start=1):
            if IM_CI_SYMBOLS.search(line) and not any(rel == where and text in line for where, text in IM_CI_ALLOWED):
                hits.append(f"{rel}:{number}")
    check("(vii) no Imbens-Manski CI symbol left", not hits, f"{hits[:8]}")


def leg_x():
    from src.experiments.configs import N_PERCENT_RANGE, PARAM_SPECS
    from src.experiments.simulation import SimulationOrchestrator
    from src.experiments.utils import set_seed

    print("(x) the m grid reads sweep_samples, the n grid is log-spaced on 10..100 %")
    for dataset in DATASETS:
        want = list(range(1, 17))
        got = PARAM_SPECS["m"].grid_fn(dataset, 16)
        check(f"(x) {dataset} m at 16 is 1..16", got.tolist() == want, f"{got.tolist()}")
        grid = PARAM_SPECS["m"].grid_fn(dataset, 32)
        check(f"(x) {dataset} m at 32: 32 points from 1", len(grid) == 32 and grid[0] == 1)
        check(f"(x) {dataset} m at 8 is 1..8", PARAM_SPECS["m"].grid_fn(dataset, 8).tolist() == list(range(1, 9)))
        grids = [PARAM_SPECS["n"].grid_fn(dataset, count) for count in (8, 16, 32)]
        check(
            f"(x) {dataset} n at 8, 16, 32 == geomspace({N_PERCENT_RANGE[0]:g}, {N_PERCENT_RANGE[1]:g}, count)",
            all(
                np.array_equal(grid, np.geomspace(*N_PERCENT_RANGE, num=count))
                for grid, count in zip(grids, (8, 16, 32), strict=True)
            ),
            f"{grids[0].tolist()}",
        )
    set_seed(42)
    orch = SimulationOrchestrator(
        seed=42,
        n_samples=2048,
        kernel_dim=0,
        treatment_dim=32,
        augmentation="translate",
        n_experiments=1,
        sweep_samples=8,
        methods=["PI"],
        hyperparameters={},
        n_jobs=1,
        recalibrate=True,
        pad=True,
        clipy=False,
        mean_match=True,
    )

    def runner_for(param):
        return orch.get_sweep_runner_cls(param)(
            methods=orch.methods, method_factory=orch.build_methods, **orch._get_clean_kwargs()
        )

    rows = runner_for("n").get_param_range().tolist()
    want_rows = [205, 285, 395, 549, 763, 1061, 1474, 2048]
    check("(x) the sim n strategy takes the grid's rows of 2048 at 8", rows == want_rows, f"{rows}")
    runner = runner_for("m")
    check("(x) the sim m strategy takes 8 steps at sweep_samples 8", len(runner.get_param_range()) == 8)
    check("(x) ... at 6.25% of 2048 = 128 rows", runner.n_samples_override == 128, f"{runner.n_samples_override}")
    tree = ast.parse(read(os.path.join(REPO, "src/experiments/configs.py")))
    literals = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values, strict=True):
            if isinstance(key, ast.Constant) and key.value in ("n", "m") and isinstance(value, ast.Call):
                for keyword in value.keywords:
                    if keyword.arg == "grid_fn":
                        numbers = [c.value for c in ast.walk(keyword.value) if isinstance(c, ast.Constant)]
                        literals[key.value] = [v for v in numbers if v in (16, 17) and not isinstance(v, bool)]
    check("(x) both grid_fns found in PARAM_SPECS", set(literals) == {"n", "m"}, f"{sorted(literals)}")
    check("(x) no 16 / 17 literal left in either", not any(literals.values()), f"{literals}")


def leg_ix():
    print("(ix) the purge: no retired symbol, every script loads")
    # case-sensitive, whole words: `EPS_TOL`, the one tolerance, is not `eps_tol`
    pattern = re.compile(r"(?<![\w-])(" + "|".join(re.escape(token) for token in RETIRED) + r")(?![\w-])")
    paths = glob.glob(os.path.join(REPO, "src", "**", "*.py"), recursive=True)
    paths += glob.glob(os.path.join(REPO, "scripts", "*.py")) + glob.glob(os.path.join(REPO, "recipes", "*.yaml"))
    paths += [os.path.join(REPO, name) for name in ("config.yaml", "README.md", os.path.join("scripts", "README.md"))]
    hits = []
    for path in paths:
        rel = os.path.relpath(path, REPO)
        if rel == os.path.join("scripts", os.path.basename(__file__)):
            continue
        for number, line in enumerate(read(path).splitlines(), start=1):
            if pattern.search(line) and not any(rel == where and text in line for where, text in IM_CI_ALLOWED):
                hits.append(f"{rel}:{number}:{pattern.search(line).group(0)}")
    check("(ix) no retired symbol left", not hits, f"{hits[:8]}")
    scripts = sorted(glob.glob(os.path.join(REPO, "scripts", "*.py")))
    broken = []
    for path in scripts:
        try:
            py_compile.compile(path, doraise=True, cfile=os.path.join(tempfile.gettempdir(), "a86_pyc"))
        except py_compile.PyCompileError as error:
            broken.append(f"{os.path.basename(path)}: {error.msg[:80]}")
    check(f"(ix) py_compile on all {len(scripts)} scripts", not broken, f"{broken}")
    ruff = subprocess.run(  # noqa: S603 - the repo's own linter
        [sys.executable, "-m", "ruff", "check", "--select", "F", "src", "scripts"],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    check("(ix) ruff check --select F", ruff.returncode == 0, ruff.stdout[-300:])
    failed = []
    # a script without a command line runs itself whatever its arguments, so its
    # load check is `py_compile` above
    for path in [path for path in scripts if "argparse" in read(path)]:
        done = subprocess.run(  # noqa: S603 - our own scripts
            [sys.executable, path, "--help"], capture_output=True, text=True, cwd=REPO, timeout=300
        )
        if done.returncode != 0:
            failed.append(os.path.basename(path))
    check("(ix) every script with a command line exits 0 on --help", not failed, f"{failed}")


def leg_xii():
    from munch import munchify

    from src.experiments.base import SweepData
    from src.experiments.configs import PARAM_SPECS, resolve_dataset_block
    from src.experiments.utils import fit_model
    from src.main import ORCHESTRATORS

    print("(xii) no exact-zero budget on the production configs")
    sources = ["config.yaml", *sorted(glob.glob(os.path.join(REPO, "recipes", "*.yaml")))]
    seen = set()
    for source in sources:
        with open(os.path.join(REPO, source)) as handle:
            config = yaml.safe_load(handle) or {}
        defaults = config.get("defaults") or {}
        for dataset in DATASETS:
            if dataset not in config:
                continue
            raw = {**defaults, **config[dataset]}
            experiment = raw.pop("experiment", None) or {}
            sweep = (experiment.get("sweep") or {}).get("param", [])
            params = [p for p in sweep if p in PARAM_SPECS] + (["query"] if experiment.get("query") else [])
            block = resolve_dataset_block(dataset, {**raw, "n_experiments": 1, "sweep_samples": 3, "n_jobs": 4})
            key = (dataset, repr(sorted((k, repr(v)) for k, v in block.items())), tuple(params))
            if key in seen:
                continue
            seen.add(key)
            orch = ORCHESTRATORS[dataset](**block, hyperparameters=munchify({}))
            want_z = {"simulation": GAMMA_Z_SIM if block.get("iv", 0) else 0.0, "cigarettes": GAMMA_Z_CIGARETTES}
            gamma_z = orch.toggles.get("gamma_z", getattr(orch, "gamma_z", 0.0))
            if dataset in want_z and (dataset == "simulation" or block.get("iv")):
                check(
                    f"(xii) {os.path.basename(source)} {dataset}: gamma_z {want_z[dataset]:g}",
                    gamma_z == want_z[dataset],
                    f"{gamma_z}",
                )
            for param in params:
                tag = f"{os.path.basename(source)} {dataset} {param}"
                if param == "query":
                    runner = orch.get_query_runner_cls()(methods=orch.methods, **orch._get_clean_kwargs())
                    models = {}
                    for name, builder in runner.methods.items():
                        model = builder()
                        if model is None or not hasattr(model, "epsilon"):
                            continue
                        fit_model(
                            model=model, method_name=name, X=runner.X, y=runner.y, GX=runner.GX, G=runner.G, Z=runner.Z
                        )
                        models[name] = model
                else:
                    runner = orch.get_sweep_runner_cls(param)(
                        methods=orch.methods, method_factory=orch.build_methods, **orch._get_clean_kwargs()
                    )
                    first = runner.get_param_range()[0]
                    data = runner.generate_data(0, first)
                    models = runner.build_models(0, 0, SweepData.coerce(data))
                    models = {name: model for name, model in models.items() if hasattr(model, "epsilon")}
                bad = []
                for name, model in models.items():
                    for part in balls(model):
                        values = {"epsilon": float(part.epsilon), "ball": part.ball_budget(part.gamma)}
                        for row in getattr(part, "rows", ()):
                            values[f"g_{row}"] = part.iv_leak()
                            values[f"r_{row}"] = part.iv_radius(row, part.gamma)
                        bad += [f"{name}:{k}={v:.3g}" for k, v in values.items() if not v > 0.0]
                check(f"(xii) {tag}: every epsilon, ball, IV leak and radius > 0", not bad, f"{bad[:6]}")


def example_lines(path):
    """The launcher's lines inside its module docstring or its `EPILOG` string, the
    one place a labelled PACE example may appear."""
    tree = ast.parse(read(path))
    spans = []
    if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant):
        spans.append((tree.body[0].lineno, tree.body[0].end_lineno))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "EPILOG" for t in node.targets):
            spans.append((node.lineno, node.end_lineno))
    return {line for start, end in spans for line in range(start, end + 1)}


def dry_run(launcher, config, out, *flags):
    return subprocess.run(  # noqa: S603 - our own script
        [sys.executable, launcher, "--config", config, "--out", out, "--dry-run", *flags],
        capture_output=True,
        text=True,
        cwd=REPO,
    )


def leg_xi():
    from src.experiments.configs import parse_experiment_plan, resolve_dataset_block

    print("(xi) the Slurm launcher, dry run")
    launcher = os.path.join(REPO, "scripts", "sbatch_sweeps.py")
    for source in ("config.yaml", "recipes/nEfficiencyFig13.yaml"):
        with open(os.path.join(REPO, source)) as handle:
            config = yaml.safe_load(handle) or {}
        expected = set()
        for dataset, blk in config.items():
            if dataset in ("defaults", "hyperparameters") or not isinstance(blk, dict):
                continue
            experiment = blk.get("experiment") or {}
            expected |= {f"{dataset}_{p}" for p in (experiment.get("sweep") or {}).get("param", [])}
            if experiment.get("perf"):
                expected.add(f"{dataset}_perf")
        with workdir("launch_") as out:
            done = dry_run(launcher, os.path.join(REPO, source), out)
            check(f"(xi) {source}: the dry run exits 0", done.returncode == 0, done.stderr[-300:])
            emitted = {os.path.basename(p)[: -len(".yaml")] for p in glob.glob(os.path.join(out, "configs", "*.yaml"))}
            check(
                f"(xi) {source}: one yaml per (dataset, param) and perf block",
                emitted == expected,
                f"{sorted(emitted)}",
            )
            stems = []
            for task in sorted(emitted):
                with open(os.path.join(out, "configs", f"{task}.yaml")) as handle:
                    cfg = yaml.safe_load(handle)
                check(
                    f"(xi) {task}.yaml carries the source's defaults and hyperparameters verbatim",
                    cfg.get("defaults") == config.get("defaults")
                    and cfg.get("hyperparameters") == config.get("hyperparameters"),
                )
                try:
                    for dataset, blk in cfg.items():
                        if dataset in ("defaults", "hyperparameters"):
                            continue
                        merged = {**(cfg.get("defaults") or {}), **blk}
                        plan = parse_experiment_plan(merged.get("experiment"))
                        resolve_dataset_block(dataset, merged)
                        stems += [(dataset, "sweep", p) for p in (plan.sweep.param if plan.sweep else ())]
                        stems += [(dataset, "perf", "epsilon")] if plan.perf else []
                    ok, detail = True, ""
                except Exception as error:
                    ok, detail = False, f"{type(error).__name__}: {error}"
                check(f"(xi) {task}.yaml resolves", ok, detail)
            check(f"(xi) {source}: no two tasks share a (dataset, subdir, stem)", len(stems) == len(set(stems)))
            script = read(os.path.join(out, "run.sbatch"))
            asked = re.findall(r"^#SBATCH\s+--(partition|account|qos|cpus-per-task|mem)", script, re.MULTILINE)
            check(f"(xi) {source}: no partition / account / QOS / cpus / mem directive unasked", not asked, f"{asked}")
            check(f"(xi) {source}: no env-setup line unasked", "module load" not in script)
            links = glob.glob(os.path.join(out, "task_*"))
            good = [
                os.path.realpath(os.path.join(t, "artifacts")) == os.path.realpath(os.path.join(REPO, "artifacts"))
                and os.path.realpath(os.path.join(t, "data")) == os.path.realpath(os.path.join(REPO, "data"))
                for t in links
                if os.path.islink(os.path.join(t, "artifacts")) and os.path.islink(os.path.join(t, "data"))
            ]
            check(
                f"(xi) {source}: every task dir links artifacts/ and data/ to the repo",
                links and len(good) == len(links) and all(good),
            )
    with workdir("launch_flags_") as out:
        flags = ["--partition", "P0", "--account", "A0", "--qos", "Q0", "--env-setup", "echo setup"]
        flags += ["--cpus-per-task", "3", "--mem", "5G"]
        done = dry_run(launcher, os.path.join(REPO, "config.yaml"), out, *flags)
        check("(xi) the flagged dry run exits 0", done.returncode == 0, done.stderr[-300:])
        script = read(os.path.join(out, "run.sbatch"))
        asked = ("--partition=P0", "--account=A0", "--qos=Q0", "--cpus-per-task=3", "--mem=5G", "echo setup")
        check("(xi) asked directives are emitted", all(s in script for s in asked))
    with workdir("launch_domnist_") as out:
        synthetic = os.path.join(out, "with_do_mnist.yaml")
        with open(synthetic, "w") as handle:
            yaml.safe_dump(
                {
                    "defaults": {"n_jobs": 1},
                    "simulation": {"seed": 42, "kernel_dim": 0, "experiment": {"sweep": {"param": ["n"]}}},
                    "do_mnist": {
                        "seed": 42,
                        "experiment": {"sweep": {"param": ["m"]}, "perf": {"metric": ["seed_var"]}},
                    },
                },
                handle,
            )
        dry_run(launcher, synthetic, out, "--tasks", "sweep,perf,query")
        emitted = sorted(os.path.basename(p) for p in glob.glob(os.path.join(out, "configs", "*.yaml")))
        check("(xi) a do_mnist block is never fanned out", emitted == ["simulation_n.yaml"], f"{emitted}")
    hits = []
    paths = glob.glob(os.path.join(REPO, "src", "**", "*.py"), recursive=True)
    paths += [os.path.join(REPO, "config.yaml"), launcher, *glob.glob(os.path.join(REPO, "recipes", "*.yaml"))]
    allowed_launcher = example_lines(launcher)
    for path in paths:
        rel = os.path.relpath(path, REPO)
        for number, line in enumerate(read(path).splitlines(), start=1):
            if not MACHINE.search(line):
                continue
            if path == launcher and number in allowed_launcher:
                continue
            if any(rel == where and text in line for where, text in MACHINE_ALLOWED):
                continue
            hits.append(f"{rel}:{number}")
    check("(xi) no NEW machine-specific value outside the launcher's labelled example", not hits, f"{hits[:6]}")


LEGS = {
    "i": leg_i,
    "ii": leg_ii,
    "iii": leg_iii,
    "iv": leg_iv,
    "v": leg_v,
    "vi": leg_vi,
    "vii": leg_vii,
    "ix": leg_ix,
    "x": leg_x,
    "xi": leg_xi,
    "xii": leg_xii,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=sorted(LEGS), default=None)
    args = parser.parse_args()
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    os.chdir(REPO)
    for name, leg in LEGS.items():
        if args.only in (None, name):
            leg()
    if not FAIL:
        for path in MADE:
            shutil.rmtree(path, ignore_errors=True)
        with contextlib.suppress(OSError):
            os.rmdir(TMPROOT)  # only when nothing else lives there
    print(f"\n{'A86 ALL PASS' if not FAIL else 'A86 FAILURES: ' + ', '.join(FAIL)}")
    sys.exit(bool(FAIL))
