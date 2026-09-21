"""A64: the (T) instrument mode, the perf sweeps and the aggregate grids.

Four commits on refactor6/iv-figures (round 12): `DA+PI+IV(T)` and its siblings
constrain the translation amounts alone at the T-side term (`parse_method`,
`build_methods`, `fit_model`, the intersection's `instrument`); `perf` is two
epsilon sweeps on the robustness grid (`src/experiments/perf.py`: a cumulative
wall clock that re-solves only the programs reading epsilon and `repad`s the rest,
and the backend cross-check D(eps); round 16 adds the feasible rate over the same
backends and drops the failure markers); `python -m src.aggregate` draws the
3 x [datasets] sweep grids and the perf grids from the pkls (both through
`_metric_grid` since round 16). Legs:

  (D)    the digest leg (scripts/digest_leg.py), as a56 to a63: no shipped block
         spells a mode or plans perf, bare names dispatch as before, `_draw_series`
         moves no number and figures are not hashed. Catches: a bare name parsed
         as the T mode (every DA+PI+IV pkl moves). Misses: everything below.
  (i)    the grammar and the display tables: the four `(T)` spellings parse as
         (base, "T"), four malformed or misplaced suffixes are rejected naming the
         entry, `IV_MODES` is the three modes, a `(T)` block resolves under an empty
         set and beside the bare and `(Z)` entries, a duplicate `(T)` pair and a
         `(T)` on do_mnist raise, every `base(T)` has its display entries on the
         base's hue, `DA+ERM+IV(T)` is a point estimate, the T style is neither solid,
         the Z pattern nor the point-estimate dash, the registry keeps the order.
         Catches: a parser without the mode, a missing display entry (a KeyError at
         plot time), the T sibling drawn like the Z one. Misses: what (T) computes.
  (ii)   dispatch on fitted attributes, a63-ii's construction (the cigarette design,
         the seed-42 DA draw, a 1-column Z, two distinct IV budgets, `rho = rho_hat`):
         the (T) instrument is G alone (width `G.shape[1]` against +1 on bare), its
         budget the T-side `epsilon_iv` with `gamma_z` 0 so `iv_bound` is exactly
         `epsilon_iv`, the intersection's DA branch the same and its baseline
         untouched, (T) predicts differently from bare and from (Z), `DA+ERM+IV(T)`
         zeroes the demeaned moment on (GX, y, G) and is the ERM optimum within it,
         beating a fresh 2SLS; under an empty Z and gamma_z 0 (the runner's values
         there) `(T)` equals `(T,Z)` to exactly 0.0 in bounds and statuses on both
         classes and `DA+ERM+IV(T)` equals `DA+ERM+IV`. Catches: Z stacked into the T
         branch (width), the real-Z radius left on the T builder (`iv_bound`).
         Misses: a wrong budget compensated by a wrong instrument.
  (iii)  the cumulation table on fitted call counts, not timings: a spy on
         `PartialR2._prepare` (one call per solve, on every branch too) under
         `perf.wall_clock` at 3 repeats on a 4-step grid: `PI` 3, `PI+INV` 12,
         `DA+PI+IV(T)` 3, `PI&DA+PI+IV(T)` 6, the ten-method list 48; the cumulative
         arrays non-decreasing, `PI+INV` rising, `PI` flat to 1e-3 s; the normaliser
         makes 4 solves and keeps the median of the last 3, and `baseline_pi` is
         serial under a block saying `n_jobs: -1`. Catches: a ball that re-solves
         (`solves_on_epsilon` on PartialR2), the INV cone that does not. Misses: the
         seconds themselves, which nothing can pin.
  (iv)   D(eps) on a synthetic record, 3 runs, 2 steps, 4 queries: the step-0 terms
         and D(0) by hand; at step 1 one FAILURE status and one lower > upper
         planted, `failures == [0, 2]`, the two queries' sd over the surviving two
         runs, D(1) by hand; a third failure on one query leaves one run and a NaN
         term that `nanmean` skips; a status of 1 counts like a 2. Catches: the
         `lower > upper` clause dropped, `ddof` 0. Misses: nothing on the formula.
  (v)    `clip_y` on rendered artists: the record {PI 1, PI+INV 1..100} rendered
         twice on log y, the frame equal to `_pad(_limits(clip=False))` under
         `clip_y=False` and to `_pad(_limits(clip=True))` by default, the former
         higher. Catches: the `clip_y` forwarding dropped. (The failure markers
         this leg used to pin are gone; a70 leg 4 pins their absence.)
  (vi)   the aggregate on a synthetic tree (simulation gamma and omega, cigarettes
         gamma without coverage and no omega, cigarettes perf, no optical): the CLI as
         a subprocess writes exactly the four pdfs (a grid per param, one per
         `PERF_FIGURES` entry); in-process the grid has the titles, the blank cells,
         one shared y on `CLAMP_YLIM`, the axis pkl's x-label, the legend in the
         repo's order in one row, the render fold (bare beside `(T,Z)` is one entry,
         and so is `(T)` beside bare), the three y-labels without " / "; the stacked
         perf grid has stability over feasible rate, the sim column blank, no marker
         line, the feasibility row alone on `CLAMP_YLIM`; ten perf methods draw five
         columns of two, with no WARNING, clear of the titles; the x-label at 0.5 on
         two columns and, with optical added, under the middle column of three (the
         sweep grid and both perf grids); and `sweep_grid` after the `_metric_grid`
         refactor matches round 16's branch 21 on four of these trees (axes count,
         cells on, titles, y-labels, legend entries, x-label and its x; RECORDED).
         Catches: rows reordered, a missing metric not blanked, the legend sorted by
         name or keyed on the spelling, the x-label left at 0.5 over three columns,
         the refactor moving the sweep grid.
  (vii)  the utility on the shipped artifacts (`--shipped DIR`): exit 0, exactly the
         pdfs the tree calls for (a grid per sweep param, one per `PERF_FIGURES`
         entry with a metric some dataset ran, the elasticity grid when the cigarette query pkls are there),
         the tree's mtimes unchanged; in-process under `captured()` no WARNING and
         the epsilon legend's 10 folded entries in two rows, five columns of two
         with green DA+PI+IV over pink PI&DA+PI+IV last. The wanted list is
         re-derived from the SAME four predicates `main` uses (`sweep_params`,
         `PERF_FIGURES`, `_has_perf`, `_has_elasticities`), so it catches `main`
         misusing one; a defect INSIDE one of them is invisible here. Catches: a
         crash on the real pkls, a grid silently not drawn. No break-it. Without
         a tree at `--shipped` the leg is SKIPPED and counted in the summary line.
  (viii) the perf path end to end on the simulation block (5 methods, 4 steps,
         `n_experiments 2` forced to 1, serial, all three metrics): the seven pkls
         and three pdfs, the grid,
         the wall-clock shapes and shape of the curves, the seed_var terms finite
         exactly where two runs survive and NaN elsewhere, `PI+INV` NaN at its
         INFEASIBLE steps and every method finite at r = 1, `failures` equal to the
         (run, query) pairs re-derived from `perf.bounds_along` per backend (612 for
         `PI+INV` there, RECORDED), the statuses pkl equal to the re-derived codes,
         the feasibility pkl equal to the re-derived share of usable runs,
         `meta`; `repeats` validation; do-MNIST perf inspected as source only.
         Catches: the `n_experiments` override dropped, the wrong grid, failures
         counted over runs, `ddof` 0. Misses: optical and cigarettes perf.
  (ix)   the legend by entry count, read back off the rendered text extents
         rather than off `ncol`: up to LEGEND_FLAT_MAX entries one row of n in the
         repo's order, more LEGEND_ROWS rows of ceil(n / 2) columns. The target
         table: the two live no-Z blocks pooled (6 entries, one row, no dash-dot),
         all three shipped datasets (12 keys folded to 10, five columns of two, the
         fifth green DA+PI+IV over pink PI&DA+PI+IV), cigarettes alone (one row of
         6), two paired groups (one row of 4), nine singletons (columns
         [2,2,2,2,1]), 11 entries (6 columns and one WARNING). On the two-row
         shapes the entry order has every paired group ahead of every singleton,
         each part in PAIR_ORDER order, and every pair in one column with member 0
         on top; `perf_grid` draws the same five columns. Catches: the render fold
         dropped (12 entries), `ncol` left on the group count (a 4-entry legend in
         2 columns), the size-first sort dropped (a pair split across columns),
         the member rule keyed on REAL_Z_METHODS (groups 6 and 9 collide), a group
         of three or a widened grid without a word. Misses: the hues, which leg (i)
         pins.

    MPLBACKEND=Agg python scripts/a64_perf_aggregate.py [--seed 42] [--reference JSON] [--skip-digest]
                                                       [--only LEG] [--shipped DIR]

`--skip-digest` drops leg (D); `--only i|ii|...|ix|D` runs one leg (the break-it
runs). The committed state is gated with (D) on. Nothing here touches do-MNIST
beyond `inspect.getsource`.
"""

import argparse
import contextlib
import inspect
import math
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import digest_leg  # noqa: E402
from munch import munchify  # noqa: E402

import src.experiments.perf as perf  # noqa: E402
import src.experiments.utils.plotting as plotting  # noqa: E402
from src import aggregate  # noqa: E402
from src.aggregate import LEGEND_FLAT_MAX, LEGEND_GRID_COLS, LEGEND_ROWS  # noqa: E402
from src.data_augmentors.cigarettes import ScaleTranslation  # noqa: E402
from src.experiments.base import METRIC_FIELDS  # noqa: E402
from src.experiments.configs import (  # noqa: E402
    EPS_TOL,
    METRIC_SPECS,
    OMEGA_XLABEL,
    PARAM_SPECS,
    MethodRegistry,
    parse_experiment_plan,
    resolve_dataset_block,
)
from src.experiments.do_mnist import DoMNISTOrchestrator  # noqa: E402
from src.experiments.utils import set_seed  # noqa: E402
from src.experiments.utils.constants import (  # noqa: E402
    ALPHA_MAP,
    ARTIFACTS_DIRECTORY,
    CLAMP_YLIM,
    COLOR_MAP,
    INSTRUMENT_Z_STYLE,
    IV_MODE_METHODS,
    IV_MODES,
    PAIR_ORDER,
    PARTIAL_IDENTIFICATION_STYLE,
    POINT_ESTIMATES,
    RC_PARAMS,
    SUBDIR_PERF,
    SUBDIR_SWEEP,
    TEX_MAPPER,
    parse_method,
    spelled_method,
)
from src.experiments.utils.metrics import rho_hat  # noqa: E402
from src.experiments.utils.model_fitting import fit_model  # noqa: E402
from src.main import ORCHESTRATORS  # noqa: E402
from src.methods.regression import TwoStageLeastSquaresIV  # noqa: E402
from src.methods.sensitivity_models import PartialR2, SolveStatus  # noqa: E402
from src.sem.cigarettes import CigaretteSEM, V, build_design  # noqa: E402

SHIPPED = os.path.expanduser("~/ICLR27/symmetry4CausalBounds/artifacts")
TMPROOT = os.path.expanduser("~/scratch/tmp/a64")
GAMMA = 0.25
TOGGLES = dict(recalibrate=True, clipy=False, mean_match=True, n_jobs=1)
T_SPELLINGS = ("DA+PI+IV(T)", "PI&DA+PI+IV(T)", "DA+ERM+IV(T)")
PERF_METHODS = ["PI", "PI+INV", "DA+PI", "DA+PI+IV(T)", "PI&DA+PI+IV(T)"]
TEN = [
    "PI",
    "PI+IV",
    "PI+INV",
    "DA+PI",
    "DA+PI+IV",
    "DA+PI+IV(T)",
    "DA+PI+IV(Z)",
    "PI&DA+PI",
    "PI&DA+PI+IV",
    "PI&DA+PI+IV(T)",
]
# the three legend shapes SS5 names. NO_Z is the live simulation and optical blocks
# (config.yaml, no observed Z anywhere); CIG the shipped cigarette spellings; ALL the
# two pooled, which is what the shipped epsilon grid draws
SHAPE_NO_Z = ("PI+INV", "PI", "DA+PI", "DA+PI+IV(T)", "PI&DA+PI", "PI&DA+PI+IV(T)")
SHAPE_CIG = ("PI+INV+IV", "PI+IV", "DA+PI+IV(Z)", "DA+PI+IV(T,Z)", "PI&DA+PI+IV(Z)", "PI&DA+PI+IV(T,Z)")
SHAPE_ALL = SHAPE_CIG + SHAPE_NO_Z
# SHAPE_ALL's target: five columns of two, the four paired families first and green
# DA+PI+IV over pink PI&DA+PI+IV last (DERIVED from PAIR_ORDER and COLOR_MAP)
TABLE_ALL = [
    ["PI+INV", "PI+INV+IV"],
    ["PI", "PI+IV"],
    ["DA+PI", "DA+PI+IV(Z)"],
    ["PI&DA+PI", "PI&DA+PI+IV(Z)"],
    ["DA+PI+IV(T)", "PI&DA+PI+IV(T)"],
]
# two paired groups and nothing else: 4 entries, one row of 4, not 6 slots
SHAPE_PAIRS = ("PI", "PI+IV", "DA+PI", "DA+PI+IV(Z)")
# nine singletons: the 7-to-9 case, ceil(9 / 2) = 5 columns, the last one short
SHAPE_WIDE = ("ATE", "PI+INV", "PI", "ERM", "DA+PI", "DA+ERM", "PI&DA+PI", "DA+PI+IV(T)", "PI&DA+PI+IV(T)")
# eleven entries: past LEGEND_GRID_COLS columns, the grid widens and a warning says so
SHAPE_OVER = SHAPE_ALL + ("ATE",)
# `sweep_grid` on leg (vi)'s trees as round 16's branch 21 drew it, before the body
# moved into `_metric_grid` (RECORDED): axes count, cells on, column titles, y-labels
# (row-major), legend entries by method, x-label and its figure x
SWEEP_GRID_R21 = {
    ("root", "gamma"): (
        6,
        (True, False, True, True, True, True),
        ("simulation", "cigarettes"),
        ("coverage", "", "width", "", "worst error", ""),
        ("PI", "DA+PI", "DA+PI+IV(T)", "PI&DA+PI+IV(T)"),
        ("gamma", 0.5),
    ),
    ("root", "omega"): (
        6,
        (True, False, True, False, True, False),
        ("simulation", "cigarettes"),
        ("coverage", "", "width", "", "worst error", ""),
        ("PI", "DA+PI", "PI&DA+PI+IV(T)"),
        ("omega", 0.5),
    ),
    ("blank", "gamma"): (
        6,
        (False, True, False, True, False, True),
        ("simulation", "cigarettes"),
        ("", "coverage", "", "width", "", "worst error"),
        ("PI", "DA+PI+IV(T)"),
        ("gamma", 0.5),
    ),
    ("three", "gamma"): (
        9,
        (True, True, False, True, True, True, True, True, True),
        ("simulation", "optical device", "cigarettes"),
        ("coverage", "", "", "width", "", "", "worst error", "", ""),
        ("PI", "DA+PI", "DA+PI+IV(T)", "PI&DA+PI+IV(T)"),
        ("gamma", 0.525930626),
    ),
}
FAIL = []
SKIPPED = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAIL.append(name)


def skip(tag, reason):
    """A leg that cannot run here: printed and counted in the summary, never a silent PASS."""
    print(f"  [SKIP] {tag} {reason}")
    SKIPPED.append(f"{tag} {reason}")


# ------------------------------------------------------------------ helpers


@contextlib.contextmanager
def captured():
    """The loguru messages at WARNING and above emitted inside the block."""
    lines = []
    handle = logger.add(lambda message: lines.append(message.record["message"]), level="WARNING")
    try:
        yield lines
    finally:
        logger.remove(handle)


class call_counter:
    """Wraps a bound method and counts its calls."""

    def __init__(self, method):
        self.method, self.calls = method, []

    def __call__(self, *args, **kwargs):
        self.calls.append(1)
        return self.method(*args, **kwargs)


class prepare_spy:
    """Counts `PartialR2._prepare` calls, one per solve, on every ball and branch."""

    def __init__(self):
        self.count = 0

    def __enter__(self):
        spy, original = self, PartialR2._prepare

        def counted(model, X, gamma):
            spy.count += 1
            return original(model, X, gamma)

        self.original = original
        PartialR2._prepare = counted
        return self

    def __exit__(self, *exc):
        PartialR2._prepare = self.original


def moment_optimum(tag, h, X, y, Z):
    """The three assertions that characterise a moment-constrained ERM fit.

    (a) is bookkeeping: 2SLS zeroes the same moment when the reduced form has full
    rank, so it passes for both estimators. (b) and (c) are the leg: they each fail
    against a 2SLS fit on an UNDER-identified instrument, where the moment alone
    leaves a whole affine set and only one point of it is the ERM optimum.
    """
    mu, off = X.mean(axis=0), float(np.mean(y))
    Xc, yc = X - mu, np.asarray(y, dtype=float) - off
    Zc = np.asarray(Z, dtype=float).reshape(len(X), -1)
    Zc = Zc - Zc.mean(axis=0)
    residual = yc - Xc @ h

    moment = float(np.linalg.norm(Zc.T @ residual)) / (np.linalg.norm(Zc, "fro") * np.linalg.norm(yc))
    check(f"{tag} zeroes the demeaned moment to 1e-9 relative", moment < 1e-9, f"{moment:.2e}")

    # ker(A) with A the moment operator on a rank-revealing basis of Zc: every step
    # along it stays feasible, so the fit is the optimum only if none of them beats it
    left, singular, _ = np.linalg.svd(Zc, full_matrices=False)
    basis = left[:, singular > max(float(singular[0]), 1.0) * 1e-12]
    A = basis.T @ Xc
    kernel = np.linalg.svd(A)[2][np.linalg.matrix_rank(A) :].T
    objective = float(np.linalg.norm(residual))
    rng = np.random.default_rng(0)
    worst = 1.0
    for _ in range(200):
        step = kernel @ rng.normal(size=kernel.shape[1])
        step /= np.linalg.norm(step)
        for t in (1e-3 * np.linalg.norm(h), -1e-3 * np.linalg.norm(h)):
            worst = min(worst, float(np.linalg.norm(yc - Xc @ (h + t * step))) / objective)
    check(f"{tag} and no feasible step beats it", worst >= 1.0 - 1e-9, f"{worst:.9f}")

    # 2SLS's own normal equations put its point INSIDE the same feasible set here,
    # so a strict inequality is exactly the claim that ERM picks better within it
    by_hand = TwoStageLeastSquaresIV(fit_intercept=True).fit(X=X, y=y, Z=Z)
    other = float(np.linalg.norm(yc - Xc @ by_hand._W))
    check(
        f"{tag} and beats a fresh 2SLS on the ERM objective",
        objective <= other * (1 - 1e-6),
        f"{objective:.4f} vs {other:.4f}",
    )


def base_block(name):
    return {
        "cigarettes": dict(seed=42, augmentation="translate", target="iv", spec="t3"),
        "do_mnist": dict(seed=42, augmentation="translate", gamma=0.067, epsilon=0.1),
    }[name]


def rejection(name, **extra):
    """The ValueError message `resolve_dataset_block` raises on the block, or None."""
    try:
        resolve_dataset_block(name, {**base_block(name), **extra})
    except ValueError as error:
        return str(error)
    return None


def draw(seed):
    """a63-ii's construction: the cigarette design, the seed-`seed` DA draw, a 1-column Z."""
    design = build_design(CigaretteSEM.panel(), spec="t3", anchor="own-tax")
    X, y = design.X, design.y
    np.random.seed(seed)
    da = ScaleTranslation(V, std=float(np.std(X @ (V / np.linalg.norm(V)))))
    GX, G = da(X)
    return X, y, GX, G, da, design.Z[:, :1]


def fitted(names, X, y, GX, G, da, Z, rho, gamma_z=2**-8):
    builders = MethodRegistry.build_methods(
        list(names),
        gamma=GAMMA,
        epsilon=EPS_TOL,
        epsilon_iv=2**-5,
        epsilon_iv_z=2**-6,
        gamma_z=gamma_z,
        rho=rho,
        pad=False,
        **TOGGLES,
    )
    models = {}
    for name in names:
        model = builders[name]()
        fit_model(model=model, method_name=name, X=X, y=y, GX=GX, G=G, Z=Z, da=da)
        models[name] = model
    return models


def width(model, k, block="Z"):
    """Instrument columns of one block on a fitted IV ball: the jitter block adds
    k rows, and an absent block is 0 columns."""
    arr = model.Z_projector_R if block == "Z" else model.T_projector_R
    return 0 if arr is None else arr.shape[0] - k


def sim_block(methods, dataset="simulation", **overrides):
    block = {
        **digest_leg.TOGGLES,
        **digest_leg.BLOCKS[dataset],
        **({"iv": 0} if dataset == "simulation" else {}),
        "methods": list(methods),
        "sweep_samples": 4,
        "n_experiments": 1,
        "n_jobs": 1,
        **overrides,
    }
    return resolve_dataset_block(dataset, block)


def perf_runner(block, dataset="simulation"):
    """The runner `_run_perf` builds, on the block: the epsilon strategy, one
    experiment, serial models."""
    set_seed(block["seed"])
    orchestrator = ORCHESTRATORS[dataset](**block, hyperparameters=munchify(digest_leg.HYPERPARAMETERS))
    methods = {k: v for k, v in orchestrator.methods.items() if k != "ATE"}
    return orchestrator.get_sweep_runner_cls("epsilon")(
        methods=methods,
        method_factory=partial(orchestrator.build_methods, n_jobs=1),
        **{**orchestrator._get_clean_kwargs(), "n_jobs": 1, "n_experiments": 1},
    )


def load(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)  # noqa: S301 - our own artifacts


def dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(obj, handle)


def legend_rows(legend):
    n = len(legend.get_texts())
    ncols = getattr(legend, "_ncols", None) or getattr(legend, "_ncol", 1)
    return math.ceil(n / ncols)


def legend_columns(fig, legend):
    """The drawn legend as columns of label strings, top entry first. matplotlib
    aligns a column's texts on one x, so the rendered x0 IS the column index and this
    reads the layout back rather than re-deriving it from `ncol`."""
    fig.canvas.draw()
    columns = {}
    for text in legend.get_texts():
        box = text.get_window_extent()
        key = next((x for x in columns if abs(x - box.x0) < 1.0), box.x0)
        columns.setdefault(key, []).append((box.y0, text.get_text()))
    return [[label for _, label in sorted(columns[x], key=lambda pair: -pair[0])] for x in sorted(columns)]


def drawn_legend(names):
    """`_legend` over one handle per name, as (columns, ncol, rows, dash patterns).
    RC_PARAMS as the aggregate sets it: the TeX labels need usetex to lay out."""
    plt.rcParams.update(RC_PARAMS)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    handles = {}
    for name in names:
        (line,) = ax.plot([0, 1], [0, 1], color=plotting._get_method_color(name), linestyle=plotting._line_style(name))
        handles.setdefault(parse_method(name), (line, name))
    legend = aggregate._legend(fig, handles)
    columns = legend_columns(fig, legend)
    ncol = getattr(legend, "_ncols", None) or getattr(legend, "_ncol", 1)
    out = (
        columns,
        ncol,
        legend_rows(legend),
        [t.get_text() for t in legend.get_texts()],
        [h.get_linestyle() for h in legend.legend_handles],
    )
    plt.close(fig)
    return out


def marker_lines(ax):
    return [line for line in ax.get_lines() if line.get_linestyle() == "None" and line.get_marker() == "x"]


def ticks_render(ax):
    """Some y tick label on `ax` carries text once drawn."""
    ax.figure.canvas.draw()
    return any(t.get_text() for t in ax.get_yticklabels())


def run_cli(artifacts, out):
    env = {**os.environ, "MPLBACKEND": "Agg"}
    return subprocess.run(
        [sys.executable, "-m", "src.aggregate", "--artifacts", artifacts, "--out", out],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO,
        timeout=1800,
    )


def mtimes(root):
    return {p: os.path.getmtime(p) for p in (os.path.join(d, f) for d, _, fs in os.walk(root) for f in fs)}


# ------------------------------------------------------------------ legs


def leg_d(reference):
    print("(D) the shipped configuration does not move")
    for label, ok, detail in digest_leg.compare(REPO, reference):
        check(f"(D) {label}", ok, detail)


def leg_i():
    print("(i) the grammar and the display tables")
    for name, base in (
        ("DA+PI+IV(T)", "DA+PI+IV"),
        ("DA+PI+IV( T )", "DA+PI+IV"),
        ("PI&DA+PI+IV(T)", "PI&DA+PI+IV"),
        ("DA+ERM+IV(T)", "DA+ERM+IV"),
    ):
        check(f"(i) {name!r} parses as ({base!r}, 'T')", parse_method(name) == (base, "T"), f"{parse_method(name)}")
    for name in ("PI+IV(T)", "DA+PI(T)", "DA+PI+IV(T,T)", "DA+PI+IV(Z,T)"):
        try:
            parse_method(name)
            check(f"(i) {name!r} is rejected", False, "parsed")
        except ValueError as error:
            check(f"(i) {name!r} is rejected naming it", name in str(error), str(error))
    check("(i) IV_MODES is ('T,Z', 'Z', 'T')", IV_MODES == ("T,Z", "Z", "T"), f"{IV_MODES}")
    check(
        "(i) [DA+PI+IV(T), DA+ERM+IV(T)] resolves under iv: [] (T needs no Z)",
        rejection("cigarettes", methods=["DA+PI+IV(T)", "DA+ERM+IV(T)"], iv=[]) is None,
        rejection("cigarettes", methods=["DA+PI+IV(T)", "DA+ERM+IV(T)"], iv=[]) or "",
    )
    stored = resolve_dataset_block(
        "cigarettes",
        {**base_block("cigarettes"), "iv": ["tax_s"], "methods": ["DA+PI+IV", "DA+PI+IV(T)", "DA+PI+IV(Z)"]},
    )["methods"]
    check(
        "(i) the three modes of one base resolve to three methods",
        stored == ["DA+PI+IV", "DA+PI+IV(T)", "DA+PI+IV(Z)"],
        f"{stored}",
    )
    message = rejection("cigarettes", methods=["DA+PI+IV(T)", "DA+PI+IV( T )"], iv=[]) or ""
    check(
        "(i) [DA+PI+IV(T), DA+PI+IV( T )] raises naming both",
        "'DA+PI+IV(T)'" in message and "'DA+PI+IV( T )'" in message,
        message or "no error",
    )
    message = rejection("do_mnist", methods=["PI", "DA+PI+IV(T)"]) or ""
    check("(i) DA+PI+IV(T) on the do_mnist block raises naming it", "'DA+PI+IV(T)'" in message, message or "no error")
    for base in IV_MODE_METHODS:
        name = f"{base}(T)"
        present = name in TEX_MAPPER and name in COLOR_MAP and name in ALPHA_MAP
        check(
            f"(i) {name} in TEX_MAPPER, COLOR_MAP, ALPHA_MAP on {base}'s hue",
            present and COLOR_MAP[name] == COLOR_MAP[base],
        )
        check(f"(i) {name} is labelled as {base}", present and TEX_MAPPER[name] == TEX_MAPPER[base])
    check("(i) DA+ERM+IV(T) is a point estimate", "DA+ERM+IV(T)" in POINT_ESTIMATES)
    check(
        "(i) _line_style draws a (T) name as its base, a (Z) name with INSTRUMENT_Z_STYLE",
        plotting._line_style("DA+PI+IV(T)") == plotting._line_style("DA+PI+IV") == PARTIAL_IDENTIFICATION_STYLE
        and plotting._line_style("DA+PI+IV(Z)") == INSTRUMENT_Z_STYLE,
    )
    order = ["PI", "DA+PI+IV(T)", "PI&DA+PI+IV(T)", "DA+ERM+IV(T)"]
    built = MethodRegistry.build_methods(
        order, gamma=GAMMA, epsilon=EPS_TOL, epsilon_iv=EPS_TOL, gamma_z=2**-8, **TOGGLES
    )
    check("(i) build_methods returns the (T) spellings in order", list(built) == order, f"{list(built)}")
    # a non-DA method fits with no translation amounts at all, as before the (T) mode
    rng = np.random.default_rng(0)
    X = rng.standard_normal((64, 3))
    y = X @ np.ones(3) + 0.1 * rng.standard_normal(64)
    built = MethodRegistry.build_methods(["ERM", "PI"], gamma=GAMMA, epsilon=EPS_TOL, **TOGGLES)
    for name in ("ERM", "PI"):
        try:
            fit_model(model=built[name](), method_name=name, X=X, y=y)
            check(f"(i) fit_model on {name} with G=None fits", True)
        except Exception as error:
            check(f"(i) fit_model on {name} with G=None fits", False, f"{type(error).__name__}: {error}")


def leg_ii(seed):
    print("(ii) the (T) mode on fitted attributes")
    X, y, GX, G, da, Z = draw(seed)
    k, n_g = X.shape[1], np.asarray(G).reshape(len(X), -1).shape[1]
    rho = float(rho_hat(X, GX, y, intercept=True))
    names = ("DA+PI+IV", "DA+PI+IV(Z)", "DA+PI+IV(T)", "PI&DA+PI+IV(T)", "DA+ERM+IV", "DA+ERM+IV(T)")
    models = fitted(names, X, y, GX, G, da, Z, rho)
    t = models["DA+PI+IV(T)"]
    check(f"(ii) DA+PI+IV(T) T width {n_g} (G alone), no Z block", width(t, k, block="T") == n_g and not t._has_z)
    bare = models["DA+PI+IV"]
    check(
        f"(ii) bare DA+PI+IV carries Z 1 and T {n_g}",
        (width(bare, k), width(bare, k, block="T")) == (1, n_g),
        f"{(width(bare, k), width(bare, k, block='T'))}",
    )
    check("(ii) DA+PI+IV(T) t_bound exactly 2^-5", t.t_bound == 2**-5, f"{t.t_bound!r}")
    inter = models["PI&DA+PI+IV(T)"]
    check(
        f"(ii) PI&DA+PI+IV(T) DA branch: T width {n_g}, no Z block, t_bound 2^-5",
        width(inter.augmented, k, block="T") == n_g and not inter.augmented._has_z and inter.augmented.t_bound == 2**-5,
    )
    base = inter.baseline
    want = float(np.hypot(2**-6, np.sqrt(base.sigma_sq / base.rho * 2**-8)))
    check(
        "(ii) PI&DA+PI+IV(T) baseline: Z width 1, epsilon_iv_z 2^-6, no T block",
        width(base, k) == 1 and base.epsilon_iv_z == 2**-6 and not base._has_t,
    )
    check(
        "(ii) and its Z radius is the declared one at its own s",
        abs(base.z_bound - want) < 1e-12,
        f"{base.z_bound:.9f}",
    )
    queries = np.eye(k)
    t_pred = np.asarray(t.predict(queries, gamma=GAMMA), dtype=float)
    for other in ("DA+PI+IV", "DA+PI+IV(Z)"):
        gap = float(np.nanmax(np.abs(t_pred - np.asarray(models[other].predict(queries, gamma=GAMMA), dtype=float))))
        check(f"(ii) DA+PI+IV(T) predicts differently from {other}", gap > 1e-6, f"{gap:.2e}")
    # G alone is one column against d_h = 4, so this fixture is UNDER-identified:
    # the moment does not pin a point and an equality with 2SLS is the wrong
    # assertion. What characterises the estimator is that it is the ERM optimum
    # WITHIN the moment set, which 2SLS is not
    moment_optimum("(ii) DA+ERM+IV(T)", models["DA+ERM+IV(T)"]._W, GX, y, np.asarray(G).reshape(len(X), -1))

    # the runner's values under an empty set: no real Z, gamma_z 0
    empty = np.zeros((len(X), 0))
    reduced = fitted(
        ("DA+PI+IV", "DA+PI+IV(T)", "PI&DA+PI+IV", "PI&DA+PI+IV(T)", "DA+ERM+IV", "DA+ERM+IV(T)"),
        X,
        y,
        GX,
        G,
        da,
        empty,
        rho,
        gamma_z=0.0,
    )
    for base in ("DA+PI+IV", "PI&DA+PI+IV"):
        a = np.asarray(reduced[base].predict(queries, gamma=GAMMA), dtype=float)
        b = np.asarray(reduced[f"{base}(T)"].predict(queries, gamma=GAMMA), dtype=float)
        gap = float(np.nanmax(np.abs(a - b)))
        same_status = np.array_equal(reduced[base].query_status, reduced[f"{base}(T)"].query_status)
        check(
            f"(ii) {base}(T) under an empty Z is {base} to exactly 0.0 with equal statuses",
            gap == 0.0 and same_status,
            f"{gap:.2e}",
        )
    gap = float(np.abs(reduced["DA+ERM+IV(T)"]._W - reduced["DA+ERM+IV"]._W).max())
    check("(ii) DA+ERM+IV(T) under an empty Z equals DA+ERM+IV to 1e-12", gap < 1e-12, f"{gap:.2e}")


def leg_iii():
    print("(iii) the cumulation table on fitted call counts")
    # the epsilon grid is now 5 points at sweep_samples 4 (`_EPSILON_RATIO_GRID`
    # forces the count odd), and a T-mode method re-solves per point like PI+INV
    expected = {("PI",): 3, ("PI+INV",): 15, ("DA+PI+IV(T)",): 15, ("PI&DA+PI+IV(T)",): 30, tuple(TEN): 123}
    for methods, want in expected.items():
        runner = perf_runner(sim_block(methods, n_samples=512, treatment_dim=32, pad=True, clipy=False, n_jobs=-1))
        x = np.asarray(runner.get_param_range(), dtype=float)
        data = runner.generate_data(0, x[0])
        with prepare_spy() as spy:
            results, _ = perf.wall_clock(runner, data, x, repeats=3, seconds_per_solve=1.0)
        label = f"{list(methods)}" if len(methods) < 5 else f"the {len(methods)}-method list"
        check(f"(iii) {label}: {want} _prepare calls", spy.count == want, f"{spy.count}")
        monotone = all(np.all(np.diff(v[:, 0]) >= -1e-12) for v in results.values())
        check(f"(iii) {label}: cumulative arrays non-decreasing", monotone)
        if "PI+INV" in methods:
            v = results["PI+INV"][:, 0]
            check("(iii) PI+INV's last > first", v[-1] > v[0], f"{v[0]:.3f} -> {v[-1]:.3f} s")
        if "PI" in methods:
            v = results["PI"][:, 0]
            check(
                "(iii) PI's last within 1e-3 s of its first", abs(v[-1] - v[0]) < 1e-3, f"{v[0]:.4f} -> {v[-1]:.4f} s"
            )
        if len(methods) == 1 and methods[0] == "PI":
            # the oracle budgets are the runner's, fitted once outside the timer
            budget_spy = call_counter(runner.fit_epsilon_iv)
            runner.fit_epsilon_iv = budget_spy
            try:
                with prepare_spy() as spy:
                    seconds, per_solve = perf.normaliser(runner, data, repeats=3)
            finally:
                del runner.fit_epsilon_iv
            calls = budget_spy.calls
            check(
                "(iii) the normaliser records 4 seconds and 4 solves",
                len(seconds) == 4 and spy.count == 4,
                f"{spy.count}",
            )
            # twice now: once for the fitted budgets and once for the predict
            # kwargs, both OUTSIDE the timed block (`perf.normaliser`)
            check("(iii) and fits the T budget twice, both outside the timed block", len(calls) == 2, f"{len(calls)}")
            check("(iii) and keeps the median of the last 3", per_solve == float(np.median(seconds[1:])))
            # the unit is one baseline PI solve, so PI's own step 0 reads 1 up to timing
            # noise: on the simulation (oracle budgets, 0.1 s of quadrature that must
            # stay outside the timer) and on cigarettes (declared budgets)
            for dataset in ("simulation", "cigarettes"):
                unit_runner = perf_runner(sim_block(["PI"], dataset=dataset), dataset=dataset)
                grid = np.asarray(unit_runner.get_param_range(), dtype=float)
                unit_data = unit_runner.generate_data(0, grid[0])
                _, unit = perf.normaliser(unit_runner, unit_data, repeats=3)
                timed, _ = perf.wall_clock(unit_runner, unit_data, grid, repeats=3, seconds_per_solve=unit)
                ratio = float(timed["PI"][0, 0])
                check(
                    f"(iii) {dataset}: PI at step 0 reads 1.0 within [0.9, 1.1] baseline solves",
                    0.9 <= ratio <= 1.1,
                    f"{ratio:.3f} (unit {unit:.3f} s)",
                )
            check(
                "(iii) baseline_pi is serial under a block saying n_jobs -1", perf.baseline_pi(runner, data).n_jobs == 1
            )


def leg_iv():
    print("(iv) D(eps) on a synthetic record")
    n_runs, n_steps, n_queries = 3, 2, 4
    w_pi = np.tile(np.array([1.0, 1.0, 2.0, 2.0]), (n_runs, 1))
    terms0 = np.array([0.05, 0.10, 0.15, 0.20])
    d = terms0 * w_pi[0]  # sd of (m - d, m, m + d) is d, so each side contributes d and the term is d / w
    centre_l, centre_u = np.array([0.0, 1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0, 4.0])
    offsets = np.array([-1.0, 0.0, 1.0])
    lower = np.zeros((n_runs, n_steps, n_queries))
    upper = np.zeros((n_runs, n_steps, n_queries))
    for r in range(n_runs):
        for i in range(n_steps):
            lower[r, i] = centre_l + offsets[r] * d
            upper[r, i] = centre_u + offsets[r] * d
    status = np.zeros((n_runs, n_steps, n_queries), dtype=int)
    terms, failures = perf.solver_stability(lower, upper, status, w_pi)
    check(
        "(iv) step-0 terms are [0.05, 0.10, 0.15, 0.20]",
        np.allclose(terms[0], terms0, atol=1e-12),
        f"{np.round(terms[0], 6)}",
    )
    check("(iv) D(0) is 0.125", np.isclose(np.nanmean(terms[0]), 0.125), f"{np.nanmean(terms[0]):.6f}")
    check("(iv) no failure", np.array_equal(failures, [0, 0]))

    status[2, 1, 1] = SolveStatus.FAILURE
    lower[0, 1, 3], upper[0, 1, 3] = upper[0, 1, 3] + 1.0, lower[0, 1, 3]  # lower > upper
    terms, failures = perf.solver_stability(lower, upper, status, w_pi)
    check("(iv) failures == [0, 2] (one status, one lower > upper)", np.array_equal(failures, [0, 2]), f"{failures}")
    # query 1: runs 0 and 1 survive, values (m - 0.1, m): sd 0.1 / sqrt 2 each side, w 1
    # query 3: runs 1 and 2 survive, values (m, m + 0.4): sd 0.4 / sqrt 2 each side, w 2
    want1 = np.array([0.05, 0.1 / np.sqrt(2), 0.15, 0.4 / np.sqrt(2) / 2])
    check(
        "(iv) step-1 terms over the surviving runs",
        np.allclose(terms[1], want1, atol=1e-12),
        f"{np.round(terms[1], 6)}",
    )
    check(
        "(iv) D(1) by hand",
        np.isclose(np.nanmean(terms[1]), want1.mean()),
        f"{np.nanmean(terms[1]):.6f} vs {want1.mean():.6f}",
    )
    check("(iv) step 0 untouched", np.allclose(terms[0], terms0, atol=1e-12))

    status[1, 1, 1] = SolveStatus.INFEASIBLE  # a 1 counts like a 2; query 1 keeps one run
    terms, failures = perf.solver_stability(lower, upper, status, w_pi)
    check("(iv) a third failure (status 1) counts: failures == [0, 3]", np.array_equal(failures, [0, 3]), f"{failures}")
    check("(iv) the one-run query's term is NaN", np.isnan(terms[1, 1]))
    want = np.mean([want1[0], want1[2], want1[3]])
    check("(iv) nanmean skips it", np.isclose(np.nanmean(terms[1]), want), f"{np.nanmean(terms[1]):.6f} vs {want:.6f}")


def leg_v():
    print("(v) clip_y on rendered artists")
    x = PARAM_SPECS["epsilon"].grid_fn("simulation", 4)
    # the epsilon grid's own length: `_EPSILON_RATIO_GRID` forces it odd
    points = len(x)
    plt.rcParams.update(plotting.RC_PARAMS)
    wall = {
        "PI": np.ones((points, 1)),
        "PI+INV": np.linspace(1.0, 100.0, points)[:, None],
    }
    means = [wall["PI"][:, 0], wall["PI+INV"][:, 0]]

    def frame(**kwargs):
        plt.close("all")
        plotting.create_sweep_plot(
            x,
            wall,
            xlabel="x",
            ylabel="y",
            xscale="log",
            yscale="log",
            fname="epsilon_wall_clock",
            bootstrapped=False,
            savefig=False,
            **kwargs,
        )
        ax = plt.gca()
        return ax, tuple(float(v) for v in ax.get_ylim())

    ax, lim_false = frame(clip_y=False)
    want_false = plotting._pad(ax.yaxis, *plotting._limits(means, clip=False))
    check(
        "(v) clip_y=False: the frame is _pad(_limits(clip=False))",
        np.allclose(lim_false, want_false, rtol=1e-9),
        f"{lim_false} vs {want_false}",
    )
    ax, lim_true = frame()
    want_true = plotting._pad(ax.yaxis, *plotting._limits(means, clip=True))
    check(
        "(v) default: the frame is _pad(_limits(clip=True))",
        np.allclose(lim_true, want_true, rtol=1e-9),
        f"{lim_true} vs {want_true}",
    )
    check("(v) the unclipped top is higher", lim_false[1] > lim_true[1], f"{lim_false[1]:.2f} > {lim_true[1]:.2f}")
    plt.close("all")


TEN_PERF = ["ERM", "DA+ERM", "PI+INV", "PI", "PI+IV", "PI+INV+IV", "DA+PI", "DA+PI+IV", "PI&DA+PI", "PI&DA+PI+IV"]


def synthetic_tree(
    root,
    sim_inter="PI&DA+PI+IV(T)",
    cig_inter="DA+PI+IV(T)",
    cig_drop=("coverage",),
    sim_gamma=True,
    perf_methods=("PI", "PI+INV"),
):
    """simulation gamma (unless `sim_gamma` is off) and omega, cigarettes gamma without
    the `cig_drop` metrics and no omega, cigarettes perf on `perf_methods`; no optical."""
    rng = np.random.default_rng(1)
    x = PARAM_SPECS["gamma"].grid_fn("simulation", 4)

    def record(names, drop=()):
        return {
            name: {key: 0.2 + 0.6 * rng.random((4, 2)) for key in METRIC_FIELDS if key not in drop} for name in names
        }

    sim = f"{root}/simulation/{SUBDIR_SWEEP}"
    if sim_gamma:
        dump(x, f"{sim}/gamma_values.pkl")
        dump(record(["PI", "DA+PI", sim_inter]), f"{sim}/gamma_results.pkl")
        dump({name: np.zeros((4, 2, 4), dtype=int) for name in ("PI", "DA+PI", sim_inter)}, f"{sim}/gamma_statuses.pkl")
    omega_x = np.array([0.3, 0.1, 0.5, 0.2])  # not ascending, as a measured axis is
    dump(omega_x, f"{sim}/omega_values.pkl")
    dump(record(["PI", "DA+PI", sim_inter]), f"{sim}/omega_results.pkl")
    dump({"knob": omega_x, "x": omega_x, "recalibrate": True, "xlabel": OMEGA_XLABEL[True]}, f"{sim}/omega_axis.pkl")
    cig = f"{root}/cigarettes/{SUBDIR_SWEEP}"
    dump(x, f"{cig}/gamma_values.pkl")
    dump(record(["PI", cig_inter], drop=cig_drop), f"{cig}/gamma_results.pkl")
    perf_dir = f"{root}/cigarettes/{SUBDIR_PERF}"
    eps = PARAM_SPECS["epsilon"].grid_fn("cigarettes", 4)
    # the epsilon grid's own length, not 4: `_EPSILON_RATIO_GRID` forces it odd
    points = len(eps)
    dump(eps, f"{perf_dir}/epsilon_values.pkl")
    dump(
        {name: (i + 1) * np.cumsum(np.ones(points))[:, None] for i, name in enumerate(perf_methods)},
        f"{perf_dir}/epsilon_wall_clock_results.pkl",
    )
    dump(
        {"PI": np.full((points, 6), 1e-9), "PI+INV": 1e-8 + 1e-10 * rng.random((points, 6))},
        f"{perf_dir}/epsilon_seed_var_results.pkl",
    )
    marks = np.zeros(points, dtype=int)
    marks[1], marks[3 % points] = 2, 1
    dump({"PI": np.zeros(points, dtype=int), "PI+INV": marks}, f"{perf_dir}/epsilon_seed_var_failures.pkl")
    # the matching rates over three backends: PI+INV loses a third of its runs at
    # the marked steps
    feasible = {"PI": np.ones((points, 6)), "PI+INV": np.ones((points, 6))}
    feasible["PI+INV"][marks > 0, :2] = 2 / 3
    dump(feasible, f"{perf_dir}/epsilon_feasibility_results.pkl")
    dump({"xlabel": PARAM_SPECS["epsilon"].xlabel, "repeats": 3}, f"{perf_dir}/epsilon_perf_meta.pkl")
    return root


def leg_vi():
    print("(vi) the aggregate on a synthetic tree")
    os.makedirs(TMPROOT, exist_ok=True)
    root = synthetic_tree(tempfile.mkdtemp(prefix="tree_", dir=TMPROOT))
    out = f"{root}/aggregate"
    proc = run_cli(root, out)
    check(
        "(vi) the CLI exits 0",
        proc.returncode == 0,
        proc.stderr.strip().splitlines()[-1][:160] if proc.returncode else "",
    )
    written = sorted(os.listdir(out)) if os.path.isdir(out) else []
    want = sorted([f"epsilon_{stem}.pdf" for stem, _ in aggregate.PERF_FIGURES] + ["gamma_grid.pdf", "omega_grid.pdf"])
    check(
        f"(vi) exactly the {len(want)} pdfs (a grid per param, one per PERF_FIGURES entry), non-empty",
        written == want and all(os.path.getsize(f"{out}/{f}") > 0 for f in written),
        f"{written}",
    )

    datasets = aggregate.columns(root)
    check("(vi) the columns are simulation, cigarettes", datasets == ["simulation", "cigarettes"], f"{datasets}")
    fig = aggregate.sweep_grid("gamma", datasets, root)
    axes = np.array(fig.axes[: 3 * len(datasets)]).reshape(3, len(datasets))
    check("(vi) gamma grid axes (3, 2)", axes.shape == (3, 2), f"{axes.shape}")
    check(
        "(vi) column titles",
        axes[0, 0].get_title() == "simulation" and axes[0, 1].get_title() == "cigarettes",
        f"{[a.get_title() for a in axes[0]]}",
    )
    check("(vi) the cigarette coverage cell is off, its width cell on", not axes[0, 1].axison and axes[1, 1].axison)
    groups = {frozenset(id(s) for s in ax.get_shared_y_axes().get_siblings(ax)) for ax in axes.ravel()}
    check("(vi) one shared-y group", len(groups) == 1, f"{len(groups)}")
    check(
        f"(vi) the frame is CLAMP_YLIM {CLAMP_YLIM}",
        tuple(float(v) for v in axes[1, 0].get_ylim()) == CLAMP_YLIM,
        f"{axes[1, 0].get_ylim()}",
    )
    labels = [ax.get_ylabel() for ax in axes[:, 0]]
    check("(vi) y-labels coverage, width, worst error", labels == ["coverage", "width", "worst error"], f"{labels}")
    check("(vi) no y-label carries ' / '", not any(" / " in ax.get_ylabel() for ax in axes.ravel()))
    sup = [t.get_text() for t in fig.texts]
    check("(vi) one x-label, the gamma spec's", sup == [PARAM_SPECS["gamma"].xlabel], f"{sup}")
    check(
        "(vi) two columns: the x-label sits at 0.5",
        fig.texts[0].get_position()[0] == 0.5,
        f"{fig.texts[0].get_position()}",
    )
    legend = fig.legends[0]
    texts = [t.get_text() for t in legend.get_texts()]
    check(
        "(vi) legend: PI, DA+PI, DA+PI+IV(T), PI&DA+PI+IV(T) in the repo's order, one row",
        texts == [TEX_MAPPER[n] for n in ("PI", "DA+PI", "DA+PI+IV(T)", "PI&DA+PI+IV(T)")] and legend_rows(legend) == 1,
        f"{len(texts)} entries, {legend_rows(legend)} row(s)",
    )
    plt.close(fig)

    fig = aggregate.sweep_grid("omega", datasets, root)
    axes = np.array(fig.axes[: 3 * len(datasets)]).reshape(3, len(datasets))
    check("(vi) every omega cigarette cell is off", not any(ax.axison for ax in axes[:, 1]))
    check(
        "(vi) the omega x-label is the axis pkl's",
        [t.get_text() for t in fig.texts] == [OMEGA_XLABEL[True]],
        f"{[t.get_text() for t in fig.texts]}",
    )
    line = axes[0, 0].get_lines()[0]
    check("(vi) the measured omega axis is drawn sorted", np.all(np.diff(line.get_xdata()) > 0), f"{line.get_xdata()}")
    plt.close(fig)

    fold = synthetic_tree(
        tempfile.mkdtemp(prefix="fold_", dir=TMPROOT), sim_inter="PI&DA+PI+IV(T,Z)", cig_inter="PI&DA+PI+IV"
    )
    fig = aggregate.sweep_grid("gamma", aggregate.columns(fold), fold)
    texts = [t.get_text() for t in fig.legends[0].get_texts()]
    check(
        "(vi) PI&DA+PI+IV(T,Z) beside PI&DA+PI+IV is one legend entry, labelled as the bare name",
        texts == [TEX_MAPPER[n] for n in ("PI", "DA+PI", "PI&DA+PI+IV")],
        f"{len(texts)} entries",
    )
    plt.close(fig)
    two = synthetic_tree(
        tempfile.mkdtemp(prefix="two_", dir=TMPROOT), sim_inter="PI&DA+PI+IV(T)", cig_inter="PI&DA+PI+IV"
    )
    fig = aggregate.sweep_grid("gamma", aggregate.columns(two), two)
    texts = [t.get_text() for t in fig.legends[0].get_texts()]
    check(
        "(vi) PI&DA+PI+IV(T) beside PI&DA+PI+IV is one entry too: they render alike",
        texts == [TEX_MAPPER[n] for n in ("PI", "DA+PI", "PI&DA+PI+IV")],
        f"{len(texts)} entries",
    )
    plt.close(fig)

    stacked = dict(aggregate.PERF_FIGURES)["seed_var"]
    fig = aggregate.perf_grid("seed_var", stacked, datasets, root)
    grid = np.array(fig.axes[: 2 * len(datasets)]).reshape(2, len(datasets))
    check(
        "(vi) the stacked perf grid is 2 x 2, the sim column off",
        len(fig.axes) == 4 and not any(ax.axison for ax in grid[:, 0]) and all(ax.axison for ax in grid[:, 1]),
        f"{len(fig.axes)} axes",
    )
    check("(vi) the cigarette seed_var panel carries no marker line", not marker_lines(grid[0, 1]))
    check("(vi) and no count text", not grid[0, 1].texts, f"{[t.get_text() for t in grid[0, 1].texts]}")
    check(
        "(vi) the stability label and its tick numbers sit on the first panel that is on",
        grid[0, 1].get_ylabel() == METRIC_SPECS["seed_var"].ylabel
        and not grid[0, 0].get_ylabel()
        and ticks_render(grid[0, 1]),
        f"{[p.get_ylabel() for p in grid[0]]!r}",
    )
    check(
        f"(vi) the feasibility row below: the cigarette panel on CLAMP_YLIM {CLAMP_YLIM}, the stability row not",
        tuple(float(v) for v in grid[1, 1].get_ylim()) == CLAMP_YLIM
        and tuple(float(v) for v in grid[0, 1].get_ylim()) != CLAMP_YLIM,
        f"{grid[1, 1].get_ylim()} / {grid[0, 1].get_ylim()}",
    )
    check(
        "(vi) the feasibility label is the grid's row label for the spec",
        grid[1, 1].get_ylabel() == aggregate.PERF_ROW_LABELS.get("feasibility", METRIC_SPECS["feasibility"].ylabel)
        and grid[1, 1].get_ylabel().replace("\n", " ") == METRIC_SPECS["feasibility"].ylabel,
        f"{grid[1, 1].get_ylabel()!r}",
    )
    plt.close(fig)

    # a blank first column: the row labels and the y tick numbers move to the first
    # column that is on
    blank = synthetic_tree(tempfile.mkdtemp(prefix="blank_", dir=TMPROOT), cig_drop=(), sim_gamma=False)
    fig = aggregate.sweep_grid("gamma", aggregate.columns(blank), blank)
    axes = np.array(fig.axes[:6]).reshape(3, 2)
    check(
        "(vi) blank first column: every simulation cell off, every cigarette cell on",
        not any(a.axison for a in axes[:, 0]) and all(a.axison for a in axes[:, 1]),
    )
    labels = [ax.get_ylabel() for ax in axes[:, 1]]
    check(
        "(vi) blank first column: the three y-labels are on the cigarette column",
        labels == ["coverage", "width", "worst error"],
        f"{labels}",
    )
    check("(vi) blank first column: its y tick numbers render", all(ticks_render(ax) for ax in axes[:, 1]))
    plt.close(fig)

    # three columns: the one x-label sits under the middle column, on the grid and
    # on both perf grids (one row, and the stacked two)
    three = synthetic_tree(tempfile.mkdtemp(prefix="three_", dir=TMPROOT))
    shutil.copytree(f"{three}/simulation/{SUBDIR_SWEEP}", f"{three}/optical_device/{SUBDIR_SWEEP}")
    datasets = aggregate.columns(three)
    check("(vi) three columns: simulation, optical_device, cigarettes", len(datasets) == 3, f"{datasets}")
    for label, fig, middle in (
        ("gamma grid", aggregate.sweep_grid("gamma", datasets, three), 2 * 3 + 1),
        ("wall-clock grid", aggregate.perf_grid("wall_clock", ("wall_clock",), datasets, three), 1),
        ("stacked perf grid", aggregate.perf_grid("seed_var", stacked, datasets, three), 3 + 1),
    ):
        fig.canvas.draw()
        box, text = fig.axes[middle].get_position(), fig.texts[0]
        extent = text.get_window_extent()
        panel = fig.axes[middle].get_window_extent()
        check(
            f"(vi) three columns: the {label} x-label is at the middle axes' centre",
            abs(text.get_position()[0] - (box.x0 + box.x1) / 2) < 1e-9,
            f"{text.get_position()[0]:.6f} vs {(box.x0 + box.x1) / 2:.6f}",
        )
        check(
            f"(vi) three columns: the {label} x-label's centre lies within the middle axes",
            panel.x0 < (extent.x0 + extent.x1) / 2 < panel.x1,
            f"{(extent.x0 + extent.x1) / 2:.0f} in ({panel.x0:.0f}, {panel.x1:.0f})",
        )
        plt.close(fig)

    # a legend that wraps must not sit on the column titles: ten methods, two rows
    ten = synthetic_tree(tempfile.mkdtemp(prefix="ten_", dir=TMPROOT), perf_methods=tuple(TEN_PERF))
    with captured() as lines:
        drawn = (
            ("perf grid", aggregate.perf_grid("wall_clock", ("wall_clock",), aggregate.columns(ten), ten)),
            ("gamma grid", aggregate.sweep_grid("gamma", aggregate.columns(ten), ten)),
        )
    check("(vi) ten methods: no WARNING", not lines, f"{lines[:3]}")
    for label, fig in drawn:
        fig.canvas.draw()
        legend = fig.legends[0]
        box = legend.get_window_extent()
        titles = [ax.title.get_window_extent() for ax in fig.axes if ax.get_title()]
        if label == "perf grid":
            sizes = [len(column) for column in legend_columns(fig, legend)]
            check(
                f"(vi) ten methods: the perf legend is {LEGEND_ROWS} rows, five columns of two",
                sizes == [2] * 5 and legend_rows(legend) == LEGEND_ROWS,
                f"columns {sizes}",
            )
        check(
            f"(vi) ten methods: the {label} legend clears the column titles",
            bool(titles) and box.y0 > max(t.y1 for t in titles),
            f"legend y0 {box.y0:.0f}, titles y1 {max(t.y1 for t in titles):.0f}",
        )
        plt.close(fig)

    # the refactor moved no part of the sweep grid: branch 21's drawing, RECORDED
    xlabels = {"gamma": PARAM_SPECS["gamma"].xlabel, "omega": OMEGA_XLABEL[True]}
    trees = {"root": root, "blank": blank, "three": three}
    for (tree, param), (n, on, titles, ylabels, names, (xname, x)) in SWEEP_GRID_R21.items():
        fig = aggregate.sweep_grid(param, aggregate.columns(trees[tree]), trees[tree])
        got = (
            len(fig.axes),
            tuple(ax.axison for ax in fig.axes),
            tuple(ax.get_title() for ax in fig.axes if ax.get_title()),
            tuple(ax.get_ylabel() for ax in fig.axes),
            tuple(t.get_text() for t in fig.legends[0].get_texts()),
            [t.get_text() for t in fig.texts],
        )
        want = (n, on, titles, ylabels, tuple(TEX_MAPPER[name] for name in names), [xlabels[xname]])
        check(
            f"(vi) sweep_grid on {tree} {param} is branch 21's: axes, cells, titles, labels, legend, x-label",
            got == want and abs(fig.texts[0].get_position()[0] - x) < 1e-6,
            f"{got if got != want else ''} x {fig.texts[0].get_position()[0]:.9f} vs {x}",
        )
        plt.close(fig)
    for folder in (root, fold, two, blank, three, ten):
        shutil.rmtree(folder, ignore_errors=True)


def leg_vii(shipped):
    print(f"(vii) the utility on the shipped artifacts {shipped}")
    if not os.path.isdir(shipped):
        skip("(vii)", "no shipped tree")
        return
    os.makedirs(TMPROOT, exist_ok=True)
    before = mtimes(shipped)
    datasets = aggregate.columns(shipped)
    params = aggregate.sweep_params(shipped, datasets)
    out = tempfile.mkdtemp(prefix="shipped_", dir=TMPROOT)
    proc = run_cli(shipped, out)
    check(
        "(vii) the CLI exits 0",
        proc.returncode == 0,
        proc.stderr.strip().splitlines()[-1][:160] if proc.returncode else "",
    )
    written = sorted(os.listdir(out))
    # what `aggregate.main` draws off THIS tree, re-derived rather than listed: one
    # grid per sweep param, one perf grid per PERF_FIGURES entry with a metric some
    # dataset ran, and the elasticity grid when the cigarette query pkls are there.
    # The leg used to say "no perf pdf" and name no elasticity grid, which went stale
    # when those landed
    want = [f"{p}_grid.pdf" for p in params]
    want += [
        f"epsilon_{stem}.{aggregate.PLOT_FORMAT}"
        for stem, metrics in aggregate.PERF_FIGURES
        if any(aggregate._has_perf(shipped, d, m) for d in datasets for m in metrics)
    ]
    if aggregate._has_elasticities(shipped):
        want.append(f"cigarettes_elasticities.{aggregate.PLOT_FORMAT}")
    check(
        f"(vii) exactly the {len(want)} non-empty pdfs this tree calls for",
        written == sorted(want) and all(os.path.getsize(f"{out}/{f}") > 0 for f in written),
        f"drawn {written}; wanted {sorted(want)}",
    )
    check("(vii) the shipped tree's mtimes are unchanged", mtimes(shipped) == before)
    second = tempfile.mkdtemp(prefix="shipped2_", dir=TMPROOT)
    with captured() as lines:
        aggregate.main(["--artifacts", shipped, "--out", second])
    check("(vii) in-process: no WARNING", not lines, f"{lines[:3]}")
    plt.close("all")
    if "epsilon" in params:
        fig = aggregate.sweep_grid("epsilon", datasets, shipped)
        legend = fig.legends[0]
        n, rows = len(legend.get_texts()), legend_rows(legend)
        drawn = legend_columns(fig, legend)
        print(f"      RECORDED epsilon grid legend: {n} entries in {rows} row(s); columns {datasets}")
        check(
            "(vii) the epsilon grid's legend folds to 10 entries in two rows", n == 10 and rows == 2, f"{n} in {rows}"
        )
        check(
            "(vii) five columns of two, green DA+PI+IV over pink PI&DA+PI+IV last",
            [len(column) for column in drawn] == [2] * 5
            and drawn[-1] == [TEX_MAPPER["DA+PI+IV"], TEX_MAPPER["PI&DA+PI+IV"]],
            f"{drawn}",
        )
        plt.close(fig)
    shutil.rmtree(out, ignore_errors=True)
    shutil.rmtree(second, ignore_errors=True)


def leg_viii():
    print("(viii) the perf path end to end on the simulation block")
    for spec, ok_want in (
        ({"perf": {"metric": ["wall_clock"], "repeats": 2}}, False),
        ({"perf": {"metric": ["wall_clock"], "repeats": 5}}, True),
        ({"perf": {"param": ["epsilon"], "metric": ["wall_clock"]}}, False),
    ):
        try:
            plan = parse_experiment_plan(spec)
            check(f"(viii) parse_experiment_plan({spec['perf']}) resolves", ok_want, f"{plan.perf}")
        except ValueError as error:
            check(f"(viii) parse_experiment_plan({spec['perf']}) raises", not ok_want, str(error))
    source = inspect.getsource(DoMNISTOrchestrator._run_perf)
    check(
        "(viii) do-MNIST perf logs a warning and skips (source only)",
        "logger.warning" in source and "perf skipped" in source and "perf_sweeps" not in source,
    )

    block = sim_block(PERF_METHODS, n_experiments=2)
    plan = parse_experiment_plan({"perf": {"metric": ["wall_clock", "seed_var", "feasibility"], "repeats": 3}})
    folder = os.path.join(ARTIFACTS_DIRECTORY, "simulation", SUBDIR_PERF)
    shutil.rmtree(folder, ignore_errors=True)
    set_seed(block["seed"])
    ORCHESTRATORS["simulation"](**block, hyperparameters=munchify(digest_leg.HYPERPARAMETERS)).run(plan)
    files = sorted(os.listdir(folder))
    want = [
        "epsilon_feasibility_results.pkl",
        "epsilon_feasibility_sweep.pdf",
        "epsilon_perf_meta.pkl",
        "epsilon_seed_var_failures.pkl",
        "epsilon_seed_var_results.pkl",
        "epsilon_seed_var_statuses.pkl",
        "epsilon_seed_var_sweep.pdf",
        "epsilon_values.pkl",
        "epsilon_wall_clock_results.pkl",
        "epsilon_wall_clock_sweep.pdf",
    ]
    check(
        "(viii) the seven pkls and three pdfs, non-empty",
        files == want and all(os.path.getsize(f"{folder}/{f}") > 0 for f in files),
        f"{files}",
    )
    x = load(f"{folder}/epsilon_values.pkl")
    # the epsilon grid's OWN length: `_EPSILON_RATIO_GRID` forces the count odd, so
    # sweep_samples 4 gives 5 points centred on r = 1
    points = len(x)
    check(
        "(viii) epsilon_values is the epsilon grid at this sweep_samples",
        np.array_equal(x, PARAM_SPECS["epsilon"].grid_fn("simulation", 4)),
        f"{np.round(x, 4).tolist()}",
    )
    wall = load(f"{folder}/epsilon_wall_clock_results.pkl")
    check(
        f"(viii) wall_clock keys are the five methods, shape ({points}, 1)",
        list(wall) == PERF_METHODS and all(v.shape == (points, 1) for v in wall.values()),
    )
    check("(viii) PI flat after step 0 to 1e-3", np.ptp(wall["PI"][:, 0]) < 1e-3, f"{np.round(wall['PI'][:, 0], 4)}")
    check(
        "(viii) PI+INV non-decreasing and rising",
        np.all(np.diff(wall["PI+INV"][:, 0]) >= 0) and wall["PI+INV"][-1, 0] > wall["PI+INV"][0, 0],
        f"{np.round(wall['PI+INV'][:, 0], 3)}",
    )
    seed = load(f"{folder}/epsilon_seed_var_results.pkl")
    statuses = load(f"{folder}/epsilon_seed_var_statuses.pkl")
    failures = load(f"{folder}/epsilon_seed_var_failures.pkl")
    feasible = load(f"{folder}/epsilon_feasibility_results.pkl")
    meta = load(f"{folder}/epsilon_perf_meta.pkl")
    n_queries = meta["n_queries"]
    check(
        f"(viii) seed_var terms shape ({points}, {n_queries})",
        all(v.shape == (points, n_queries) for v in seed.values()),
        f"{ {k: v.shape for k, v in seed.items()} }",
    )
    n_runs = len(meta["backends"])
    check(
        f"(viii) statuses shape (R, {points}, n_queries)",
        all(v.shape == (n_runs, points, n_queries) for v in statuses.values()),
    )

    # re-derive the bounds per backend, as the seed_var loop does, and count the pairs
    runner = perf_runner(block)
    data = runner.generate_data(0, x[0])
    build = perf.builders(runner, data)
    models = {}
    for name in PERF_METHODS:
        model = build[name]()
        fit_model(
            model=model,
            method_name=name,
            hyperparameters=runner.hyperparameters,
            da=runner.get_da(0),
            **data.fit_arrays,
        )
        models[name] = model
    runs = []
    for backend in perf.installed_backends():
        for model in models.values():
            perf.set_backend(model, backend)
        runs.append(perf.bounds_along(models, runner, data, x))
    for name in PERF_METHODS:
        lower = np.array([run[name][0] for run in runs])
        upper = np.array([run[name][1] for run in runs])
        status = np.array([run[name][2] for run in runs])
        failed = (status != SolveStatus.OK) | ~np.isfinite(lower) | ~np.isfinite(upper) | (lower > upper)
        survivors = (~failed).sum(axis=0)  # (points, n_queries)
        finite = np.isfinite(seed[name])
        check(f"(viii) {name}: terms finite exactly where two runs survive", np.array_equal(finite, survivors >= 2))
        check(f"(viii) {name}: statuses pkl equals the re-derived codes", np.array_equal(statuses[name], status))
        check(
            f"(viii) {name}: feasibility equals the re-derived share of usable runs",
            np.array_equal(feasible[name], 1.0 - failed.mean(axis=0)),
            f"per step {np.round(feasible[name].mean(axis=1), 3)}",
        )
        pairs = failed.sum(axis=(0, 2))
        check(
            f"(viii) {name}: failures equal the re-derived (run, query) pairs",
            np.array_equal(failures[name], pairs),
            f"{failures[name]} (RECORDED)",
        )
        d = np.nanmean(seed[name], axis=1)
        check(f"(viii) {name}: D finite at r = 1 and above", np.all(np.isfinite(d[points // 2 :])), f"{d}")
        # the grid now starts BELOW the oracle budget, so a method that re-solves
        # can be refuted over a prefix of it and its D line is empty there. The leg
        # pins that D is NaN exactly where every run failed, whichever methods those
        # are, and RECORDS the prefix rather than hard-coding one
        refuted = failures[name] == n_runs * n_queries
        check(
            f"(viii) {name}: D is NaN exactly at the all-failed steps",
            np.array_equal(np.isnan(d), refuted),
            f"D {np.round(d, 12)} failures {failures[name]}",
        )
        print(f"      RECORDED {name}: refuted at r = {np.round(np.asarray(x)[refuted], 3).tolist()}")
    names = [name for name, _ in meta["backends"]]
    check(
        "(viii) meta backends are the installed names with the tolerances",
        meta["backends"] == [b for b in perf.BACKENDS if b[0] in names] and len(names) == 3,
        f"{names}",
    )
    check("(viii) meta seconds_per_solve > 0", meta["seconds_per_solve"] > 0, f"{meta['seconds_per_solve']:.3f}")
    check(
        "(viii) meta n_experiments is 1 although the block said 2",
        meta["n_experiments"] == 1 and block["n_experiments"] == 2,
        f"{meta['n_experiments']}",
    )
    print("      RECORDED wall clock (baseline solves): " + ", ".join(f"{k} {v[-1, 0]:.2f}" for k, v in wall.items()))
    print("      RECORDED D(eps) at r = 1: " + ", ".join(f"{k} {np.nanmean(v[-1]):.2e}" for k, v in seed.items()))


def shape_check(tag, names, want_columns):
    """One legend shape: the columns as drawn against the target table, then the
    layout rule's invariants on the ENTRY order (matplotlib's fill makes the rendered
    column sizes non-increasing whatever the sort does, so only the order can
    falsify it)."""
    columns, ncol, rows, ordered_texts, styles = drawn_legend(names)
    want_rows = max(len(c) for c in want_columns)
    n = sum(len(c) for c in columns)
    check(
        f"(ix) {tag}: {sum(len(c) for c in want_columns)} entries in {len(want_columns)} columns, {want_rows} row(s)",
        n == sum(len(c) for c in want_columns) and ncol == len(want_columns) and rows == want_rows,
        f"{n} entries, ncol {ncol}, {rows} row(s)",
    )
    want = [[TEX_MAPPER[spelled_method(name)] for name in column] for column in want_columns]
    check(f"(ix) {tag}: the columns are the target table", columns == want, f"{columns}")
    # the (group, member) of each drawn label: the render fold keeps one spelling per
    # label, and the mode spellings it folds share their (group, member)
    slot = {TEX_MAPPER[spelled_method(name)]: PAIR_ORDER[spelled_method(name)] for name in names}
    sequence = [slot[label][0] for label in ordered_texts]
    if n <= LEGEND_FLAT_MAX:
        check(
            f"(ix) {tag}: one row in the repo's order",
            rows == 1 and sequence == sorted(sequence),
            f"groups {sequence}",
        )
        return styles
    counts = Counter(sequence)
    paired = [counts[g] == 2 for g in sequence]
    check(
        f"(ix) {tag}: every paired group precedes every singleton",
        paired == sorted(paired, reverse=True),
        f"groups {sequence} paired {paired}",
    )
    singles = [g for g, p in zip(sequence, paired, strict=True) if not p]
    doubles = [g for g, p in zip(sequence, paired, strict=True) if p]
    check(
        f"(ix) {tag}: paired groups and singletons each in PAIR_ORDER order",
        singles == sorted(singles) and doubles == sorted(doubles),
        f"paired {doubles}, singletons {singles}",
    )
    # every pair in one column, member 0 on top
    split = [
        g
        for g in counts
        if counts[g] == 2 and not any([slot[label] for label in column] == [(g, 0), (g, 1)] for column in columns)
    ]
    check(f"(ix) {tag}: every pair sits in one column, member 0 on top", not split, f"split {split}")
    return styles


def leg_ix():
    print("(ix) the legend by entry count: one row up to six, else two rows, pairs intact")
    # (i) the live simulation and optical blocks pooled: no observed Z anywhere
    styles = shape_check(
        "(i) no observed Z",
        SHAPE_NO_Z,
        [["PI+INV"], ["PI"], ["DA+PI"], ["DA+PI+IV(T)"], ["PI&DA+PI"], ["PI&DA+PI+IV(T)"]],
    )
    check(
        "(ix) (i) no observed Z: nothing is drawn dash-dot",
        all(style != INSTRUMENT_Z_STYLE for style in styles),
        f"{styles}",
    )
    # (ii) all three shipped datasets: the 12 pooled keys fold to 10, five columns of two
    shape_check("(ii) all three datasets", SHAPE_ALL, TABLE_ALL)
    # (iii) one dataset alone -- cigarettes: six entries, one flat row, dash-dot on
    # the four real-Z entries
    shape_check(
        "(iii) cigarettes alone",
        SHAPE_CIG,
        [
            ["PI+INV+IV"],
            ["PI+IV"],
            ["DA+PI+IV(Z)"],
            ["DA+PI+IV(T,Z)"],
            ["PI&DA+PI+IV(Z)"],
            ["PI&DA+PI+IV(T,Z)"],
        ],
    )
    # (iv) two paired groups: `ncol` is the entry count, one row of 4, no empty slots
    shape_check("(iv) two paired groups", SHAPE_PAIRS, [["PI"], ["PI+IV"], ["DA+PI"], ["DA+PI+IV(Z)"]])
    # (v) nine singletons: the 7-to-9 case, stacked two to a column in PAIR_ORDER order
    shape_check(
        "(v) nine singletons",
        SHAPE_WIDE,
        [["ATE", "PI+INV"], ["PI", "ERM"], ["DA+PI", "DA+ERM"], ["DA+PI+IV(T)", "PI&DA+PI"], ["PI&DA+PI+IV(T)"]],
    )
    # past LEGEND_GRID_COLS columns: the grid widens and one WARNING names it
    with captured() as lines:
        over, ncol, rows, _, _ = drawn_legend(SHAPE_OVER)
    sizes = [len(column) for column in over]
    check(
        f"(ix) 11 entries: {LEGEND_ROWS} rows of 6 columns and one WARNING naming LEGEND_GRID_COLS",
        ncol == 6 > LEGEND_GRID_COLS and rows == LEGEND_ROWS and len(lines) == 1 and "LEGEND_GRID_COLS" in lines[0],
        f"ncol {ncol}, {rows} row(s), warnings {lines}",
    )
    check("(ix) 11 entries: the rendered columns are [2]*5 + [1]", sizes == [2] * 5 + [1], f"{sizes}")
    # every group of size <= 2 is the rule's precondition. Unreachable with today's
    # PAIR_ORDER, so the only way to exercise it is to break the table: fold PI+INV
    # into PI's group and it holds three kept entries
    patched = dict(aggregate.PAIR_ORDER)
    patched["PI+INV"] = (2, 0)
    original, aggregate.PAIR_ORDER = aggregate.PAIR_ORDER, patched
    try:
        with captured() as lines:
            drawn_legend(("PI+INV", "PI", "PI+IV"))
    finally:
        aggregate.PAIR_ORDER = original
    check(
        "(ix) a group of three entries warns, rather than breaking the columns quietly",
        len(lines) == 1 and "[2]" in lines[0],
        f"warnings {lines}",
    )

    # the second consumer: `perf_grid` pools the same way `sweep_grid` does
    tree = synthetic_tree(tempfile.mkdtemp(prefix="shapes_", dir=TMPROOT), perf_methods=SHAPE_ALL)
    fig = aggregate.perf_grid("wall_clock", ("wall_clock",), aggregate.columns(tree), tree)
    columns = legend_columns(fig, fig.legends[0])
    want = [[TEX_MAPPER[spelled_method(name)] for name in column] for column in TABLE_ALL]
    check(
        "(ix) perf_grid folds the same 12 keys to the same five columns of two",
        columns == want,
        f"{[len(column) for column in columns]}",
    )
    plt.close(fig)
    shutil.rmtree(tree, ignore_errors=True)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    os.chdir(REPO)
    os.environ.setdefault("MPLBACKEND", "Agg")
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reference", default=digest_leg.DEFAULT_REFERENCE)
    parser.add_argument("--skip-digest", action="store_true")
    parser.add_argument("--only", default=None, help="one leg: i, ii, iii, iv, v, vi, vii, viii, ix or D")
    parser.add_argument("--shipped", default=SHIPPED, help="the shipped artifacts tree for leg (vii)")
    args = parser.parse_args()
    print(f"tree: {perf.__file__}")
    legs = {
        "i": leg_i,
        "ii": partial(leg_ii, args.seed),
        "iii": leg_iii,
        "iv": leg_iv,
        "v": leg_v,
        "vi": leg_vi,
        "vii": partial(leg_vii, os.path.expanduser(args.shipped)),
        "viii": leg_viii,
        "ix": leg_ix,
    }
    if args.only:
        selected = {args.only: legs[args.only]} if args.only in legs else {}
        if args.only == "D":
            selected = {"D": partial(leg_d, args.reference)}
        if not selected:
            parser.error(f"unknown leg {args.only!r}")
    else:
        selected = dict(legs)
        if not args.skip_digest:
            selected["D"] = partial(leg_d, args.reference)
    for tag, leg in selected.items():
        try:
            leg()
        except Exception as error:  # a raise is a FAIL line, and the later legs still report
            check(f"({tag}) ran without raising", False, f"{type(error).__name__}: {error}")
    if "D" not in selected:
        print("(D) SKIPPED: a break-it or --skip-digest run, not the committed state")
    skipped = f" ({len(SKIPPED)} SKIPPED: {'; '.join(SKIPPED)})" if SKIPPED else ""
    if not FAIL:
        print(f"A64 PASS{skipped}")
    else:
        print(f"A64 FAIL: {FAIL}{skipped}")
        sys.exit(1)
