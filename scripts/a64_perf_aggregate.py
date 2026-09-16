"""A64: the (T) instrument mode, the perf sweeps and the aggregate grids.

Four commits on refactor6/iv-figures (round 12): `DA+PI+IV(T)` and its siblings
constrain the translation amounts alone at the T-side term (`parse_method`,
`build_methods`, `fit_model`, the intersection's `instrument`); `perf` is two
epsilon sweeps on the robustness grid (`src/experiments/perf.py`: a cumulative
wall clock that re-solves only the programs reading epsilon and `repad`s the rest,
and the backend cross-check D(eps) with failure markers); `python -m src.aggregate`
draws the 3 x [datasets] sweep grids and the perf rows from the pkls. Legs:

  (D)    the digest leg (scripts/digest_leg.py), as a56 to a63: no shipped block
         spells a mode or plans perf, bare names dispatch as before, `_draw_series`
         moves no number and figures are not hashed. Catches: a bare name parsed
         as the T mode (every DA+PI+IV pkl moves). Misses: everything below.
  (i)    the grammar and the display tables: the four `(T)` spellings parse as
         (base, "T"), four malformed or misplaced suffixes are rejected naming the
         entry, `IV_MODES` is the three modes, a `(T)` block resolves under an empty
         set and beside the bare and `(Z)` entries, a duplicate `(T)` pair and a
         `(T)` on do_mnist raise, every `base(T)` has its display entries on the
         base's hue, `DA+IV(T)` is a point estimate, the T style is neither solid,
         the Z pattern nor the point-estimate dash, the registry keeps the order.
         Catches: a parser without the mode, a missing display entry (a KeyError at
         plot time), the T sibling drawn like the Z one. Misses: what (T) computes.
  (ii)   dispatch on fitted attributes, a63-ii's construction (the cigarette design,
         the seed-42 DA draw, a 1-column Z, two distinct IV budgets, `rho = rho_hat`):
         the (T) instrument is G alone (width `G.shape[1]` against +1 on bare), its
         budget the T-side `epsilon_iv` with `gamma_z` 0 so `iv_bound` is exactly
         `epsilon_iv`, the intersection's DA branch the same and its baseline
         untouched, (T) predicts differently from bare and from (Z), `DA+IV(T)` is a
         fresh 2SLS on (GX, y, G); under an empty Z and gamma_z 0 (the runner's
         values there) `(T)` equals `(T,Z)` to exactly 0.0 in bounds and statuses on
         both classes and `DA+IV(T)` equals `DA+IV`. Catches: Z stacked into the T
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
  (v)    `FAILURE_MARKER` on rendered artists: with the constant on, a marker-only
         line (`x`, 2 points, the method's colour) and two count texts; off, neither,
         the mean lines and bands untouched. And `clip_y`: the record {PI 1, PI+INV
         1..100} rendered twice on log y, the frame equal to `_pad(_limits(clip=False))`
         under `clip_y=False` and to `_pad(_limits(clip=True))` by default, the former
         higher. Catches: the flag ignored, the `clip_y` forwarding dropped. Misses:
         how the markers look.
  (vi)   the aggregate on a synthetic tree (simulation gamma and trS, cigarettes
         gamma without coverage and no trS, cigarettes perf, no optical): the CLI as
         a subprocess writes exactly the four pdfs; in-process the grid has the
         titles, the blank cells, one shared y on `CLAMP_YLIM`, the axis pkl's
         x-label, the legend in the repo's order in one row, the (base, mode) fold
         (bare beside `(T,Z)` is one entry, `(T)` beside bare is two), the three
         y-labels without " / "; the perf row has the blank sim panel and the
         cigarette marker line. Catches: rows reordered, a missing metric not
         blanked, the legend sorted by name or keyed on the spelling.
  (vii)  the utility on the shipped artifacts (`--shipped DIR`): exit 0, one grid
         per shipped param, no perf pdf, the tree's mtimes unchanged; in-process
         under `captured()` no WARNING and the epsilon legend's 10 folded entries in
         two rows. Catches: a crash on the real pkls. No break-it.
  (viii) the perf path end to end on the simulation block (5 methods, 4 steps,
         `n_experiments 2` forced to 1, serial): the six pkls and two pdfs, the grid,
         the wall-clock shapes and shape of the curves, the seed_var terms finite
         exactly where two runs survive and NaN elsewhere, `PI+INV` NaN at its
         INFEASIBLE steps and every method finite at r = 1, `failures` equal to the
         (run, query) pairs re-derived from `perf.bounds_along` per backend (612 for
         `PI+INV` there, RECORDED), the statuses pkl equal to the re-derived codes,
         `meta`; `repeats` validation; do-MNIST perf inspected as source only.
         Catches: the `n_experiments` override dropped, the wrong grid, failures
         counted over runs, `ddof` 0. Misses: optical and cigarettes perf.

    MPLBACKEND=Agg python scripts/a64_perf_aggregate.py [--seed 42] [--reference JSON] [--skip-digest]
                                                       [--only LEG] [--shipped DIR]

`--skip-digest` drops leg (D); `--only i|ii|...|viii|D` runs one leg (the break-it
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
from src.data_augmentors.cigarettes import ScaleTranslation  # noqa: E402
from src.experiments.base import METRIC_FIELDS  # noqa: E402
from src.experiments.configs import (  # noqa: E402
    EPS_TOL,
    METRIC_SPECS,
    PARAM_SPECS,
    TRS_XLABEL,
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
    INSTRUMENT_T_STYLE,
    INSTRUMENT_Z_STYLE,
    IV_MODE_METHODS,
    IV_MODES,
    POINT_ESTIMATE_STYLE,
    POINT_ESTIMATES,
    SUBDIR_PERF,
    SUBDIR_SWEEP,
    TEX_MAPPER,
    parse_method,
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
T_SPELLINGS = ("DA+PI+IV(T)", "PI&DA+PI+IV(T)", "DA+IV(T)")
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
FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAIL.append(name)


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


def width(model, k):
    """Instrument columns of a fitted IV ball: the jitter block adds k rows."""
    return model.Z_projector_R.shape[0] - k


def sim_block(methods, **overrides):
    block = {
        **digest_leg.TOGGLES,
        **digest_leg.BLOCKS["simulation"],
        "iv": 0,
        "methods": list(methods),
        "sweep_samples": 4,
        "n_experiments": 1,
        "n_jobs": 1,
        **overrides,
    }
    return resolve_dataset_block("simulation", block)


def perf_runner(block):
    """The runner `_run_perf` builds, on the block: the epsilon strategy, one
    experiment, serial models."""
    set_seed(block["seed"])
    orchestrator = ORCHESTRATORS["simulation"](**block, hyperparameters=munchify(digest_leg.HYPERPARAMETERS))
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


def marker_lines(ax):
    return [line for line in ax.get_lines() if line.get_linestyle() == "None" and line.get_marker() == "x"]


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
        ("DA+IV(T)", "DA+IV"),
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
        "(i) [DA+PI+IV(T), DA+IV(T)] resolves under iv: [] (T needs no Z)",
        rejection("cigarettes", methods=["DA+PI+IV(T)", "DA+IV(T)"], iv=[]) is None,
        rejection("cigarettes", methods=["DA+PI+IV(T)", "DA+IV(T)"], iv=[]) or "",
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
        check(f"(i) {name} differs from {base} in TeX", present and TEX_MAPPER[name] != TEX_MAPPER[base])
    check("(i) DA+IV(T) is a point estimate", "DA+IV(T)" in POINT_ESTIMATES)
    check(
        "(i) INSTRUMENT_T_STYLE is neither solid, the Z pattern nor the point-estimate dash",
        INSTRUMENT_T_STYLE not in ("-", INSTRUMENT_Z_STYLE, POINT_ESTIMATE_STYLE),
        f"{INSTRUMENT_T_STYLE}",
    )
    check(
        "(i) _line_style draws a (T) name with INSTRUMENT_T_STYLE",
        plotting._line_style("DA+PI+IV(T)") == INSTRUMENT_T_STYLE,
    )
    order = ["PI", "DA+PI+IV(T)", "PI&DA+PI+IV(T)", "DA+IV(T)"]
    built = MethodRegistry.build_methods(
        order, gamma=GAMMA, epsilon=EPS_TOL, epsilon_iv=EPS_TOL, gamma_z=2**-8, **TOGGLES
    )
    check("(i) build_methods returns the (T) spellings in order", list(built) == order, f"{list(built)}")


def leg_ii(seed):
    print("(ii) the (T) mode on fitted attributes")
    X, y, GX, G, da, Z = draw(seed)
    k, n_g = X.shape[1], np.asarray(G).reshape(len(X), -1).shape[1]
    rho = float(rho_hat(X, GX, y, intercept=True))
    names = ("DA+PI+IV", "DA+PI+IV(Z)", "DA+PI+IV(T)", "PI&DA+PI+IV(T)", "DA+IV", "DA+IV(T)")
    models = fitted(names, X, y, GX, G, da, Z, rho)
    t = models["DA+PI+IV(T)"]
    check(f"(ii) DA+PI+IV(T) instrument width {n_g} (G alone)", width(t, k) == n_g, f"{width(t, k)}")
    check(f"(ii) bare DA+PI+IV instrument width {n_g + 1}", width(models["DA+PI+IV"], k) == n_g + 1)
    check("(ii) DA+PI+IV(T) epsilon_iv 2^-5", t.epsilon_iv == 2**-5, f"{t.epsilon_iv!r}")
    check("(ii) DA+PI+IV(T) gamma_z 0.0", t.gamma_z == 0.0, f"{t.gamma_z!r}")
    check("(ii) DA+PI+IV(T) iv_bound exactly 2^-5", t.iv_bound == 2**-5, f"{t.iv_bound!r}")
    inter = models["PI&DA+PI+IV(T)"]
    check(
        f"(ii) PI&DA+PI+IV(T) DA branch: width {n_g}, gamma_z 0, iv_bound 2^-5",
        width(inter.augmented, k) == n_g and inter.augmented.gamma_z == 0.0 and inter.augmented.iv_bound == 2**-5,
    )
    check(
        "(ii) PI&DA+PI+IV(T) baseline: width 1, epsilon_iv 2^-6",
        width(inter.baseline, k) == 1 and inter.baseline.epsilon_iv == 2**-6,
    )
    queries = np.eye(k)
    t_pred = np.asarray(t.predict(queries, gamma=GAMMA), dtype=float)
    for other in ("DA+PI+IV", "DA+PI+IV(Z)"):
        gap = float(np.nanmax(np.abs(t_pred - np.asarray(models[other].predict(queries, gamma=GAMMA), dtype=float))))
        check(f"(ii) DA+PI+IV(T) predicts differently from {other}", gap > 1e-6, f"{gap:.2e}")
    by_hand = TwoStageLeastSquaresIV(fit_intercept=True).fit(X=GX, y=y, Z=np.asarray(G).reshape(len(X), -1))
    gap = float(np.abs(models["DA+IV(T)"]._W - by_hand._W).max())
    check("(ii) DA+IV(T) coefficients equal a fresh 2SLS on (GX, y, G) to 1e-12", gap < 1e-12, f"{gap:.2e}")

    # the runner's values under an empty set: no real Z, gamma_z 0
    empty = np.zeros((len(X), 0))
    reduced = fitted(
        ("DA+PI+IV", "DA+PI+IV(T)", "PI&DA+PI+IV", "PI&DA+PI+IV(T)", "DA+IV", "DA+IV(T)"),
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
    gap = float(np.abs(reduced["DA+IV(T)"]._W - reduced["DA+IV"]._W).max())
    check("(ii) DA+IV(T) under an empty Z equals DA+IV to 1e-12", gap < 1e-12, f"{gap:.2e}")


def leg_iii():
    print("(iii) the cumulation table on fitted call counts")
    expected = {("PI",): 3, ("PI+INV",): 12, ("DA+PI+IV(T)",): 3, ("PI&DA+PI+IV(T)",): 6, tuple(TEN): 48}
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
            with prepare_spy() as spy:
                seconds, per_solve = perf.normaliser(runner, data, repeats=3)
            check(
                "(iii) the normaliser records 4 seconds and 4 solves",
                len(seconds) == 4 and spy.count == 4,
                f"{spy.count}",
            )
            check("(iii) and keeps the median of the last 3", per_solve == float(np.median(seconds[1:])))
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
    print("(v) FAILURE_MARKER and clip_y on rendered artists")
    x = PARAM_SPECS["epsilon"].grid_fn("simulation", 4)
    rng = np.random.default_rng(0)
    seed = {"PI": 1e-8 + 1e-10 * rng.random((4, 6)), "PI+INV": 3e-8 + 1e-10 * rng.random((4, 6))}
    failures = {"PI+INV": np.array([0, 2, 0, 1])}
    plt.rcParams.update(plotting.RC_PARAMS)
    import seaborn as sns

    sns.set_palette("deep")
    colour = sns.color_palette()[COLOR_MAP["PI+INV"]]

    def render(flag):
        plt.close("all")
        saved = plotting.FAILURE_MARKER
        plotting.FAILURE_MARKER = flag
        try:
            plotting.create_sweep_plot(
                x,
                seed,
                xlabel="x",
                ylabel="y",
                xscale="log",
                fname="epsilon_seed_var",
                failures=failures,
                promote_y=False,
                savefig=False,
            )
        finally:
            plotting.FAILURE_MARKER = saved
        ax = plt.gca()
        means = [line for line in ax.get_lines() if line.get_linestyle() != "None" and len(line.get_xdata()) == 4]
        return ax, marker_lines(ax), means, len(ax.collections)

    ax, markers, means, bands = render(True)
    check("(v) on: one marker-only line", len(markers) == 1, f"{len(markers)}")
    if markers:
        line = markers[0]
        check(
            "(v) on: 2 marked points at the failed steps",
            np.allclose(line.get_xdata(), x[[1, 3]]),
            f"{line.get_xdata()}",
        )
        check(
            "(v) on: in PI+INV's colour",
            np.allclose(line.get_color()[:3], colour[:3]) if not isinstance(line.get_color(), str) else True,
        )
    texts = sorted(t.get_text() for t in ax.texts)
    check("(v) on: the texts read 1 and 2", texts == ["1", "2"], f"{texts}")
    check("(v) on: two mean lines and two bands", len(means) == 2 and bands == 2, f"{len(means)} {bands}")
    ax, markers, means_off, bands_off = render(False)
    check("(v) off: no marker line", not markers, f"{len(markers)}")
    check("(v) off: no text", not ax.texts, f"{[t.get_text() for t in ax.texts]}")
    check("(v) off: the mean lines and bands untouched", len(means_off) == 2 and bands_off == 2)
    check("(v) FAILURE_MARKER defaults to True", plotting.FAILURE_MARKER is True)

    wall = {"PI": np.array([[1.0], [1.0], [1.0], [1.0]]), "PI+INV": np.array([[1.0], [10.0], [50.0], [100.0]])}
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


def synthetic_tree(root, sim_inter="PI&DA+PI+IV(T)", cig_inter="DA+PI+IV(T)"):
    """simulation gamma and trS, cigarettes gamma without coverage and no trS,
    cigarettes perf; no optical."""
    rng = np.random.default_rng(1)
    x = PARAM_SPECS["gamma"].grid_fn("simulation", 4)

    def record(names, drop=()):
        return {
            name: {key: 0.2 + 0.6 * rng.random((4, 2)) for key in METRIC_FIELDS if key not in drop} for name in names
        }

    sim = f"{root}/simulation/{SUBDIR_SWEEP}"
    dump(x, f"{sim}/gamma_values.pkl")
    dump(record(["PI", "DA+PI", sim_inter]), f"{sim}/gamma_results.pkl")
    dump({name: np.zeros((4, 2, 4), dtype=int) for name in ("PI", "DA+PI", sim_inter)}, f"{sim}/gamma_statuses.pkl")
    trs_x = np.array([0.3, 0.1, 0.5, 0.2])  # not ascending, as a measured axis is
    dump(trs_x, f"{sim}/trS_values.pkl")
    dump(record(["PI", "DA+PI", sim_inter]), f"{sim}/trS_results.pkl")
    dump({"knob": trs_x, "x": trs_x, "recalibrate": True, "xlabel": TRS_XLABEL[True]}, f"{sim}/trS_axis.pkl")
    cig = f"{root}/cigarettes/{SUBDIR_SWEEP}"
    dump(x, f"{cig}/gamma_values.pkl")
    dump(record(["PI", cig_inter], drop=("coverage",)), f"{cig}/gamma_results.pkl")
    perf_dir = f"{root}/cigarettes/{SUBDIR_PERF}"
    eps = PARAM_SPECS["epsilon"].grid_fn("cigarettes", 4)
    dump(eps, f"{perf_dir}/epsilon_values.pkl")
    dump(
        {"PI": np.ones((4, 1)), "PI+INV": np.cumsum(np.ones(4))[:, None]}, f"{perf_dir}/epsilon_wall_clock_results.pkl"
    )
    dump(
        {"PI": np.full((4, 6), 1e-9), "PI+INV": 1e-8 + 1e-10 * rng.random((4, 6))},
        f"{perf_dir}/epsilon_seed_var_results.pkl",
    )
    dump({"PI": np.zeros(4, dtype=int), "PI+INV": np.array([0, 2, 0, 1])}, f"{perf_dir}/epsilon_seed_var_failures.pkl")
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
    want = ["epsilon_seed_var.pdf", "epsilon_wall_clock.pdf", "gamma_grid.pdf", "trS_grid.pdf"]
    check(
        "(vi) exactly the four pdfs, non-empty",
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
    legend = fig.legends[0]
    texts = [t.get_text() for t in legend.get_texts()]
    check(
        "(vi) legend: PI, DA+PI, DA+PI+IV(T), PI&DA+PI+IV(T) in the repo's order, one row",
        texts == [TEX_MAPPER[n] for n in ("PI", "DA+PI", "DA+PI+IV(T)", "PI&DA+PI+IV(T)")] and legend_rows(legend) == 1,
        f"{len(texts)} entries, {legend_rows(legend)} row(s)",
    )
    plt.close(fig)

    fig = aggregate.sweep_grid("trS", datasets, root)
    axes = np.array(fig.axes[: 3 * len(datasets)]).reshape(3, len(datasets))
    check("(vi) every trS cigarette cell is off", not any(ax.axison for ax in axes[:, 1]))
    check(
        "(vi) the trS x-label is the axis pkl's",
        [t.get_text() for t in fig.texts] == [TRS_XLABEL[True]],
        f"{[t.get_text() for t in fig.texts]}",
    )
    line = axes[0, 0].get_lines()[0]
    check("(vi) the measured trS axis is drawn sorted", np.all(np.diff(line.get_xdata()) > 0), f"{line.get_xdata()}")
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
        "(vi) PI&DA+PI+IV(T) beside PI&DA+PI+IV is two entries, the default first",
        texts == [TEX_MAPPER[n] for n in ("PI", "DA+PI", "PI&DA+PI+IV", "PI&DA+PI+IV(T)")],
        f"{len(texts)} entries",
    )
    plt.close(fig)

    fig = aggregate.perf_row("seed_var", datasets, root)
    panels = fig.axes[: len(datasets)]
    check(
        "(vi) the perf row has two panels, the sim panel off",
        len(panels) == 2 and not panels[0].axison and panels[1].axison,
    )
    check("(vi) the cigarette seed_var panel carries a marker line", bool(marker_lines(panels[1])))
    check(
        "(vi) the perf y-label is the metric's",
        panels[0].get_ylabel() == METRIC_SPECS["seed_var"].ylabel,
        f"{panels[0].get_ylabel()!r}",
    )
    plt.close(fig)
    shutil.rmtree(root, ignore_errors=True)
    shutil.rmtree(fold, ignore_errors=True)
    shutil.rmtree(two, ignore_errors=True)


def leg_vii(shipped):
    print(f"(vii) the utility on the shipped artifacts {shipped}")
    if not os.path.isdir(shipped):
        print("      skipped: no shipped tree at that path")
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
    check(
        f"(vii) one non-empty grid per shipped param {params}, no perf pdf",
        written == sorted(f"{p}_grid.pdf" for p in params) and all(os.path.getsize(f"{out}/{f}") > 0 for f in written),
        f"{written}",
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
        print(f"      RECORDED epsilon grid legend: {n} entries in {rows} row(s); columns {datasets}")
        check(
            "(vii) the epsilon grid's legend folds to 10 entries in two rows", n == 10 and rows == 2, f"{n} in {rows}"
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
    plan = parse_experiment_plan({"perf": {"metric": ["wall_clock", "seed_var"], "repeats": 3}})
    folder = os.path.join(ARTIFACTS_DIRECTORY, "simulation", SUBDIR_PERF)
    shutil.rmtree(folder, ignore_errors=True)
    set_seed(block["seed"])
    ORCHESTRATORS["simulation"](**block, hyperparameters=munchify(digest_leg.HYPERPARAMETERS)).run(plan)
    files = sorted(os.listdir(folder))
    want = [
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
        "(viii) the six pkls and two pdfs, non-empty",
        files == want and all(os.path.getsize(f"{folder}/{f}") > 0 for f in files),
        f"{files}",
    )
    x = load(f"{folder}/epsilon_values.pkl")
    check(
        "(viii) epsilon_values is the 4-point epsilon grid",
        np.array_equal(x, PARAM_SPECS["epsilon"].grid_fn("simulation", 4)),
        f"{x}",
    )
    wall = load(f"{folder}/epsilon_wall_clock_results.pkl")
    check(
        "(viii) wall_clock keys are the five methods, shape (4, 1)",
        list(wall) == PERF_METHODS and all(v.shape == (4, 1) for v in wall.values()),
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
    meta = load(f"{folder}/epsilon_perf_meta.pkl")
    n_queries = meta["n_queries"]
    check(
        f"(viii) seed_var terms shape (4, {n_queries})",
        all(v.shape == (4, n_queries) for v in seed.values()),
        f"{ {k: v.shape for k, v in seed.items()} }",
    )
    n_runs = len(meta["backends"])
    check("(viii) statuses shape (R, 4, n_queries)", all(v.shape == (n_runs, 4, n_queries) for v in statuses.values()))

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
        survivors = (~failed).sum(axis=0)  # (4, n_queries)
        finite = np.isfinite(seed[name])
        check(f"(viii) {name}: terms finite exactly where two runs survive", np.array_equal(finite, survivors >= 2))
        check(f"(viii) {name}: statuses pkl equals the re-derived codes", np.array_equal(statuses[name], status))
        pairs = failed.sum(axis=(0, 2))
        check(
            f"(viii) {name}: failures equal the re-derived (run, query) pairs",
            np.array_equal(failures[name], pairs),
            f"{failures[name]} (RECORDED)",
        )
        d = np.nanmean(seed[name], axis=1)
        check(f"(viii) {name}: D finite at r = 1", np.isfinite(d[-1]), f"{d}")
        if name == "PI+INV":
            check(
                "(viii) PI+INV: D NaN at the three INFEASIBLE steps",
                np.all(np.isnan(d[:3])) and np.all(failures[name][:3] == n_runs * n_queries),
                f"{d} failures {failures[name]}",
            )
        else:
            check(f"(viii) {name}: D finite at every step", np.all(np.isfinite(d)), f"{d}")
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


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    os.chdir(REPO)
    os.environ.setdefault("MPLBACKEND", "Agg")
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reference", default=digest_leg.DEFAULT_REFERENCE)
    parser.add_argument("--skip-digest", action="store_true")
    parser.add_argument("--only", default=None, help="one leg: i, ii, iii, iv, v, vi, vii, viii or D")
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
    if not FAIL:
        print("A64 PASS")
    else:
        print(f"A64 FAIL: {FAIL}")
        sys.exit(1)
