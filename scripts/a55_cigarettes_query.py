"""A55: the cigarette query figures are the panel's, and the trim is the angle.

refactor6 overrides `_run_query_sweep` with the PC panel plus six query figures, a
width-ratio figure and two tables. None of them is a new plotting function: Q2-Q6 go
through `create_query_sweep_plot` and Q7 through `create_sweep_plot` with the
`subdir` argument refactor4 added. Legs:

  (0)   the precondition on every other leg: the query design IS the full panel.
        2450 rows, every one distinct, elementwise `sem.pool[0]`. Then the headline:
        PI+INV / PI on the log-CPI coefficient. Catches: a query runner handed
        `bootstrap=True` (the headline would be one resample of about 30 states).
  (i)   seven PDFs and two tex tables land in artifacts/cigarettes/query.
        `create_sweep_plot` swallows every exception and only logs it, so THIS
        EXISTENCE CHECK IS THE ONLY GUARD on the width-ratio figure. Catches: a
        dropped `subdir` (the figure lands in sweep/), two figures sharing an fname.
  (ii)  the grids. Four per-dimension rays hold the other three coordinates at
        exactly 0, span +-3 sd of their own column, carry an EVEN number of points
        and never touch the origin, where `PartialR2._solve_single` short-circuits
        to [0, 0] and every method scores a spurious miss. The shipped block's
        `sweep_samples` is even, so the figures it draws inherit that. Catches: an
        odd grid, a grid centred off the mean, a span in the wrong units.
  (iii) the width law. ratio^2 = 1 - cos^2(x, v) in the Sigma^-1 metric, to 1e-12,
        recomputed here from the two closed forms; the SOLVER's own ratio agrees
        with it to 0.01 over the plotted queries (the SOCP's accuracy, and a
        positive epsilon budget, are what the gap is); and the four coefficient
        |cos| are 0.028 / 0.306 / 0.100 / 0.772, and the figure's four marked
        queries are those same cosines. Catches: the Euclidean angle in place of
        the Sigma^-1 one -- with v = (1,1,1,1) every e_j then reads 0.5.
  (iv)  the per-figure trims: the log-CPI grid under 0.70, the log p grid at 1.000
        +- 0.005, and the ray along v-hat two orders of magnitude under either,
        where a homogeneous h is 0 and the symmetry alone identifies the point.
        Catches: the dimension order swapped in the grid builder.
  (v)   coefficients.tex, cell by cell. The b_r and b_u rows against a50(vi) and
        SS0.5 IN RAW LOG UNITS, the cosine row against SS9, and PI+INV's log-CPI
        width ratio against the trim leg (iv) measured. Then the plumbing check the
        plan asks for: one row per configured method and every interval contains
        b_r, which is analytic once h_* is strictly interior (a linear functional
        of a point inside the ball is inside its image) and is labelled as such.
        The cell pins are what catch a units error: the table is the main-text
        table and a row-count check reads none of it.
  (vi)  ladder.tex: five data rows, one per spec, and the t3 row carries the
        numbers a50 and a52 pin (rho_max, W, J2 under three clusterings, gamma*).

    MPLBACKEND=Agg python scripts/a55_cigarettes_query.py [--seed 42]

Runs the query sweep ONCE at `sweep_samples=32` and reads the artifacts back.
Never touches do-MNIST.
"""

import argparse
import os
import pickle
import re
import shutil
import sys

import numpy as np
import yaml
from loguru import logger

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from src.experiments.cigarettes import DIM_IDS, RAY_ID, CigaretteOrchestrator  # noqa: E402
from src.experiments.configs import QUERY_GAMMA, resolve_dataset_block  # noqa: E402
from src.experiments.utils import set_seed  # noqa: E402
from src.experiments.utils.constants import ARTIFACTS_DIRECTORY, SUBDIR_QUERY, TEX_MAPPER  # noqa: E402
from src.sem.cigarettes import V, null_basis  # noqa: E402

GRID_POINTS = 32
N_JOBS = 4
ARTIFACTS = os.path.join(REPO, ARTIFACTS_DIRECTORY, "cigarettes", SUBDIR_QUERY)
FIGURES = (
    "query_sweep_panel",
    "dim_p_sweep",
    "dim_y_sweep",
    "dim_pn_sweep",
    "dim_cpi_sweep",
    "ray_v_sweep",
    "ratio_cos_sweep",
)
TABLES = ("coefficients", "ladder")

PANEL_ROWS = 2450
GRID_SPAN = 3.0
# plan SS9: |cos(e_j, v)| in the Sigma^-1 metric, the one number behind each
# coefficient's trim
COSINES = (0.028, 0.306, 0.100, 0.772)
COSINE_TOL = 5e-4
LAW_TOL = 1e-12
# the solver against the eps = 0 law. It is not exact and is not meant to be: the
# SOCP solves to its own tolerance and the INV budget is eps* + eps_tol > 0.
SOLVER_TOL = 0.01
# leg (0) and (iv). The log-CPI trim at the declared budget with eps* = 0: the
# analytic eps = 0 limit is sqrt(1 - 0.772^2) = 0.636 and the measured value sits
# just above it. SS0.7's 0.664 was measured at the LOOSER EPS_TOL (2^-5) slack;
# `CigaretteConfig.eps_tol` is 2^-8, which is sharper -- see the batch C report.
CPI_RATIO = (0.63, 0.66)
CPI_CEILING = 0.70
PRICE_RATIO = 1.000
PRICE_TOL = 0.005
RAY_CEILING = 0.01
SPEC_ROWS = 5
# leg (v): the table is written in RAW log units, so its cells are a50(vi)'s b_r and
# SS0.5's unrestricted point, not the solver's sigma-normalised numbers
B_R_RAW = (-1.992, 0.507, 1.237, 0.249)
B_U_RAW = (-2.000, 0.457, 1.229, 0.044)
CELL_TOL = 5e-4
# a50 / a52 pins for the t3 row of the ladder
T3_LADDER = {"rho_max": 1.0020, "W": 4.8, "J2_state": 2.74, "J2_year": 1.75, "J2_iid": 6.62, "gamma": 0.1922}

FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAIL.append(name)


def shipped_block():
    with open(os.path.join(REPO, "config.yaml")) as handle:
        config = yaml.safe_load(handle) or {}
    return {**(config.get("defaults") or {}), **(config.get("cigarettes") or {})}, config.get("cigarettes") or {}


def load(name):
    """A saved query record, or None when the run did not write it. None rather
    than an exception: a figure that never got drawn must turn a leg RED, not stop
    the gate before the legs that would say why."""
    path = os.path.join(ARTIFACTS, f"{name}.pkl")
    if not os.path.exists(path):
        logger.warning(f"{path} is missing.")
        return None
    with open(path, "rb") as handle:
        return pickle.load(handle)  # noqa: S301 - our own artifacts, no untrusted input


def widths(record):
    """Per-query interval width of every PI method in a saved query record."""
    if record is None:
        return {}
    return {name: bounds[:, 0, 1] - bounds[:, 0, 0] for name, bounds in record.items() if bounds.ndim == 3}


def trim(name):
    """PI+INV over PI, per query, from a saved record; None when either is absent."""
    width = widths(load(name))
    if "PI+INV" not in width or "PI" not in width:
        return None
    return width["PI+INV"] / width["PI"]


def table_rows(text):
    """The data rows of a booktabs table, the row terminator stripped."""
    return [line.rstrip().removesuffix(r"\\").strip() for line in text.splitlines() if line.rstrip().endswith(r"\\")]


def cells(row):
    return [cell.strip() for cell in row.split("&")]


def row_named(rows, label):
    """The row whose first cell is `label`, or None."""
    return next((row for row in rows if cells(row)[0] == label), None)


def cosines(points, precision):
    direction = V / np.linalg.norm(V)
    numerator = np.abs(np.asarray(points) @ precision @ direction)
    denominator = np.sqrt(
        np.einsum("ij,jk,ik->i", points, precision, points) * float(direction @ precision @ direction)
    )
    return numerator / denominator


# =============================================================================
# LEG (0): THE QUERY DESIGN IS THE PANEL
# =============================================================================


def leg_0(runner):
    print("(0) the query figures are fit on the full panel")
    design, pool = runner.X, runner.sem.pool[0]
    check("(0) the query design IS the pool", np.array_equal(design, pool), f"{design.shape}")
    check("(0) the query design has the whole panel", len(design) == PANEL_ROWS, f"{len(design)}")
    check(
        "(0) every row appears once",
        len(np.unique(design, axis=0)) == len(design),
        f"{len(np.unique(design, axis=0))} distinct",
    )

    ratios = trim("dim_cpi_outcomes")
    ratio = None if ratios is None else float(ratios[0])
    check(
        "(0) the log-CPI PI+INV / PI trim",
        ratio is not None and CPI_RATIO[0] <= ratio <= CPI_RATIO[1],
        f"{ratio} in {CPI_RATIO}",
    )


# =============================================================================
# LEG (i): THE ARTIFACTS
# =============================================================================


def leg_i():
    print("(i) seven figures and two tables land in query/")
    for name in FIGURES:
        path = os.path.join(ARTIFACTS, f"{name}.pdf")
        check(f"(i) {name}.pdf", os.path.exists(path) and os.path.getsize(path) > 0)
    for name in TABLES:
        path = os.path.join(ARTIFACTS, f"{name}.tex")
        check(f"(i) {name}.tex", os.path.exists(path) and os.path.getsize(path) > 0)
    stray = os.path.join(REPO, ARTIFACTS_DIRECTORY, "cigarettes", "sweep", "ratio_cos_sweep.pdf")
    check("(i) the width-ratio figure is not in sweep/", not os.path.exists(stray))


# =============================================================================
# LEG (ii): THE GRIDS
# =============================================================================


def leg_ii(orch, runner, block):
    print("(ii) the per-dimension grids")
    X = runner.X
    grids = orch._query_grids(X)
    for column, name in enumerate(DIM_IDS):
        x, points = grids[name]
        others = np.delete(points, column, axis=1)
        span = GRID_SPAN * float(np.std(X[:, column]))
        check(f"(ii) {name}: {GRID_POINTS} points", len(x) == GRID_POINTS, f"{len(x)}")
        check(
            f"(ii) {name}: even, so no query at the origin", np.min(np.abs(x)) > 0.0, f"min |x| {np.abs(x).min():.2e}"
        )
        check(
            f"(ii) {name}: the other three coordinates are exactly 0",
            not np.any(others),
            f"max {np.abs(others).max():g}",
        )
        check(f"(ii) {name}: spans +-3 sd of its own column", abs(float(np.max(x)) - span) < 1e-12, f"{span:.4f}")
    x, points = grids[RAY_ID]
    direction = V / np.linalg.norm(V)
    residual = float(np.max(np.abs(points - np.outer(x, direction))))
    check("(ii) the ray runs along v-hat", residual < 1e-12, f"max residual {residual:.2e}")

    shipped = int(block.get("sweep_samples", GRID_POINTS))
    check("(ii) the shipped block's sweep_samples is even", shipped % 2 == 0, f"{shipped}")


# =============================================================================
# LEG (iii): THE WIDTH LAW
# =============================================================================


def leg_iii(runner):
    print("(iii) ratio^2 = 1 - cos^2 in the Sigma^-1 metric")
    X = runner.X
    Sigma = X.T @ X / len(X)
    precision = np.linalg.inv(Sigma)
    N = null_basis()
    inverse_restricted = N @ np.linalg.inv(N.T @ Sigma @ N) @ N.T

    saved = load("ratio_cos_values")
    if saved is None:
        check("(iii) the width-ratio record exists", False)
        return
    queries = np.asarray(saved)
    rows = np.linspace(0, len(X) - 1, len(queries), dtype=int)
    points = X[rows]
    angle = cosines(points, precision)
    check("(iii) the saved x IS |cos(x, v)|", float(np.max(np.abs(angle - queries))) < 1e-12)

    # the two closed forms, at eps = 0: PI's half-width is sqrt(gamma x' Sigma^-1 x)
    # and PI+INV's the same quadratic form in the restricted metric
    free = np.einsum("ij,jk,ik->i", points, precision, points)
    restricted = np.einsum("ij,jk,ik->i", points, inverse_restricted, points)
    law = float(np.max(np.abs(restricted / free - (1.0 - angle**2))))
    check("(iii) the analytic law", law < LAW_TOL, f"max deviation {law:.2e}")

    measured = np.asarray(load("ratio_cos_outcomes")["PI+INV"]).ravel()  # guarded by the record check above
    gap = float(np.max(np.abs(measured - np.sqrt(1.0 - angle**2))))
    check("(iii) the solver agrees with the law", gap < SOLVER_TOL, f"max gap {gap:.4f}")

    marks = load("ratio_cos_marks")
    coefficient_cos = cosines(np.eye(X.shape[1]), precision)
    check(
        "(iii) the figure marks the four coefficient queries",
        marks is not None and float(np.max(np.abs(np.asarray(marks) - coefficient_cos))) < 1e-12,
        "" if marks is None else f"{np.round(marks, 4)}",
    )
    for j, (measured_cos, expected) in enumerate(zip(coefficient_cos, COSINES, strict=True)):
        check(
            f"(iii) |cos(e_{j}, v)|",
            abs(float(measured_cos) - expected) < COSINE_TOL,
            f"{float(measured_cos):.4f} vs {expected}",
        )


# =============================================================================
# LEG (iv): THE TRIMS
# =============================================================================


def leg_iv():
    print("(iv) the trim per figure")
    trims = {name: trim(f"{name}_outcomes") for name in DIM_IDS + (RAY_ID,)}
    missing = sorted(name for name, value in trims.items() if value is None)
    check("(iv) every per-dimension record exists", not missing, f"missing {missing}" if missing else "")
    if missing:
        return

    cpi = float(np.max(trims["dim_cpi"]))
    check("(iv) the log-CPI grid trims below 0.70", cpi < CPI_CEILING, f"{cpi:.4f}")
    price = float(np.max(np.abs(trims["dim_p"] - PRICE_RATIO)))
    check("(iv) the log p grid is 1.000 +- 0.005", price < PRICE_TOL, f"{float(np.mean(trims['dim_p'])):.4f}")
    ray = float(np.max(trims[RAY_ID]))
    # not 0 to machine precision: the INV budget is eps* + eps_tol > 0, so the
    # constraint admits a sliver of non-homogeneous h. It IS 0 at eps = 0.
    check("(iv) the ray along v-hat collapses", ray < RAY_CEILING, f"{ray:.2e}")
    check("(iv) the ray is two orders under the CPI grid", ray * 100 < cpi, f"{ray:.2e} vs {cpi:.4f}")

    # the grids are not interchangeable, and this is keyed on the NAMES rather than
    # on DIM_IDS' order: |cos(e_j, v)| ranks the four coordinates CPI, y, p_n, p, so
    # the trims must rank the same way whatever order the builder walks them in
    ranked = [float(np.mean(trims[name])) for name in ("dim_cpi", "dim_y", "dim_pn", "dim_p")]
    check(
        "(iv) the trims rank as |cos(e_j, v)| does: CPI, y, p_n, p",
        bool(np.all(np.diff(ranked) > 0.0)),
        f"{np.round(ranked, 4)}",
    )


# =============================================================================
# LEGS (v) AND (vi): THE TABLES
# =============================================================================


def leg_v(orch, runner):
    print("(v) coefficients.tex")
    with open(os.path.join(ARTIFACTS, "coefficients.tex")) as handle:
        table = handle.read()
    rows = table_rows(table)
    check(
        "(v) one row per configured method plus five reference rows",
        len(rows) == 1 + len(orch.kwargs["methods"]) + 5,
        f"{len(rows)} rows",
    )

    # the cells themselves, in the RAW log units the table declares. A row count
    # reads no number, so a units error (sigma dropped) leaves the headline
    # elasticity at -0.29 and every other leg green.
    for label, expected in ((r"$b_r$", B_R_RAW), (r"$b_u$ (2SLS)", B_U_RAW)):
        row = row_named(rows, label)
        if row is None:
            check(f"(v) the {label} row", False, "missing")
            continue
        values = [float(cell) for cell in cells(row)[1:]]
        gap = float(np.max(np.abs(np.asarray(values) - np.asarray(expected))))
        check(f"(v) the {label} row in raw log units", gap < CELL_TOL, f"{values} vs {list(expected)}")

    row = row_named(rows, r"$|\cos(e_j, v)|$")
    if row is None:
        check("(v) the cosine row", False, "missing")
    else:
        values = [float(cell) for cell in cells(row)[1:]]
        gap = float(np.max(np.abs(np.asarray(values) - np.asarray(COSINES))))
        check("(v) the cosine row", gap < CELL_TOL, f"{values}")

    trims = trim("dim_cpi_outcomes")
    row = row_named(rows, TEX_MAPPER["PI+INV"])
    printed = None if row is None else re.search(r"\(([0-9.]+)\)\s*$", cells(row)[-1])
    check(
        "(v) PI+INV's log-CPI width ratio matches the figure's trim",
        printed is not None and trims is not None and abs(float(printed.group(1)) - float(trims[0])) < CELL_TOL,
        f"{None if printed is None else printed.group(1)} vs {None if trims is None else round(float(trims[0]), 4)}",
    )

    design = runner.sem.design
    b_r = runner.sem.solution.ravel()
    results, _ = _coefficient_bounds(runner)
    inside = True
    for bounds in results.values():
        if bounds.ndim != 3:
            continue
        for j in range(design.k):
            inside &= bool(bounds[j, 0, 0] <= b_r[j] <= bounds[j, 0, 1])
    check("(v) every interval contains b_r", inside, f"b_r {np.round(design.sigma * b_r, 3)}")


def _coefficient_bounds(runner):
    """The fitted methods' intervals at the four coefficient queries."""
    from src.experiments.utils import PanelBuilder

    panel = PanelBuilder(runner, "cigarettes", False)
    panel._fit_all_models()
    return panel.predict(np.eye(runner.X.shape[1]))


def leg_vi():
    print("(vi) ladder.tex")
    with open(os.path.join(ARTIFACTS, "ladder.tex")) as handle:
        table = handle.read()
    rows = [line for line in table.splitlines() if line.endswith(r"\\") and not line.startswith("spec")]
    check("(vi) one row per spec", len(rows) == SPEC_ROWS, f"{len(rows)} rows")

    t3_row = next(row for row in rows if row.startswith("t3")).removesuffix(r"\\")
    t3 = [cell.strip() for cell in t3_row.split("&")]
    measured = dict(
        rho_max=float(t3[2]),
        W=float(t3[3]),
        J2_state=float(t3[4]),
        J2_year=float(t3[5]),
        J2_iid=float(t3[6]),
        gamma=float(t3[8]),
    )
    for key, expected in T3_LADDER.items():
        check(f"(vi) t3 {key}", abs(measured[key] - expected) < 5e-3, f"{measured[key]} vs {expected}")
    check("(vi) t3 is the only non-empty sliver", sum(row.strip().endswith(r"ok \\") for row in rows) == 1)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    os.chdir(REPO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    seed = parser.parse_args().seed

    block, raw = shipped_block()
    block.pop("experiment", None)
    resolved = resolve_dataset_block("cigarettes", dict(block))
    resolved.update(seed=seed, n_experiments=1, sweep_samples=GRID_POINTS, n_jobs=N_JOBS)
    set_seed(seed)
    orch = CigaretteOrchestrator(**resolved, hyperparameters={})

    shutil.rmtree(ARTIFACTS, ignore_errors=True)
    orch._run_query_sweep()

    runner = orch.get_query_runner_cls()(methods=orch.methods, **{**orch._get_clean_kwargs(), "n_experiments": 1})
    print(f"query budget gamma = {QUERY_GAMMA[orch.spec]:g}, epsilon = {runner.default_epsilon:.6g}")
    leg_i()
    leg_0(runner)
    leg_ii(orch, runner, raw)
    leg_iii(runner)
    leg_iv()
    leg_v(orch, runner)
    leg_vi()

    if not FAIL:
        print("A55 PASS")
    else:
        print(f"A55 FAIL: {FAIL}")
        sys.exit(1)
