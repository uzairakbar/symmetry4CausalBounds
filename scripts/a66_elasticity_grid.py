"""A66: the family colour convention, the paper's coefficient notation, DA+PI+IV(Z)
on the headline pair, and the 2 x 2 elasticity grid of `src/aggregate.py`.

refactor9 is on top of refactor8. The hue is the family and a real Z on top of it
changes the line style only (`REAL_Z_METHODS`); a (T) spelling is its base in hue,
line and label; the query figures draw both band edges in the method's line style
(`_draw_bands`) so two bands of one hue read apart; the price coefficients are
labelled theta_{state price} and theta_{neighbour price} (`COEFFICIENT_LABELS`);
`HEADLINE_METHODS` carries DA+PI+IV(Z); `elasticity_grid` tiles the four headline
figures with y shared per row, x per column, the legend inside one panel. Legs:

  (i)   the convention: PI and PI+IV one hue, DA+PI and DA+PI+IV(Z) one hue,
        PI+INV and PI+INV+IV one hue, DA+PI+IV and DA+PI+IV(T) one hue AND one
        label AND one line; the real-Z names dash-dotted, their families solid.
        Catches: any of the maps moved back.
  (ii)  the labels: both COEFFICIENT_LABELS carry \\theta and neither carries
        \\beta or p_n; T1's benchmark column is labelled with the neighbour-price
        theta. Catches: the old notation on a figure or the table.
  (iii) `_draw_bands` on a synthetic pair: one fill and two edge lines per
        interval method, the edges in `_line_style`'s pattern, a (patch, line)
        legend handle; `_mark_frame` keeps a log frame positive with a mark far
        above the grid. Catches: the edges dropped, the F2 blank frame back.
  (iv)  the recipe's query leg at reduced scale (1 experiment, 4 grid points): the
        F1 and F2 outcomes of both coefficients are keyed by HEADLINE_METHODS with
        DA+PI+IV(Z) among them; DA+PI+IV(Z) lies inside PI to 0.02 (its ball is recalibrated) on every grid point of
        every figure, equals PI on beta_pn's F2 at the largest radius to 0.02 and sits
        above PI's lower end by more than 0.5 at the smallest,
        as PI+IV does. Catches: the (Z) headline missing, a band that leaves PI.
  (v)   `elasticity_grid` on those artifacts: the file exists; four live panels;
        one legend, in the first panel, five entries; x labels on the bottom row
        only, y labels on the first column only, the row labels the two thetas;
        on a copy with the neighbour-price pkls removed the bottom row is off and
        the top row still draws. Catches: the grid not wired, the legend shared,
        a missing pair taking the grid down.
  (D)   the digest leg (scripts/digest_leg.py): the shipped artifacts hash as
        before (pkls and tex only; a figure's edges are not hashed).

    MPLBACKEND=Agg python scripts/a66_elasticity_grid.py [--reference JSON] [--skip-digest]
"""

import argparse
import os
import pickle
import shutil
import sys
import tempfile
from functools import partial

import matplotlib
import numpy as np
import yaml
from loguru import logger

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import digest_leg  # noqa: E402
from munch import munchify  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src import aggregate  # noqa: E402
from src.experiments.cigarettes import HEADLINE_COEFFICIENTS, HEADLINE_METHODS  # noqa: E402
from src.experiments.configs import parse_experiment_plan, resolve_dataset_block  # noqa: E402
from src.experiments.utils import plotting, set_seed  # noqa: E402
from src.experiments.utils.constants import (  # noqa: E402
    ALPHA_MAP,
    ARTIFACTS_DIRECTORY,
    COEFFICIENT_LABELS,
    COLOR_MAP,
    INSTRUMENT_Z_STYLE,
    PARTIAL_IDENTIFICATION_STYLE,
    REAL_Z_METHODS,
    SUBDIR_QUERY,
    TEX_MAPPER,
)
from src.main import ORCHESTRATORS  # noqa: E402

FAIL = []
# raw log units: the DA branch's recalibrated ball against PI's (measured 0.003 to 0.012)
DA_TOL: float = 0.02


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAIL.append(name)


def recipe_block(**overrides):
    with open(os.path.join(REPO, "recipes", "neighbour-price_fig12.yaml")) as handle:
        config = yaml.safe_load(handle)
    defaults = config.pop("defaults", {}) or {}
    block = {**defaults, **config["cigarettes"]}
    block.pop("experiment", None)
    return resolve_dataset_block("cigarettes", {**block, "n_jobs": 1, **overrides})


def load(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)  # noqa: S301 - our own artifact


# ------------------------------------------------------------------ legs


def leg_d(reference):
    print("(D) the shipped configuration does not move")
    for label, ok, detail in digest_leg.compare(REPO, reference):
        check(f"(D) {label}", ok, detail)


def leg_i():
    print("(i) the family colour convention")
    for a, b in (("PI", "PI+IV"), ("DA+PI", "DA+PI+IV(Z)"), ("PI+INV", "PI+INV+IV"), ("DA+PI+IV", "DA+PI+IV(T)")):
        check(f"(i) {a} and {b} share a hue", COLOR_MAP[a] == COLOR_MAP[b], f"{COLOR_MAP[a]} vs {COLOR_MAP[b]}")
    check("(i) DA+PI+IV(Z) takes DA+PI's alpha", ALPHA_MAP["DA+PI+IV(Z)"] == ALPHA_MAP["DA+PI"])
    check("(i) PI+IV and DA+PI+IV(Z) differ in hue (blue vs red)", COLOR_MAP["PI+IV"] != COLOR_MAP["DA+PI+IV(Z)"])
    check("(i) DA+PI+IV(T) is labelled as DA+PI+IV", TEX_MAPPER["DA+PI+IV(T)"] == TEX_MAPPER["DA+PI+IV"])
    check("(i) PI&DA+PI+IV(T) is labelled as PI&DA+PI+IV", TEX_MAPPER["PI&DA+PI+IV(T)"] == TEX_MAPPER["PI&DA+PI+IV"])
    check("(i) DA+PI+IV(Z) keeps its own label", TEX_MAPPER["DA+PI+IV(Z)"] != TEX_MAPPER["DA+PI+IV"])
    for name in ("PI+IV", "PI+INV+IV", "DA+PI+IV(Z)", "PI&DA+PI+IV(Z)"):
        check(
            f"(i) {name} is a real-Z name, dash-dotted",
            name in REAL_Z_METHODS and plotting._line_style(name) == INSTRUMENT_Z_STYLE,
        )
    for name in ("PI", "PI+INV", "DA+PI", "DA+PI+IV", "DA+PI+IV(T)", "DA+PI+IV(T,Z)", "PI&DA+PI+IV(T)"):
        check(f"(i) {name} is solid", plotting._line_style(name) == PARTIAL_IDENTIFICATION_STYLE)


def leg_ii():
    print("(ii) the coefficient notation")
    for key, label in COEFFICIENT_LABELS.items():
        check(
            f"(ii) {key} label carries theta and no beta or p_n",
            "theta" in label and "beta" not in label and "p_n" not in label,
            label,
        )
    check("(ii) both headline coefficients are labelled", set(HEADLINE_COEFFICIENTS) <= set(COEFFICIENT_LABELS))


def leg_iii():
    print("(iii) the band edges and the log frame")
    x = np.linspace(1.0, 2.0, 5)
    lower, upper = -np.ones(5), np.ones(5)
    band = np.stack([lower, upper], axis=1)[:, None, :]  # (points, 1, 2)
    fig, ax = plt.subplots()
    handles, lo, hi = plotting._draw_bands(ax, x, {"PI": band, "PI+IV": 0.5 * band})
    lines = ax.get_lines()
    check("(iii) two interval methods draw four edge lines", len(lines) == 4, f"{len(lines)}")
    check("(iii) one fill per method", len(ax.collections) == 2, f"{len(ax.collections)}")
    check(
        "(iii) the handles are (patch, line) pairs", all(isinstance(h, tuple) and len(h) == 2 for h in handles.values())
    )
    iv_edges = [
        line for line in lines if line.get_color() == handles["PI+IV"][1].get_color() and line is not handles["PI"][1]
    ]
    pattern = getattr(handles["PI+IV"][1], "_unscaled_dash_pattern", None)
    check("(iii) PI+IV's edges carry INSTRUMENT_Z_STYLE", pattern == INSTRUMENT_Z_STYLE, f"{pattern}")
    check("(iii) PI's edges are solid", handles["PI"][1].get_linestyle() == "-", handles["PI"][1].get_linestyle())
    check("(iii) the drawn range is the union of the bands", lo == -1.0 and hi == 1.0, f"{lo} {hi}")
    check("(iii) PI and PI+IV share a colour on the axes", len({line.get_color() for line in lines}) == 1)
    plt.close(fig)
    x_lo, x_hi, marks = plotting._mark_frame(np.geomspace(2**-8, 2**-0.5, 4), (0.5579, np.nan), "log")
    check(
        "(iii) a log frame with a far mark stays positive and covers the mark",
        x_lo > 0 and x_hi >= 0.5579 and marks == [0.5579],
    )
    del iv_edges


def leg_iv():
    print("(iv) the query leg at reduced scale, DA+PI+IV(Z) on both coefficients")
    block = recipe_block(n_experiments=1, sweep_samples=4)
    folder = os.path.join(ARTIFACTS_DIRECTORY, "cigarettes", SUBDIR_QUERY)
    shutil.rmtree(folder, ignore_errors=True)
    set_seed(block["seed"])
    with threadpool_limits(limits=1):
        ORCHESTRATORS["cigarettes"](**block, hyperparameters=munchify(digest_leg.HYPERPARAMETERS)).run(
            parse_experiment_plan({"query": True})
        )
    check("(iv) HEADLINE_METHODS carries DA+PI+IV(Z)", "DA+PI+IV(Z)" in HEADLINE_METHODS, f"{HEADLINE_METHODS}")
    for coefficient in HEADLINE_COEFFICIENTS:
        for axis in ("gamma", "budget"):
            outcomes = load(os.path.join(folder, f"beta_{coefficient}_{axis}_outcomes.pkl"))
            check(f"(iv) beta_{coefficient}_{axis} keyed by HEADLINE_METHODS", tuple(outcomes) == HEADLINE_METHODS)
            if "DA+PI+IV(Z)" not in outcomes:
                continue
            # the DA branch's ball is recalibrated on the translated data, so the
            # (Z) band sits inside PI's up to that rescaling, not exactly
            z, pi = outcomes["DA+PI+IV(Z)"], outcomes["PI"]
            inside = np.all(z[:, 0, 0] >= pi[:, 0, 0] - DA_TOL) and np.all(z[:, 0, 1] <= pi[:, 0, 1] + DA_TOL)
            check(f"(iv) beta_{coefficient}_{axis}: DA+PI+IV(Z) inside PI to {DA_TOL}", bool(inside))
    outcomes = load(os.path.join(folder, "beta_pn_budget_outcomes.pkl"))
    slack = np.abs(outcomes["DA+PI+IV(Z)"][-1, 0] - outcomes["PI"][-1, 0]).max()
    check(f"(iv) beta_pn F2: DA+PI+IV(Z) equals PI at the largest radius to {DA_TOL}", slack < DA_TOL, f"{slack:.5f}")
    tight = outcomes["DA+PI+IV(Z)"][0, 0, 0] - outcomes["PI"][0, 0, 0]
    check(
        "(iv) beta_pn F2: DA+PI+IV(Z)'s lower end above PI's by > 0.5 at the smallest radius",
        tight > 0.5,
        f"{tight:.3f}",
    )
    return ARTIFACTS_DIRECTORY


def leg_v(artifacts):
    print("(v) the elasticity grid")
    out = os.path.join(artifacts, "aggregate")
    fig = aggregate.elasticity_grid(artifacts, out)
    path = os.path.join(out, "cigarettes_elasticities.pdf")
    check("(v) cigarettes_elasticities.pdf written", os.path.isfile(path) and os.path.getsize(path) > 0)
    axes = np.array(fig.axes).reshape(2, 2) if len(fig.axes) == 4 else None
    check("(v) four panels", axes is not None)
    if axes is not None:
        check("(v) every panel live", all(ax.axison for ax in axes.ravel()))
        legends = [ax.get_legend() for ax in axes.ravel() if ax.get_legend() is not None]
        check("(v) one legend, in the first panel", len(legends) == 1 and axes[0, 0].get_legend() is not None)
        if legends:
            check(
                "(v) the legend lists the five headline methods",
                len(legends[0].get_texts()) == len(HEADLINE_METHODS),
                f"{len(legends[0].get_texts())}",
            )
        xlabels = [[bool(ax.get_xlabel()) for ax in row] for row in axes]
        ylabels = [[bool(ax.get_ylabel()) for ax in row] for row in axes]
        check("(v) x labels on the bottom row only", xlabels == [[False, False], [True, True]], f"{xlabels}")
        check("(v) y labels on the first column only", ylabels == [[True, False], [True, False]], f"{ylabels}")
        check(
            "(v) the row labels are the two thetas",
            axes[0, 0].get_ylabel() == COEFFICIENT_LABELS["p"] and axes[1, 0].get_ylabel() == COEFFICIENT_LABELS["pn"],
        )
        check(
            "(v) y shared per row",
            axes[0, 0].get_ylim() == axes[0, 1].get_ylim() and axes[0, 0].get_ylim() != axes[1, 0].get_ylim(),
        )
        check(
            "(v) the leak column is log, the budget column linear",
            axes[0, 1].get_xscale() == "log" and axes[0, 0].get_xscale() == "linear",
        )
    plt.close(fig)
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(artifacts, "cigarettes", SUBDIR_QUERY)
        dst = os.path.join(tmp, "cigarettes", SUBDIR_QUERY)
        os.makedirs(dst)
        for name in os.listdir(src):
            if name.startswith("beta_p_"):
                shutil.copy(os.path.join(src, name), dst)
        fig = aggregate.elasticity_grid(tmp, None)
        axes = np.array(fig.axes).reshape(2, 2)
        check(
            "(v) without the neighbour-price pkls the bottom row is off, the top row draws",
            all(ax.axison for ax in axes[0]) and not any(ax.axison for ax in axes[1]),
        )
        check("(v) the legend still sits in the first panel", axes[0, 0].get_legend() is not None)
        plt.close(fig)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    os.chdir(REPO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", default=digest_leg.DEFAULT_REFERENCE)
    parser.add_argument("--skip-digest", action="store_true")
    args = parser.parse_args()
    state = {}
    legs = [
        ("(i)", leg_i),
        ("(ii)", leg_ii),
        ("(iii)", leg_iii),
        ("(iv)", lambda: state.update(artifacts=leg_iv())),
        ("(v)", lambda: leg_v(state.get("artifacts", ARTIFACTS_DIRECTORY))),
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
        print("A66 PASS")
    else:
        print(f"A66 FAIL: {FAIL}")
        sys.exit(1)
