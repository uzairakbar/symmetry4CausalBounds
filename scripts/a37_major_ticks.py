"""A37: at least two labelled major ticks on every axis of every figure.

`_at_least_two_major_ticks` (plotting.py) runs after the last limit or scale
change at every figure site: where fewer than two major ticks fall inside the
view, a log axis gets a (1, 2, 5) x 10^k LogLocator with plain-number labels
and a linear axis a MaxNLocator that insists on two. Synthetic inputs, no
experiment. Three legs:

  (i)   every `PARAM_SPECS` param on both datasets, x = `grid_fn(dataset, 16)`,
        y random (16, 3), through `create_sweep_plot`: x and y each hold >= 2
        major tick LOCATIONS inside the view (matplotlib keeps Text objects for
        out-of-view ticks, so labels are not what is counted), no plotting error
        swallowed. Fails without the helper on n (0 or 1 majors);
  (ii)  the n sweep: the in-view majors are exactly [200, 500, 1000] on both
        datasets (x runs to the grid's last point, 1024 / 1000), and their label
        texts, stripped of `$\\mathdefault{...}$`, read 200, 500, 1000;
  (iii) `create_query_sweep_plot` on synthetic positive angles with a log x scale
        (the function has no y-scale argument; y is a linear `plt.ylim`), and
        `create_perf_plot` on a record in `_run_perf`'s schema: >= 2 in-view
        majors on every axis, zero non-empty minor labels (a32's property);
  (iv)  every param on both datasets: xlim == (x.min(), x.max()) exactly (no
        top-tail clip on the x grid, no pad), and the reference lines drawn are
        exactly the spec's values inside the grid: r = 1 on gamma and epsilon,
        1.0 on the sim trS grid, none on the optical one (0.2..0.99);
  (v)   the gamma sweep from `<artifacts>/<dataset>/sweep/gamma_{values,results}.pkl`
        when present (`--artifacts DIR`, default the repo's `artifacts/`; read
        only): the grid ends at r = 1, xlim is its exact [min, max], and the
        r = 1 line is drawn. Skipped, not failed, without the pkls.

    MPLBACKEND=Agg python scripts/a37_major_ticks.py [--artifacts DIR]
"""

import argparse
import os
import pickle
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from loguru import logger  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from src.experiments.configs import METRIC_SPECS, PARAM_SPECS  # noqa: E402
from src.experiments.utils import plotting  # noqa: E402

DATASETS = ("simulation", "optical_device")
EXPECT_N_MAJORS = {"simulation": [200.0, 500.0, 1000.0], "optical_device": [200.0, 500.0, 1000.0]}
FAIL = []
_errors = []
logger.add(lambda m: _errors.append(m), level="ERROR")


def check(tag, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {tag} {detail}")
    if not ok:
        FAIL.append(tag)


def majors_in_view(axis):
    lo, hi = sorted(axis.get_view_interval())
    return [float(t) for t in axis.get_majorticklocs() if lo <= t <= hi]


def minor_labels(fig):
    fig.canvas.draw()
    return sum(
        bool(t.get_text()) for ax in fig.axes for axis in (ax.xaxis, ax.yaxis) for t in axis.get_minorticklabels()
    )


def plain(text):
    """A tick label with the mathtext wrapper removed."""
    return re.sub(r"^\$\\mathdefault\{(.*)\}\$$", r"\1", text.strip())


def in_view_labels(axis):
    """Label text of every major tick whose location is inside the view, in order."""
    axis.figure.canvas.draw()
    lo, hi = sorted(axis.get_view_interval())
    out = []
    for tick in axis.get_major_ticks():
        loc = float(tick.get_loc())
        if lo <= loc <= hi:
            out.append(plain(tick.label1.get_text()))
    return out


def drawn_vlines(ax):
    """x of every axvline on the axes (two points at one x); the data lines have 16."""
    out = []
    for line in ax.lines:
        xdata = np.asarray(line.get_xdata(), dtype=float)
        if len(xdata) == 2 and xdata[0] == xdata[1]:
            out.append(float(xdata[0]))
    return out


def render_sweep(dataset, param, seed=0, x=None, y=None):
    rng = np.random.default_rng(seed)
    if x is None:
        x = PARAM_SPECS[param].grid_fn(dataset, 16)
    if y is None:
        y = {"PI": 1.0 + 0.3 * rng.random((len(x), 3)), "DA+PI": 0.8 + 0.3 * rng.random((len(x), 3))}
    plt.close("all")
    plotting.create_sweep_plot(
        x,
        y,
        xlabel=PARAM_SPECS[param].xlabel,
        ylabel=METRIC_SPECS["width"].ylabel,
        xscale=PARAM_SPECS[param].xscale,
        yscale=METRIC_SPECS["width"].yscale,
        experiment=dataset,
        fname=f"{param}_width",
        savefig=False,
        vlines=PARAM_SPECS[param].vlines,
    )
    return plt.gca()


def leg_i():
    print("(i) every sweep param, both datasets: >= 2 majors in view on x and y")
    for dataset in DATASETS:
        for param in PARAM_SPECS:
            before = len(_errors)
            ax = render_sweep(dataset, param)
            xs, ys = majors_in_view(ax.xaxis), majors_in_view(ax.yaxis)
            check(
                f"(i) {dataset} {param}: x majors >= 2",
                len(xs) >= 2 and len(_errors) == before,
                f"x {ax.get_xscale()} {np.round(xs, 4).tolist()} in {np.round(ax.get_xlim(), 4).tolist()}",
            )
            check(f"(i) {dataset} {param}: y majors >= 2", len(ys) >= 2, f"y {ax.get_yscale()} {len(ys)} majors")
            plt.close("all")


def leg_ii():
    print("(ii) the n sweep: (1, 2, 5) majors with plain labels")
    for dataset in DATASETS:
        ax = render_sweep(dataset, "n")
        xs = majors_in_view(ax.xaxis)
        labels = in_view_labels(ax.xaxis)
        want = EXPECT_N_MAJORS[dataset]
        check(
            f"(ii) {dataset} n: in-view majors == {want}",
            np.allclose(xs, want) if len(xs) == len(want) else False,
            f"{xs}",
        )
        check(
            f"(ii) {dataset} n: labels read {[str(int(v)) for v in want]}",
            labels == [str(int(v)) for v in want],
            f"{labels}",
        )
        plt.close("all")


def leg_iii():
    print("(iii) query sweep and perf figure")
    rng = np.random.default_rng(1)
    angles = np.geomspace(0.3, 6.0, 24)
    base = np.sin(angles)
    query = {
        "PI": np.stack([base - 0.3, base + 0.3], -1)[:, None, :].repeat(3, 1) + 0.01 * rng.standard_normal((24, 3, 2)),
        "ERM": base[:, None] + 0.01 * rng.standard_normal((24, 3)),
    }
    before = len(_errors)
    plt.close("all")
    plotting.create_query_sweep_plot(
        angles, query, xlabel="angle", xscale="log", experiment="simulation", savefig=False
    )
    fig = plt.gcf()
    fewest = min(len(majors_in_view(axis)) for ax in fig.axes for axis in (ax.xaxis, ax.yaxis))
    check(
        "(iii) query sweep (log x): >= 2 majors in view on every axis",
        fewest >= 2 and len(_errors) == before,
        f"fewest {fewest}, x majors {majors_in_view(fig.axes[0].xaxis)}",
    )
    check("(iii) query sweep: no minor labels", minor_labels(fig) == 0)
    plt.close("all")

    methods = ["PI", "DA+PI", "PI+INV"]
    record = {
        name: {
            "rates": np.array([0.7, 0.1, 0.1, 0.1]),
            "wall_clock": float(10.0 ** (i - 1)),  # 0.1, 1, 10 s: about a decade each
            "seed_var": 0.1 * (i + 1),
            "coverage_sd": 0.01,
            "midpoint_sd": 0.05,
            "n_experiments": 3,
        }
        for i, name in enumerate(methods)
    }
    before = len(_errors)
    plotting.create_perf_plot(
        record, overlay_metrics=["wall_clock", "seed_var"], experiment="simulation", savefig=False
    )
    fig = plt.gcf()
    counts = {i: (len(majors_in_view(ax.xaxis)), len(majors_in_view(ax.yaxis))) for i, ax in enumerate(fig.axes)}
    fewest = min(min(v) for v in counts.values())
    check(
        "(iii) perf figure: >= 2 majors in view on every axis (twin included)",
        fewest >= 2 and len(_errors) == before,
        f"(x, y) per axes {counts}",
    )
    check("(iii) perf figure: no minor labels", minor_labels(fig) == 0)
    plt.close("all")


def leg_iv():
    print("(iv) x keeps its exact [min, max]; the in-grid reference lines are drawn")
    for dataset in DATASETS:
        for param in PARAM_SPECS:
            x = PARAM_SPECS[param].grid_fn(dataset, 16)
            ax = render_sweep(dataset, param)
            exact = (float(x.min()), float(x.max()))
            check(
                f"(iv) {dataset} {param}: xlim == (x.min(), x.max())",
                ax.get_xlim() == exact,
                f"{ax.get_xlim()} vs {exact}",
            )
            want = [v for v in PARAM_SPECS[param].vlines if exact[0] <= v <= exact[1]]
            drawn = drawn_vlines(ax)
            check(f"(iv) {dataset} {param}: reference lines drawn == {want}", drawn == want, f"{drawn}")
            plt.close("all")


def leg_v(artifacts):
    print(f"(v) the gamma sweep pkls under {artifacts}")
    for dataset in DATASETS:
        values = f"{artifacts}/{dataset}/sweep/gamma_values.pkl"
        results = f"{artifacts}/{dataset}/sweep/gamma_results.pkl"
        if not (os.path.exists(values) and os.path.exists(results)):
            print(f"      {dataset}: no gamma pkls, skipped")
            continue
        with open(values, "rb") as fh:
            x = np.asarray(pickle.load(fh), dtype=float)  # noqa: S301 - our own artifacts
        with open(results, "rb") as fh:
            record = pickle.load(fh)  # noqa: S301 - our own artifacts
        key = METRIC_SPECS["width"].key
        y = {name: rec[key] for name, rec in record.items() if name != "ATE"}
        ax = render_sweep(dataset, "gamma", x=x, y=y)
        exact = (float(x.min()), float(x.max()))
        check(f"(v) {dataset} gamma pkl: grid ends at r = 1", np.isclose(exact[1], 1.0), f"x.max() {exact[1]}")
        check(f"(v) {dataset} gamma pkl: xlim == (x.min(), x.max())", ax.get_xlim() == exact, f"{ax.get_xlim()}")
        drawn = drawn_vlines(ax)
        check(f"(v) {dataset} gamma pkl: the r = 1 line is drawn", drawn == [1.0], f"{drawn}")
        plt.close("all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default=os.path.join(REPO, "artifacts"))
    args = parser.parse_args()
    leg_i()
    leg_ii()
    leg_iii()
    leg_iv()
    leg_v(os.path.expanduser(args.artifacts))
    print(f"\n{'A37 ALL PASS' if not FAIL else 'A37 FAILURES: ' + ', '.join(FAIL)}")
    sys.exit(bool(FAIL))
