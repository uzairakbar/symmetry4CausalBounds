"""A31: the trS sweep's x-axis under both budget conventions.

`ExpansionStrategy` plots a MEASURED x: rho tr(S)/k under calibrated budgets and,
under raw budgets, tr(S)/k (both balls then have the radius sqrt(gamma), so Prop. 2
holds with rho = 1; the label stays `rho tr(S)/k`). The runner and the pkls keep
knob order; `create_sweep_plot` holds the one sort. Two experiments, a 4-step knob
grid, both datasets, both toggles, through the production path (`sweep_record`,
`_run_sweeps`), with a second runner recomputing the factors independently:

  (0)    `generate_data(j, knob)` is deterministic (common random numbers), so the
         recomputation in (i) measures the same draw;
  (i)    every stored x is exactly rho * trS (calibrated) or trS (raw), with rho
         and trS recomputed by the gate on the runner's own data;
  (ii)   the returned x is the experiment mean in KNOB order, `trS_values.pkl` is
         bit-identical to it, and the width array is (n_steps, n_exp);
  (iii)  raw: x falls strictly with the knob, and on an unbootstrapped render the
         DA+PI line carries `sort(x)` against the width means in that order, exactly,
         and its CI band (the `fill_between` polygon in the line's colour) carries the
         2.5/97.5 width percentiles in that same order, so a sorted line over an
         unsorted band is caught;
  (iv)   calibrated: x is the mean of rho * trS and the fixture argsort is the
         fold-back one ([3, 2, 0, 1] on sim, [0, 1, 2, 3] on optical), which the
         raw convention cannot produce;
  (v)    the xlabel is `PARAM_SPECS["trS"].xlabel` and contains `\\rho` under BOTH
         toggles, and no plotting error was swallowed;
  (vi)   exactly n_exp * n_steps `trS step` INFO lines, each with the stored x to
         5 decimals and the marker of its convention;
  (vii)  every spec vline inside the resolved xlim is drawn and none outside it;
  (viii) `trS_axis.pkl` holds knob, rho, trS, x and the toggle in knob order, with
         x equal to `trS_values.pkl` and to the mean of the right factor;
  (ix)   the other sweeps write no axis record: the gamma runner returns None and
         `_run_sweeps` on the gamma grid writes no `gamma_axis.pkl`.

Writes only into a fresh directory under `~/scratch/tmp/a31/`, never into the
repo's `artifacts/`; the directory is removed when its sweep passes and kept
(path printed) when it fails. Nothing here touches do-MNIST (the trS sweep is
not wired for it).

    MPLBACKEND=Agg python scripts/a31_trs_axis.py [simulation] [optical_device]
"""

import glob
import os
import pickle
import re
import shutil
import sys
import tempfile
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from loguru import logger  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from src.experiments.base import SweepData  # noqa: E402
from src.experiments.configs import PARAM_SPECS, SweepSpec  # noqa: E402
from src.experiments.generic_runner import SPECTRUM_KEEP  # noqa: E402
from src.experiments.optical_device import OpticalOrchestrator  # noqa: E402
from src.experiments.simulation import SimulationOrchestrator  # noqa: E402
from src.experiments.utils import set_seed  # noqa: E402
from src.experiments.utils.constants import TEX_MAPPER  # noqa: E402
from src.experiments.utils.metrics import rho_hat, trace_S_over_k  # noqa: E402
from src.experiments.utils.plotting import create_sweep_plot  # noqa: E402

METHODS = ["PI", "DA+PI"]
N_EXP, N_STEPS = 2, 4
LINE = re.compile(r"trS step (\S+): rho (\S+) tr\(S\)/k (\S+) \(untruncated (\S+)\) x (\S+)(.*)$")
MARK = {True: "calibrated: x = rho tr(S)/k", False: "raw budgets: rho := 1, x = tr(S)/k"}
# the calibrated product folds back on the sim fixture (steps 0 and 1 swap); the
# raw axis is strictly decreasing, argsort [3, 2, 1, 0], on both fixtures
EXPECT_ARGSORT_CAL = {"simulation": [3, 2, 0, 1], "optical_device": [0, 1, 2, 3]}
AXIS_KEYS = {"knob", "rho", "trS", "x", "calibrate"}
TMPROOT = os.path.expanduser("~/scratch/tmp/a31")
FAIL = []


def check(tag, row, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {row:6s} {tag} {detail}")
    if not ok:
        FAIL.append(f"{tag} {row} {detail}")


def build(experiment, calibrate):
    set_seed(42)
    common = dict(
        seed=42,
        n_experiments=N_EXP,
        sweep_samples=N_STEPS,
        methods=METHODS,
        hyperparameters={},
        n_jobs=4,
        calibrate=calibrate,
        pad=False,
        clipy=True,
        mean_match=True,
    )
    if experiment == "simulation":
        return SimulationOrchestrator(n_samples=2048, kernel_dim=0, treatment_dim=32, **common)
    return OpticalOrchestrator(n_samples=1000, augmentation="rotation > gaussian-noise", **common)


def second_runner(orch, param, **override):
    return orch.get_sweep_runner_cls(param)(
        methods=orch.methods, method_factory=orch.build_methods, **override, **orch._get_clean_kwargs()
    )


def band_edges(ax, rgb):
    """Sorted x, lower and upper edge of the fill_between polygon drawn in the colour `rgb`."""
    for coll in ax.collections:
        fc = coll.get_facecolor()
        if len(fc) and np.allclose(fc[0][:3], rgb):
            v = coll.get_paths()[0].vertices
            xs = np.unique(v[:, 0])
            low = np.array([v[v[:, 0] == xv, 1].min() for xv in xs])
            high = np.array([v[v[:, 0] == xv, 1].max() for xv in xs])
            return xs, low, high
    return None


def vline_positions(ax):
    return [
        float(ln.get_xdata()[0])
        for ln in ax.lines
        if len(ln.get_xdata()) == 2 and ln.get_xdata()[0] == ln.get_xdata()[1]
    ]


def run_one(experiment, calibrate):
    tag = f"[{experiment} calibrate={calibrate}]"
    print(f"\n===== {tag}")
    n_fail_before = len(FAIL)
    captured, errors = [], []
    sink = logger.add(lambda m: captured.append(m.record["message"]), level="INFO")
    err_sink = logger.add(lambda m: errors.append(m.record["message"]), level="ERROR")
    orch = build(experiment, calibrate)
    t0 = time.perf_counter()
    x, results, _ = orch.sweep_record("trS")
    t_sweep = time.perf_counter() - t0
    tmp = tempfile.mkdtemp(prefix=f"{experiment}_{'cal' if calibrate else 'raw'}_", dir=TMPROOT)
    os.chdir(tmp)
    plt.close("all")
    orch._run_sweeps(SweepSpec(param=("trS",), metric=("width",)))
    logger.remove(sink)
    logger.remove(err_sink)
    ax = plt.gcf().axes[0]  # the production render
    os.chdir(REPO)
    x = np.asarray(x)
    sweep_dir = f"{tmp}/artifacts/{experiment}/sweep"

    runner = second_runner(orch, "trS")
    knobs = np.asarray(runner.get_param_range(), dtype=float)

    # (0) determinism of the draw the factors are measured on
    ok0, det0 = True, []
    for knob in knobs:
        for j in range(N_EXP):
            a = SweepData.coerce(runner.generate_data(j, knob))
            b = SweepData.coerce(runner.generate_data(j, knob))
            if not (np.array_equal(a.X, b.X) and np.array_equal(a.GX, b.GX)):
                ok0 = False
                det0.append(f"({j},{knob:.4g}) X/GX differ between calls")
    check(tag, "(0)", ok0, "; ".join(det0[:2]))

    # (i) stored x against an independent recomputation of the factors
    rho = np.full((N_STEPS, N_EXP), np.nan)
    trs = np.full((N_STEPS, N_EXP), np.nan)
    ok_i, det_i = True, []
    for i, knob in enumerate(knobs):
        for j in range(N_EXP):
            data = SweepData.coerce(runner.generate_data(j, knob))
            rho[i, j] = rho_hat(data.X, data.GX, data.y, intercept=runner.mean_match)
            trs[i, j] = trace_S_over_k(data.X, data.GX, keep=SPECTRUM_KEEP)
            want = rho[i, j] * trs[i, j] if calibrate else trs[i, j]
            got = runner._measured[(j, float(knob))]
            if got != want:
                ok_i = False
                det_i.append(f"({j},{knob:.4g}) measured {got:.6f} want {want:.6f}")
    check(tag, "(i)", ok_i, "; ".join(det_i[:3]))

    # (ii) knob order, pkl bit-identity, width shape
    meas = np.array([[runner._measured[(j, float(k))] for j in range(N_EXP)] for k in knobs])
    x_knob = np.nanmean(meas, axis=1)
    with open(f"{sweep_dir}/trS_values.pkl", "rb") as fh:
        pkl = np.asarray(pickle.load(fh))  # noqa: S301 - the gate wrote this file itself
    w = results["DA+PI"]["interval_width"]
    ok_ii = np.array_equal(x, x_knob) and np.array_equal(pkl, x) and w.shape == (N_STEPS, N_EXP)
    check(
        tag,
        "(ii)",
        ok_ii,
        f"x {np.round(x, 5).tolist()} knob-mean {np.round(x_knob, 5).tolist()} "
        f"pkl==x {np.array_equal(pkl, x)} width shape {w.shape}",
    )

    # second render without the bootstrap, so the drawn y is the plain width mean
    plt.close("all")
    create_sweep_plot(
        x,
        {m: results[m]["interval_width"] for m in results},
        xlabel=PARAM_SPECS["trS"].xlabel,
        ylabel="w",
        experiment=experiment,
        fname="trS_width",
        savefig=False,
        bootstrapped=False,
        vlines=PARAM_SPECS["trS"].vlines,
    )
    ax2 = plt.gcf().axes[0]
    lines = [ln for ln in ax2.lines if ln.get_label() == TEX_MAPPER["DA+PI"]]
    order = np.argsort(x, kind="stable")
    if not calibrate:
        # (iii) raw: monotone axis, sorted line, y reordered with x, and the CI band
        # reordered with the line (the band is drawn from the same reordered array)
        dec = bool((np.diff(x) < 0).all())
        xd = np.asarray(lines[0].get_xdata(), dtype=float) if len(lines) == 1 else None
        yd = np.asarray(lines[0].get_ydata(), dtype=float) if len(lines) == 1 else None
        xs_ok = xd is not None and np.array_equal(xd, np.sort(x))
        ys_ok = yd is not None and np.array_equal(yd, np.nanmean(w, axis=1)[order])
        band = band_edges(ax2, to_rgb(lines[0].get_color())) if len(lines) == 1 else None
        band_ok = (
            band is not None
            and np.array_equal(band[0], np.sort(x))
            and np.array_equal(band[1], np.nanpercentile(w, 2.5, axis=1)[order])
            and np.array_equal(band[2], np.nanpercentile(w, 97.5, axis=1)[order])
        )
        check(
            tag,
            "(iii)",
            dec and xs_ok and ys_ok and band_ok,
            f"decreasing {dec} xdata==sort(x) {xs_ok} ydata exact {ys_ok} band exact {band_ok}; "
            f"xdata {None if xd is None else np.round(xd, 5).tolist()}",
        )
    else:
        # (iv) calibrated: the product, with its fixture fold-back
        ok_iv = np.array_equal(x, np.nanmean(rho * trs, axis=1))
        srt = order.tolist()
        check(
            tag,
            "(iv)",
            ok_iv and srt == EXPECT_ARGSORT_CAL[experiment],
            f"x==mean(rho*trS) {ok_iv} argsort {srt} expected {EXPECT_ARGSORT_CAL[experiment]}",
        )

    # (v) the label on the production render, and no swallowed plotting error
    lab = ax.get_xlabel()
    check(
        tag,
        "(v)",
        lab == PARAM_SPECS["trS"].xlabel and r"\rho" in lab and not errors,
        f"{lab!r} errors {len(errors)}" + (f": {errors[0][:120]}" if errors else ""),
    )

    # (vi) the INFO lines against the stored x and the convention marker
    rows = [(m, LINE.search(m)) for m in captured if LINE.search(m)]
    ok_vi = len(rows) == N_EXP * N_STEPS
    det_vi, seen = [], {}
    for _, g in rows:
        knob, xf, tail = float(g.group(1)), float(g.group(5)), g.group(6)
        j = seen.get(knob, 0)
        seen[knob] = j + 1
        kk = min(knobs, key=lambda k, knob=knob: abs(k - knob))
        want = runner._measured.get((j, float(kk)), np.nan)
        if abs(round(want, 5) - xf) > 1e-5:
            ok_vi = False
            det_vi.append(f"log x {xf} vs measured {want:.5f}")
        if not tail.strip().endswith(f"({MARK[calibrate]})"):
            ok_vi = False
            det_vi.append(f"marker missing/wrong: {tail.strip()!r}")
    check(tag, "(vi)", ok_vi, f"{len(rows)} lines (want {N_EXP * N_STEPS}); " + "; ".join(det_vi[:2]))

    # (vii) vlines on the production render, gated on the resolved xlim
    drawn = vline_positions(ax)
    x_lo, x_hi = ax.get_xlim()
    spec_in = [v for v in PARAM_SPECS["trS"].vlines if x_lo <= v <= x_hi]
    drawn_in = [v for v in drawn if x_lo <= v <= x_hi]
    drawn_out = [v for v in drawn if not (x_lo <= v <= x_hi)]
    check(
        tag,
        "(vii)",
        len(drawn_in) == len(spec_in) and not drawn_out,
        f"xlim ({x_lo:.5f}, {x_hi:.5f}) expected {len(spec_in)} drawn {drawn}",
    )

    # (viii) the axis record next to the values pkl
    axis_path = f"{sweep_dir}/trS_axis.pkl"
    if not hasattr(runner, "axis_record"):
        check(tag, "(viii)", False, "no axis_record on the trS runner")
    elif not os.path.exists(axis_path):
        check(tag, "(viii)", False, "trS_axis.pkl missing")
    else:
        with open(axis_path, "rb") as fh:
            rec = pickle.load(fh)  # noqa: S301 - the gate wrote this file itself
        keys_ok = isinstance(rec, dict) and set(rec) == AXIS_KEYS
        shapes_ok = (
            keys_ok
            and np.shape(rec["rho"]) == (N_STEPS, N_EXP)
            and np.shape(rec["trS"]) == (N_STEPS, N_EXP)
            and np.shape(rec["x"]) == (N_STEPS,)
        )
        knob_ok = keys_ok and np.array_equal(rec["knob"], knobs)
        x_ok = keys_ok and np.array_equal(rec["x"], pkl)
        fac = None
        if shapes_ok:
            fac = np.nanmean(rec["rho"] * rec["trS"], 1) if calibrate else np.nanmean(rec["trS"], 1)
        fac_ok = fac is not None and np.array_equal(fac, pkl)
        cal_ok = keys_ok and rec["calibrate"] == calibrate
        check(
            tag,
            "(viii)",
            keys_ok and shapes_ok and knob_ok and x_ok and fac_ok and cal_ok,
            f"keys {keys_ok} shapes {shapes_ok} knob==grid {knob_ok} x==values {x_ok} "
            f"factors==x {fac_ok} calibrate {cal_ok}",
        )

    # (ix) no axis record on a designed grid: runner level, then file level on sim raw
    if experiment == "simulation" and not calibrate:
        gamma_runner = second_runner(orch, "gamma", param_grid_override=[1.0])
        r_ok = hasattr(gamma_runner, "axis_record") and gamma_runner.axis_record() is None
        os.chdir(tmp)
        t0 = time.perf_counter()
        orch._run_sweeps(SweepSpec(param=("gamma",), metric=("width",)))
        t9 = time.perf_counter() - t0
        os.chdir(REPO)
        files = sorted(os.path.basename(f) for f in glob.glob(f"{sweep_dir}/gamma_*"))
        f_ok = "gamma_axis.pkl" not in files and {"gamma_values.pkl", "gamma_results.pkl", "gamma_statuses.pkl"} <= set(
            files
        )
        check(tag, "(ix)", r_ok and f_ok, f"runner axis_record None {r_ok}; files {files} ({t9:.1f}s)")

    plt.close("all")
    if len(FAIL) == n_fail_before:
        shutil.rmtree(tmp, ignore_errors=True)
        kept = "removed"
    else:
        kept = f"kept at {tmp}"
    print(f"  DA+PI width knob order {np.round(np.nanmean(w, 1), 4).tolist()}  sweep {t_sweep:.1f}s  out {kept}")


def main():
    experiments = sys.argv[1:] or ["simulation", "optical_device"]
    os.makedirs(TMPROOT, exist_ok=True)
    for experiment in experiments:
        for calibrate in (True, False):
            run_one(experiment, calibrate)
    print("\nRESULT:", "ALL PASS" if not FAIL else f"{len(FAIL)} FAIL")
    for f in FAIL:
        print("  ", f)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
