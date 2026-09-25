"""A39: the recalibrate sweep, the continuous version of the `recalibrate` toggle.

`experiment.sweep.param: [recalibrate]` sweeps t on linspace(0, 1) and re-solves the
DA+ methods at gamma~ = gamma ((1 - t) + t / rho) through the predict-time knob, on
data held constant (fit once per experiment, no re-fit, no re-augment). The plotted
x is the experiment mean of gamma~/gamma = (1 - t) + t/rho_j, from 1/rho (t = 1)
up to 1 (t = 0). One sim experiment, 5 steps, PI / DA+PI / PI&DA+PI:

  (i)    spec: data-constant, linear, no vline, the grid's endpoints are exactly 0 and 1;
  (ii)   production run (`ParamSweepRunner.run`, the models captured) under the toggles
         in config.yaml: one `build_models` and one `generate_data` call per experiment
         (fit once), PI widths bit-identical across steps, DA+PI narrower at every
         later t per query and strictly in the mean at t = 1, `PI&DA+PI` == max/min of
         PI and DA+PI at every step, and the hand re-solve at each t reproduces the
         recorded widths exactly;
  (ii')  identity leg (`pad=False, clipy=False`): width(t)/width(0) == sqrt((1 - t) + t/rho_hat)
         per query at every t (rtol 1e-6, rho_hat recomputed here), so the endpoints'
         ratio is sqrt(rho_hat);
  (iii)  axis: x == mean_j((1 - t) + t/rho_j) with rho_j == the fitted `PI&DA+PI.rho`,
         x[0] == 1 exactly, x[-1] == mean(1/rho_j), x decreasing in t, and a sample
         rho_j < 1 reads as 1 on the axis as it does in the solver;
         `recalibrate_axis.pkl` holds {knob, rho, x, xlabel} with x == `recalibrate_values.pkl`;
  (iv)   `_run_sweeps` renders without a swallowed error, the xlabel is the spec's,
         the DA+PI line carries sort(x) against the width means in that order;
  (v)    endpoints == the toggle: the t = 0 bounds equal a `recalibrate: false` run's and
         the t = 1 bounds a `recalibrate: true` run's, at the same gamma* on the same
         draw (PI bit-identical as the anchor), rtol 1e-6;
  (vi)   plumbing: `param: [recalibrate]` parses, a recipe-shaped block with the toggle
         AND the sweep resolves through main.py's steps, the yaml toggle is bool-only
         (`recalibrate: 0.5` rejected), both orchestrators resolve the strategy, and it
         takes no `augment_kwargs_fn` or `n_samples_override`.

`--optical` adds one optical experiment (5 steps) through (ii) and (iii). Writes only
into a fresh directory under `~/scratch/tmp/a39/`, removed when it passes. Nothing
here touches do-MNIST (its sweeps stop at `m`).

    MPLBACKEND=Agg python scripts/a39_recalibrate_sweep.py [--optical]
"""

import os
import pickle
import shutil
import sys
import tempfile
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from loguru import logger  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from src.experiments.base import SweepData  # noqa: E402
from src.experiments.configs import (  # noqa: E402
    PARAM_SPECS,
    SweepSpec,
    parse_experiment_plan,
    resolve_dataset_block,
)
from src.experiments.generic_runner import STRATEGIES, GenericParamSweep, RecalibrationStrategy  # noqa: E402
from src.experiments.optical_device import OpticalOrchestrator  # noqa: E402
from src.experiments.simulation import SimulationOrchestrator  # noqa: E402
from src.experiments.utils import set_seed  # noqa: E402
from src.experiments.utils.constants import label as method_label  # noqa: E402
from src.experiments.utils.metrics import rho_hat  # noqa: E402
from src.experiments.utils.plotting import create_sweep_plot  # noqa: E402
from src.methods.sensitivity_models import recalibrated_gamma  # noqa: E402

METHODS = ["PI", "DA+PI", "PI&DA+PI"]
N_STEPS = 5
N_JOBS = 4
TMPROOT = os.path.expanduser("~/scratch/tmp/a39")
FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAIL.append(name)


def width(bounds):
    return bounds[:, 1] - bounds[:, 0]


def configured_toggles():
    with open(os.path.join(REPO, "config.yaml")) as handle:
        defaults = (yaml.safe_load(handle) or {}).get("defaults") or {}
    return dict(
        recalibrate=bool(defaults.get("recalibrate", True)),
        pad=bool(defaults.get("pad", False)),
        clipy=bool(defaults.get("clipy", True)),
        mean_match=bool(defaults.get("mean_match", True)),
    )


def build(experiment, **toggles):
    set_seed(42)
    common = dict(
        seed=42, n_experiments=1, sweep_samples=N_STEPS, methods=METHODS, hyperparameters={}, n_jobs=N_JOBS, **toggles
    )
    if experiment == "simulation":
        return SimulationOrchestrator(n_samples=2048, kernel_dim=0, treatment_dim=32, **common)
    return OpticalOrchestrator(n_samples=1000, augmentation="rotation > hflip > vflip > random-permutation", **common)


def runner_for(orch, param, **override):
    return orch.get_sweep_runner_cls(param)(
        methods=orch.methods, method_factory=orch.build_methods, **override, **orch._get_clean_kwargs()
    )


def production_run(orch):
    """The production loop with its data and models captured, then re-solved at
    every t by hand. Returns (runner, x, results, ts, data, models, bounds[name][i])."""
    runner = runner_for(orch, "recalibrate")
    captured = {}
    calls = {"build_models": 0, "generate_data": 0}
    original_build, original_generate = runner.build_models, runner.generate_data

    def build_models(j, i, data):
        calls["build_models"] += 1
        models = original_build(j, i, data)
        captured[j] = (data, models)
        return models

    def generate_data(j, param):
        calls["generate_data"] += 1
        return original_generate(j, param)

    runner.build_models = build_models
    runner.generate_data = generate_data
    t0 = time.perf_counter()
    x, results, statuses = runner.run("recalibrate sweep")
    elapsed = time.perf_counter() - t0
    ts = np.asarray(runner.get_param_range(), dtype=float)
    data, models = captured[0]
    bounds = {
        name: [models[name].predict(data.X_test, **runner.get_predict_kwargs(t, 0)) for t in ts] for name in models
    }
    # leave the models where the loop left them (t = 1), not that anything reads them
    for name in models:
        models[name].predict(data.X_test[:1], **runner.get_predict_kwargs(ts[-1], 0))
    # the orchestrator's memo, so `_run_sweeps` renders exactly this record
    orch._sweep_cache["recalibrate"] = (x, results, statuses)
    orch._sweep_vlines["recalibrate"] = runner.vlines
    orch._sweep_axis["recalibrate"] = runner.axis_record()
    orch._sweep_xlabel["recalibrate"] = runner.xlabel
    print(f"      sweep {elapsed:.1f}s, {len(data.X_test)} queries, rho_hat {runner._rho[0]:.6f}")
    check(
        "(ii) fit once: one build_models and one generate_data call per experiment",
        calls == {"build_models": runner.n_experiments, "generate_data": runner.n_experiments},
        f"{calls} for {len(ts)} steps",
    )
    return runner, np.asarray(x, dtype=float), results, ts, data, models, bounds


# --------------------------------------------------------------------- (i) spec


def leg_i():
    print("(i) spec")
    spec = PARAM_SPECS["recalibrate"]
    grid = spec.grid_fn("simulation", N_STEPS)
    check("(i) data_constant", spec.data_constant)
    check("(i) linear scale, no vline, no ATE", spec.xscale == "linear" and spec.vlines == () and not spec.include_ate)
    check("(i) grid endpoints exactly 0 and 1", grid[0] == 0.0 and grid[-1] == 1.0 and len(grid) == N_STEPS, f"{grid}")
    check(
        "(i) STRATEGIES['recalibrate'] is RecalibrationStrategy", STRATEGIES.get("recalibrate") is RecalibrationStrategy
    )


# ------------------------------------------------------------ (ii) production


def leg_ii(experiment, toggles, label):
    print(f"(ii) production run, {experiment}, toggles {toggles}")
    orch = build(experiment, **toggles)
    runner, x, results, ts, data, models, bounds = production_run(orch)
    w_pi = results["PI"]["interval_width"][:, 0]
    w_da = results["DA+PI"]["interval_width"][:, 0]
    check(f"(ii) {label} PI width bit-identical across steps", bool(np.all(w_pi == w_pi[0])), f"{w_pi}")
    check(
        f"(ii) {label} DA+PI width non-increasing in t, strictly smaller at t = 1",
        bool(np.all(np.diff(w_da) <= 1e-12)) and w_da[-1] < w_da[0],
        f"{np.round(w_da, 5)}",
    )
    pi_same = all(np.array_equal(bounds["PI"][i], bounds["PI"][0]) for i in range(len(ts)))
    check(f"(ii) {label} PI bounds bit-identical across t (rho = 1 ignores the knob)", pi_same)
    per_query = all(
        np.all(width(bounds["DA+PI"][i + 1]) <= width(bounds["DA+PI"][i]) + 1e-9) for i in range(len(ts) - 1)
    )
    check(f"(ii) {label} DA+PI narrower at every later t per query", per_query)
    worst = 0.0
    for i in range(len(ts)):
        got = bounds["PI&DA+PI"][i]
        want = np.column_stack(
            [
                np.maximum(bounds["PI"][i][:, 0], bounds["DA+PI"][i][:, 0]),
                np.minimum(bounds["PI"][i][:, 1], bounds["DA+PI"][i][:, 1]),
            ]
        )
        worst = max(worst, float(np.nanmax(np.abs(got - want))))
    check(f"(ii) {label} PI&DA+PI == max/min of PI and DA+PI at every step", worst <= 1e-9, f"max |d| {worst:.2e}")
    # the hand re-solve reproduces the record: the loop solved the same models at the same t
    # `interval_width` is the plain mean width (DEFAULT_NORMALIZE_ERROR is off),
    # so the hand re-solve must reproduce the record exactly, no factor allowed
    hand = np.array([np.nanmean(width(bounds["DA+PI"][i])) for i in range(len(ts))])
    check(
        f"(ii) {label} hand re-solve at each t reproduces the recorded DA+PI widths",
        np.allclose(hand, w_da, rtol=1e-9),
        f"max rel |d| {float(np.max(np.abs(hand / w_da - 1.0))):.2e}",
    )
    return orch, runner, x, results, ts, data, models, bounds


# ------------------------------------------------------------- (ii') identity


def leg_ii_identity(toggles):
    print("(ii') identity leg: pad off, clipy off")
    toggles = dict(toggles, pad=False, clipy=False)
    orch = build("simulation", **toggles)
    runner, x, results, ts, data, models, bounds = production_run(orch)
    rho = rho_hat(data.X, data.GX, data.y, intercept=runner.mean_match)
    w0 = width(bounds["DA+PI"][0])
    ok, detail = True, []
    for i, t in enumerate(ts):
        ratio = width(bounds["DA+PI"][i]) / w0
        want = np.sqrt((1.0 - t) + t / rho)
        if not np.allclose(ratio, want, rtol=1e-6):
            ok = False
        detail.append(f"t={t:.2f}: {ratio.min():.6f}..{ratio.max():.6f} want {want:.6f}")
    check("(ii') width(t)/width(0) == sqrt((1 - t) + t/rho_hat) per query", ok, "; ".join(detail))
    ratio = w0 / width(bounds["DA+PI"][-1])
    check(
        "(ii') width(t=0)/width(t=1) == sqrt(rho_hat)",
        np.allclose(ratio, np.sqrt(rho), rtol=1e-6),
        f"{ratio.min():.6f}..{ratio.max():.6f} vs {np.sqrt(rho):.6f}",
    )
    check("(ii') runner._rho == rho_hat recomputed", np.isclose(runner._rho[0], rho, rtol=1e-12), f"{rho:.6f}")
    return orch, runner, x, ts, data, models, bounds


# ------------------------------------------------------------------ (iii) axis


def leg_iii(runner, x, ts, models, label):
    print(f"(iii) axis, {label}")
    rho = runner._rho[0]
    want = (1.0 - ts) + ts / rho
    check(f"(iii) {label} x == (1 - t) + t/rho_hat", np.allclose(x, want, rtol=1e-12), f"x {np.round(x, 5)}")
    # a sample rho_hat < 1 is read as 1 by the solver (DPI); the axis must say the same
    runner._rho[0] = 0.9
    clamped = runner.observed_x(ts)
    runner._rho[0] = rho
    check(
        f"(iii) {label} a rho_hat < 1 gives x == 1 at every t, as the solver's recalibrated_gamma",
        np.array_equal(clamped, np.ones_like(ts))
        and np.allclose([recalibrated_gamma(1.0, 0.9, t) for t in ts], 1.0, rtol=0, atol=0),
        f"{clamped}",
    )
    check(
        f"(iii) {label} rho_hat == the fitted PI&DA+PI.rho",
        np.isclose(rho, models["PI&DA+PI"].rho, rtol=1e-9),
        f"{rho:.6f} vs {models['PI&DA+PI'].rho:.6f}",
    )
    check(f"(iii) {label} x[0] == 1 exactly, x[-1] == 1/rho", x[0] == 1.0 and np.isclose(x[-1], 1.0 / rho, rtol=1e-12))
    check(f"(iii) {label} x decreasing in t", bool(np.all(np.diff(x) < 0)))
    rec = runner.axis_record()
    check(
        f"(iii) {label} axis_record keys and values",
        set(rec) == {"knob", "rho", "x", "xlabel"}
        and np.array_equal(rec["knob"], ts)
        and np.array_equal(rec["x"], x)
        and rec["rho"].shape == (1,)
        and rec["rho"][0] == rho
        and rec["xlabel"] == PARAM_SPECS["recalibrate"].xlabel,
    )


# ---------------------------------------------------------------- (iv) render


def leg_iv(orch, x, results, experiment):
    print(f"(iv) render, {experiment}")
    errors = []
    err_sink = logger.add(lambda m: errors.append(m.record["message"]), level="ERROR")
    tmp = tempfile.mkdtemp(prefix=f"{experiment}_", dir=TMPROOT)
    os.chdir(tmp)
    plt.close("all")
    orch._run_sweeps(SweepSpec(param=("recalibrate",), metric=("width",)))
    ax = plt.gcf().axes[0]
    os.chdir(REPO)
    logger.remove(err_sink)
    check("(iv) no swallowed plotting error", not errors, errors[0][:120] if errors else "")
    check("(iv) xlabel is the spec label", ax.get_xlabel() == PARAM_SPECS["recalibrate"].xlabel, f"{ax.get_xlabel()!r}")
    sweep_dir = f"{tmp}/artifacts/{experiment}/sweep"
    with open(f"{sweep_dir}/recalibrate_values.pkl", "rb") as fh:
        values = np.asarray(pickle.load(fh))  # noqa: S301 - the gate wrote this file itself
    axis_path = f"{sweep_dir}/recalibrate_axis.pkl"
    if os.path.exists(axis_path):
        with open(axis_path, "rb") as fh:
            rec = pickle.load(fh)  # noqa: S301 - the gate wrote this file itself
        check(
            "(iii) recalibrate_axis.pkl written with {knob, rho, x, xlabel}, x == values pkl",
            set(rec) == {"knob", "rho", "x", "xlabel"}
            and np.array_equal(rec["x"], values)
            and np.array_equal(values, x),
        )
    else:
        check("(iii) recalibrate_axis.pkl written", False, "missing")
    # unbootstrapped render: the drawn y is the plain width mean, in sorted-x order
    plt.close("all")
    create_sweep_plot(
        x,
        {m: results[m]["interval_width"] for m in results},
        xlabel=PARAM_SPECS["recalibrate"].xlabel,
        ylabel="w",
        experiment=experiment,
        fname="recalibrate_width",
        savefig=False,
        bootstrapped=False,
        has_z=False,
    )
    ax2 = plt.gcf().axes[0]
    lines = [ln for ln in ax2.lines if ln.get_label() == method_label("DA+PI", False)]
    order = np.argsort(x, kind="stable")
    w = results["DA+PI"]["interval_width"]
    xd = np.asarray(lines[0].get_xdata(), dtype=float) if len(lines) == 1 else None
    yd = np.asarray(lines[0].get_ydata(), dtype=float) if len(lines) == 1 else None
    check(
        "(iv) DA+PI line carries sort(x) with y reordered (the sort reverses the knob order)",
        xd is not None
        and np.array_equal(xd, np.sort(x))
        and np.array_equal(yd, np.nanmean(w, axis=1)[order])
        and order.tolist() == list(range(len(x)))[::-1],
        f"xdata {None if xd is None else np.round(xd, 5).tolist()} argsort {order.tolist()}",
    )
    plt.close("all")
    return tmp


# --------------------------------------------------------- (v) the toggle's ends


def toggle_bounds(recalibrate, toggles, X_test):
    """A fresh orchestrator under the toggle, through the gamma runner at gamma*
    (grid [1.0]), the same fixture and draw sequence as the sweep runner's."""
    orch = build("simulation", **dict(toggles, recalibrate=recalibrate))
    runner = runner_for(orch, "gamma", param_grid_override=[1.0])
    data = SweepData.coerce(runner.generate_data(0, 1.0))
    models = runner.build_models(0, 0, data)
    same_draw = np.array_equal(data.X_test, X_test)
    return {
        name: model.predict(X_test, **runner.get_predict_kwargs(1.0, 0)) for name, model in models.items()
    }, same_draw


def leg_v(toggles, data, bounds):
    print("(v) endpoints against the toggle (configured pad / clipy / mean_match)")
    off, same_off = toggle_bounds(False, toggles, data.X_test)
    on, same_on = toggle_bounds(True, toggles, data.X_test)
    check("(v) the toggle runs see the sweep's draw", same_off and same_on)
    check(
        "(v) PI bit-identical: sweep, toggle off, toggle on",
        np.array_equal(off["PI"], bounds["PI"][0]) and np.array_equal(on["PI"], bounds["PI"][-1]),
    )
    for name in ("DA+PI", "PI&DA+PI"):
        d0 = float(np.nanmax(np.abs(bounds[name][0] - off[name])))
        d1 = float(np.nanmax(np.abs(bounds[name][-1] - on[name])))
        check(
            f"(v) {name}: t = 0 == `recalibrate: false`, t = 1 == `recalibrate: true` (rtol 1e-6)",
            np.allclose(bounds[name][0], off[name], rtol=1e-6) and np.allclose(bounds[name][-1], on[name], rtol=1e-6),
            f"max |d| {d0:.2e} / {d1:.2e}",
        )
    print(
        f"      DA+PI mean width: t = 0 {np.nanmean(width(bounds['DA+PI'][0])):.4f} "
        f"(toggle off {np.nanmean(width(off['DA+PI'])):.4f}), t = 1 {np.nanmean(width(bounds['DA+PI'][-1])):.4f} "
        f"(toggle on {np.nanmean(width(on['DA+PI'])):.4f})"
    )


# --------------------------------------------------------------- (vi) plumbing


def leg_vi(sim_orch):
    print("(vi) plumbing")
    plan = parse_experiment_plan({"sweep": {"param": ["recalibrate"], "metric": ["width", "coverage"]}})
    check("(vi) `param: [recalibrate]` parses", plan.sweep is not None and plan.sweep.param == ("recalibrate",))
    recipe = yaml.safe_load(
        "defaults:\n  recalibrate: false\n  pad: true\n  clipy: false\n  mean_match: true\n  n_jobs: 2\n"
        "simulation:\n  seed: 42\n  n_samples: 256\n  kernel_dim: 0\n  n_experiments: 1\n  sweep_samples: 3\n"
        "  augmentation: translate\n  experiment:\n    sweep:\n"
        "      param: [recalibrate, gamma]\n      metric: [width]\n"
    )
    defaults = recipe.pop("defaults")
    block = {**defaults, **recipe["simulation"], "im-ci": 0}
    plan = parse_experiment_plan(block.get("experiment"))
    block = resolve_dataset_block("simulation", block)
    check(
        "(vi) a recipe with the toggle AND the sweep resolves through main.py's steps",
        block.get("recalibrate") is False and plan.sweep.param == ("recalibrate", "gamma"),
        f"recalibrate {block.get('recalibrate')!r} param {plan.sweep.param}",
    )
    for bad in (0.5, 1, "yes", "true"):
        try:
            resolve_dataset_block("simulation", {"seed": 42, "kernel_dim": 0, "recalibrate": bad})
            rejected = False
        except ValueError:
            rejected = True
        check(f"(vi) the yaml toggle `recalibrate: {bad!r}` is rejected (bool only)", rejected)
    check(
        "(vi) the yaml toggle accepts both bools",
        all(
            resolve_dataset_block("simulation", {"seed": 42, "kernel_dim": 0, "recalibrate": v})["recalibrate"] is v
            for v in (True, False)
        ),
    )
    cls = sim_orch.get_sweep_runner_cls("recalibrate")
    check("(vi) SimulationOrchestrator resolves the strategy", issubclass(cls, RecalibrationStrategy))
    try:
        opt = OpticalOrchestrator(
            seed=42,
            n_samples=1000,
            n_experiments=1,
            sweep_samples=N_STEPS,
            methods=METHODS,
            hyperparameters={},
            augmentation="rotation > hflip > vflip > random-permutation",
        )
        opt_ok = issubclass(opt.get_sweep_runner_cls("recalibrate"), RecalibrationStrategy)
    except Exception as error:  # the class must resolve without a run
        opt_ok, error_text = False, str(error)[:120]
    else:
        error_text = ""
    check("(vi) OpticalOrchestrator resolves the strategy", opt_ok, error_text)
    check(
        "(vi) the strategy is a plain GenericParamSweep (no knob kwargs)",
        issubclass(RecalibrationStrategy, GenericParamSweep)
        and "augment_kwargs_fn" not in RecalibrationStrategy.__init__.__code__.co_varnames
        and "n_samples_override" not in RecalibrationStrategy.__init__.__code__.co_varnames,
    )


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    os.chdir(REPO)
    os.makedirs(TMPROOT, exist_ok=True)
    optical = "--optical" in sys.argv[1:]
    toggles = configured_toggles()
    leg_i()
    orch, runner, x, results, ts, data, models, bounds = leg_ii("simulation", toggles, "sim")
    leg_iii(runner, x, ts, models, "sim")
    tmp = leg_iv(orch, x, results, "simulation")
    leg_v(toggles, data, bounds)
    leg_vi(orch)
    orch_id, runner_id, x_id, ts_id, data_id, models_id, bounds_id = leg_ii_identity(toggles)
    leg_iii(runner_id, x_id, ts_id, models_id, "sim identity leg")
    tmps = [tmp]
    if optical:
        orch_o, runner_o, x_o, results_o, ts_o, data_o, models_o, bounds_o = leg_ii(
            "optical_device", toggles, "optical"
        )
        leg_iii(runner_o, x_o, ts_o, models_o, "optical")
        tmps.append(leg_iv(orch_o, x_o, results_o, "optical_device"))
    if not FAIL:
        for path in tmps:
            shutil.rmtree(path, ignore_errors=True)
    else:
        print(f"  outputs kept at {tmps}")
    print(f"\n{'A39 ALL PASS' if not FAIL else 'A39 FAILURES: ' + chr(10) + chr(10).join('  ' + f for f in FAIL)}")
    sys.exit(bool(FAIL))
