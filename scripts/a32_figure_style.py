"""A32: the figure style, re-rendered from the sweep pkls.

Every figure labels only its major ticks, and the five style keys of
`constants._STYLE_KEYS` (`legend`, `x_color`, `y_color`, `title`, `title_color`)
reach the figures through `PLOT_CONFIGS` and `ANNOTATE_SWEEP_PLOT`. Pkl-driven: it
pairs `{param}_values.pkl` with `{param}_results.pkl` under `<artifacts>/<experiment>/
sweep/` (so `trS_axis.pkl` and the statuses are ignored), and takes the query pair
and `perf/perf.pkl` when present. No experiment is run.

  (a) every re-rendered figure (each sweep param x metric, the query sweep, the
      perf figure with its twin axis): zero non-empty minor tick labels on every
      axes, both axes, at least one minor tick mark on every log axis, and at
      least two major ticks inside the view on every axis; no plotting error
      was swallowed;
  (b) `legend`: False removes the legend artist, a loc string places it, True
      shows it over a `hide_legend=True` argument, and the argument still hides
      it when the key is absent;
  (c) `x_color` / `y_color`: the axis label and every major tick label carry the
      colour; black by default;
  (d) `title` / `title_color`: the title text and colour; no title by default;
  (e) the same keys through `ANNOTATE_SWEEP_PLOT["pc12"]` kwargs on the query
      sweep, and `PLOT_CONFIGS[experiment]["query"]` overriding them key by key;
  (f) the perf figure takes `title`, `x_color` (categorical tick labels) and
      `y_color` (every y label, the twin included) from `PLOT_CONFIGS["*"]["perf"]`;
  (g) an unknown key is an import-time ValueError: copies of `constants.py` and
      `configs.py` with `legnd` injected fail to import in a subprocess, and
      `validate_plot_keys` raises when called directly;
  (h) the query panel (with a log row injected, where the labelling formatter
      would fire) and the digit sweep, on synthetic inputs, show no minor labels.

    MPLBACKEND=Agg python scripts/a32_figure_style.py [--artifacts DIR] [--save]

`--artifacts` defaults to the repo's untracked `artifacts/`. Without `--save`
nothing is written. With `--save` every sweep and perf pdf is re-rendered INTO
`--artifacts` (every sweep vline is a PARAM_SPECS constant, so the pkls carry all
a figure needs). The query sweep and panel are the orchestrator's (`query: true`)
and are left alone (checked by mtime). The run chdirs to a scratch directory whose `artifacts` is a
symlink to that tree, so the repo's own `artifacts/` is written only when it IS
the argument. Nothing here touches do-MNIST: the digit sweep is drawn from a
synthetic (n, 3, 8, 8) array.
"""

import argparse
import glob
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from loguru import logger  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from src.experiments.configs import ANNOTATE_SWEEP_PLOT, METRIC_SPECS, PARAM_SPECS  # noqa: E402
from src.experiments.utils import plotting  # noqa: E402
from src.experiments.utils.constants import PANEL_CONFIGS, PLOT_CONFIGS, plot_keys_for, validate_plot_keys  # noqa: E402

TMPROOT = os.path.expanduser("~/scratch/tmp/a32")
SYNTHETIC = "_a32"
FAIL = []
_errors = []
logger.add(lambda m: _errors.append(m), level="ERROR")


def check(row, tag, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {row:4s} {tag} {detail}")
    if not ok:
        FAIL.append(f"{row} {tag} {detail}")


def load(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)  # noqa: S301 - our own artifacts, no untrusted input


def same_colour(a, b):
    return to_rgba(a) == to_rgba(b)


# ------------------------------------------------------------------ inspection
def tick_report(fig):
    """(non-empty minor labels over all axes, log axes without a minor mark, log axes,
    minimum count of in-view major ticks over all axes)."""
    fig.canvas.draw()
    labels, bare, n_log, fewest = 0, 0, 0, None
    for ax in fig.axes:
        for axis, scale in ((ax.xaxis, ax.get_xscale()), (ax.yaxis, ax.get_yscale())):
            labels += sum(bool(t.get_text()) for t in axis.get_minorticklabels())
            if scale == "log":
                n_log += 1
                bare += len(axis.get_minorticklocs()) == 0
            lo, hi = sorted(axis.get_view_interval())
            majors = int(sum(lo <= t <= hi for t in axis.get_majorticklocs()))
            fewest = majors if fewest is None else min(fewest, majors)
    return labels, bare, n_log, 0 if fewest is None else fewest


def style_report(ax):
    ax.figure.canvas.draw()
    legend = ax.get_legend()
    return dict(
        legend=None if legend is None else legend._loc,
        xlabel=ax.xaxis.label.get_color(),
        xticks={t.get_color() for t in ax.get_xticklabels() if t.get_text()},
        ylabel=ax.yaxis.label.get_color(),
        yticks={t.get_color() for t in ax.get_yticklabels() if t.get_text()},
        title=ax.get_title(),
        title_color=ax.title.get_color(),
    )


def colours_are(report, x_color, y_color):
    ok_x = same_colour(report["xlabel"], x_color) and all(same_colour(c, x_color) for c in report["xticks"])
    ok_y = same_colour(report["ylabel"], y_color) and all(same_colour(c, y_color) for c in report["yticks"])
    return ok_x and report["xticks"], ok_y and report["yticks"]


class injected:
    """Temporarily set PLOT_CONFIGS[experiment][plot_id]; restore on exit."""

    def __init__(self, experiment, plot_id, cfg):
        self.experiment, self.plot_id, self.cfg = experiment, plot_id, cfg

    def __enter__(self):
        self.had = self.experiment in PLOT_CONFIGS
        self.before = dict(PLOT_CONFIGS.get(self.experiment, {}))
        PLOT_CONFIGS.setdefault(self.experiment, {})[self.plot_id] = self.cfg
        return self

    def __exit__(self, *exc):
        if self.had:
            PLOT_CONFIGS[self.experiment] = self.before
        else:
            del PLOT_CONFIGS[self.experiment]


# -------------------------------------------------------------------- renders
def render_sweep(experiment, param, metric, x, results, save, **kwargs):
    spec = METRIC_SPECS[metric]
    plotting.create_sweep_plot(
        x,
        {name: record[spec.key] for name, record in results.items() if spec.include_ate or name != "ATE"},
        experiment=experiment,
        fname=f"{param}_{metric}",
        xlabel=PARAM_SPECS[param].xlabel,
        ylabel=spec.ylabel,
        xscale=PARAM_SPECS[param].xscale,
        yscale=spec.yscale,
        vlines=PARAM_SPECS[param].vlines,
        savefig=save,
        **kwargs,
    )
    return plt.gcf()


def render_query(experiment, x, results, save, **kwargs):
    plotting.create_query_sweep_plot(
        x, results, **{**ANNOTATE_SWEEP_PLOT["pc12"], **kwargs}, experiment=experiment, savefig=save
    )
    return plt.gcf()


def render_perf(experiment, record, save):
    overlays = [m for m in ("wall_clock", "seed_var") if all(m in r for r in record.values())]
    plotting.create_perf_plot(record, overlay_metrics=overlays, experiment=experiment, savefig=save)
    return plt.gcf()


# ---------------------------------------------------------------------- rows
def pdf_mtimes(artifacts):
    return {p: os.path.getmtime(p) for p in glob.glob(f"{artifacts}/*/*/*.pdf")}


def row_a(artifacts, experiments, save):
    """Every figure from the pkls: no minor labels, marks kept on log axes."""
    n_fig = 0
    before_mtimes = pdf_mtimes(artifacts)
    for experiment in experiments:
        sweep_dir = f"{artifacts}/{experiment}/sweep"
        params = sorted(
            os.path.basename(p)[: -len("_values.pkl")]
            for p in glob.glob(f"{sweep_dir}/*_values.pkl")
            if os.path.exists(p.replace("_values.pkl", "_results.pkl"))
        )
        unknown = [p for p in params if p not in PARAM_SPECS]
        params = [p for p in params if p in PARAM_SPECS]
        ignored = sorted(
            os.path.basename(p)
            for p in glob.glob(f"{sweep_dir}/*.pkl")
            if not any(p.endswith(f"{q}_{kind}.pkl") for q in params for kind in ("values", "results"))
        )
        print(f"  {experiment}: sweeps {params}; ignored {ignored}; not a PARAM_SPECS param {unknown}")
        check("(a)", f"{experiment} discovery", bool(params) and not unknown, f"{len(params)} sweep params")

        for param in params:
            x, results = load(f"{sweep_dir}/{param}_values.pkl"), load(f"{sweep_dir}/{param}_results.pkl")
            metrics = [m for m, spec in METRIC_SPECS.items() if not spec.perf_only]
            labels = bare = n_log = 0
            fewest = None
            before = len(_errors)
            for metric in metrics:
                fig = render_sweep(experiment, param, metric, x, results, save)
                a, b, c, d = tick_report(fig)
                labels, bare, n_log = labels + a, bare + b, n_log + c
                fewest = d if fewest is None else min(fewest, d)
                plt.close("all")
                n_fig += 1
            check(
                "(a)",
                f"{experiment} {param}",
                labels == 0 and bare == 0 and fewest >= 2 and len(_errors) == before,
                f"{len(metrics)} figures: minor labels {labels}, log axes without marks {bare}/{n_log}, "
                f"fewest majors in view {fewest}, errors {len(_errors) - before}",
            )

        query = f"{artifacts}/{experiment}/query"
        if os.path.exists(f"{query}/treatment_values.pkl") and os.path.exists(f"{query}/outcome_values.pkl"):
            before = len(_errors)
            fig = render_query(
                experiment, load(f"{query}/treatment_values.pkl"), load(f"{query}/outcome_values.pkl"), False
            )
            labels, bare, n_log, fewest = tick_report(fig)
            plt.close("all")
            n_fig += 1
            check(
                "(a)",
                f"{experiment} query",
                labels == 0 and bare == 0 and fewest >= 2 and len(_errors) == before,
                f"minor labels {labels}, log axes without marks {bare}/{n_log}, fewest majors in view {fewest}",
            )
        else:
            print(f"  {experiment}: no query pkls")

        perf = f"{artifacts}/{experiment}/perf/perf.pkl"
        if os.path.exists(perf):
            before = len(_errors)
            fig = render_perf(experiment, load(perf), save)
            labels, bare, n_log, fewest = tick_report(fig)
            n_axes = len(fig.axes)
            plt.close("all")
            n_fig += 1
            check(
                "(a)",
                f"{experiment} perf",
                labels == 0 and bare == 0 and n_log >= 1 and fewest >= 2 and len(_errors) == before,
                f"{n_axes} axes (twin included): minor labels {labels}, log axes without marks {bare}/{n_log}, "
                f"fewest majors in view {fewest}",
            )
        else:
            print(f"  {experiment}: no perf.pkl")
    print(f"  {n_fig} figures re-rendered")
    if save:
        after = pdf_mtimes(artifacts)
        touched = sorted(p for p in after if after[p] != before_mtimes.get(p))
        kept = [p for p in before_mtimes if "/query/" in p]
        wrong = [p for p in kept if p in touched]
        expected = [p for p in before_mtimes if p not in kept]
        missing = [p for p in expected if p not in touched]
        check(
            "(a)",
            "--save wrote the sweep and perf pdfs only",
            not wrong and not missing,
            f"{len(touched)} rewritten; untouched as required {len(kept) - len(wrong)}/{len(kept)}; "
            f"not rewritten {[os.path.relpath(p, artifacts) for p in missing]}; "
            f"wrongly rewritten {[os.path.relpath(p, artifacts) for p in wrong]}",
        )


def first_sweep(artifacts, experiments):
    """The (experiment, param, x, results) the style rows render; simulation gamma when present."""
    for experiment in ["simulation", *experiments]:
        for param in ["gamma", *PARAM_SPECS]:
            values = f"{artifacts}/{experiment}/sweep/{param}_values.pkl"
            results = values.replace("_values.pkl", "_results.pkl")
            if os.path.exists(values) and os.path.exists(results):
                return experiment, param, load(values), load(results)
    return None


def rows_bcd(artifacts, experiments):
    """legend, colours and title through PLOT_CONFIGS on one sweep figure."""
    found = first_sweep(artifacts, experiments)
    if found is None:
        check("(b)", "style rows", False, "no sweep pkl pair to render")
        return
    experiment, param, x, results = found
    metric = "coverage"
    plot_id = f"{param}_{metric}"
    tag = f"{experiment} {plot_id}"

    def render(cfg=None, **kwargs):
        with injected(experiment, plot_id, cfg or {}):
            fig = render_sweep(experiment, param, metric, x, results, False, **kwargs)
        report = style_report(fig.axes[0])
        plt.close("all")
        return report

    # (b) legend
    default = render()
    hidden = render({"legend": False})
    placed = render({"legend": "upper left"})
    forced = render({"legend": True}, hide_legend=True)
    by_arg = render(hide_legend=True)
    check("(b)", f"{tag} legend default on", default["legend"] is not None, f"loc {default['legend']}")
    check("(b)", f"{tag} legend False", hidden["legend"] is None, f"legend {hidden['legend']}")
    check("(b)", f"{tag} legend 'upper left'", placed["legend"] == 2, f"loc {placed['legend']} (2 = upper left)")
    check("(b)", f"{tag} legend True over hide_legend", forced["legend"] is not None, f"loc {forced['legend']}")
    check("(b)", f"{tag} hide_legend argument alone", by_arg["legend"] is None, f"legend {by_arg['legend']}")

    # (c) colours
    ok_x, ok_y = colours_are(default, "black", "black")
    check(
        "(c)",
        f"{tag} default black",
        ok_x and ok_y,
        f"{default['xlabel']} {default['xticks']} {default['ylabel']} {default['yticks']}",
    )
    coloured = render({"x_color": "red", "y_color": "tab:blue"})
    ok_x, ok_y = colours_are(coloured, "red", "tab:blue")
    check(
        "(c)",
        f"{tag} x_color red, y_color tab:blue",
        ok_x and ok_y,
        f"xlabel {coloured['xlabel']} xticks {coloured['xticks']} "
        f"ylabel {coloured['ylabel']} yticks {coloured['yticks']}",
    )
    star = None
    with injected("*", plot_id, {"y_color": "green"}):
        star = render()
    ok_x, ok_y = colours_are(star, "black", "green")
    check(
        "(c)",
        f"{tag} y_color through PLOT_CONFIGS['*']",
        ok_x and ok_y,
        f"ylabel {star['ylabel']} yticks {star['yticks']}",
    )

    # (d) title
    check("(d)", f"{tag} no title by default", default["title"] == "", repr(default["title"]))
    titled = render({"title": r"a title", "title_color": "tab:orange"})
    check(
        "(d)",
        f"{tag} title and title_color",
        titled["title"] == "a title" and same_colour(titled["title_color"], "tab:orange"),
        f"{titled['title']!r} {titled['title_color']}",
    )
    by_arg = render(title="by argument", title_color="m")
    check(
        "(d)",
        f"{tag} title by argument",
        by_arg["title"] == "by argument" and same_colour(by_arg["title_color"], "m"),
        f"{by_arg['title']!r} {by_arg['title_color']}",
    )


def row_e(artifacts, experiments):
    """The keys through ANNOTATE_SWEEP_PLOT kwargs on the query sweep, then the query id overriding."""
    pair = None
    for experiment in experiments:
        query = f"{artifacts}/{experiment}/query"
        if os.path.exists(f"{query}/treatment_values.pkl") and os.path.exists(f"{query}/outcome_values.pkl"):
            pair = experiment, load(f"{query}/treatment_values.pkl"), load(f"{query}/outcome_values.pkl")
            break
    if pair is None:
        print("  (e) skipped: no query pkl pair under the artifacts")
        return
    experiment, x, results = pair
    kwargs = {"legend": False, "x_color": "red", "y_color": "tab:blue", "title": "radial", "title_color": "tab:orange"}
    validate_plot_keys(
        "a32 kwargs", {"pc12": {**ANNOTATE_SWEEP_PLOT["pc12"], **kwargs}}, {"xlabel", "xscale"} | set(kwargs)
    )

    default = style_report(render_query(experiment, x, results, False).axes[0])
    plt.close("all")
    ok_x, ok_y = colours_are(default, "black", "black")
    check(
        "(e)",
        f"{experiment} query default",
        default["legend"] is not None and ok_x and ok_y and default["title"] == "",
        f"legend {default['legend']} title {default['title']!r}",
    )

    via_kwargs = style_report(render_query(experiment, x, results, False, **kwargs).axes[0])
    plt.close("all")
    ok_x, ok_y = colours_are(via_kwargs, "red", "tab:blue")
    check(
        "(e)",
        f"{experiment} query keys as kwargs (ANNOTATE_SWEEP_PLOT)",
        via_kwargs["legend"] is None
        and ok_x
        and ok_y
        and via_kwargs["title"] == "radial"
        and same_colour(via_kwargs["title_color"], "tab:orange"),
        f"legend {via_kwargs['legend']} xlabel {via_kwargs['xlabel']} ylabel {via_kwargs['ylabel']} "
        f"title {via_kwargs['title']!r} {via_kwargs['title_color']}",
    )

    override = {"legend": "lower left", "x_color": "green", "y_color": "k", "title": "over", "title_color": "k"}
    with injected(experiment, "query", override):
        via_cfg = style_report(render_query(experiment, x, results, False, **kwargs).axes[0])
    plt.close("all")
    ok_x, ok_y = colours_are(via_cfg, "green", "k")
    check(
        "(e)",
        f"{experiment} PLOT_CONFIGS[{experiment!r}]['query'] wins over the kwargs",
        via_cfg["legend"] == 3
        and ok_x
        and ok_y
        and via_cfg["title"] == "over"
        and same_colour(via_cfg["title_color"], "k"),
        f"legend {via_cfg['legend']} (3 = lower left) xlabel {via_cfg['xlabel']} ylabel {via_cfg['ylabel']} "
        f"title {via_cfg['title']!r} {via_cfg['title_color']}",
    )


def row_f(artifacts, experiments):
    """The perf figure takes title, x_color and y_color from PLOT_CONFIGS['*']['perf']."""
    path = None
    for experiment in experiments:
        if os.path.exists(f"{artifacts}/{experiment}/perf/perf.pkl"):
            path, exp = f"{artifacts}/{experiment}/perf/perf.pkl", experiment
            break
    if path is None:
        print("  (f) skipped: no perf.pkl under the artifacts")
        return
    record = load(path)
    base = dict(PLOT_CONFIGS.get("*", {}).get("perf", {}))
    with injected(
        "*", "perf", {**base, "title": "perf title", "title_color": "red", "x_color": "green", "y_color": "tab:blue"}
    ):
        fig = render_perf(exp, record, False)
        fig.canvas.draw()
        top = fig.axes[0]
        title_ok = top.get_title() == "perf title" and same_colour(top.title.get_color(), "red")
        y_ok = all(
            same_colour(ax.yaxis.label.get_color(), "tab:blue")
            and all(same_colour(t.get_color(), "tab:blue") for t in ax.get_yticklabels() if t.get_text())
            for ax in fig.axes
        )
        # the twin shares the categorical labels but hides its x axis
        xticks = [t for ax in fig.axes if ax.xaxis.get_visible() for t in ax.get_xticklabels() if t.get_text()]
        x_ok = bool(xticks) and all(same_colour(t.get_color(), "green") for t in xticks)
        plt.close("all")
    check(
        "(f)",
        f"{exp} perf title, x_color, y_color",
        title_ok and y_ok and x_ok,
        f"title {title_ok} y {y_ok} x {x_ok} on {len(fig.axes)} axes",
    )


def row_g():
    """A typo in either table is an import-time ValueError."""
    try:
        validate_plot_keys("T", {"gamma_coverage": {"legnd": False}}, plot_keys_for)
        direct = False
    except ValueError:
        direct = True
    check("(g)", "validate_plot_keys raises on 'legnd'", direct)

    cases = {
        "constants": (
            "src/experiments/utils/constants.py",
            '"perf": {"bars": False}',
            '"perf": {"bars": False, "legnd": True}',
        ),
        "configs": (
            "src/experiments/configs.py",
            '    "pc12": {\n        "xlabel": r"$\\vartheta$",',
            '    "pc12": {\n        "legnd": True,\n        "xlabel": r"$\\vartheta$",',
        ),
    }
    tmp = tempfile.mkdtemp(prefix="typo_", dir=TMPROOT)
    for name, (rel, old, new) in cases.items():
        with open(f"{REPO}/{rel}") as fh:
            src = fh.read()
        if src.count(old) != 1:
            check("(g)", f"{name} anchor", False, f"anchor found {src.count(old)} times")
            continue
        module = f"a32_{name}_typo"
        with open(f"{tmp}/{module}.py", "w") as fh:
            fh.write(src.replace(old, new))
        env = {**os.environ, "PYTHONPATH": f"{tmp}{os.pathsep}{REPO}", "MPLBACKEND": "Agg"}
        proc = subprocess.run(  # noqa: S603 - our own interpreter on a file this gate wrote
            [sys.executable, "-c", f"import {module}"], capture_output=True, text=True, env=env, cwd=tmp, timeout=600
        )
        ok = proc.returncode != 0 and "ValueError" in proc.stderr and "legnd" in proc.stderr
        last = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "(no stderr)"
        check("(g)", f"{name}.py copy with 'legnd' fails at import", ok, f"exit {proc.returncode}: {last[:110]}")
    shutil.rmtree(tmp, ignore_errors=True)


def row_h():
    """Panel (with a log row injected) and digit sweep on synthetic inputs: no minor labels."""
    rng = np.random.default_rng(0)
    n, m = 24, 3
    grid = np.linspace(-1.0, 1.0, n)
    band = np.stack([np.sin(grid) - 0.3, np.sin(grid) + 0.3], -1)[:, None, :].repeat(m, 1)
    results = {
        "PI": band + 0.02 * rng.standard_normal(band.shape),
        "ERM": np.sin(grid)[:, None] + 0.01 * rng.standard_normal((n, m)),
    }
    columns = [(results, np.sin(grid), grid) for _ in range(3)]
    hists = {
        "pc1": (rng.standard_normal(200), rng.standard_normal(200) + 0.5),
        "pc2": (rng.standard_normal(200), rng.standard_normal(200)),
    }
    # a log row with one major tick in view is where LogFormatter labels the minors
    PANEL_CONFIGS[SYNTHETIC] = {
        0: {"scale": "log", "ylim": (0.2, 8.0)},
        1: {"scale": "log", "ylim": (0.2, 8.0)},
        3: {"ylim": (-2, 2)},
    }
    captured = []
    original = plotting.save
    plotting.save = lambda fig, *a, **k: captured.append(fig)  # the panel saves unconditionally; keep it in memory
    try:
        before = len(_errors)
        plotting.create_panel_plot(SYNTHETIC, columns, hists)
    finally:
        plotting.save = original
        del PANEL_CONFIGS[SYNTHETIC]
    ok = bool(captured) and len(_errors) == before
    if ok:
        labels, bare, n_log, fewest = tick_report(captured[0])
        ok = labels == 0 and bare == 0 and n_log >= 6 and fewest >= 2
        detail = (
            f"{len(captured[0].axes)} axes, minor labels {labels}, log axes without marks {bare}/{n_log}, "
            f"fewest majors in view {fewest}"
        )
    else:
        detail = "panel did not render"
    plt.close("all")
    check("(h)", "query panel (log rows injected)", ok, detail)

    exemplars = rng.random((5, 3, 8, 8))
    digits = {"PI": np.stack([np.full((5, m), 0.2), np.full((5, m), 0.8)], -1), "ERM": np.full((5, m), 0.5)}
    before = len(_errors)
    plotting.create_digit_sweep_plot(exemplars, digits, labels=[0, 1, 2, 3, 4], experiment=SYNTHETIC, savefig=False)
    labels, bare, n_log, _ = tick_report(plt.gcf())  # its x ticks are the thumbnails, not counted
    plt.close("all")
    check("(h)", "digit sweep (synthetic)", labels == 0 and len(_errors) == before, f"minor labels {labels}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifacts", default="artifacts", help="artifacts tree (default: the repo's artifacts/)")
    parser.add_argument("--save", action="store_true", help="re-render the sweep and perf pdfs into --artifacts")
    parser.add_argument("--experiments", nargs="*", default=None, help="subset of experiment directories")
    args = parser.parse_args()

    artifacts = os.path.abspath(args.artifacts)
    experiments = args.experiments or sorted(
        d for d in os.listdir(artifacts) if os.path.isdir(f"{artifacts}/{d}") and not d.startswith("_")
    )
    os.makedirs(TMPROOT, exist_ok=True)
    workdir = tempfile.mkdtemp(prefix="run_", dir=TMPROOT)
    if args.save:
        # save() writes to artifacts/ relative to the CWD (data_operations.py)
        os.symlink(artifacts, f"{workdir}/artifacts")
    os.chdir(workdir)
    print(f"artifacts {artifacts}; experiments {experiments}; save {args.save}; cwd {workdir}")

    t0 = time.perf_counter()
    for row in (
        lambda: row_a(artifacts, experiments, args.save),
        lambda: rows_bcd(artifacts, experiments),
        lambda: row_e(artifacts, experiments),
        lambda: row_f(artifacts, experiments),
        row_g,
        row_h,
    ):
        row()
    print(f"  swallowed plot errors: {len(_errors)}")
    for message in _errors[:5]:
        print(f"      {str(message).strip()[:200]}")
    if _errors:
        FAIL.append(f"{len(_errors)} plot errors logged")

    os.chdir(REPO)
    if not FAIL:
        shutil.rmtree(workdir, ignore_errors=True)
    print(f"\nRESULT: {'ALL PASS' if not FAIL else f'{len(FAIL)} FAIL'}  ({time.perf_counter() - t0:.0f}s)")
    for f in FAIL:
        print("  ", f)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
