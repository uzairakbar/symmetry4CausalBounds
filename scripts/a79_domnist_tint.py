"""A79: the do-MNIST tint sweep, its aggregate figure and the results table (two
MNIST loads, no nets, CPU; about a minute, most of it LaTeX).

(i)   the exemplar pin: `exemplars(420)` draws the same 10 indices and tints, and
      both image arrays (the SEM's subsample and `subsample=1`) hash to the same
      sha1, as on `develop` before `_exemplar_indices` was factored out. The values
      below were read once off that tree.
(ii)  `tinted`: one image per call, (N, 3, H, W) float32, `tint_of` returns the
      grid, the ink (`grey_of`) is the same in every row, the image is the
      exemplar's on the same SEM and seed, `subsample` only changes the resolution,
      and a digit outside 0..9 raises.
(iii) `run_tint_sweep` on a stub runner in a scratch cwd: the image is MNIST test's
      (the configured source), not the training exemplar; the digits come out
      sorted; ATE is constant along each sweep; intervals are (n, 1, 2) and points
      (n, 1); a NaN interval reaches the status split; the density pkl holds the B
      rows' tints before and after DA, bimodal at 0.1 and 0.9.
(iv)  `tint_stack` on a synthetic tree written out of order (7, 0, 3): the rows
      read 0, 3, 7 from the top, a missing digit is absent, the left image is the
      blue endpoint and the right the red one, each at 2/3 of its former size (narrower image columns), the
      title is $h({\bm{x}})$ and the x-label `tint`; ERM and DA+ERM are not drawn;
      the one legend is the methods' in the bottom-right cell; the histogram is
      the query panel's (`draw_da_density`); a failed tint gets its cross; the
      y ticks read 0 and 1; each interval method is two solid lines, no fill,
      PI+INV's lowest, the legend all line handles, a light-grey dashed line at
      0.5 behind them; the title and the x-label
      are the density label's size.
(v)   `domnist_table` on the same tree: one row per interval method in
      ALL_METHODS order, the point estimators absent, `fit_parts` charging the
      centre by `inv_recenter`, latency = fit + mean per-query solve (not divided
      by the query count), n_jobs and the BLAS cap in the comment lines, and the
      tex compiles under `pdflatex` when it is on PATH ([SKIP] otherwise);
      `aggregate.main` on a tree holding only `do_mnist/query/` writes both files.
(vi)  the table's gamma provenance: with no selection beside the run it claims no
      calibration; with a selection of another gamma, or of the same gamma on
      another split or with other nets, it says "gamma not from the selection
      beside it" and prints no split-C line; with the matching one it names the
      selection's `calibrated_on` method and its split-C coverage; a run.json
      without a `net` key reads as the flat nets.

  uv run python scripts/a79_domnist_tint.py
"""

import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import matplotlib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger  # noqa: E402

from src.experiments.utils.data_operations import load  # noqa: E402
from src.sem.do_mnist import DoMNISTSEM, grey_of, tint_of  # noqa: E402

SEM_KW = dict(seed=42, alpha=0.0, beta=0.4, eta=0.25, subsample=2)
EXEMPLAR_SEED = 420
# frozen from develop 52469a7 (`exemplars(420)` on the training SEM above)
PIN_IDX = [37510, 46046, 7432, 56697, 52945, 59351, 881, 24743, 23582, 38048]
PIN_TINTS = [
    0.1015180416847707,
    0.9151335640197078,
    0.023706396410535138,
    0.9639775547139551,
    0.07594752991750797,
    0.9531975189500955,
    0.0857911814572391,
    0.8841451110628251,
    0.06432985716155085,
    1.0,
]
PIN_SHA1 = {
    2: "2dd36cf57ea807da2353e5bd145af0088091881d",  # the SEM's subsample, (10, 3, 14, 14)
    1: "f4b80b88ccb0657e00a1cddfc626745f07b28204",  # full resolution, (10, 3, 28, 28)
}
FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail and not ok else ""))
    if not ok:
        FAIL.append(name)


def sha1(array) -> str:
    return hashlib.sha1(np.ascontiguousarray(array).tobytes()).hexdigest()  # noqa: S324 (provenance digest)


def leg_i(sem):
    print("(i) the exemplar pin")
    idx = sem._exemplar_indices(np.random.default_rng(EXEMPLAR_SEED))
    check("(i) the exemplar indices are the pinned ten", idx.tolist() == PIN_IDX, str(idx.tolist()))
    for subsample in (None, 1):
        images, digits = sem.exemplars(EXEMPLAR_SEED, subsample=subsample)
        key = SEM_KW["subsample"] if subsample is None else subsample
        check(f"(i) subsample {key}: digits 0..9", digits.tolist() == list(range(10)))
        check(f"(i) subsample {key}: the image sha1 is pinned", sha1(images) == PIN_SHA1[key], sha1(images))
        check(
            f"(i) subsample {key}: the tints are pinned",
            np.allclose(tint_of(images), PIN_TINTS, atol=1e-6),
            str(tint_of(images)),
        )


def leg_ii(sem):
    print("(ii) tinted")
    grid = np.linspace(0.0, 1.0, 8)
    for digit in (0, 7):
        images = sem.tinted(EXEMPLAR_SEED, digit, grid)
        check(
            f"(ii) digit {digit}: shape (8, 3, 14, 14) float32",
            images.shape == (8, 3, 14, 14) and images.dtype == np.float32,
        )
        check(f"(ii) digit {digit}: tint_of returns the grid", np.allclose(tint_of(images), grid, atol=1e-6))
        ink = grey_of(images)
        check(f"(ii) digit {digit}: the ink is the same in every row", np.allclose(ink, ink[:1], atol=1e-6))
        exemplar, _ = sem.exemplars(EXEMPLAR_SEED, subsample=1)
        full = sem.tinted(EXEMPLAR_SEED, digit, grid[[0, -1]], subsample=1)
        check(f"(ii) digit {digit}: subsample=1 gives (2, 3, 28, 28)", full.shape == (2, 3, 28, 28))
        check(
            f"(ii) digit {digit}: the image is the exemplar's",
            np.allclose(grey_of(full)[0], grey_of(exemplar)[digit], atol=1e-6),
        )
        check(
            f"(ii) digit {digit}: subsample changes the resolution only",
            np.allclose(full[:, :, ::2, ::2], images[[0, -1]], atol=1e-6),
        )
    for bad in (-1, 10):
        try:
            sem.tinted(EXEMPLAR_SEED, bad, grid)
            raised = False
        except ValueError:
            raised = True
        check(f"(ii) digit {bad} raises", raised)


class IntervalStub:
    """An interval method: [0.1, 0.9] everywhere, NaN on every fifth query."""

    def predict(self, X):
        bounds = np.tile([0.1, 0.9], (len(X), 1)).astype(float)
        bounds[::5] = np.nan
        self.query_status = np.where(np.isnan(bounds[:, 0]), 2, 0)
        return bounds


class PointStub:
    def predict(self, X):
        return np.full((len(X), 1), 0.5)


def leg_iii(sem, sem_test, scratch):
    print("(iii) run_tint_sweep on a stub runner")
    from src.experiments.configs import DOMNIST_CONFIG, TintSpec
    from src.experiments.do_mnist import flat, run_tint_sweep

    X, _ = sem.sample(2_000, seed=0)
    GX = X.copy()
    runner = SimpleNamespace(
        sem=sem,
        sem_test=sem_test,
        exemplar_seed=EXEMPLAR_SEED,
        methods={"ATE": None, "ERM": None, "PI": None},
        models_={"ERM": PointStub(), "PI": IntervalStub()},
        data_=SimpleNamespace(X=flat(X), GX=flat(GX), split_key="stub"),
        inv_recenter="inv",
        default_gamma=0.1,
        default_epsilon=0.04,
    )
    spec = TintSpec(digits=(7, 2), sweep_samples=5)
    cwd = os.getcwd()
    os.chdir(scratch)
    try:
        record = run_tint_sweep(runner, spec)
    finally:
        os.chdir(cwd)
    folder = os.path.join(scratch, "artifacts", "do_mnist", "query")

    def pkl(name):
        return load(os.path.join(folder, f"{name}.pkl"))

    check("(iii) the source is MNIST test by default", DOMNIST_CONFIG.tint_image_source == "test")
    check("(iii) the record says so", record["image_source"] == "test" and record["digits"] == [2, 7])
    for digit in (2, 7):
        images = pkl(f"tint_{digit}_images")
        expected = sem_test.tinted(EXEMPLAR_SEED, digit, [0.0, 1.0], subsample=1)
        check(f"(iii) digit {digit}: the endpoints are the test image", np.array_equal(images, expected))
        train = sem.tinted(EXEMPLAR_SEED, digit, [0.0, 1.0], subsample=1)
        check(f"(iii) digit {digit}: not the training exemplar", not np.array_equal(images, train))
        outcomes = pkl(f"tint_{digit}_outcomes")
        ate = np.asarray(outcomes["ATE"]).ravel()
        check(f"(iii) digit {digit}: ATE constant along the sweep", np.all(ate == ate[0]) and len(ate) == 5)
        check(f"(iii) digit {digit}: ATE is h_* of the digit", ate[0] == float(sem.ate_of([digit])[0, 0]))
        check(f"(iii) digit {digit}: the interval is (5, 1, 2)", np.shape(outcomes["PI"]) == (5, 1, 2))
        check(f"(iii) digit {digit}: the point estimate is (5, 1)", np.shape(outcomes["ERM"]) == (5, 1))
        check(f"(iii) digit {digit}: the grid", np.allclose(pkl(f"tint_{digit}_values"), np.linspace(0, 1, 5)))
        check(f"(iii) digit {digit}: the figure", os.path.exists(os.path.join(folder, f"tint_{digit}_sweep.pdf")))
    check("(iii) a NaN interval reaches the status split", record["status"]["PI"]["solver_failure"] == 2)
    density = pkl("tint_density")
    before = np.asarray(density["before"])
    bimodal = abs(np.median(before[before < 0.5]) - 0.1) < 0.03 and abs(np.median(before[before > 0.5]) - 0.9) < 0.03
    check("(iii) the density is the B rows' tints, bimodal at 0.1 and 0.9", len(before) == 2_000 and bimodal)
    check("(iii) the density carries the post-DA tints", len(density["after"]) == 2_000)


def _tree(root, digits=(7, 0, 3), n=6):
    """A synthetic `do_mnist/query/` tree: tint sweeps with ATE = d / 10, one NaN
    tint on PI+INV, the population pkls and a run.json."""
    folder = os.path.join(root, "do_mnist", "query")
    os.makedirs(folder, exist_ok=True)

    def dump(name, obj):
        with open(os.path.join(folder, f"{name}.pkl"), "wb") as fh:
            pickle.dump(obj, fh)

    grid = np.linspace(0.0, 1.0, n)
    for d in digits:
        ink = np.zeros((28, 28), dtype=np.float32)
        ink[6:22, 12:16] = 1.0
        blue = np.stack([0 * ink, 0 * ink, ink])
        red = np.stack([ink, 0 * ink, 0 * ink])
        pi_inv = np.tile([[0.2, 0.9]], (n, 1))[:, None, :].astype(float)
        pi_inv[2] = np.nan
        dump(f"tint_{d}_values", grid)
        dump(f"tint_{d}_images", np.stack([blue, red]))
        dump(
            f"tint_{d}_outcomes",
            {
                "ATE": np.full((n, 1), d / 10),
                "ERM": np.full((n, 1), 0.5),
                "PI": np.tile([[0.0, 1.0]], (n, 1))[:, None, :],
                "PI+INV": pi_inv,
            },
        )
    rng = np.random.default_rng(0)
    dump("tint_density", {"before": rng.normal(0.1, 0.05, 500), "after": rng.normal(0.9, 0.05, 500)})
    q = 50
    target = np.full((q, 1), 0.8)
    wide = np.tile([0.0, 1.0], (q, 1))
    narrow = np.tile([0.85, 0.95], (q, 1))
    dump("population_values", target)
    dump("population_outcomes", {"ERM": np.full((q, 1), 0.7), "PI+INV": narrow, "DA+PI": wide, "PI": wide})
    run = {
        "gamma": 0.05,
        "epsilon": 0.04,
        "inv_recenter": "inv",
        "erm_inv_tau": 4e-4,
        "target_coverage": 0.995,
        "n_queries": q,
        "pop_seed": 44,
        "rho": 1.5,
        "tr_S_over_k": 0.5,
        "iv_rho": 1.1,
        "toggle_n_jobs": -1,
        "cpu_count": 32,
        "blas_threads": 8,
        "train_seconds_X": 1.0,
        "train_seconds_GX": 2.0,
        "train_seconds_INV": 4.0,
        "augment_seconds_A": 0.5,
        "augment_seconds_B": 0.25,
        "fit_seconds_PI": 2.0,
        "fit_seconds_DA+PI": 3.0,
        "fit_seconds_PI+INV": 5.0,
        "floor_seconds_PI+INV": 0.5,
        "wall_clock_PI": 0.5,
        "wall_clock_DA+PI": 0.25,
        "wall_clock_PI+INV": 1.0,
        "worst_error_PI": 0.64,
        "worst_error_DA+PI": 0.64,
        "worst_error_PI+INV": 0.0225,
        "rmse_ERM": 0.1,
        "split_key": "k",
        "tint": {"digits": sorted(digits), "grid": grid.tolist()},
    }
    with open(os.path.join(folder, "run.json"), "w") as fh:
        json.dump(run, fh)
    return run


def leg_iv(scratch):
    print("(iv) tint_stack on a synthetic tree")
    import matplotlib.pyplot as plt

    from src.aggregate import tint_stack
    from src.experiments.utils.constants import TEX_MAPPER

    root = os.path.join(scratch, "tree")
    _tree(root)
    fig = tint_stack(root)
    bands = [ax for ax in fig.axes if any(line.get_label() == TEX_MAPPER["ATE"] for line in ax.get_lines())]
    bands.sort(key=lambda ax: -ax.get_position().y0)
    order = [
        round(float(next(ln for ln in ax.get_lines() if ln.get_label() == TEX_MAPPER["ATE"]).get_ydata()[0]) * 10)
        for ax in bands
    ]
    check("(iv) the rows read 0, 3, 7 from the top", order == [0, 3, 7], str(order))
    check("(iv) a missing digit is absent (three rows)", len(bands) == 3)
    images = [ax for ax in fig.axes if ax.images]

    def centre(ax):
        box = ax.get_position()
        return (box.x0 + box.x1) / 2

    left = [ax for ax in images if centre(ax) < bands[0].get_position().x0]
    right = [ax for ax in images if centre(ax) > bands[0].get_position().x1]
    blue = all(ax.images[0].get_array()[..., 2].sum() > ax.images[0].get_array()[..., 0].sum() for ax in left)
    red = all(ax.images[0].get_array()[..., 0].sum() > ax.images[0].get_array()[..., 2].sum() for ax in right)
    check("(iv) three blue images on the left, three red on the right", len(left) == len(right) == 3 and blue and red)
    background = left[0].images[0].get_array()[0, 0]
    check("(iv) the image background is transparent (white page)", float(background[3]) == 0.0)
    check("(iv) the title is h(x) on the top row", bands[0].get_title() == r"$h({\bm{x}})$")
    bottom = min(fig.axes, key=lambda ax: ax.get_position().y0 if ax.axison else 9)
    check("(iv) the x-label tint sits under the histogram", bottom.get_xlabel() == "tint" and bottom.patches)
    ticks = all(list(ax.get_yticks()) == [0.0, 1.0] for ax in bands)
    texts = [[t.get_text() for t in ax.get_yticklabels()] for ax in bands]
    check(
        "(iv) the bounds' y ticks are 0 and 1, labelled without decimals", ticks and all(t == ["0", "1"] for t in texts)
    )
    import seaborn as sns
    from matplotlib.collections import PolyCollection

    from src.experiments.utils.constants import COLOR_MAP

    palette = sns.color_palette("deep")
    fills = [c for ax in bands for c in ax.collections if isinstance(c, PolyCollection)]
    check("(iv) no filled bands", not fills, str(len(fills)))
    for name in ("PI", "PI+INV"):
        hue = np.asarray(palette[COLOR_MAP[name]])
        rows = [
            [ln for ln in ax.get_lines() if np.allclose(matplotlib.colors.to_rgb(ln.get_color()), hue)] for ax in bands
        ]
        solid = all(len(r) == 2 and all(ln.get_linestyle() == "-" for ln in r) for r in rows)
        same = all(len({ln.get_linewidth() for ln in r}) == 1 for r in rows)
        check(f"(iv) {name}: two solid bound lines of one width per row", solid and same)
    under = all(
        max(
            ln.get_zorder()
            for ln in ax.get_lines()
            if np.allclose(matplotlib.colors.to_rgb(ln.get_color()), palette[7])
        )
        < min(
            ln.get_zorder()
            for ln in ax.get_lines()
            if np.allclose(matplotlib.colors.to_rgb(ln.get_color()), palette[0])
        )
        for ax in bands
    )
    check("(iv) PI+INV's lines sit under the other methods'", under)
    mid = [ln for ax in bands for ln in ax.get_lines() if list(ln.get_ydata()) == [0.5, 0.5]]
    behind = all(
        ln.get_linestyle() == "--" and ln.get_color() == "0.8" and ln.get_zorder() < 2 and ln.get_linewidth() < 1
        for ln in mid
    )
    check("(iv) a thin light-grey dashed line at 0.5 behind the bounds", len(mid) == len(bands) and behind)
    sizes = {bands[0].title.get_fontsize(), bottom.xaxis.label.get_fontsize(), bottom.yaxis.label.get_fontsize()}
    check("(iv) the title, the x-label and the density label share one size", len(sizes) == 1, str(sizes))
    labels = {line.get_label() for ax in bands for line in ax.get_lines()}
    check("(iv) ERM and DA+ERM are not drawn", not labels & {TEX_MAPPER["ERM"], TEX_MAPPER["DA+ERM"]})
    legends = [ax.get_legend() for ax in fig.axes if ax.get_legend() is not None] + list(fig.legends)
    check("(iv) one legend", len(legends) == 1, str(len(legends)))
    entries = [t.get_text() for t in legends[0].get_texts()] if legends else []
    check("(iv) it is the methods' legend", TEX_MAPPER["PI"] in entries and "pre-DA" not in entries, str(entries))
    from matplotlib.lines import Line2D

    kinds = {type(h) for h in (legends[0].legend_handles if legends else [])}
    check("(iv) every legend handle is a line", kinds == {Line2D}, str(kinds))
    cell = legends[0].axes if legends else None
    corner = cell is not None and cell.get_position().x0 >= bands[0].get_position().x1 and cell.get_position().y0 < 0.2
    check("(iv) the legend sits in the bottom-right cell", corner)
    from src.aggregate import TINT_WIDTHS, TINT_WSPACE

    share = TINT_WIDTHS[0] / sum(TINT_WIDTHS) * 4 / (4 + 3 * TINT_WSPACE)
    check(
        "(iv) the image columns hold 2/3 of their former 0.14-ratio share",
        np.isclose(share, 2 / 3 * 0.14 / 1.28 * 3 / 3.08),
    )
    full = fig.subplotpars.right - fig.subplotpars.left
    fig.canvas.draw()
    to_figure = fig.transFigure.inverted()
    label_left = min(
        to_figure.transform((t.get_window_extent().x0, 0))[0]
        for ax in bands
        for t in ax.get_yticklabels()
        if t.get_text()
    )
    clear = []
    for ax in left:
        box, alpha = ax.get_position(), ax.images[0].get_array()[..., 3]
        cols = np.flatnonzero(alpha.max(axis=0) > 0.1)
        clear.append(box.x0 + (cols[-1] + 1) / alpha.shape[1] * box.width < label_left)
    check("(iv) the blue ink stops short of the y tick labels", all(clear))
    drawn = left[0].get_position().width / full
    check(
        "(iv) the drawn image takes that share of the row", abs(drawn / share - 1) < 0.05, f"{drawn:.4f} vs {share:.4f}"
    )
    from src.experiments.utils.plotting import DENSITY_HIST

    alphas = {round(p.get_alpha(), 3) for p in bottom.patches}
    check("(iv) the histogram uses the query panel's style", alphas == {DENSITY_HIST["alpha"]}, str(alphas))
    crosses = [c for ax in bands for c in ax.collections if getattr(c, "get_offsets", None) and len(c.get_offsets())]
    check("(iv) each row marks its failed tint", len(crosses) >= 3)
    plt.close(fig)


def leg_v(scratch):
    print("(v) domnist_table on the synthetic tree")
    from src.aggregate import domnist_table, fit_parts
    from src.aggregate import main as aggregate_main

    root = os.path.join(scratch, "tree")
    tex = domnist_table(root)
    lines = tex.split("\\midrule\n")[1].split("\\bottomrule")[0].strip("\n").splitlines()
    body = [row for row in lines if not row.startswith(" & ")]
    check("(v) each row has its band line under it", len(lines) == 2 * len(body))
    check("(v) one row per interval method", len(body) == 3, str(len(body)))
    from src.experiments.utils.constants import TEX_MAPPER

    names = [row.split(" & ")[0] for row in body]
    check("(v) ALL_METHODS order: PI+INV, PI, DA+PI", names == [TEX_MAPPER[m] for m in ("PI+INV", "PI", "DA+PI")])
    check("(v) the point estimators are absent", TEX_MAPPER["ERM"] not in names)
    pi = body[1].split(" & ")
    check("(v) PI: fit = X net + fit (3.00 s)", pi[5] == "$3.00$", pi[5])
    check("(v) PI: solve 500.00 ms per query", pi[6] == "$500.00$", pi[6])
    check("(v) PI: latency = fit + solve, not fit / n_queries", pi[7].startswith("$3.50$"), pi[7])
    inv = body[0].split(" & ")
    check("(v) PI+INV (inv): INV net + both DA passes + fit + floor (10.25 s)", inv[5] == "$10.25$", inv[5])
    check("(v) PI+INV covers nothing (0.000)", inv[1].startswith("$0.000$"))
    check("(v) Omega-hat on DA+PI only", body[2].split(" & ")[3] == "$0.750$" and pi[3] == "--")
    check("(v) the centre follows inv_recenter", fit_parts("PI+INV", "off")[0] == "train_seconds_X")
    check("(v) n_jobs and the BLAS cap in the comments", "n_jobs -1 on 32 cores" in tex and "BLAS at 8" in tex)
    if shutil.which("pdflatex"):
        doc = os.path.join(scratch, "doc")
        os.makedirs(doc, exist_ok=True)
        with open(os.path.join(doc, "table.tex"), "w") as fh:
            fh.write(tex)
        with open(os.path.join(doc, "main.tex"), "w") as fh:
            fh.write(
                "\\documentclass{article}\\usepackage{booktabs,amsmath}\\usepackage[landscape]{geometry}"
                "\\begin{document}\\input{table.tex}\\end{document}\n"
            )
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"], cwd=doc, capture_output=True
        )
        check("(v) the tex compiles under pdflatex", result.returncode == 0)
    else:
        print("  [SKIP] (v) pdflatex not on PATH")

    only = os.path.join(scratch, "only")
    _tree(only, digits=(5,))
    aggregate_main(["--artifacts", only])
    written = [os.path.exists(os.path.join(only, "aggregate", f)) for f in ("do_mnist_tint.pdf", "do_mnist_table.tex")]
    check("(v) aggregate.main on do_mnist/query alone writes the figure and the table", all(written))


def leg_vi(scratch):
    print("(vi) the table's gamma provenance")
    from src.aggregate import domnist_table

    root = os.path.join(scratch, "tree")
    select = os.path.join(root, "do_mnist", "select")
    os.makedirs(select, exist_ok=True)
    path = os.path.join(select, "gamma_selection.json")
    if os.path.exists(path):
        os.remove(path)
    tex = domnist_table(root)
    check("(vi) no selection: no calibration claimed", "calibrated on" not in tex and "not checked" in tex)

    def write(**overrides):
        selection = {
            "shared_gamma": 0.05,
            "calibrated_on": "DA+PI",
            "split_key": "k",
            "target_coverage": 0.995,
            "n_select": 5000,
            "DA+PI": {"coverage": 0.9952},
            "PI": {"coverage": 0.9990, "at_shared_gamma": {"coverage": 0.9991}},
        }
        with open(path, "w") as fh:
            json.dump({**selection, **overrides}, fh)
        return domnist_table(root)

    others = (
        ("another gamma", {"shared_gamma": 0.12188}),
        ("another split", {"split_key": "x"}),
        ("other nets", {"net": "domnist-pool"}),
    )
    for why, overrides in others:
        tex = write(**overrides)
        check(f"(vi) {why}: flagged", "gamma not from the selection beside it" in tex)
        check(f"(vi) {why}: no calibration claim, no split-C line", "calibrated on" not in tex and "split-C" not in tex)
    tex = write()
    check("(vi) the matching selection: the calibration names calibrated_on", "calibrated on DA+PI" in tex)
    check("(vi) the matching selection: split-C coverage of both", "DA+PI 0.9952, PI 0.9991" in tex)
    check("(vi) a run without a net key reads as the flat nets", "nets domnist-fast" in tex)
    os.remove(path)


def main():
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    sem = DoMNISTSEM(train=True, **SEM_KW)
    leg_i(sem)
    leg_ii(sem)
    sem_test = DoMNISTSEM(train=False, **SEM_KW)
    with tempfile.TemporaryDirectory() as scratch:
        leg_iii(sem, sem_test, scratch)
        leg_iv(scratch)
        leg_v(scratch)
        leg_vi(scratch)
    print(f"\nA79 {'PASS' if not FAIL else 'FAIL'}")
    for name in FAIL:
        print(f"  FAILED: {name}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
