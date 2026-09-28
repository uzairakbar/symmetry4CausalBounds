"""A81: the do-MNIST population across seeds and its table (no MNIST, no nets, CPU;
seconds, most of it LaTeX).

(i)   `domnist_seed_table` on synthetic records: one row per interval method in
      ALL_METHODS order, ATE and the point estimators absent from the rows and
      their RMSE (mean +- SE) in a comment line; each cell the mean over seeds
      +- std(ddof 1) / sqrt(n finite); coverage ranked highest first, width and
      worst error lowest first, `\\bm` the best and `\\mathit` the second best;
      a tie at the printed precision shares its mark; an all-NaN method prints
      `--` in every column and is not ranked; a partly NaN method is averaged over
      its finite seeds and the count is named; the seeds, `target_coverage`, the
      fixed seeds and `\\usepackage{bm}` in the comments; one seed prints no SE;
      the tex compiles under `pdflatex` when it is on PATH ([SKIP] otherwise).
(ii)  `evaluate_population(save_outcomes=False)` writes nothing (`save` patched on
      a stub runner and a stub population), `True` writes the two population
      pkls, and the returned metrics are the same either way.
(iii) `_run_seeds` on a stub orchestrator and stub runners in a scratch cwd: seeds
      `seed + j` for j < n_experiments, one fresh runner per further seed at
      `n_experiments: 1`, every one scored with `save_outcomes=False`, the first
      runner's replicate released before the second is built and every runner's
      released after it is scored, `seeds.json` holding only the population
      metrics per seed and the fixed provenance, `seeds_table.tex` equal to
      `domnist_seed_table` on it; one experiment builds no further runner.
(iv)  `aggregate.main` on a tree holding only `do_mnist/query/seeds.json` writes
      `do_mnist_seeds_table.tex`, equal to the run's table; without the json
      `domnist_seeds` returns None and nothing is written.
(v)   static: `_run_seeds` builds its runners at `seed + j`, the gamma sweep's
      replicate seed (`_draw_base`), and frees the GPU cache.

  uv run python scripts/a81_domnist_seeds.py
"""

import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger  # noqa: E402

FAIL = []

FIXED = dict(gamma=0.0621, n_queries=2000, split_seed=420, pop_seed=44, exemplar_seed=420)
# PI 0.90 / 0.92 / 0.94 (mean 0.92, SE 0.01155); DA+PI 0.95 / 0.95 / 0.95; DA+PI+IV one NaN seed;
# PI+INV never solved; widths: DA+PI+IV and PI&DA+PI tie at the printed precision
RECORDS = [
    {
        "seed": 42,
        "coverage_PI": 0.90,
        "width_PI": 0.80,
        "worst_error_PI": 0.40,
        "nan_PI": 0.0,
        "coverage_DA+PI": 0.95,
        "width_DA+PI": 0.70,
        "worst_error_DA+PI": 0.30,
        "nan_DA+PI": 0.0,
        "coverage_DA+PI+IV": 0.50,
        "width_DA+PI+IV": 0.5001,
        "worst_error_DA+PI+IV": 0.20,
        "nan_DA+PI+IV": 0.0,
        "coverage_PI&DA+PI": 0.60,
        "width_PI&DA+PI": 0.5002,
        "worst_error_PI&DA+PI": 0.25,
        "nan_PI&DA+PI": 0.0,
        "coverage_PI+INV": float("nan"),
        "width_PI+INV": float("nan"),
        "worst_error_PI+INV": float("nan"),
        "nan_PI+INV": 1.0,
        "rmse_ERM": 0.22,
        "rmse_DA+ERM": 0.26,
    },
    {
        "seed": 43,
        "coverage_PI": 0.92,
        "width_PI": 0.82,
        "worst_error_PI": 0.42,
        "nan_PI": 0.0,
        "coverage_DA+PI": 0.95,
        "width_DA+PI": 0.72,
        "worst_error_DA+PI": 0.32,
        "nan_DA+PI": 0.0,
        "coverage_DA+PI+IV": float("nan"),
        "width_DA+PI+IV": float("nan"),
        "worst_error_DA+PI+IV": float("nan"),
        "nan_DA+PI+IV": 1.0,
        "coverage_PI&DA+PI": 0.62,
        "width_PI&DA+PI": 0.5002,
        "worst_error_PI&DA+PI": 0.27,
        "nan_PI&DA+PI": 0.0,
        "coverage_PI+INV": float("nan"),
        "width_PI+INV": float("nan"),
        "worst_error_PI+INV": float("nan"),
        "nan_PI+INV": 1.0,
        "rmse_ERM": 0.24,
        "rmse_DA+ERM": 0.28,
    },
    {
        "seed": 44,
        "coverage_PI": 0.94,
        "width_PI": 0.84,
        "worst_error_PI": 0.44,
        "nan_PI": 0.0,
        "coverage_DA+PI": 0.95,
        "width_DA+PI": 0.74,
        "worst_error_DA+PI": 0.34,
        "nan_DA+PI": 0.0,
        "coverage_DA+PI+IV": 0.54,
        "width_DA+PI+IV": 0.5003,
        "worst_error_DA+PI+IV": 0.22,
        "nan_DA+PI+IV": 0.0,
        "coverage_PI&DA+PI": 0.64,
        "width_PI&DA+PI": 0.5002,
        "worst_error_PI&DA+PI": 0.29,
        "nan_PI&DA+PI": 0.0,
        "coverage_PI+INV": float("nan"),
        "width_PI+INV": float("nan"),
        "worst_error_PI+INV": float("nan"),
        "nan_PI+INV": 1.0,
        "rmse_ERM": 0.26,
        "rmse_DA+ERM": 0.30,
    },
]


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail and not ok else ""))
    if not ok:
        FAIL.append(name)


def body_rows(tex: str) -> dict[str, list[str]]:
    """The table's rows by label, cells stripped of the row end."""
    rows = tex.split("\\midrule\n")[1].split("\\bottomrule")[0].strip("\n").splitlines()
    cells = [row.removesuffix(r" \\").split(" & ") for row in rows]
    return {c[0]: c[1:] for c in cells}


def leg_i(scratch):
    print("(i) domnist_seed_table on synthetic records")
    from src.experiments.do_mnist import domnist_seed_table
    from src.experiments.utils.constants import label as method_label

    tex = domnist_seed_table(RECORDS, has_z=False, target_coverage=0.995, fixed=FIXED)
    rows = body_rows(tex)
    order = ["PI+INV", "PI", "DA+PI", "DA+PI+IV", "PI&DA+PI"]
    check(
        "(i) one row per interval method, ALL_METHODS order",
        list(rows) == [method_label(m, False) for m in order],
        str(list(rows)),
    )
    labels = {m: method_label(m, False) for m in order}
    check(
        "(i) ATE and the point estimators are not rows",
        all(method_label(m, False) not in rows for m in ("ERM", "DA+ERM", "ATE")),
    )
    pi = rows[labels["PI"]]
    se = np.std([0.90, 0.92, 0.94], ddof=1) / np.sqrt(3)
    check("(i) PI coverage: mean and SE (ddof 1)", f"0.920 \\pm {se:.3f}" in pi[0] and f"{se:.3f}" == "0.012", pi[0])
    check(
        "(i) coverage highest is best: DA+PI bold",
        rows[labels["DA+PI"]][0] == r"$\bm{0.950 \pm 0.000}$",
        rows[labels["DA+PI"]][0],
    )
    check("(i) coverage second best: PI italic", pi[0] == rf"$\mathit{{0.920 \pm {se:.3f}}}$", pi[0])
    check(
        "(i) width lowest is best, a tie at .3f shares the bold",
        all(rows[labels[m]][1].startswith(r"$\bm{0.500") for m in ("DA+PI+IV", "PI&DA+PI")),
        str([rows[labels[m]][1] for m in ("DA+PI+IV", "PI&DA+PI")]),
    )
    check(
        "(i) width after the tie: DA+PI italic",
        rows[labels["DA+PI"]][1].startswith(r"$\mathit{0.720"),
        rows[labels["DA+PI"]][1],
    )
    check("(i) width: PI unmarked", pi[1] == r"$0.820 \pm 0.012$", pi[1])
    check(
        "(i) worst error lowest is best (.4f)",
        rows[labels["DA+PI+IV"]][2] == r"$\bm{0.2100 \pm 0.0100}$",
        rows[labels["DA+PI+IV"]][2],
    )
    check(
        "(i) worst error second best",
        rows[labels["PI&DA+PI"]][2].startswith(r"$\mathit{0.2700"),
        rows[labels["PI&DA+PI"]][2],
    )
    check("(i) an all-NaN method prints -- and is unranked", rows[labels["PI+INV"]] == ["--", "--", "--"])
    check(
        "(i) a partly NaN method averages its finite seeds",
        rows[labels["DA+PI+IV"]][0] == r"$0.520 \pm 0.020$",
        rows[labels["DA+PI+IV"]][0],
    )
    check("(i) the finite-seed counts are named", "PI+INV 0/3" in tex and "DA+PI+IV 2/3" in tex)
    rmse = next((line for line in tex.splitlines() if "point estimators" in line), "")
    check(
        "(i) the point estimators' RMSE in a comment",
        rmse.startswith("%") and "ERM 0.2400 +- 0.0115" in rmse and "DA+ERM 0.2800 +- 0.0115" in rmse,
        rmse,
    )
    check("(i) the seeds in the comments", "across 3 seeds [42, 43, 44]" in tex)
    check("(i) target_coverage and gamma in the comments", "target_coverage 0.995" in tex and "gamma 0.0621" in tex)
    check("(i) the fixed seeds in the comments", "split_seed 420 / pop_seed 44 / exemplar_seed 420 fixed" in tex)
    check("(i) needs bm", r"% needs \usepackage{bm}" in tex)
    check(
        "(i) every comment line is a comment",
        all(line.startswith(("%", "{", "\\", "}", " &", "$")) for line in tex.splitlines()),
    )
    one = body_rows(domnist_seed_table(RECORDS[:1], fixed=FIXED))
    check("(i) one seed: no SE", one[labels["PI"]][0] == r"$\mathit{0.900}$", one[labels["PI"]][0])
    if shutil.which("pdflatex"):
        doc = os.path.join(scratch, "doc")
        os.makedirs(doc, exist_ok=True)
        with open(os.path.join(doc, "table.tex"), "w") as fh:
            fh.write(tex)
        with open(os.path.join(doc, "main.tex"), "w") as fh:
            fh.write(
                "\\documentclass{article}\\usepackage{booktabs,amsmath,bm}"
                "\\begin{document}\\input{table.tex}\\end{document}\n"
            )
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"], cwd=doc, capture_output=True
        )
        check("(i) the tex compiles under pdflatex", result.returncode == 0)
    else:
        print("  [SKIP] (i) pdflatex not on PATH")


class IntervalStub:
    def predict(self, X):
        bounds = np.tile([0.1, 0.9], (len(X), 1)).astype(float)
        self.query_status = np.zeros(len(X), dtype=int)
        return bounds


class PointStub:
    def predict(self, X):
        return np.full((len(X), 1), 0.5)


def leg_ii():
    print("(ii) evaluate_population(save_outcomes=False) writes nothing")
    import src.experiments.do_mnist as dm

    n = 20
    runner = dm.DoMNISTQuerySweep.__new__(dm.DoMNISTQuerySweep)
    runner.sem_test, runner.n_queries, runner.pop_seed = None, n, 44
    runner.models_ = {"ERM": PointStub(), "PI": IntervalStub()}
    runner.fit_seconds_ = {"ERM": 1.0, "PI": 2.0}
    runner.nets, runner.inv_recenter = None, "off"
    target = np.linspace(0.0, 1.0, n)
    calls = []
    population, save = dm.population, dm.save
    dm.population = lambda sem, k, seed: (np.zeros((k, 4), dtype=np.float32), target, None, None)
    dm.save = lambda obj, fname, *args, **kwargs: calls.append(fname)
    try:
        quiet = runner.evaluate_population(save_outcomes=False)
        check("(ii) save_outcomes=False: no save", calls == [], str(calls))
        loud = runner.evaluate_population()
        check(
            "(ii) the default writes the two population pkls",
            calls == ["population_values", "population_outcomes"],
            str(calls),
        )
    finally:
        dm.population, dm.save = population, save
    from src.experiments.do_mnist import SEED_METRICS

    same = {k: v for k, v in quiet.items() if k.startswith(SEED_METRICS)} == {
        k: v for k, v in loud.items() if k.startswith(SEED_METRICS)
    }
    check("(ii) the population metrics are the same either way", same)
    check(
        "(ii) coverage, width, worst error and RMSE are scored",
        {"coverage_PI", "width_PI", "worst_error_PI", "rmse_ERM"} <= set(quiet),
    )


def stub_orchestrator(n_experiments: int, seed: int = 42):
    """A DoMNISTOrchestrator with stub runners: each scores its seed's metrics and
    records its kwargs, the state `_release` must free and what it saw."""
    from src.experiments.do_mnist import RUNNER_STATE, DoMNISTOrchestrator

    built = []

    class StubRunner:
        def __init__(self, methods=None, **kwargs):
            self.kwargs = kwargs
            self.seed = kwargs["seed"]
            # the previous runner must already be released when this one is built
            self.previous_released = all(all(getattr(r, name, None) is None for name in RUNNER_STATE) for r in built)
            for name in RUNNER_STATE:
                setattr(self, name, np.ones(3))
            self.flags = []
            built.append(self)

        def run(self, desc):
            self.ran = desc

        def evaluate_population(self, save_outcomes=True):
            self.flags.append(save_outcomes)
            return scores(self.seed)

    class Orchestrator(DoMNISTOrchestrator):
        methods = {"ATE": None, "PI": None}

        def __init__(self):
            self.name = "do_mnist"
            self.kwargs = dict(seed=seed, n_experiments=n_experiments, n_samples=10, methods=["ATE", "PI"])
            self.target_coverage, self.gamma = 0.995, FIXED["gamma"]
            self.n_queries, self.split_seed = FIXED["n_queries"], FIXED["split_seed"]
            self.pop_seed, self.exemplar_seed = FIXED["pop_seed"], FIXED["exemplar_seed"]

        def get_query_runner_cls(self):
            return StubRunner

    return Orchestrator(), StubRunner, built


def scores(seed: int) -> dict:
    return {
        "coverage_PI": 0.9 + 0.01 * (seed - 42),
        "width_PI": 0.8,
        "worst_error_PI": 0.4,
        "nan_PI": 0.0,
        "rmse_ERM": 0.2,
        "fit_seconds_PI": 3.0,
        "status_PI": {"feasible_and_covers": 1},
        "load_PI": {"fit_before": 0.0},
        "wall_clock_PI": 0.5,
    }


def leg_iii(scratch):
    print("(iii) _run_seeds on stub runners")
    from src.experiments.do_mnist import RUNNER_STATE, domnist_seed_table

    cwd = os.getcwd()
    os.chdir(scratch)
    try:
        orchestrator, StubRunner, built = stub_orchestrator(3)
        first = StubRunner.__new__(StubRunner)
        for name in RUNNER_STATE:
            setattr(first, name, np.ones(3))
        built.append(first)
        orchestrator._run_seeds(first, scores(42))
        folder = os.path.join(scratch, "artifacts", "do_mnist", "query")
        with open(os.path.join(folder, "seeds.json")) as fh:
            seeds = json.load(fh)
        with open(os.path.join(folder, "seeds_table.tex")) as fh:
            tex = fh.read()
        runners = built[1:]
        check("(iii) one fresh runner per further seed", len(runners) == 2, str(len(runners)))
        check("(iii) runners at seed + j", [r.seed for r in runners] == [43, 44])
        check("(iii) each at n_experiments 1", all(r.kwargs["n_experiments"] == 1 for r in runners))
        check("(iii) each runs the query sweep", all(getattr(r, "ran", None) == "Query Sweep" for r in runners))
        check("(iii) each scored with save_outcomes=False", all(r.flags == [False] for r in runners))
        check("(iii) the previous replicate released before each build", all(r.previous_released for r in runners))
        check(
            "(iii) every runner released after scoring, the first included",
            all(getattr(r, name) is None for r in built for name in RUNNER_STATE),
        )
        check("(iii) seeds.json: the seeds", [r["seed"] for r in seeds["records"]] == [42, 43, 44])
        check(
            "(iii) seeds.json: the population metrics only",
            all(
                set(r) == {"seed", "coverage_PI", "width_PI", "worst_error_PI", "nan_PI", "rmse_ERM"}
                for r in seeds["records"]
            ),
            str(seeds["records"][0]),
        )
        check(
            "(iii) seeds.json: each seed's own numbers",
            [r["coverage_PI"] for r in seeds["records"]] == [scores(s)["coverage_PI"] for s in (42, 43, 44)],
        )
        check("(iii) seeds.json: the fixed provenance", seeds["fixed"] == FIXED and seeds["target_coverage"] == 0.995)
        expected = domnist_seed_table(seeds["records"], has_z=False, target_coverage=0.995, fixed=FIXED)
        check("(iii) seeds_table.tex is domnist_seed_table on seeds.json", tex == expected)

        orchestrator, StubRunner, built = stub_orchestrator(1)
        only = StubRunner.__new__(StubRunner)
        orchestrator._run_seeds(only, scores(42))
        with open(os.path.join(folder, "seeds.json")) as fh:
            seeds = json.load(fh)
        check("(iii) one experiment: no further runner, one record", built == [] and len(seeds["records"]) == 1)
    finally:
        os.chdir(cwd)


def leg_iv(scratch):
    print("(iv) aggregate re-render")
    from src.aggregate import domnist_seeds
    from src.aggregate import main as aggregate_main
    from src.experiments.do_mnist import domnist_seed_table

    root = os.path.join(scratch, "only")
    folder = os.path.join(root, "do_mnist", "query")
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(root, "do_mnist", "labels.json"), "w") as fh:
        json.dump({"has_z": False, "methods": ["PI"]}, fh)
    check("(iv) without seeds.json: None", domnist_seeds(root) is None)
    aggregate_main(["--artifacts", root])
    check(
        "(iv) without seeds.json: nothing written",
        not os.path.exists(os.path.join(root, "aggregate", "do_mnist_seeds_table.tex")),
    )
    with open(os.path.join(folder, "seeds.json"), "w") as fh:
        json.dump({"records": RECORDS, "target_coverage": 0.995, "fixed": FIXED}, fh)
    aggregate_main(["--artifacts", root])
    path = os.path.join(root, "aggregate", "do_mnist_seeds_table.tex")
    written = ""
    if os.path.exists(path):
        with open(path) as fh:
            written = fh.read()
    expected = domnist_seed_table(RECORDS, has_z=False, target_coverage=0.995, fixed=FIXED)
    check("(iv) aggregate.main writes do_mnist_seeds_table.tex, equal to the run's table", written == expected)
    check(
        "(iv) no results table without its pkls",
        not os.path.exists(os.path.join(root, "aggregate", "do_mnist_table.tex")),
    )


def leg_v():
    print("(v) static")
    from src.experiments.do_mnist import DoMNISTMixin, DoMNISTOrchestrator

    source = inspect.getsource(DoMNISTOrchestrator._run_seeds)
    check("(v) _run_seeds builds its runners at seed + j", re.search(r'"seed": seed \+ j\b', source) is not None)
    check(
        "(v) the gamma sweep's replicate seed is seed + j too",
        "seed=self.seed + experiment_index" in inspect.getsource(DoMNISTMixin._draw_base),
    )
    release = inspect.getsource(DoMNISTOrchestrator._release)
    check(
        "(v) _release collects and frees the GPU cache",
        "gc.collect()" in release and "torch.cuda.empty_cache()" in release,
    )


def main():
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    with tempfile.TemporaryDirectory() as scratch:
        leg_i(scratch)
        leg_ii()
        leg_iii(scratch)
        leg_iv(scratch)
    leg_v()
    print(f"\nA81 {'PASS' if not FAIL else 'FAIL'}")
    for name in FAIL:
        print(f"  FAILED: {name}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
