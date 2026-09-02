# Acceptance gates

Run with `uv run python` (the repo `.venv`), from the repo root.

Gates A2 and A3 compare against SOURCE
(`../doMNIST/symmetry4CausalBoundsDoMNIST`), so they run in two stages: `--source`
under SOURCE's own `.venv`, then `--check` under this repo's `.venv`. A4 keeps the
same two-stage shape but both stages run here: `--dump` freezes, `--check` compares.

| gate | script | what it pins |
|---|---|---|
| A1 | `a1_a2_sem.py --check` | SEM structural laws vs closed form |
| A2 | `a1_a2_sem.py` | SEM draws bit-identical to SOURCE |
| A3 | `a3_da_parity.py` | DA `(GX, G)` bit-identical to SOURCE |
| A4 | `a4_partial_r2_net_regression.py` | `partial_r2_net` bounds + intermediates vs a frozen dump |
| A5 | `a5_njobs_exactness.py` | `n_jobs` changes nothing but wall clock |
| A6-A8, A11-A14 | `a6_a14_pipeline.py` | perf fairness, status, JAX≡FD, config, recipe, cost, memory |
| A9 | `a6_a14_pipeline.py --full` | the estimand is the CAUSAL one (needs 1.2M draws) |
| A10 | `a10_partial_r2_regression.py` | `PartialR2` unchanged by the `BoundedSA` hoist |
| A21 | `a6_a14_pipeline.py` | intersection wiring: branch nets, fit ball, `pad`, `n_jobs` |
| A24 | `a24_budget_selection.py` | bisection contract, floor cache, budget-report schema |
| A25 | `a25_floor_guard.py` | closed-form floor vs cvxpy; guard is a no-op when feasible, rescues when not |
| A27 | `a27_domnist_r2.py` | partial_r2_net backend: nesting, h_* membership at gamma*, JAX≡FD, l=2 path |

`smoke_do_mnist.py` is an end-to-end query-sweep + perf run at reduced scale;
`--full` runs it at the config's own numbers.

## Budget selection

`select_domnist_budgets.py` picks `gamma`, `epsilon` and `epsilon_iv` by POPULATION
coverage and writes `artifacts/domnist-budget_report.json`. DEMOTED to a sanity
check: the pipeline consumes oracle `gamma* = bias_sq/sigma_sq` directly (the
`partial_r2_net` ball is the Lemma-2 ball in function space) and A27 gates
membership, so nothing here needs a coverage-selected budget.

Three sequential legs at one target coverage `X` (`--target-coverage`, or
`do_mnist.target_coverage`; 0.95 or 0.99):

1. lowest `gamma` with coverage ≥ `X` on `PI` — **fixed, never re-selected**
2. lowest `epsilon_iv` with coverage ≥ `X` on `DA+PI+IV`, at that `gamma`
3. lowest `epsilon` with coverage ≥ `X` on `PI+INV`, at that `gamma`

One `gamma` for every method, because that is how the pipeline consumes it.
`DA+PI+IV` and `PI+INV` are subsets of `DA+PI`, so if `DA+PI` misses `X` at the
selected `gamma`, no budget can reach it — the leg is marked
`target_reachable: false` and reports the lowest budget attaining the ceiling.

```bash
python scripts/select_domnist_budgets.py            # 1.2M draws, 10k eval, hours
python scripts/select_domnist_budgets.py --smoke    # 60k / 6k / 2k, minutes
python scripts/a24_budget_selection.py artifacts/domnist-budget_report_smoke.json
```

Budgets are conditional on every setting the report records (`pad`, `calibrate`,
`n_pi`, `net`, ...). Change one and they are stale. Read `warnings[]` first: it flags
an inert budget and a `DA+PI` ceiling below target.

## Two-stage gates

```bash
SRC=../doMNIST/symmetry4CausalBoundsDoMNIST
$SRC/.venv/bin/python scripts/a1_a2_sem.py    --source /tmp/a2.npz
$SRC/.venv/bin/python scripts/a3_da_parity.py --source /tmp/a3.npz

uv run python scripts/a1_a2_sem.py    --check /tmp/a2.npz
uv run python scripts/a3_da_parity.py --check /tmp/a3.npz

uv run python scripts/a4_partial_r2_net_regression.py --dump  ~/scratch/a4
uv run python scripts/a4_partial_r2_net_regression.py --check ~/scratch/a4
```

## A10

Digest the current tree, then the tree without the change, and diff:

```bash
python scripts/a10_partial_r2_regression.py > /tmp/after.json
git stash && python scripts/a10_partial_r2_regression.py > /tmp/before.json && git stash pop
diff /tmp/before.json /tmp/after.json
```
