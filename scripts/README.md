# Acceptance gates

Run with `~/scratch/envs/s4cb/bin/python`, from the repo root.

Gates A2, A3 and A4 compare against SOURCE
(`../doMNIST/symmetry4CausalBoundsDoMNIST`), so they run in two stages: `--source`
under SOURCE's own `.venv`, then `--check` under `s4cb`.

| gate | script | what it pins |
|---|---|---|
| A1 | `a1_a2_sem.py --check` | SEM structural laws vs closed form |
| A2 | `a1_a2_sem.py` | SEM draws bit-identical to SOURCE |
| A3 | `a3_da_parity.py` | DA `(GX, G)` bit-identical to SOURCE |
| A4 | `a4_copsens_parity.py` | CopSens bounds + intermediates vs SOURCE |
| A5 | `a5_njobs_exactness.py` | `n_jobs` changes nothing but wall clock |
| A6-A8, A11-A14 | `a6_a14_pipeline.py` | perf fairness, status, JAX≡FD, config, recipe, cost, memory |
| A9 | `a6_a14_pipeline.py --full` | the estimand is the CAUSAL one (needs 1.2M draws) |
| A10 | `a10_partial_r2_regression.py` | `PartialR2` unchanged by the `BoundedSA` hoist |

`smoke_do_mnist.py` is an end-to-end query-sweep + perf run at reduced scale;
`--full` runs it at the config's own numbers.

## Two-stage gates

```bash
SRC=../doMNIST/symmetry4CausalBoundsDoMNIST
$SRC/.venv/bin/python scripts/a1_a2_sem.py       --source /tmp/a2
$SRC/.venv/bin/python scripts/a3_da_parity.py    --source /tmp/a3.npz
$SRC/.venv/bin/python scripts/a4_copsens_parity.py --source /tmp/a4

~/scratch/envs/s4cb/bin/python scripts/a1_a2_sem.py        --check /tmp/a2
~/scratch/envs/s4cb/bin/python scripts/a3_da_parity.py     --check /tmp/a3.npz
~/scratch/envs/s4cb/bin/python scripts/a4_copsens_parity.py --check /tmp/a4
```

## A10

Digest the current tree, then the tree without the change, and diff:

```bash
python scripts/a10_partial_r2_regression.py > /tmp/after.json
git stash && python scripts/a10_partial_r2_regression.py > /tmp/before.json && git stash pop
diff /tmp/before.json /tmp/after.json
```
