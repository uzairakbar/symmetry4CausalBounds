# Symmetry-Informed Causal Partial Identification
> Implementation for *"Symmetry-Informed Causal Partial Identification"* (Preprint 2026).
<p align="center">
    <img src="https://uzairakbar.github.io/symmetry4CausalBounds/card.png"
    alt="Symmetry for Causal Bounds"
    width="33%">
</p>
<p align="center">
  <a href="https://arxiv.org/abs/#"><img src="https://img.shields.io/badge/arXiv-2510.25128-B31B1B.svg?logo" alt="arXiv Manuscript"></a>
  <a href="https://uzairakbar.github.io/symmetry4CausalBounds"><img src="https://img.shields.io/badge/WEB-page-0eb077.svg" alt="Project Webpage"></a>
  <a href="https://colab.research.google.com/github/uzairakbar/symmetry4CausalBounds/blob/colab/s4cb.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Google Colab"></a>
</p>

## Setup
### Dependencies
Python 3.13, managed with [`uv`](https://docs.astral.sh/uv/). `uv.lock` pins the
exact versions the paper figures were generated with.

```bash
uv sync
uv run python src/main.py
```

### Docker
Build provided `Dockerfile` and run.
```bash
image=symmetry4CausalBoundsImage
container=symmetry4CausalBoundsContainer
docker build --tag "$image" .
docker run --name "$container" \
    --volume "$PWD"/data:/app/data/ \
    --volume "$PWD"/artifacts:/app/artifacts/ \
    "$image"
```

To delete docker artifacts after finishing experiments, run the following commands.
```bash
image=symmetry4CausalBoundsImage
container=symmetry4CausalBoundsContainer
docker rm "$container"
docker image rm -f "$image"
```

## Experiment configuration
Use the `./config.yaml` file to specify the experiment parameters. The provided (default) configuration was used to generate the figures of the paper.

Comment out (or remove) the experiemnts from `./config.yaml` that you are not interested in, and then run the `./src/main.py` script to run the remaining experiments.

The generated figures and artifacts are saved in the `./artifacts/` directory after the experiments finish execution.

## Finite-sample tolerance (gamma_n)
Every linear PI program carries finite-sample pads, so one solve gives a
simultaneous 95% band. The level is the `gamma_n` key, in percent, set to `95` in
`config.yaml`'s `defaults:` and in every recipe but `opticalDeviceFig6` (below).

- **The pads.** With `gamma_n(d; g) = F^-1_{chi2_d(n g)}(1 - a) / n`
  (`sensitivity_models.finite_sample_budget`), the ERM ball is `sigma-hat sqrt((1 +
  gamma~) gamma_n(k; gamma~ / (1 + gamma~)))`, the mean-matched slice gets a mean
  row `|delta| <= sigma-hat sqrt((1 + gamma~) gamma_n(1; 0))` (delta inside the
  ball; none with `mean_match: false`), and every IV row is `sqrt(s^2 (1 + gamma~)
  gamma_n(d; g / (1 + gamma~)))`. alpha = 0.05 is split the same way on every
  method: alpha/3 to the ball, alpha/3 to the mean row, alpha/3 shared by the IV
  rows. A non-DA IV row's leak `g` is the declared `gamma_z`; a DA row's is App. D's
  post-DA `(eps / sigma~ + sqrt(gamma_z / rho))^2`. A DA fit with both the
  translation amounts T and an observed instrument Z carries one joint row on
  span(T, Z) (`IV_LAYOUT`, chosen by a pilot). The invariance budget and the +-eps
  pad stay unpadded: eps = eps* + 2^-8 (`EPS_TOL`), eps* the q0.95 of |W| on the
  sweeps and its RMS on the query panels.
- **Raw mode.** Without the key (or `gamma_n: 0` / `false`) the programs use the
  population budgets: the ball `sigma-hat sqrt(gamma~)`, the exact mean match and
  the IV rows `s sqrt(g)`. A genuine but optimistic band. Every model built
  directly (`PartialR2(...)`, `MethodRegistry.build_methods(...)` without
  `gamma_n_alpha`) is raw.
- **n per dataset.** Every padded row counts the fit's original samples (the m
  sweep's rows are m copies of them): the simulation and optical rows, the
  cigarette plasmode rows (220 to 2205; its errors are independent given X), and
  the real-panel query `cigarettesFig7`'s state-year rows, counted the same way.
- **Intersections.** PI & DA+... intersects two simultaneous 95% sets, so its band
  is 90%.
- **Declared instrument leaks.** The cigarettes declare `gamma_z: 0.01`; the
  simulation declares `SimulationConfig.gamma_z = 2^-8` for its generated
  instrument (not to be confused with `CigaretteConfig.gamma_z`, a sliver guard);
  optical has no observed instrument.
- **No nesting of (T,Z) in (Z).** Under the pads the joint (T,Z) row has more
  columns than the Z row at the same level, so a `(T,Z)` interval need not sit
  inside the `(Z)` one; raw, it does. Every +IV interval still sits inside its
  non-IV counterpart.
- do-MNIST accepts the key and ignores it: its CopSens balls are oracle-calibrated.
- **The unpadded exceptions.** Besides do-MNIST, `opticalDeviceFig6` runs
  `gamma_n: 0` with `oracle_t_leak: true`, a population-level illustration: the
  DA T row's radius is the oracle T leak `||E-hat[W# | T]|| / sqrt(N)` (W# the part
  of W that OLS on Phi(GX) leaves; RMS-pooled over the oracle's seeded DA draws) +
  2^-8, raw, in place of the row formula at `(eps / sigma~)^2`. The T row's bite on
  optical is a large-sample effect: at n = 1000 the padded T row is slack. The
  toggle (default false) is the optical query's alone; anywhere else it is a
  config error.

## do-MNIST
The `do_mnist:` block runs the query path on the CopSens latent-factor ball around
prefit nets. The main knobs:

- `inv_recenter` picks PI+INV's centre. `inv` (shipped) is the ERM+INV net, the ERM
  trained by an augmented Lagrangian to E[(h(X) - h(GX))^2] <= `erm_inv_tau` (4e-4,
  i.e. 0.02^2 in epsilon units) on the DA pairs; the ball sits on X with the pairs
  (X, GX) and the budget eps^2 = 0.0016. `off` centres PI+INV on the ERM (its floor,
  0.0198, is above eps^2, so it is infeasible at every query); `on` on DA+ERM.
  `scripts/diagnose_domnist_erm_inv.py` checks the ERM+INV net on the full draw.
- `ERM+INV` is also a do-MNIST-only point method, listed commented out under
  `methods:`; listing it trains the net and plots it.
- `gamma` is ONE value for every method: PI's smallest gamma reaching
  `target_coverage` (0.995) on split C, from `scripts/select_domnist_gamma.py`
  (0.062082 with the pooled nets). It replaced the earlier max over PI and DA+PI
  (0.0851), so the F1 figure's numbers moved with it.
- `net: domnist-pool` trains all three nets (ERM, DA+ERM, ERM+INV) as a pooled CNN
  (global average pooling over the last conv map). The flat head, `domnist-fast`,
  stays the code default. Under the flat head the ERM+INV net met tau by giving up
  most of its digit signal (f-accuracy 0.825 on split C); pooled, it keeps 0.959.
- `experiment.query.tint: {digit: [...], sweep_samples: 8, range: [0, 1]}` adds one
  tint sweep per digit: one MNIST-test image rendered from blue (0) to red (1) and
  scored by every method, with the target constant along it
  (`recipes/doMnistTintFigF2.yaml` sweeps all ten).
- `n_experiments` is the number of seeds the population metrics are read at:
  `seed`, `seed + 1`, ... Every seed draws its own replicate, retrains the nets and
  refits every method; the split, the population and the exemplars stay fixed. The
  figure, `run.json` and the population pkls are the first seed's. The run writes
  `do_mnist/query/seeds.json` (coverage, width, worst error, NaN share and RMSE per
  seed) and `seeds_table.tex`: coverage, width and worst error of each interval
  method, mean +- standard error over the seeds, the best `\bm`, the second best
  `\mathit` (coverage highest, width and worst error lowest), the point estimators'
  RMSE in a comment line. It needs `\usepackage{bm}`.

`python -m src.aggregate --artifacts DIR` then writes `DIR/aggregate/do_mnist_tint.pdf`
(the sweeps stacked, 0 at the top, the tint histogram before and after DA at the
bottom) and `DIR/aggregate/do_mnist_table.tex`: coverage and width with 95% bootstrap
bands over the 2,000 population queries, Omega-hat, worst error, and the latency of
each method, its one-time fit (every net, DA pass, model fit and floor it needs) plus
its mean per-query solve at the run's `n_jobs`. The fit and the per-query solve are
also columns of their own. With `seeds.json` present it also re-renders the table
across seeds to `DIR/aggregate/do_mnist_seeds_table.tex`.

## CPU vs. GPU backend
PyTorch picks CUDA/MPS automatically when available (only do-MNIST trains nets; `optical_device` and `simulation` never touch torch). To force CPU, set `CPU_ONLY = True` in `./src/methods/nets.py`.

## PACE
Creates the env under `~/scratch/uv_envs/` and symlinks it to `./.venv`.

```bash
bash setup_uv.sh
```

In a later shell, set the same variables before any `uv run`, or uv builds a
multi-GB `.venv` inside the repo:

```bash
module load uv
export UV_CACHE_DIR=$HOME/scratch/.cache/uv
export UV_PYTHON_INSTALL_DIR=$HOME/scratch/.local/uv/python
export UV_PROJECT_ENVIRONMENT=$HOME/scratch/uv_envs/symmetry4CausalBounds-py313
export REPO=$HOME/close-this/final/symmetry4CausalBounds
uv sync --frozen
```

Run experiments from a scratch directory holding the `config.yaml` (`save` writes
`./artifacts` relative to it): `cd ~/scratch/domnist_runs/NAME && uv run --frozen
--project $REPO python $REPO/src/main.py`.

## Citation
If you find our work helpful, consider citing our paper and leaving a star :star:.
```bibtex
@misc{akbar2026symmetry4CausalBounds,
      title={Symmetry-Constrained Causal Partial Identification},
      author={Uzair Akbar and Zulfiqar Zaidi and Niki Kilbertus and Krikamol Muandet and Bo Dai},
      year={2026},
      eprint={TBD},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={TBD},
}
```
