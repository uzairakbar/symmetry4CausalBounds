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
  (0.059352). It replaced the earlier max over PI and DA+PI (0.0851), so the F1
  figure's numbers moved with it.
- `experiment.query.tint: {digit: [...], sweep_samples: 8, range: [0, 1]}` adds one
  tint sweep per digit: one MNIST-test image rendered from blue (0) to red (1) and
  scored by every method, with the target constant along it
  (`recipes/doMnistTintFigF2.yaml` sweeps all ten).

`python -m src.aggregate --artifacts DIR` then writes `DIR/aggregate/do_mnist_tint.pdf`
(the sweeps stacked, 0 at the top, the tint histogram before and after DA at the
bottom) and `DIR/aggregate/do_mnist_table.tex`: coverage and width with 95% bootstrap
bands over the 2,000 population queries, Omega-hat, worst error, and the latency of
each method, its one-time fit (every net, DA pass, model fit and floor it needs) plus
its mean per-query solve at the run's `n_jobs`. The fit and the per-query solve are
also columns of their own.

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
