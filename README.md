# Symmetry-Constrained Causal Partial Identification
> Implementation for *"Symmetry-Constrained Causal Partial Identification"* (Preprint 2026).
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

## CPU vs. GPU backend
PyTorch picks CUDA/MPS automatically when available (only do-MNIST trains nets; `optical_device` and `simulation` never touch torch). To force CPU, set `CPU_ONLY = True` in `./src/methods/utils.py` and `./src/methods/nets.py`.

## PACE
Creates the env under `~/scratch/uv_envs/` and symlinks it to `./.venv`.

```bash
bash setup_uv.sh
```

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
