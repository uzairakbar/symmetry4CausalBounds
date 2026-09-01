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
This code was tested on **MacOS** (Apple silicon) and **Linux**. We recommend running this code using `conda`, however we have also provided Docker and python `venv` setup scripts as alternatives.

### Conda environment (recommended)
```bash
environment=symmetry4CausalBounds
conda env create -f environment.yaml
conda activate "$environment"
export PYTORCH_ENABLE_MPS_FALLBACK=1
```
Then run experiments with `conda run --no-capture-output -n symmetry4CausalBounds python src/main.py`.

### Python `venv` (tested with `3.10.14`)
```bash
environment='.symmetry4CausalBounds'
python -m venv "$environment"
"$environment"/bin/python -m pip install -r requirements.txt
source "$environment"/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1
```
Then run the main script `symmetry4CausalBounds/bin/python src/main.py`.

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
The code uses a CPU backend for PyTroch by default (recommended for `optical_device` and `simulation` sweep experiments). To use a GPU or MPS backend, however, change the `CPU_ONLY` variable specified in `./src/regressors/utils.py` to `False`.

## PACE

```bash
bash setup_uv.sh
```

```bash
bash fix_torch.sh
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