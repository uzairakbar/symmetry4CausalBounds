"""A3: the ported do-MNIST DA must reproduce SOURCE's `_augment_fast` exactly.

Two processes, two environments. Stage 1 dumps SOURCE's (GX, G) under a fixed
torch seed; stage 2 recomputes them here and compares.

    <source-venv>/bin/python scripts/a3_da_parity.py --source /tmp/a3.npz
    <s4cb>/bin/python       scripts/a3_da_parity.py --check  /tmp/a3.npz
"""

import argparse
import os
import sys

import numpy as np

SOURCE = os.path.expanduser("~/neurips26/ZULFI/doMNIST/symmetry4CausalBoundsDoMNIST")
AUGMENTATION = "translate > rotation > contrast > saturation > hue"
SEED, N = 1234, 2048


def _path(path):
    """np.savez_compressed appends .npz, np.load does not. Normalise both ends."""
    return path if path.endswith(".npz") else path + ".npz"


def _images(root, n):
    """One SEM draw, from whichever repo we are running inside."""
    sys.path.insert(0, root)
    from src.sem.domnist import DoMNISTSEM as SourceSEM  # noqa: PLC0415

    sem = SourceSEM(train=False, alpha=0.0, beta=0.4, eta=0.25, subsample=2)
    return sem.sample(N=n, seed=7)[0]


def dump(path):
    import torch

    sys.path.insert(0, SOURCE)
    os.chdir(SOURCE)
    from src.data_augmentors.domnist import DoMNISTDA  # noqa: PLC0415

    X = _images(SOURCE, N)
    torch.manual_seed(SEED)
    GX, G = DoMNISTDA(AUGMENTATION)(X)
    np.savez_compressed(_path(path), X=X, GX=GX, G=G)
    print(f"wrote {_path(path)}: GX {GX.shape} G {G.shape}")


def check(path):
    import torch

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.data_augmentors.do_mnist import DoMNISTDA  # noqa: PLC0415

    blob = np.load(_path(path))
    torch.manual_seed(SEED)
    GX, G = DoMNISTDA(AUGMENTATION)(blob["X"])

    d_gx = float(np.abs(GX - blob["GX"]).max())
    d_g = float(np.abs(G - blob["G"]).max())
    print(f"max|dGX| = {d_gx:g}   max|dG| = {d_g:g}")
    ok = d_gx == 0.0 and d_g == 0.0
    print("A3 PASS" if ok else "A3 FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source")
    parser.add_argument("--check")
    args = parser.parse_args()
    if args.source:
        dump(args.source)
    else:
        sys.exit(check(args.check))
