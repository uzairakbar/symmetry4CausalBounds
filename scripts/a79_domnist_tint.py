"""A79: the do-MNIST tint sweep's images (one MNIST load, no nets, CPU, seconds).

(i)   the exemplar pin: `exemplars(420)` draws the same 10 indices and tints, and
      both image arrays (the SEM's subsample and `subsample=1`) hash to the same
      sha1, as on `develop` before `_exemplar_indices` was factored out. The values
      below were read once off that tree.
(ii)  `tinted`: one image per call, (N, 3, H, W) float32, `tint_of` returns the
      grid, the ink (`grey_of`) is the same in every row, the image is the
      exemplar's on the same SEM and seed, `subsample` only changes the resolution,
      and a digit outside 0..9 raises.

  uv run python scripts/a79_domnist_tint.py
"""

import hashlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger  # noqa: E402

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


def main():
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    sem = DoMNISTSEM(train=True, **SEM_KW)
    leg_i(sem)
    leg_ii(sem)
    print(f"\nA79 {'PASS' if not FAIL else 'FAIL'}")
    for name in FAIL:
        print(f"  FAILED: {name}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
