"""
do-MNIST: the canonical confounding triangle X <- U -> Y, X -> Y.

U ~ Bern(1/2) is hidden and reaches X only through a NOISY colour readout (eta > 0),
so it is a genuine confounder, not an adjustable covariate. Colour never causes Y;
digit identity does. The confounding is therefore ADDITIVE in probability and
CONSTANT in x: |h_erm - h_*| = beta(1/2 - eta).
"""

import copy
import hashlib
import os
from collections.abc import Sequence

import numpy as np
from loguru import logger
from numpy.typing import NDArray
from torchvision import datasets

from src.sem.abstract import StructuralEquationModel

DATA_DIR: str = os.environ.get("MNIST_DIR", os.path.expanduser("~/scratch/data/mnist"))
TINT_LO, TINT_HI, TINT_JITTER = 0.1, 0.9, 0.05
# image-level partition of the 60k training images: A trains the nets, B is the only
# thing the PI machinery sees, C selects gamma. The SEM owns the split, so the default
# lives here.
SPLIT_DEFAULT: dict[str, int] = {"A": 40_000, "B": 10_000, "C": 10_000}


def tint(grey: NDArray, t: NDArray) -> NDArray:
    """RGB = [t, 0, 1-t] * grey. grey (N,H,W) in [0,1], t (N,) -> (N,3,H,W)."""
    t = np.asarray(t, dtype=np.float32).reshape(-1, 1, 1)
    grey = np.asarray(grey, dtype=np.float32)
    return np.stack([t * grey, np.zeros_like(grey), (1.0 - t) * grey], axis=1)


def tint_of(X: NDArray) -> NDArray:
    """t = sumR / (sumR + sumB). Exact, invariant to the ink amount."""
    X = np.asarray(X)
    R, B = X[:, 0].sum((-2, -1)), X[:, 2].sum((-2, -1))
    return R / np.maximum(R + B, 1e-12)


def grey_of(X: NDArray) -> NDArray:
    """Ink mask: R + B = grey (the green channel is identically 0)."""
    X = np.asarray(X)
    return X[:, 0] + X[:, 2]


def split_indices(n: int, sizes: dict[str, int], seed: int) -> dict[str, NDArray]:
    """Disjoint image-index sets from ONE permutation of range(n); sizes must sum to n.

    Asserts pairwise disjointness and full coverage every time: the acceptance check
    is not a test-only property.
    """
    sizes = {k: int(v) for k, v in sizes.items()}
    if sum(sizes.values()) != n:
        raise ValueError(f"split sizes {sizes} sum to {sum(sizes.values())}, not {n}")
    perm = np.random.default_rng(seed).permutation(n)
    cuts = np.cumsum([0] + list(sizes.values()))
    out = {k: np.sort(perm[a:b]) for k, a, b in zip(sizes, cuts[:-1], cuts[1:], strict=True)}
    if len(np.unique(np.concatenate(list(out.values())))) != n:
        raise AssertionError("split parts must be disjoint and cover every image")
    return out


def split_key(parts: dict[str, "DoMNISTSEM"]) -> str:
    """ONE sha1 over 'A' + sorted A indices + 'B' + ...: a single provenance string for
    the whole partition, written as `split_key` by every entry point that uses it."""
    h = hashlib.sha1()  # noqa: S324 (provenance digest, not security)
    for k in sorted(parts):
        h.update(k.encode())
        h.update(np.sort(parts[k].subset_).astype(np.int64).tobytes())
    return h.hexdigest()


def _bern(p, n, rng) -> NDArray:
    return (rng.random(n) < p).astype(float)


def _flat(A) -> NDArray:
    return np.asarray(A).reshape(len(A), -1)


class DoMNISTSEM(StructuralEquationModel):
    """
    f := 1[digit >= 5] is the invariant causal label; colour carries the confounding.

    h_* is NOT linear in pixels, so `solution` does not exist, and it is a function
    of the digit label alone, so `f` does not exist either: the analytic target is
    `ate_of(digits)` for a set of queries and `h_star(f)` for a draw (`last_["f"]`),
    which is what every runner scores against.
    """

    def __init__(
        self,
        seed: int,
        train: bool = True,
        alpha: float = 0.0,
        beta: float = 0.4,
        eta: float = 0.25,
        jitter: float = TINT_JITTER,
        subsample: int = 2,
        directory: str = DATA_DIR,
    ):
        ds = datasets.MNIST(directory, train=train, download=True)
        self.seed, self.train = seed, train
        self.images, self.targets = ds.data.numpy(), ds.targets.numpy()
        self.alpha, self.beta, self.eta, self.jitter = alpha, beta, eta, jitter
        self.sub = subsample
        is_high = self.targets >= 5
        self._pools = (np.flatnonzero(~is_high), np.flatnonzero(is_high))
        self.subset_ = None  # global MNIST indices this SEM may draw from; None = all
        logger.info(
            f"do-mnist: n={len(self.images):,} train={train} alpha={alpha} beta={beta} eta={eta} sub={subsample}"
        )

    def __len__(self) -> int:
        return len(self.subset_) if self.subset_ is not None else len(self.images)

    # ------------------------------------------------------------------ split

    def restrict(self, idx) -> "DoMNISTSEM":
        """A shallow copy that draws only from the images `idx` (GLOBAL MNIST indices).

        Shares images/targets with self; only the class pools change, so `_balanced`
        still rebalances to P(f=1) = 1/2 and `last_['idx']` stays a global index.
        """
        sem = copy.copy(self)
        sem.__dict__.pop("last_", None)
        sem.subset_ = np.sort(np.asarray(idx, dtype=np.int64))
        is_high = self.targets[sem.subset_] >= 5
        sem._pools = (sem.subset_[~is_high], sem.subset_[is_high])
        return sem

    def split(self, sizes: dict[str, int] | None = None, seed: int = 420) -> dict[str, "DoMNISTSEM"]:
        """{'A': sem_A, 'B': sem_B, 'C': sem_C}: disjoint restrictions covering every image."""
        if self.subset_ is not None:
            raise ValueError("split() partitions the full image set; call it on an unrestricted SEM")
        idx = split_indices(len(self.images), sizes or SPLIT_DEFAULT, seed)
        return {k: self.restrict(v) for k, v in idx.items()}

    # ------------------------------------------------- structural (closed form)

    def h_star(self, f) -> NDArray:
        """E[Y|do(x)]: U is resampled, so colour drops out. {0.2, 0.8} canonically."""
        f = np.asarray(f, dtype=float)
        return (1 - self.beta) * (self.alpha + f * (1 - 2 * self.alpha)) + self.beta / 2

    def h_erm(self, f, c) -> NDArray:
        """E[Y|X=x]: colour leaks E[U|C=c]. Cells {.1,.3,.7,.9} canonically."""
        f, c = np.asarray(f, dtype=float), np.asarray(c, dtype=float)
        eu = np.where(c > 0.5, 1 - self.eta, self.eta)
        return (1 - self.beta) * (self.alpha + f * (1 - 2 * self.alpha)) + self.beta * eu

    def ate_of(self, digits) -> NDArray:
        """Analytic h_*(x) straight off the digit label. (N,1), like `f`."""
        return self.h_star((np.asarray(digits) >= 5).astype(float))[:, None]

    @property
    def solution(self) -> NDArray:
        raise NotImplementedError("do-MNIST h_* is not linear in pixels; use ate_of / h_star instead.")

    @property
    def bias_sq(self) -> float:
        """|| h_erm - h_* ||^2 = (beta(1/2 - eta))^2. Constant in x, so no draw needed."""
        return float((self.beta * (0.5 - self.eta)) ** 2)

    @property
    def sigma_sq(self) -> float:
        """E[Var(Y|X)] = E[h_erm(1 - h_erm)]. f and C are independent and both
        balanced, so the four cells are equiprobable."""
        cells = np.array([self.h_erm(f, c) for f in (0.0, 1.0) for c in (0.0, 1.0)])
        return float(np.mean(cells * (1 - cells)))

    @property
    def attainable(self) -> tuple[float, float]:
        """Range h_erm can occupy: [(1-beta)alpha + beta*eta, 1 - that]. mu_y outside
        it is impossible."""
        lo = (1 - self.beta) * self.alpha + self.beta * self.eta
        return float(lo), float(1.0 - lo)

    # ---------------------------------------------------------------- sampling

    def _grey(self, idx, subsample: int | None = None) -> NDArray:
        step = self.sub if subsample is None else int(subsample)
        return self.images[idx][:, ::step, ::step] / 255.0

    def _balanced(self, N, rng) -> NDArray:
        """P(f=1) = 1/2 exactly, whatever MNIST's own class balance is."""
        lo, hi = self._pools
        n1 = N // 2
        idx = np.concatenate(
            [rng.choice(lo, N - n1, replace=N - n1 > len(lo)), rng.choice(hi, n1, replace=n1 > len(hi))]
        )
        rng.shuffle(idx)
        return idx

    def _draw(self, N, rng) -> dict:
        """Everything except U_out. Drawing U_out LAST is what makes the obs and do
        draws at one seed share idx, U, C, t, S and y_causal exactly."""
        idx = self._balanced(N, rng)
        digits = self.targets[idx]
        f = (digits >= 5).astype(float)
        U = _bern(0.5, N, rng)
        C = np.logical_xor(U > 0.5, _bern(self.eta, N, rng) > 0.5).astype(float)
        t = np.clip(np.where(C > 0.5, TINT_HI, TINT_LO) + rng.normal(0, self.jitter, N), 0.0, 1.0)
        X = tint(self._grey(idx), t)
        S = _bern(self.beta, N, rng)
        y_causal = np.logical_xor(f > 0.5, _bern(self.alpha, N, rng) > 0.5).astype(float)
        return dict(idx=idx, digits=digits, f=f, U=U, C=C, t=t, S=S, X=X, y_causal=y_causal)

    def _stash(self, d, mode):
        """Per-draw provenance. MUTABLE: every sample/sample_paired overwrites it,
        so read it immediately after its own draw."""
        self.last_ = dict(idx=d["idx"], digits=d["digits"], f=d["f"], U=d["U"], C=d["C"], t=d["t"], S=d["S"], mode=mode)

    def sample(
        self, N: int | None = 1, intervention: bool = False, seed: int | None = None, **kwargs
    ) -> tuple[NDArray, NDArray]:
        """intervention=False uses the image's own U; True resamples it for Y only."""
        rng = np.random.default_rng(seed) if seed is not None else np.random
        N = len(self) if (N is None or N <= 0) else int(N)  # N > len(self) is fine: fresh (U,C,S) per draw
        d = self._draw(N, rng)
        U_out = _bern(0.5, N, rng) if intervention else d["U"]
        y = np.where(d["S"] > 0.5, U_out, d["y_causal"])
        self._stash(d, "do" if intervention else "obs")
        return d["X"].astype(np.float32), y[:, None]

    def sample_paired(self, N: int, seed: int):
        """(X, y_obs, y_do) from ONE draw: same images and (U,C,t,S), only U_out differs.

        Common random numbers for the obs and do label lines, so the ERM report's
        accuracies on both are measured on the same images. `seed` is REQUIRED.
        """
        rng = np.random.default_rng(seed)
        N = len(self) if (N is None or N <= 0) else int(N)
        d = self._draw(N, rng)
        y_obs = np.where(d["S"] > 0.5, d["U"], d["y_causal"])
        y_do = np.where(d["S"] > 0.5, _bern(0.5, N, rng), d["y_causal"])
        self._stash(d, "paired")
        return d["X"].astype(np.float32), y_obs[:, None], y_do[:, None]

    # ------------------------------------------------------- the estimand h_*

    def f(self, X) -> NDArray:
        raise NotImplementedError("do-MNIST h_* is a function of the digit: use ate_of(digits) or h_star(last_['f'])")

    # ------------------------------------------------------------- exemplars

    def _exemplar_indices(self, rng, digits: Sequence[int] = range(10)) -> NDArray:
        """One image index per digit, off `rng` in digit order. Takes the generator,
        not a seed, so `exemplars` draws its tints from the same stream after it."""
        return np.array([int(rng.choice(np.flatnonzero(self.targets == d))) for d in digits])

    def exemplars(
        self,
        seed: int = 420,
        digits: Sequence[int] = range(10),
        colors: str = "alternating",
        subsample: int | None = None,
    ):
        """One frozen sample per digit, for the query-sweep x-axis.

        colors='alternating': C = 0,1,0,1,... so the colour-driven ERM zig-zag reads
        against a flat h_*. That is OFF the SEM's colour law, so these are a
        VISUALISATION set only -- never use them to measure coverage.

        `subsample` overrides the SEM's own. The draws (which digit, which tint) do
        not depend on it, so `subsample=1` returns the SAME exemplars at full
        resolution -- for the figure, while the models keep the subsampled ones.
        """
        rng = np.random.default_rng(seed)
        idx = self._exemplar_indices(rng, digits)
        n = len(idx)
        if colors == "alternating":
            C = (np.arange(n) % 2).astype(float)  # 0=blue, 1=red
        else:
            U = _bern(0.5, n, rng)
            C = np.logical_xor(U > 0.5, _bern(self.eta, n, rng) > 0.5).astype(float)
        t = np.clip(np.where(C > 0.5, TINT_HI, TINT_LO) + rng.normal(0, self.jitter, n), 0.0, 1.0)
        return tint(self._grey(idx, subsample), t).astype(np.float32), self.targets[idx]

    def tinted(self, seed: int, digit: int, tints: Sequence[float], subsample: int | None = None) -> NDArray:
        """ONE image of `digit` rendered at every tint in `tints`: (N, 3, H, W).

        The image is the one `exemplars(seed)` draws for that digit on this SEM, so
        the grey ink is identical along the row and only the tint moves. On the
        training SEM it IS the exemplar image; on the test SEM it is an image no net
        trained on. A VISUALISATION set, like the exemplars, never scored.
        """
        if int(digit) not in range(10):
            raise ValueError(f"digit must be in 0..9; got {digit!r}")
        idx = self._exemplar_indices(np.random.default_rng(seed))[int(digit)]
        t = np.asarray(tints, dtype=np.float32).ravel()
        grey = np.repeat(self._grey(np.array([idx]), subsample), len(t), axis=0)
        return tint(grey, t).astype(np.float32)
