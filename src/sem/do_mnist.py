"""
do-MNIST: the canonical confounding triangle X <- U -> Y, X -> Y.

U ~ Bern(1/2) is hidden and reaches X only through a NOISY colour readout (eta > 0),
so it is a genuine confounder, not an adjustable covariate. Colour never causes Y;
digit identity does. The confounding is therefore ADDITIVE in probability and
CONSTANT in x: |h_erm - h_*| = beta(1/2 - eta).
"""

import os
import numpy as np
from loguru import logger
from typing import Tuple, Optional, Sequence
from numpy.typing import NDArray
from torchvision import datasets

from src.sem.abstract import StructuralEquationModel

DATA_DIR: str = os.environ.get("MNIST_DIR", os.path.expanduser("~/scratch/data/mnist"))
TINT_LO, TINT_HI, TINT_JITTER = 0.1, 0.9, 0.05


def tint(grey: NDArray, t: NDArray) -> NDArray:
    """RGB = [t, 0, 1-t] * grey. grey (N,H,W) in [0,1], t (N,) -> (N,3,H,W)."""
    t = np.asarray(t, dtype=np.float32).reshape(-1, 1, 1)
    grey = np.asarray(grey, dtype=np.float32)
    return np.stack([t * grey, np.zeros_like(grey), (1.0 - t) * grey], axis=1)


def grey_of(X: NDArray) -> NDArray:
    """Ink mask: R + B = grey (the green channel is identically 0)."""
    X = np.asarray(X)
    return X[:, 0] + X[:, 2]


def _bern(p, n, rng) -> NDArray:
    return (rng.random(n) < p).astype(float)


def _flat(A) -> NDArray:
    return np.asarray(A).reshape(len(A), -1)


class DoMNISTSEM(StructuralEquationModel):
    """
    f := 1[digit >= 5] is the invariant causal label; colour carries the confounding.

    h_* is NOT linear in pixels, so `solution` does not exist and `f` is a fitted
    net on interventional draws -- sound because sample(intervention=True) resamples
    U for the outcome line only, so E[Y | X=x, do] = h_*(x) exactly.
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
        net: str = "domnist-fast",
        target_samples: int = 1_200_000,
        target_kw: Optional[dict] = None,
        directory: str = DATA_DIR,
    ):
        ds = datasets.MNIST(directory, train=train, download=True)
        self.seed, self.train = seed, train
        self.images, self.targets = ds.data.numpy(), ds.targets.numpy()
        self.alpha, self.beta, self.eta, self.jitter = alpha, beta, eta, jitter
        self.sub = subsample
        self.net, self.target_samples = net, target_samples
        self.target_kw = dict(target_kw or {})
        self._target = None
        is_high = self.targets >= 5
        self._pools = (np.flatnonzero(~is_high), np.flatnonzero(is_high))
        logger.info(
            f"do-mnist: n={len(self.images):,} train={train} alpha={alpha} beta={beta} eta={eta} sub={subsample}"
        )

    def __len__(self) -> int:
        return len(self.images)

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
        raise NotImplementedError("do-MNIST h_* is not linear in pixels; use sem.f(X) instead.")

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
    def attainable(self) -> Tuple[float, float]:
        """Range h_erm can occupy: [(1-beta)alpha + beta*eta, 1 - that]. mu_y outside
        it is impossible, which is what makes clipping it safe (`mu_clip`)."""
        lo = (1 - self.beta) * self.alpha + self.beta * self.eta
        return float(lo), float(1.0 - lo)

    # ---------------------------------------------------------------- sampling

    def _grey(self, idx, subsample: Optional[int] = None) -> NDArray:
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
        self, N: int = 1, intervention: bool = False, seed: Optional[int] = None, **kwargs
    ) -> Tuple[NDArray, NDArray]:
        """intervention=False uses the image's own U; True resamples it for Y only."""
        rng = np.random.default_rng(seed) if seed is not None else np.random
        N = len(self.images) if (N is None or N <= 0) else int(N)
        d = self._draw(N, rng)
        U_out = _bern(0.5, N, rng) if intervention else d["U"]
        y = np.where(d["S"] > 0.5, U_out, d["y_causal"])
        self._stash(d, "do" if intervention else "obs")
        return d["X"].astype(np.float32), y[:, None]

    def sample_paired(self, N: int, seed: int):
        """(X, y_obs, y_do) from ONE draw: same images and (U,C,t,S), only U_out differs.

        Common random numbers for the ERM/target pair -- the estimation error the two
        nets share then cancels in |mu_erm - h^|, whose p95 sets the PI width.
        `seed` is REQUIRED: the estimand must not depend on which call site happened
        to trigger the lazy target fit.
        """
        rng = np.random.default_rng(seed)
        N = len(self.images) if (N is None or N <= 0) else int(N)
        d = self._draw(N, rng)
        y_obs = np.where(d["S"] > 0.5, d["U"], d["y_causal"])
        y_do = np.where(d["S"] > 0.5, _bern(0.5, N, rng), d["y_causal"])
        self._stash(d, "paired")
        return d["X"].astype(np.float32), y_obs[:, None], y_do[:, None]

    # ------------------------------------------------------- the estimand h_*

    @property
    def target(self):
        """h_*, estimated. One net per SEM: every method in an experiment must be
        scored against the same estimand. Fitted lazily, then frozen."""
        if self._target is None:
            from src.methods.regression import GradientDescentERM

            logger.info(f"do-mnist: fitting the target net on {self.target_samples:,} interventional draws")
            X, _, y_do = self.sample_paired(self.target_samples, seed=self.seed)
            # Same draw and same init_seed as the runner's ERM pair (which draws at
            # this same (N, seed)) => the two nets are coupled and share error.
            self._target = GradientDescentERM(self.net).fit(_flat(X), y_do, init_seed=self.seed, **self.target_kw)
            del X
        return self._target

    def f(self, X) -> NDArray:
        """h_*(x) on FLAT (N, d) pixels -> (N, 1), matching the linear SEMs' shape."""
        return np.asarray(self.target.predict_mean(X), dtype=float).reshape(-1, 1)

    # ------------------------------------------------------------- exemplars

    def exemplars(
        self,
        seed: int = 420,
        digits: Sequence[int] = range(10),
        colors: str = "alternating",
        subsample: Optional[int] = None,
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
        idx = np.array([int(rng.choice(np.flatnonzero(self.targets == d))) for d in digits])
        n = len(idx)
        if colors == "alternating":
            C = (np.arange(n) % 2).astype(float)  # 0=blue, 1=red
        else:
            U = _bern(0.5, n, rng)
            C = np.logical_xor(U > 0.5, _bern(self.eta, n, rng) > 0.5).astype(float)
        t = np.clip(np.where(C > 0.5, TINT_HI, TINT_LO) + rng.normal(0, self.jitter, n), 0.0, 1.0)
        return tint(self._grey(idx, subsample), t).astype(np.float32), self.targets[idx]
