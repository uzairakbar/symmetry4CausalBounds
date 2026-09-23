"""
do-MNIST augmentors, batched on the device.

NAMING: the colour ops are NOT their torchvision namesakes. Pixels here live on the
2-parameter manifold ink*(t, 0, 1-t), so every op just reparametrises (ink, t):
  contrast   -- mean over FOREGROUND only, background pinned at 0. Ink contrast.
  saturation -- MISNOMER. Pulls t -> 0.5, i.e. equal red and blue = MAGENTA, not grey
                (grey needs the green channel we do not have). A chroma contraction
                on the R-B line.
  hue        -- MISNOMER. t -> t + d translates along one axis. A tint shift.
  brightness -- honest: scaling ink scales R and B equally, = torchvision brightness.
Staying on the R/B line keeps P_X~ on do-MNIST's support. saturation and hue are both
affine in t, so as IV columns they are not independent directions the way rotation
and translation are.
"""

import numpy as np
import torch
import torch.nn.functional as Fn
from numpy.typing import NDArray

from src.data_augmentors.abstract import DataAugmenter
from src.data_augmentors.utils import BetaStandardScaler
from src.methods.nets import device

# geometric amounts at strength 1
ROT_DEG, TRANS_FRAC = 10.0, 0.2
# colour amounts at strength 1; contrast/saturation are multiplicative factors around
# one, hue is an additive movement on the red-blue tint coordinate t
COLOR_AMOUNTS = {"brightness": 0.15, "contrast": 0.15, "saturation": 0.10, "hue": 0.025}
# the raw parameter value at which each colour op is the identity (factor 1, shift 0)
COLOR_IDENTITY = {"brightness": 1.0, "contrast": 1.0, "saturation": 1.0, "hue": 0.0}
COLOR_OPS = tuple(COLOR_AMOUNTS)
GEOMETRIC_OPS = ("translate", "rotation")
OPS = GEOMETRIC_OPS + COLOR_OPS

CHUNK: int = 16_384


def _scaler(name: str, amount: float) -> BetaStandardScaler:
    """The op's parameter range. hue is centred on 0, the factors on 1."""
    if name == "hue":
        return BetaStandardScaler(-amount, amount)
    return BetaStandardScaler(max(0.0, 1.0 - amount), 1.0 + amount)


def _ink_and_tint(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """x is (N,3,H,W) -> (ink, tint), both (N,H,W)."""
    ink = x[:, 0] + x[:, 2]
    eps = torch.finfo(x.dtype).eps
    tint = torch.where(ink > 0, x[:, 0] / ink.clamp_min(eps), torch.zeros_like(ink))
    return ink, tint


def _restore(x: torch.Tensor, ink: torch.Tensor, tint: torch.Tensor) -> torch.Tensor:
    """Rebuild the R/B channels; the empty green channel is left alone."""
    out = x.clone()
    out[:, 0] = tint * ink
    out[:, 2] = (1.0 - tint) * ink
    return out


class DoMNISTDA(DataAugmenter):
    """`DoMNISTDA('translate > rotation > contrast > saturation > hue')`.

    Every op runs batched on the device (~1000x the per-image torchvision loop, same
    affine + ink/tint maths). translate and rotation are fused into ONE affine and
    always applied first; the colour ops then run in the order named, since they do
    not commute.

    Exactly invariant: rotation is +-10 deg so it cannot turn a 2 into a 5,
    translation is +-20%, and the colour ops provably cannot change E[Y|do(x)] since
    colour never causes Y. The configured epsilon is finite-sample slack, not a real
    invariance defect.
    """

    exact_invariance = True

    def __init__(
        self,
        augmentations: str = "translate > rotation > contrast > saturation > hue",
        strength: float = 1.0,
        chunk: int = CHUNK,
        amounts: dict[str, float] | None = None,
    ):
        names: list[str] = [n for n in augmentations.replace(" ", "").split(">") if n]
        unknown = sorted(set(names) - set(OPS))
        if unknown:
            raise ValueError(f"unknown do-MNIST augmentation(s) {unknown}; valid: {sorted(OPS)}")
        # per-op overrides of the frozen colour amounts, e.g. {'hue': 0.4}; the
        # effective amounts are `amounts_`, which `_color` reads
        amounts = dict(amounts or {})
        bad = sorted(set(amounts) - set(COLOR_AMOUNTS))
        if bad:
            raise ValueError(f"unknown amounts {bad}; valid: {sorted(COLOR_AMOUNTS)}")
        for name, value in amounts.items():
            if float(value) < 0:
                raise ValueError(f"amount for {name} must be non-negative, got {value}")
        self.amounts = amounts
        self.amounts_ = {n: float(amounts.get(n, a)) for n, a in COLOR_AMOUNTS.items()}
        self.names, self.chunk = names, chunk
        self._strength = float(strength)

    @property
    def augmentation(self):
        return " > ".join(self.names)

    @property
    def strength(self) -> float:
        """One multiplier over every op's amount; the knob the omega/epsilon sweeps turn."""
        return self._strength

    @strength.setter
    def strength(self, value: float):
        self._strength = float(value)

    # ------------------------------------------------------------------ batched

    def _color(self, name: str, x: torch.Tensor, generator) -> tuple[torch.Tensor, torch.Tensor]:
        """One batched colour op, plus its standardised report column."""
        scaler = _scaler(name, self.amounts_[name] * self._strength)
        u = torch.rand(len(x), device=x.device, generator=generator)
        report = (u - float(scaler.mean)) / float(scaler.std)
        v = scaler.rescale(u)[:, None, None]  # U[0,1] -> the op's range

        ink, tint = _ink_and_tint(x)
        if name == "brightness":
            ink = (ink * v).clamp(0.0, 1.0)
        elif name == "contrast":
            # mean over FOREGROUND pixels only, per image; background stays exactly 0
            foreground = ink > 0
            mean = (ink.sum((1, 2)) / foreground.sum((1, 2)).clamp_min(1))[:, None, None]
            ink = torch.where(foreground, ((ink - mean) * v + mean).clamp(0.0, 1.0), ink)
        elif name == "saturation":
            tint = (0.5 + v * (tint - 0.5)).clamp(0.0, 1.0)
        else:  # hue
            tint = (tint + v).clamp(0.0, 1.0)
        return _restore(x, ink, tint), report

    def _augment_chunk(self, chunk: NDArray, dev, generator) -> tuple[NDArray, NDArray]:
        x = torch.as_tensor(chunk, dtype=torch.float, device=dev)
        n, params = len(x), []

        if "translate" in self.names:
            # integer pixels, as torchvision Translate does: sub-pixel bilinear shifts
            # blur high frequencies and wreck the covariance geometry
            width = x.shape[-1]
            tx, ty = (
                torch.round(
                    (torch.rand(2, n, device=dev, generator=generator) * 2 - 1) * TRANS_FRAC * self._strength * width
                )
                / width
            )
            params += [tx, ty]
        else:
            tx = ty = torch.zeros(n, device=dev)

        rotating = "rotation" in self.names
        angle = (
            (torch.rand(n, device=dev, generator=generator) * 2 - 1) * ROT_DEG * self._strength * np.pi / 180.0
            if rotating
            else torch.zeros(n, device=dev)
        )
        cos, sin = torch.cos(angle), torch.sin(angle)
        if rotating:
            params += [sin, cos]

        if "translate" in self.names or rotating:
            theta = torch.zeros(n, 2, 3, device=dev)
            theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = cos, -sin, 2 * tx
            theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = sin, cos, 2 * ty
            grid = Fn.affine_grid(theta, list(x.shape), align_corners=False)
            # nearest: torchvision F.rotate defaults to NEAREST, and bilinear blurring
            # moves tr(S)/k from 0.65 to 1.55
            x = Fn.grid_sample(x, grid, mode="nearest", padding_mode="zeros", align_corners=False)

        for name in self.names:
            if name in COLOR_OPS:
                x, report = self._color(name, x, generator)
                params.append(report)

        return (
            x.cpu().numpy().astype(np.float32),
            torch.stack(params, 1).cpu().numpy() if params else np.zeros((n, 0)),
        )

    # ------------------------------------------------------------------ mix-in

    def identity_params(self) -> NDArray:
        """G row of the identity element, in `_augment_chunk`'s column order: tx, ty = 0;
        (sin, cos) = (0, 1); each colour op's standardised report at its identity value."""
        p: list[float] = []
        if "translate" in self.names:
            p += [0.0, 0.0]
        if "rotation" in self.names:
            p += [0.0, 1.0]
        for name in self.names:
            if name in COLOR_OPS:
                p.append(float(_scaler(name, self.amounts_[name] * self._strength)(COLOR_IDENTITY[name])))
        return np.asarray(p, dtype=float)

    def mix_in(
        self, X: NDArray, GX: NDArray, G: NDArray | None, frac: float, seed, inplace: bool = False
    ) -> tuple[NDArray, NDArray | None, NDArray]:
        """Observed-data mix-in (Asm. 1b): the DA measure becomes (1-frac) p_X~ + frac p_X.

        Exactly round(frac*n) rows, chosen by np.random.default_rng(seed) (NOT the torch
        or the global numpy RNG), take their original X row and the identity G. Returns
        (GX_mixed, G_mixed, mask).

        frac == 0 returns (GX, G, all-False) untouched: same objects, no RNG draw, so the
        default path is bit-identical. inplace=True writes into GX (and G) instead of
        copying, for the big A draw once nothing else needs the unmixed pairs. It MUTATES
        GX/G, so use it only on arrays no non-DA consumer (ERM, PI, the PI+INV pairs)
        reads afterwards.
        """
        n = len(GX)
        if not 0.0 <= frac < 1.0:
            raise ValueError(f"mix_in fraction must be in [0, 1), got {frac}")
        if len(X) != n or (G is not None and len(G) != n):
            raise ValueError("X, GX and G must be row-aligned")
        mask = np.zeros(n, dtype=bool)
        if frac == 0:
            return GX, G, mask
        ident = self.identity_params()
        mask[np.random.default_rng(seed).choice(n, int(round(frac * n)), replace=False)] = True
        if not inplace:
            GX = GX.copy()
            G = None if G is None else G.copy()
        GX[mask] = X[mask]
        if G is not None:
            G[mask] = ident
        return GX, G, mask

    # ------------------------------------------------------------------ augment

    def augment(
        self, X: NDArray, strength: float | None = None, seed: int | None = None, **kwargs
    ) -> tuple[NDArray, NDArray]:
        """
        (GX, G) for X of shape (N,3,H,W).

        `seed` gives the batch its own torch Generator. Common random numbers across a
        knob grid need it: the draws here are torch, so seeding numpy alone does
        nothing. Left None the global torch stream is used.
        """
        if strength is not None:
            self.strength = strength

        dev = device()
        generator = None
        if seed is not None:
            generator = torch.Generator(device=dev)
            generator.manual_seed(int(seed))

        augmented, params = [], []
        for i in range(0, len(X), self.chunk):
            gx, g = self._augment_chunk(X[i : i + self.chunk], dev, generator)
            augmented.append(gx)
            params.append(g)
        return np.concatenate(augmented), np.concatenate(params)
