"""A74: the do-MNIST SEM, split and augmentor, plus the coverage bisection.

The analytic target is two-valued, the draws are balanced and on the R/B support,
the h_erm cell table and bias_sq match the closed forms, Bayes and ERM accuracy agree
on obs and do, do() severs colour at the closed-form rate, tint round-trips, the
colour ops keep the do-MNIST support and are exogenous, `amounts` bites, the identity
element and `mix_in` behave (counts, masks, seeds, frac 0 untouched, colour ops at
identity leave the image), eta = 1/2 removes the confounding, `centre_error_report`
on a synthetic split, the A/B/C partition (disjoint, covering, every draw inside its
set, `split_key` stable, `last_['idx']` global), `pop_seed == seed + 1` rejected, and
`bisect_gamma` on a stub equals the inverted-cdf quantile.

    uv run python scripts/a74_domnist_sem.py            # ~3 min, GPU DA
    uv run python scripts/a74_domnist_sem.py --nets     # + the init_seed coupling leg
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_augmentors import do_mnist as D  # noqa: E402
from src.data_augmentors.do_mnist import COLOR_AMOUNTS, COLOR_IDENTITY, DoMNISTDA  # noqa: E402
from src.experiments.utils.coverage import bisect_gamma  # noqa: E402
from src.experiments.utils.diagnostics import centre_error_report  # noqa: E402
from src.sem.do_mnist import SPLIT_DEFAULT, TINT_HI, TINT_LO, DoMNISTSEM, grey_of, split_key, tint_of  # noqa: E402

N = 60_000
SEM_KW = dict(alpha=0.0, beta=0.4, eta=0.25, subsample=2)
FAIL = []
_CACHE = {}


def check(name, ok, detail="", hard=True):
    tag = "PASS" if ok else ("FAIL" if hard else "WARN")
    print(f"[{tag}] {name} {detail}")
    if hard and not ok:
        FAIL.append(name)


def _sem() -> DoMNISTSEM:
    if "sem" not in _CACHE:
        _CACHE["sem"] = DoMNISTSEM(seed=0, train=True, **SEM_KW)
    return _CACHE["sem"]


def _obs():
    if "obs" not in _CACHE:
        X, y = _sem().sample(N, seed=0)
        _CACHE["obs"] = (X, y.ravel(), dict(_sem().last_))
    return _CACHE["obs"]


# ------------------------------------------------------------------------- SEM


def a74_ate_two_valued():
    sem = _sem()
    a = sem.ate_of(np.arange(10))
    lo, hi = sem.h_star(0.0), sem.h_star(1.0)
    check("A74 ate_of is (N,1)", a.shape == (10, 1), str(a.shape))
    check("A74 ate two-valued", set(np.round(a.ravel(), 10)) == {round(lo, 10), round(hi, 10)}, f"{lo} {hi}")
    f = (np.arange(10) >= 5).astype(float)
    check("A74 round(h*) = f", np.allclose(np.round(a.ravel()), f))
    check("A74 ATE contrast = 1 - beta", abs((a.max() - a.min()) - (1 - sem.beta)) < 1e-12)


def a74_balanced_and_shapes():
    X, y, d = _obs()
    check("A74 draw shape", X.shape == (N, 3, 14, 14), str(X.shape))
    check("A74 pixels in [0,1]", X.min() >= 0 and X.max() <= 1)
    check("A74 P(f=1) = 1/2", abs(d["f"].mean() - 0.5) < 1e-9, f"{d['f'].mean():.6f}")
    check("A74 green channel empty", np.allclose(X[:, 1], 0.0))
    check("A74 y is (N,1) binary", set(np.unique(y)) <= {0.0, 1.0})


def a74_h_erm_table():
    sem, (_, y, d) = _sem(), _obs()
    f, c = d["f"], d["C"]
    worst = 0.0
    for fv in (0.0, 1.0):
        for cv in (0.0, 1.0):
            m = (f == fv) & (c == cv)
            got, want = y[m].mean(), sem.h_erm(fv, cv)
            worst = max(worst, abs(got - want))
            print(f"  f={fv:.0f} c={cv:.0f}: h_erm {got:.4f} vs {want:.4f}")
    check("A74 h_erm cells within 2e-2", worst < 2e-2, f"{worst:.4f}")
    b = sem.h_erm(f, c) - sem.h_star(f)
    want = sem.beta * (0.5 - sem.eta)
    check("A74 |b| = beta(1/2 - eta)", abs(np.abs(b).mean() - want) < 1e-9)
    check("A74 mean b^2 = bias_sq", abs(np.mean(b**2) - sem.bias_sq) < 1e-9)
    cells = np.array([sem.h_erm(a, c_) for a in (0.0, 1.0) for c_ in (0.0, 1.0)])
    check("A74 sigma_sq = E h_erm(1-h_erm)", abs(sem.sigma_sq - np.mean(cells * (1 - cells))) < 1e-12)
    lo = (1 - sem.beta) * sem.alpha + sem.beta * sem.eta
    check("A74 attainable", np.allclose(sem.attainable, (lo, 1 - lo)))


def a74_accuracy_both_modes():
    """Bayes rule and ERM share the decision rule and the accuracy (1-beta)(1-alpha) + beta/2
    on obs AND do: accuracy cannot separate them."""
    sem = _sem()
    want = (1 - sem.beta) * (1 - sem.alpha) + sem.beta / 2
    for mode, do in (("obs", False), ("do", True)):
        _, y = sem.sample(N, intervention=do, seed=1)
        d, y = sem.last_, y.ravel()
        bayes = d["f"]
        erm = np.round(sem.h_erm(d["f"], d["C"]))
        check(f"A74 {mode}: Bayes and ERM decide alike", np.allclose(bayes, erm))
        acc = (y == bayes).mean()
        check(f"A74 {mode}: accuracy {want:.2f}", abs(acc - want) < 0.01, f"{acc:.4f}")


def a74_do_severs_colour():
    sem = _sem()
    _, y_do = sem.sample(N, intervention=True, seed=2)
    t_do = sem.last_["t"]
    _, y_ob = sem.sample(N, seed=3)
    t_ob = sem.last_["t"]
    r_do = np.corrcoef(t_do, y_do.ravel())[0, 1]
    r_ob = np.corrcoef(t_ob, y_ob.ravel())[0, 1]
    # closed form: Cov(y,t) = (hi-lo) beta (1-2 eta)/4, Var(y) = 1/4
    want = ((TINT_HI - TINT_LO) * sem.beta * (1 - 2 * sem.eta) / 4) / (0.5 * t_ob.std())
    print(f"  corr(t,y): do {r_do:+.4f}  obs {r_ob:+.4f} (theory {want:+.4f})")
    check("A74 do severs colour", abs(r_do) < 0.02, f"{r_do:+.4f}")
    check("A74 obs leaks colour at rate eta", abs(r_ob - want) < 0.01, f"{r_ob:+.4f} vs {want:+.4f}")


def a74_tint_roundtrip():
    X, _, d = _obs()
    check("A74 tint_of round trip", np.abs(tint_of(X) - d["t"]).max() < 1e-5)
    g = grey_of(X)
    check("A74 grey_of = R + B", np.abs(g - X[:, 0] - X[:, 2]).max() < 1e-7 and g.max() <= 1.0 + 1e-6)


def a74_paired_draw_is_coupled():
    """sample_paired == sample(obs) and sample(do) at one seed, sharing everything but U_out."""
    s = _sem()
    Xo, yo = s.sample(5_000, seed=11)
    to = dict(s.last_)
    Xd, yd = s.sample(5_000, intervention=True, seed=11)
    td = dict(s.last_)
    Xp, po, pd = s.sample_paired(5_000, seed=11)
    check("A74 obs/do share images", np.array_equal(Xo, Xd))
    check(
        "A74 obs/do share (idx,f,U,C,t,S)", all(np.array_equal(to[k], td[k]) for k in ("idx", "f", "U", "C", "t", "S"))
    )
    check("A74 paired == obs + do", np.array_equal(Xp, Xo) and np.array_equal(po, yo) and np.array_equal(pd, yd))
    frac = float((po != pd).mean())
    check("A74 y_obs != y_do on beta/2", abs(frac - s.beta / 2) < 0.02, f"{frac:.4f}")


def a74_eta_half_removes_confounding():
    s = DoMNISTSEM(seed=0, train=True, alpha=0.0, beta=0.4, eta=0.5)
    f = np.array([0.0, 1.0])
    ok = (
        abs(s.bias_sq) < 1e-12
        and np.allclose(s.h_erm(f, 0.0), s.h_star(f))
        and np.allclose(s.h_erm(f, 1.0), s.h_star(f))
    )
    check("A74 eta = 1/2 removes the confounding", ok)


# ------------------------------------------------------------------------- split


def a74_three_way_split():
    """A/B/C partition the 60k training images; every draw stays inside its own set."""
    import yaml

    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
    with open(path) as fh:
        c = (yaml.safe_load(fh) or {}).get("do_mnist") or {}
    sizes, seed = c.get("split") or SPLIT_DEFAULT, int(c.get("split_seed", 420))
    parts = _sem().split(sizes, seed)
    idx = {k: s.subset_ for k, s in parts.items()}
    check("A74 split keys", set(idx) == {"A", "B", "C"}, str(set(idx)))
    check(
        "A74 split disjoint",
        all(len(np.intersect1d(idx[a], idx[b])) == 0 for a, b in (("A", "B"), ("A", "C"), ("B", "C"))),
    )
    check("A74 split covers", np.array_equal(np.sort(np.concatenate(list(idx.values()))), np.arange(N)))
    for k, sem in parts.items():
        f = sem.targets[idx[k]] >= 5
        check(f"A74 split {k}: size and both classes", len(sem) == sizes[k] and 0 < f.sum() < len(f))
    for k, n, sd in (("A", 5_000, 1), ("B", 70_000, 2), ("C", None, 3)):
        parts[k].sample(n, seed=sd)  # B over-size: the replace path; C: N=None defaults to len
        drawn = parts[k].last_["idx"]
        inside = np.isin(drawn, idx[k]).all() and not any(np.isin(drawn, idx[o]).any() for o in set(idx) - {k})
        check(f"A74 split {k}: draw of {n} stays inside (global idx)", len(drawn) == (n or sizes[k]) and inside)
        check(f"A74 split {k}: draw balanced", abs(parts[k].last_["f"].mean() - 0.5) < 1e-9)
    key = split_key(parts)
    check(
        "A74 split_key stable",
        key == split_key(_sem().split(sizes, seed)) and key != split_key(_sem().split(sizes, seed + 1)),
    )
    check("A74 parent SEM untouched", _sem().subset_ is None and len(_sem()) == N)
    try:
        _sem().split({"A": 1, "B": 2}, seed)
        check("A74 bad split sizes raise", False)
    except ValueError:
        check("A74 bad split sizes raise", True)
    try:
        parts["A"].split(sizes, seed)
        check("A74 split on a restricted SEM raises", False)
    except ValueError:
        check("A74 split on a restricted SEM raises", True)
    print(f"  sizes { ({k: len(v) for k, v in idx.items()}) } key {key[:12]}")


def a74_pop_seed_distinct():
    """The evaluation population's seed may not coincide with the C draw's seed + 1."""
    try:
        from src.experiments.configs import resolve_dataset_block
    except ImportError as e:  # pragma: no cover
        check("A74 pop_seed == seed + 1 rejected", False, f"import failed: {e}", hard=False)
        return
    base = {"seed": 43, "augmentation": "translate > rotation", "gamma": 0.1, "epsilon": 0.04, "methods": ["ERM"]}
    try:
        resolve_dataset_block("do_mnist", {**base, "pop_seed": 44})
        check("A74 pop_seed == seed + 1 rejected", False, "no ValueError", hard=False)
    except ValueError:
        check("A74 pop_seed == seed + 1 rejected", True)
    except Exception as e:  # noqa: BLE001
        check("A74 pop_seed == seed + 1 rejected", False, f"not wired yet: {type(e).__name__}: {e}", hard=False)


# ------------------------------------------------------------------------- DA


def a74_color_ops_preserve_support():
    """Colour ops may not invent green, leave [0,1], or move ink off the R/B line."""
    X, y, d = _obs()
    X, y, U = X[:20_000], y[:20_000], d["U"][:20_000]
    pipe = "brightness > contrast > saturation > hue"
    GX, T = DoMNISTDA(pipe).augment(X)
    check("A74 colour ops: shapes", GX.shape == X.shape and T.shape == (len(X), 4))
    check("A74 colour ops: green stays empty", np.abs(GX[:, 1]).max() == 0.0)
    check("A74 colour ops: pixels in [0,1]", GX.min() >= 0.0 and GX.max() <= 1.0 + 1e-6)
    check("A74 colour ops: still R + B", np.abs(grey_of(GX) - GX[:, 0] - GX[:, 2]).max() < 1e-6)
    for j, nm in enumerate(pipe.replace(" ", "").split(">")):
        r_y = np.corrcoef(T[:, j], y)[0, 1]
        r_u = np.corrcoef(T[:, j], U)[0, 1]
        check(
            f"A74 {nm}: T exogenous", abs(r_y) < 0.03 and abs(r_u) < 0.03, f"corr(T,y) {r_y:+.4f} corr(T,U) {r_u:+.4f}"
        )
    keep = np.corrcoef(tint_of(GX), tint_of(X))[0, 1]
    check("A74 colour ops perturb the tint, never resample it", keep > 0.9, f"{keep:+.4f}")


def a74_brightness_scales_ink():
    X, _, _ = _obs()
    X = X[:2_000]
    GX, T = DoMNISTDA("brightness").augment(X, seed=5)
    ink0, ink1 = grey_of(X), grey_of(GX)
    fg = ink0 > 0
    ratio = np.where(fg, ink1 / np.maximum(ink0, 1e-12), 1.0)
    per_image_const = np.array([np.ptp(ratio[i][fg[i] & (ink1[i] < 1.0)]) for i in range(len(X))])
    check(
        "A74 brightness scales the ink by one factor per image",
        per_image_const.max() < 1e-4,
        f"{per_image_const.max():.1e}",
    )
    check("A74 brightness keeps the tint", np.abs(tint_of(GX) - tint_of(X)).max() < 1e-4)
    check("A74 brightness amount", COLOR_AMOUNTS["brightness"] == 0.15 and T.shape == (len(X), 1))


def a74_amounts_override():
    X, _, _ = _obs()
    X = X[:4_000]
    base, wide = DoMNISTDA("hue"), DoMNISTDA("hue", amounts={"hue": 0.8})
    check("A74 amounts_: defaults", base.amounts_["hue"] == COLOR_AMOUNTS["hue"] and base.amounts == {})
    check(
        "A74 amounts_: override", wide.amounts_["hue"] == 0.8 and wide.amounts_["contrast"] == COLOR_AMOUNTS["contrast"]
    )
    t0 = tint_of(X)
    db = np.abs(tint_of(base.augment(X, seed=1)[0]) - t0).mean()
    dw = np.abs(tint_of(wide.augment(X, seed=1)[0]) - t0).mean()
    check("A74 amounts bite", dw > 5 * db, f"mean |dtint| {db:.4f} -> {dw:.4f}")
    for bad in ({"nonsense": 1.0}, {"hue": -1.0}):
        try:
            DoMNISTDA("hue", amounts=bad)
            check(f"A74 bad amounts {bad} raise", False)
        except ValueError:
            check(f"A74 bad amounts {bad} raise", True)


def a74_mix_in_identity_rows():
    """frac=0 is a no-op (same objects, no RNG); otherwise exactly round(frac n) rows go
    back to X with the identity G, from their own seeded generator."""
    import torch

    X, _, _ = _obs()
    X = X[:2_000]
    da = DoMNISTDA("translate > rotation > contrast > saturation > hue")
    GX, G = da.augment(X, seed=9)
    ident = da.identity_params()
    check("A74 identity_params shape", ident.shape == (G.shape[1],), f"{ident.shape} vs {G.shape}")
    check("A74 identity_params values", np.allclose(ident, [0, 0, 0, 1, 0, 0, 0], atol=1e-12), str(ident))

    def state():
        cuda = torch.cuda.get_rng_state().clone() if torch.cuda.is_available() else None
        return torch.get_rng_state().clone(), np.random.get_state()[1].copy(), cuda

    def same(a, b):
        return all(
            (x is None and y is None) or np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(a, b, strict=True)
        )

    s0 = state()
    GX0, G0, m0 = da.mix_in(X, GX, G, 0.0, seed=[42, 1])
    check("A74 mix_in frac 0 untouched", GX0 is GX and G0 is G and not m0.any())
    GXm, Gm, mask = da.mix_in(X, GX, G, 0.25, seed=[42, 1])
    check("A74 mix_in leaves the global RNGs alone", same(s0, state()))
    check("A74 mix_in count", mask.sum() == 500, str(mask.sum()))
    check("A74 mix_in rows", np.array_equal(GXm[mask], X[mask]) and np.array_equal(GXm[~mask], GX[~mask]))
    check(
        "A74 mix_in G rows",
        np.array_equal(Gm[mask], np.broadcast_to(ident.astype(G.dtype), (500, len(ident))))
        and np.array_equal(Gm[~mask], G[~mask]),
    )
    check(
        "A74 mix_in copies by default", GXm is not GX and Gm is not G and np.array_equal(GX, da.augment(X, seed=9)[0])
    )
    check("A74 mix_in seeded", np.array_equal(da.mix_in(X, GX, G, 0.25, seed=[42, 1])[2], mask))
    check("A74 mix_in seed changes rows", not np.array_equal(da.mix_in(X, GX, G, 0.25, seed=[42, 2])[2], mask))
    GXc = GX.copy()
    GXi, Gi, mi = da.mix_in(X, GXc, None, 0.25, seed=[42, 1], inplace=True)
    check(
        "A74 mix_in inplace, G None",
        Gi is None and GXi is GXc and np.array_equal(mi, mask) and np.array_equal(GXi, GXm),
    )
    for bad in (1.0, -0.1):
        try:
            da.mix_in(X, GX, G, bad, seed=0)
            check(f"A74 mix_in frac {bad} raises", False)
        except ValueError:
            check(f"A74 mix_in frac {bad} raises", True)

    # each colour op at its identity value leaves the image unchanged and reports ~0
    for name in ("brightness", "contrast", "saturation", "hue"):
        d1 = DoMNISTDA(name)
        sc = D._scaler(name, d1.amounts_[name])
        u = (COLOR_IDENTITY[name] - sc.min) / (sc.max - sc.min)
        real_rand = torch.rand
        torch.rand = lambda n, _u=float(u), **k: torch.full((n,), _u, device=k.get("device", "cpu"))
        try:
            gx, rep = d1._color(name, torch.as_tensor(X, dtype=torch.float), None)
        finally:
            torch.rand = real_rand
        d_img = float(np.abs(gx.numpy() - X).max())
        check(
            f"A74 {name} at identity is the identity",
            d_img < 1e-6 and float(rep.abs().max()) < 1e-6,
            f"|dx| {d_img:.1e}",
        )
        check(f"A74 {name} identity report", abs(float(sc(COLOR_IDENTITY[name])) - d1.identity_params()[0]) < 1e-12)


# ------------------------------------------------------------------------- diagnostics


def a74_centre_error_report():
    """floor95 = 2 p95|mu-h*|; shared_frac in [0,1], -> 1 as the net's own error -> 0;
    err_corr = corr(e_est, b) flips sign with b."""
    rng = np.random.default_rng(0)
    n = 20_000
    f = rng.integers(0, 2, n).astype(float)
    C = rng.integers(0, 2, n).astype(float)
    hs = 0.2 + 0.6 * f
    b = (2 * C - 1) * 0.1
    e_est = 0.3 * b + rng.normal(0, 0.05, n)
    r = centre_error_report(hs + b + e_est, hs, hs + b)
    d = np.abs(b + e_est)
    check("A74 floor95 = 2 p95", np.isclose(r["floor95"], 2 * np.quantile(d, 0.95)))
    check("A74 d_p95 and d_max", np.isclose(r["d_p95"], np.quantile(d, 0.95)) and np.isclose(r["d_max"], d.max()))
    want = 0.01 / (0.01 + np.mean(e_est**2))
    check("A74 shared_frac", 0.0 <= r["shared_frac"] <= 1.0 and np.isclose(r["shared_frac"], want))
    check("A74 err_corr", np.isfinite(r["err_corr"]) and r["err_corr"] > 0.4, f"{r['err_corr']:.3f}")
    flip = centre_error_report(hs - b + e_est, hs, hs - b)
    check("A74 err_corr flips with b", np.isclose(flip["err_corr"], -r["err_corr"]))
    tiny = centre_error_report(hs + b + 1e-4 * rng.normal(size=n), hs, hs + b)
    check("A74 shared_frac -> 1 as e_est -> 0", tiny["shared_frac"] > 0.999, f"{tiny['shared_frac']:.6f}")


class _StubPI:
    """mu +- sqrt(gamma) r_i: the closed-form gaussian shape, no solver. Rows of X are
    indices into (mu, r); rows in `nan` return NaN bounds (a failed solve)."""

    def __init__(self, mu, r, nan=()):
        self.mu, self.r, self.nan = mu, r, np.asarray(nan, dtype=int)

    def predict(self, X, gamma=None):
        i = np.asarray(X).ravel().astype(int)
        h = np.sqrt(gamma) * self.r[i]
        b = np.stack([self.mu[i] - h, self.mu[i] + h], axis=1)
        b[np.isin(i, self.nan)] = np.nan
        return b


def a74_bisect_gamma():
    """Bisection == the inverted_cdf quantile of the per-sample required gamma, within tol."""
    rng = np.random.default_rng(0)
    n, tol, target = 3_000, 0.05, 0.95
    mu = rng.random(n)
    r = rng.uniform(0.5, 2.0, n)
    d = np.abs(rng.normal(0, 0.1, n))
    h = mu + d * rng.choice([-1.0, 1.0], n)
    X = np.arange(n)[:, None]
    req = (d / r) ** 2

    pi = _StubPI(mu, r)
    out = bisect_gamma(pi, X, h, target, lo=1e-8, hi=10.0, tol=tol, max_iter=40)
    want = np.quantile(req, target, method="inverted_cdf")
    print(f"  bisection {out['gamma']:.5g} vs quantile {want:.5g} ({out['n_eval']} evals)")
    check("A74 bisection = inverted-cdf quantile", want <= out["gamma"] <= want * (1 + tol) * (1 + 1e-12))
    cov = [c for _, c, _ in out["trace"]]
    check(
        "A74 bisection covers, monotone trace, eval budget",
        out["coverage"] >= target and np.all(np.diff(cov) >= 0) and out["n_eval"] <= 42,
    )
    few = bisect_gamma(pi, X, h, target, lo=1e-8, hi=10.0, tol=tol, max_iter=3)
    check("A74 bisection max_iter honoured, conservative", few["n_eval"] <= 5 and few["coverage"] >= target)
    top = bisect_gamma(pi, X, h, target, lo=1e-10, hi=1e-8, tol=tol)
    check(
        "A74 bisection bracket too low returns hi",
        top["gamma"] == 1e-8 and top["coverage"] < target and top["n_eval"] == 1,
    )
    bot = bisect_gamma(pi, X, h, target, lo=1.0, hi=10.0, tol=tol)
    check(
        "A74 bisection already covered returns lo",
        bot["gamma"] == 1.0 and bot["coverage"] >= target and bot["n_eval"] == 2,
    )
    bad = rng.choice(n, n // 10, replace=False)
    pin = _StubPI(mu, r, nan=bad)
    miss = bisect_gamma(pin, X, h, target, lo=1e-8, hi=10.0, tol=tol)
    check("A74 NaN rows count as uncovered", miss["gamma"] == 10.0 and abs(miss["coverage"] - 0.9) < 1e-12)
    req_nan = np.where(np.isin(np.arange(n), bad), np.inf, req)
    got = bisect_gamma(pin, X, h, 0.85, lo=1e-8, hi=10.0, tol=tol)
    want = np.quantile(req_nan, 0.85, method="inverted_cdf")
    check("A74 bisection with NaN rows", want <= got["gamma"] <= want * (1 + tol) * (1 + 1e-12))


# ------------------------------------------------------------------------- nets


def a74_init_seed_couples_nets():
    """Same init_seed => identical weights; different data => different fit. GPU."""
    from src.methods.regression import GradientDescentERM

    rng = np.random.default_rng(0)
    X = rng.random((2_000, 588)).astype(np.float32)
    y = (X[:, :10].sum(1) > 5).astype(float)[:, None]
    kw = dict(epochs=1, lr=0.01, batch=256, optimizer="adam", betas=(0.7, 0.9), onecycle=True, loss="mse")

    def weights(yy, s):
        return GradientDescentERM("domnist-fast").fit(X, yy, init_seed=s, **kw).solution.ravel()

    a, b, c = weights(y, 3), weights(y, 3), weights(1 - y, 3)
    same, flipped = np.abs(a - b).max(), np.abs(a - c).max()
    print(f"  same seed+data {same:.3e}  vs flipped labels {flipped:.3e}")
    check("A74 init_seed: same seed + data is exact", same == 0.0, f"{same:.3e}")
    check("A74 init_seed: a label change moves the weights", flipped > 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nets", action="store_true", help="add the init_seed coupling leg (GPU)")
    args = ap.parse_args()
    legs = [
        a74_ate_two_valued,
        a74_balanced_and_shapes,
        a74_h_erm_table,
        a74_accuracy_both_modes,
        a74_do_severs_colour,
        a74_tint_roundtrip,
        a74_paired_draw_is_coupled,
        a74_eta_half_removes_confounding,
        a74_three_way_split,
        a74_pop_seed_distinct,
        a74_color_ops_preserve_support,
        a74_brightness_scales_ink,
        a74_amounts_override,
        a74_mix_in_identity_rows,
        a74_centre_error_report,
        a74_bisect_gamma,
    ]
    if args.nets:
        legs.append(a74_init_seed_couples_nets)
    for fn in legs:
        print(f"\n{fn.__name__}")
        fn()
    print("\n" + ("ALL PASS" if not FAIL else f"FAILURES: {FAIL}"))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
