"""A78: the do-MNIST ERM+INV centre (`InvariantGradientDescentERM`, `inv_recenter: inv`).

Static legs (no MNIST; the tiny AL fit runs on whatever `device()` picks, seconds):

  (i)   the registry: PI+INV is `InvarianceConstrainedCopSens` on `nets["X"]` under
        `off`, `RecentredInvCopSens` on `nets["GX"]` under `on`, and
        `InvarianceConstrainedCopSens` on `nets["INV"]` under `inv`, each taking
        sigma-hat from `nets["X"]` (`sigma_model`, PI's radius); `ERM+INV` is the
        INV net itself; a bogus `inv_recenter` raises when PI+INV is built.
  (ii)  nesting: `nested_in("inv")` has no PI+INV entry (off/on keep theirs), and
        `log_nesting` under `inv` stays quiet on a PI+INV wider than PI.
  (iii) the AL schedule: each epoch's batches split into `updates_per_epoch`
        near-equal windows, so both the 60k and the 1.2M draw update exactly that
        many times per epoch (one per batch when an epoch is shorter), and a tiny
        fit at two sizes updates at exactly the window ends.
  (iv)  the plumbing: `erm_inv_fit_kwargs` carries the `DOMNIST_CONFIG` constants and
        the epoch override, which `train_erm_inv` honours over the pair's own
        `epochs`, the default `erm_inv_tau` is 4e-4, and the orchestrator
        trains the ERM+INV net iff PI+INV runs under `inv` or ERM+INV is listed.

GPU legs (`--nets`, a 60k draw, about two minutes, then (vii)):

  (v)   the AL fit is deterministic at a fixed `init_seed`; the replicate's ERM+INV
        net has a lower invariance error than the ERM on the held-out B pairs;
        `draw_replicate(train_inv=True)` leaves the ERM and DA+ERM nets (state sha1)
        and the B arrays (`X`, `GX`, `GX_inv`, `y`, `G`) bit-identical to
        `train_inv=False`; an `inv_fits` hook that fits the configured knobs returns
        a net whose sha1 equals the plain `train_inv=True` path's; PI+INV's sigma2_
        equals PI's, the ERM's clipped mean mu(1-mu) on the B rows, under off, on
        and inv.

Static leg (the pooled head, seconds):

  (vi)  `NETS` carries `domnist-fast` and `domnist-pool`; on a 3x14x14 input the
        pooled net ends in global average pooling and a 64-unit dense head (no
        flatten over positions), the flat net does not pool, both map to (0, 1),
        and the orchestrator's `net` defaults to the flat head and takes the block's.

GPU leg (`--nets`, two more 60k replicates, about two minutes):

  (vii) the flat head is bit-identical to 421549d: the (v) replicate's ERM, DA+ERM
        and ERM+INV state sha1s equal the pins recorded there (L40S, CUDA); under
        `net: domnist-pool` all three nets carry the pooled head, a second replicate
        reproduces their sha1s, they differ from the flat ones, and the B arrays
        equal the flat replicate's (the head changes the nets and nothing else).

    uv run python scripts/a78_domnist_erm_inv.py            # static legs
    uv run python scripts/a78_domnist_erm_inv.py --nets     # + the GPU leg
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger  # noqa: E402

from src.experiments.configs import DOMNIST_CONFIG, MethodRegistry, resolve_dataset_block  # noqa: E402
from src.experiments.do_mnist import (  # noqa: E402
    DoMNISTOrchestrator,
    draw_replicate,
    erm_inv_fit_kwargs,
    log_nesting,
    nested_in,
    train_erm_inv,
)
from src.methods.copsens import InvarianceConstrainedCopSens, RecentredInvCopSens  # noqa: E402
from src.methods.regression import InvariantGradientDescentERM  # noqa: E402

FAIL = []
MINIMAL = dict(seed=42, augmentation="translate > rotation > contrast > saturation > hue", gamma=0.085, epsilon=0.04)
TRAIN_KW = dict(lr=0.01, batch=256, epochs=1, optimizer="adam", betas=(0.7, 0.9), onecycle=True, loss="mse")


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail and not ok else ""))
    if not ok:
        FAIL.append(name)


def leg_i():
    print("(i) the registry dispatches PI+INV on the centre")
    nets = {"X": object(), "GX": object(), "INV": object()}

    def build(recenter, name="PI+INV"):
        return MethodRegistry.build_methods(
            [name], gamma=0.085, epsilon=0.04, backend="copsens", outcome_models=nets, inv_recenter=recenter
        )[name]()

    off, on, inv = build("off"), build("on"), build("inv")
    check("(i) off: InvarianceConstrainedCopSens on X", type(off) is InvarianceConstrainedCopSens)
    check("(i) off: centre is nets['X']", off.outcome_model is nets["X"])
    check("(i) on: RecentredInvCopSens on GX", type(on) is RecentredInvCopSens and on.outcome_model is nets["GX"])
    check("(i) inv: InvarianceConstrainedCopSens", type(inv) is InvarianceConstrainedCopSens)
    check("(i) inv: centre is nets['INV']", inv.outcome_model is nets["INV"])
    check("(i) inv: the block epsilon is the budget", inv.epsilon == 0.04 and inv._budget() == 0.04**2)
    erm_sigma = all(m.sigma_model is nets["X"] for m in (off, on, inv))
    check("(i) PI+INV's sigma model is nets['X'] under off, on and inv", erm_sigma)
    check("(i) PI carries no sigma model", build("off", "PI").sigma_model is None)
    check("(i) ERM+INV is the INV net", build("off", "ERM+INV") is nets["INV"])
    try:
        build("fallback")
        raised = False
    except ValueError:
        raised = True
    check("(i) a bogus inv_recenter raises when PI+INV is built", raised)


def leg_ii():
    print("(ii) nesting")
    check("(ii) nested_in('inv') has no PI+INV", "PI+INV" not in nested_in("inv"))
    check("(ii) nested_in('off') nests PI+INV in PI", nested_in("off")["PI+INV"] == "PI")
    check("(ii) nested_in('on') nests PI+INV in DA+PI", nested_in("on")["PI+INV"] == "DA+PI")
    check(
        "(ii) the other entries do not move",
        {k: v for k, v in nested_in("off").items() if k != "PI+INV"} == nested_in("inv"),
    )
    bounds = {"PI": np.array([[[0.4, 0.6]]] * 3), "PI+INV": np.array([[[0.1, 0.9]]] * 3)}
    messages = []
    sink = logger.add(lambda message: messages.append(str(message)), level="WARNING")
    try:
        log_nesting(bounds, "inv")
        quiet = not any("PI+INV wider" in m for m in messages)
        log_nesting(bounds, "off")
        loud = any("PI+INV wider" in m for m in messages)
    finally:
        logger.remove(sink)
    check("(ii) log_nesting under inv is quiet on PI+INV", quiet)
    check("(ii) log_nesting under off still warns", loud)


def leg_iii():
    print("(iii) the AL windows split each epoch evenly")
    ends_of = InvariantGradientDescentERM.al_window_ends
    upe = DOMNIST_CONFIG.erm_inv_updates_per_epoch
    for n in (60_000, 1_200_000):
        n_batches = int(np.ceil(n / 256))
        ends = ends_of(n_batches, upe)
        sizes = np.diff([-1, *ends])
        check(f"(iii) n={n:,}: exactly {upe} updates per epoch", len(ends) == upe, str(len(ends)))
        check(f"(iii) n={n:,}: the last window ends the epoch", ends[-1] == n_batches - 1)
        check(f"(iii) n={n:,}: window sizes differ by at most 1", sizes.max() - sizes.min() <= 1, str(sizes))
    check("(iii) fewer batches than updates: one per batch", ends_of(3, 20) == [0, 1, 2])

    rng = np.random.default_rng(0)
    for n_batches, epochs in ((10, 1), (107, 2)):
        n = 64 * n_batches
        X = rng.random((n, 3 * 14 * 14), dtype=np.float32)
        GX = np.clip(X + 0.2 * rng.standard_normal(X.shape).astype(np.float32), 0, 1)
        y = (X[:, :50].mean(axis=1) > 0.5).astype(np.float32)
        kw = erm_inv_fit_kwargs({**TRAIN_KW, "batch": 64}, 1e-4, epochs=epochs)
        net = InvariantGradientDescentERM("domnist-fast").fit(X, y, GX=GX, init_seed=0, **kw)
        per_epoch = [end + 1 for end in ends_of(n_batches, upe)]
        expected = [e * n_batches + step for e in range(epochs) for step in per_epoch]
        steps = [row[0] for row in net.al_trace_]
        check(f"(iii) {n_batches} batches x {epochs} epochs: updates at the window ends", steps == expected, str(steps))
        check(
            f"(iii) {n_batches} batches x {epochs} epochs: {min(upe, n_batches)} updates per epoch",
            len(steps) == min(upe, n_batches) * epochs,
            str(len(steps)),
        )
        mus = [row[3] for row in net.al_trace_]
        capped = max(mus) <= DOMNIST_CONFIG.erm_inv_mu_max
        check("(iii) mu never shrinks and stays capped", mus == sorted(mus) and capped)
        check("(iii) lam stays non-negative", min(row[2] for row in net.al_trace_) >= 0.0)


def leg_iv():
    print("(iv) the plumbing")
    kw = erm_inv_fit_kwargs(TRAIN_KW, 4e-4)
    check("(iv) the default erm_inv_tau is 4e-4", DOMNIST_CONFIG.erm_inv_tau == 4e-4)
    check(
        "(iv) the AL constants reach the fit",
        kw["al_tau"] == 4e-4
        and kw["al_mu0"] == DOMNIST_CONFIG.erm_inv_mu0
        and kw["al_growth"] == DOMNIST_CONFIG.erm_inv_growth
        and kw["al_updates_per_epoch"] == DOMNIST_CONFIG.erm_inv_updates_per_epoch
        and kw["al_mu_max"] == DOMNIST_CONFIG.erm_inv_mu_max,
    )
    import inspect

    defaults = inspect.signature(InvariantGradientDescentERM._fit).parameters
    check(
        "(iv) the class defaults are the run's AL constants",
        defaults["al_tau"].default == DOMNIST_CONFIG.erm_inv_tau
        and defaults["al_mu0"].default == DOMNIST_CONFIG.erm_inv_mu0
        and defaults["al_growth"].default == DOMNIST_CONFIG.erm_inv_growth
        and defaults["al_updates_per_epoch"].default == DOMNIST_CONFIG.erm_inv_updates_per_epoch
        and defaults["al_mu_max"].default == DOMNIST_CONFIG.erm_inv_mu_max,
    )
    expected = TRAIN_KW["epochs"] if DOMNIST_CONFIG.erm_inv_epochs is None else DOMNIST_CONFIG.erm_inv_epochs
    check("(iv) the epoch count is erm_inv_epochs, else the pair's", kw["epochs"] == expected)
    check("(iv) an explicit epoch count wins", erm_inv_fit_kwargs(TRAIN_KW, 4e-4, epochs=2)["epochs"] == 2)
    # the pair's dict carries `epochs` too: train_erm_inv must not let it shadow the
    # override (20 batches, 20 updates per epoch, so the trace length counts the epochs)
    rng = np.random.default_rng(1)
    X = rng.random((64 * 20, 3 * 14 * 14), dtype=np.float32)
    GX, y = X[::-1].copy(), (X[:, 0] > 0.5).astype(np.float32)
    small = {**TRAIN_KW, "batch": 64}
    two = train_erm_inv(X, GX, y, 0, 1e-3, small, epochs=2)
    check("(iv) train_erm_inv honours the epoch override over the pair's", len(two.al_trace_) == 40)
    default = train_erm_inv(X, GX, y, 0, 1e-3, small)
    check("(iv) train_erm_inv defaults to erm_inv_epochs, else the pair's", len(default.al_trace_) == 20 * expected)
    cases = (
        ("inv", ["PI", "PI+INV"], True),
        ("off", ["PI", "PI+INV"], False),
        ("on", ["PI", "PI+INV"], False),
        ("off", ["ERM", "ERM+INV"], True),
        ("inv", ["PI", "DA+PI"], False),
    )
    for recenter, methods, expected in cases:
        block = resolve_dataset_block("do_mnist", {**MINIMAL, "methods": methods, "inv_recenter": recenter, "im-ci": 0})
        block.pop("experiment", None)
        orchestrator = DoMNISTOrchestrator(**block, hyperparameters=TRAIN_KW)
        check(f"(iv) train_inv is {expected} for {recenter} {methods}", orchestrator.train_inv is expected)
        check(
            f"(iv) the runner kwargs carry train_inv for {recenter} {methods}",
            orchestrator._runner_kwargs()["train_inv"] is expected,
        )
    block = resolve_dataset_block("do_mnist", {**MINIMAL, "methods": ["PI"], "erm_inv_tau": 1e-3, "im-ci": 0})
    block.pop("experiment", None)
    check(
        "(iv) the block's erm_inv_tau reaches the runner",
        DoMNISTOrchestrator(**block)._runner_kwargs()["erm_inv_tau"] == 1e-3,
    )


def leg_v():
    print("(v) the replicate, GPU (60k draw)")
    from src.experiments.utils import set_seed

    block = resolve_dataset_block(
        "do_mnist", {**MINIMAL, "methods": ["PI", "PI+INV"], "inv_recenter": "inv", "n_samples": 60_000, "im-ci": 0}
    )
    block.pop("experiment", None)
    orchestrator = DoMNISTOrchestrator(**block, hyperparameters=TRAIN_KW)
    tau = orchestrator.erm_inv_tau

    def replicate(net=orchestrator.net, **kwargs):
        set_seed(block["seed"])
        return draw_replicate(
            orchestrator._sem_factory(),
            orchestrator._sem_test_factory(),
            orchestrator._da_factory(),
            seed=block["seed"],
            n_samples=60_000,
            n_pi=6_000,
            mix_in=orchestrator.mix_in,
            hyperparameters=TRAIN_KW,
            net=net,
            split=orchestrator.split,
            split_seed=orchestrator.split_seed,
            **kwargs,
        )

    plain = replicate()
    with_inv = replicate(train_inv=True, erm_inv_tau=tau)
    fitted = {}

    def hook(X, GX, y, init_seed, train_kw):
        # the diagnostic pattern: another point first, then the configured one
        train_erm_inv(X, GX, y, init_seed, tau * 10, train_kw)
        fitted["net"] = train_erm_inv(X, GX, y, init_seed, tau, train_kw)
        fitted["pairs"] = (X.copy(), GX.copy(), np.asarray(y).copy())
        return fitted["net"]

    hooked = replicate(inv_fits=hook)

    for key in ("X", "GX"):
        same = plain.nets[key].state_sha1() == with_inv.nets[key].state_sha1() == hooked.nets[key].state_sha1()
        check(f"(v) nets['{key}'] is bit-identical with and without the ERM+INV fit", same)
    for field in ("X", "GX", "GX_inv", "y", "G", "mask_a", "mask_b"):
        a, b, c = (np.asarray(getattr(d, field)) for d in (plain, with_inv, hooked))
        check(f"(v) the B array {field} is bit-identical", np.array_equal(a, b) and np.array_equal(a, c))
    check("(v) no INV net unless asked", "INV" not in plain.nets)
    same_key = plain.split_key == with_inv.split_key == hooked.split_key and len(plain.split_key) == 40
    check("(v) the split key is the partition sha1 on every path", same_key, with_inv.split_key)
    sha = with_inv.nets["INV"].state_sha1()
    check("(v) the inv_fits hook trains the run's net (sha1)", hooked.nets["INV"].state_sha1() == sha)
    check("(v) run.json provenance carries the sha1", with_inv.diagnostics["erm_inv_state_sha1"] == sha)

    X, GX, y = fitted["pairs"]
    again = train_erm_inv(X, GX, y, block["seed"], tau, TRAIN_KW)
    check("(v) the AL fit is deterministic at a fixed init_seed", again.state_sha1() == sha)

    d = with_inv.diagnostics
    check(
        "(v) ERM+INV is more invariant than the ERM on the held-out B pairs",
        d["E_inv_B_INV"] < d["E_inv_B_X"],
        f"{d['E_inv_B_INV']:.4g} vs {d['E_inv_B_X']:.4g}",
    )
    print(
        f"  E_inv_B: ERM {d['E_inv_B_X']:.4g} DA+ERM {d['E_inv_B_GX']:.4g} ERM+INV {d['E_inv_B_INV']:.4g} "
        f"(tau {tau:g}); train s X {d['train_seconds_X']:.1f} GX {d['train_seconds_GX']:.1f} "
        f"INV {d['train_seconds_INV']:.1f}"
    )
    for key in ("train_seconds_X", "train_seconds_GX", "train_seconds_INV", "erm_inv_al_trace", "erm_inv_al_tau"):
        check(f"(v) the diagnostics carry {key}", key in d)

    # PI+INV's radius on the replicate's nets: the ERM's sigma-hat under every centre
    lo_hi = DOMNIST_CONFIG.attainable if DOMNIST_CONFIG.mu_clip else (0.0, 1.0)
    mu = np.clip(np.asarray(with_inv.nets["X"].predict_mean(with_inv.X), dtype=float).ravel(), *lo_hi)
    erm_sigma2 = float(np.mean(mu * (1 - mu)))
    for recenter in ("off", "on", "inv"):
        built = MethodRegistry.build_methods(
            ["PI", "PI+INV"],
            gamma=0.085,
            epsilon=0.04,
            backend="copsens",
            outcome_models=with_inv.nets,
            inv_recenter=recenter,
        )
        pi = built["PI"]().fit(with_inv.X, with_inv.y)
        pi_inv = built["PI+INV"]().fit(with_inv.X, with_inv.y, GX=with_inv.GX_inv)
        check(
            f"(v) {recenter}: PI+INV's sigma2_ is the ERM's mean mu(1-mu), PI's",
            pi_inv.sigma2_ == pi.sigma2_
            and pi_inv._radius(0.085) == pi._radius(0.085)
            and np.isclose(pi.sigma2_, erm_sigma2, rtol=1e-6),
            f"{pi_inv.sigma2_!r} vs {pi.sigma2_!r} vs {erm_sigma2!r}",
        )
    return replicate, with_inv


#: the (v) replicate's flat-head state sha1s at 421549d, before `domnist-pool` existed
FLAT_SHA1 = {
    "X": "359805f5fe039701ac5d013b014cba30bc7656e4",
    "GX": "e5d4f0f68da809c1440c077c901f8fcad6adf8ee",
    "INV": "c5f8eb50f0cf0ec6931390928e5a378ce01fed52",
}
NET_KEYS = ("X", "GX", "INV")


def pooled(model) -> bool:
    from torch import nn

    return any(isinstance(m, nn.AdaptiveAvgPool2d) for m in model.modules())


def leg_vi():
    print("(vi) the pooled head")
    import torch
    from torch import nn

    from src.methods.nets import NETS

    check("(vi) NETS carries domnist-fast and domnist-pool", {"domnist-fast", "domnist-pool"} <= set(NETS))
    d = 3 * 14 * 14
    pool, fast = NETS["domnist-pool"](d), NETS["domnist-fast"](d)
    layers = list(pool)
    check(
        "(vi) domnist-pool ends in GAP, a 64-unit dense layer and a sigmoid",
        pooled(pool) and isinstance(layers[-1], nn.Sigmoid),
    )
    dense = [m for m in layers if isinstance(m, nn.Linear)]
    check("(vi) domnist-pool's dense head reads 64 pooled channels", [m.in_features for m in dense] == [64, 64])
    check("(vi) domnist-fast does not pool globally", not pooled(fast))
    torch.manual_seed(0)
    x = torch.rand(8, d)
    with torch.no_grad():
        outs = [net(x) for net in (pool, fast)]
    check("(vi) both heads map to (0, 1)", all(o.shape == (8, 1) and bool(((o > 0) & (o < 1)).all()) for o in outs))
    block = resolve_dataset_block("do_mnist", {**MINIMAL, "methods": ["PI"], "im-ci": 0})
    block.pop("experiment", None)
    check("(vi) the orchestrator defaults to the flat head", DoMNISTOrchestrator(**block).net == "domnist-fast")
    block = resolve_dataset_block("do_mnist", {**MINIMAL, "methods": ["PI"], "net": "domnist-pool", "im-ci": 0})
    block.pop("experiment", None)
    check("(vi) the block's net reaches the orchestrator", DoMNISTOrchestrator(**block).net == "domnist-pool")


def leg_vii(replicate, flat):
    print("(vii) the head, GPU (60k draw)")
    tau = flat.diagnostics["erm_inv_al_tau"]
    for key in NET_KEYS:
        sha = flat.nets[key].state_sha1()
        check(f"(vii) flat nets['{key}'] is bit-identical to 421549d", sha == FLAT_SHA1[key], sha)
    first = replicate(net="domnist-pool", train_inv=True, erm_inv_tau=tau)
    again = replicate(net="domnist-pool", train_inv=True, erm_inv_tau=tau)
    for key in NET_KEYS:
        check(f"(vii) pooled nets['{key}'] carries the pooled head", pooled(first.nets[key].f))
        sha = first.nets[key].state_sha1()
        check(f"(vii) pooled nets['{key}'] is deterministic", sha == again.nets[key].state_sha1())
        check(f"(vii) pooled nets['{key}'] differs from the flat one", sha != flat.nets[key].state_sha1())
    for field in ("X", "GX", "GX_inv", "y", "G", "mask_a", "mask_b"):
        same = np.array_equal(np.asarray(getattr(first, field)), np.asarray(getattr(flat, field)))
        check(f"(vii) the B array {field} does not depend on the head", same)
    d = first.diagnostics
    print(
        f"  pooled E_inv_B: ERM {d['E_inv_B_X']:.4g} DA+ERM {d['E_inv_B_GX']:.4g} ERM+INV {d['E_inv_B_INV']:.4g} "
        f"(tau {tau:g}); train s X {d['train_seconds_X']:.1f} GX {d['train_seconds_GX']:.1f} "
        f"INV {d['train_seconds_INV']:.1f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nets", action="store_true", help="add the GPU replicate legs (60k draws)")
    args = parser.parse_args()
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    leg_i()
    leg_ii()
    leg_iii()
    leg_iv()
    leg_vi()
    if args.nets:
        leg_vii(*leg_v())
    print(f"\nA78 {'PASS' if not FAIL else 'FAIL'}")
    for name in FAIL:
        print(f"  FAILED: {name}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
