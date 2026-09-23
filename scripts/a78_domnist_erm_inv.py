"""A78: the do-MNIST ERM+INV centre (`InvariantGradientDescentERM`, `inv_recenter: inv`).

Static legs (no MNIST; the tiny AL fit runs on whatever `device()` picks, seconds):

  (i)   the registry: PI+INV is `InvarianceConstrainedCopSens` on `nets["X"]` under
        `off`, `RecentredInvCopSens` on `nets["GX"]` under `on`, and
        `InvarianceConstrainedCopSens` on `nets["INV"]` under `inv`; `ERM+INV` is the
        INV net itself; a bogus `inv_recenter` raises when PI+INV is built.
  (ii)  nesting: `nested_in("inv")` has no PI+INV entry (off/on keep theirs), and
        `log_nesting` under `inv` stays quiet on a PI+INV wider than PI.
  (iii) the AL schedule: the window is `n_batches // updates_per_epoch` (at least 1)
        at the 60k and the 1.2M draw sizes, and a tiny fit at two sizes updates at
        exactly those steps, `updates_per_epoch` times per epoch at the larger one.
  (iv)  the plumbing: `erm_inv_fit_kwargs` carries the `DOMNIST_CONFIG` constants and
        the epoch override, the default `erm_inv_tau` is 4e-4, and the orchestrator
        trains the ERM+INV net iff PI+INV runs under `inv` or ERM+INV is listed.

GPU leg (`--nets`, a 60k draw, about two minutes):

  (v)   the AL fit is deterministic at a fixed `init_seed`; the replicate's ERM+INV
        net has a lower invariance error than the ERM on the held-out B pairs;
        `draw_replicate(train_inv=True)` leaves the ERM and DA+ERM nets (state sha1)
        and the B arrays (`X`, `GX`, `GX_inv`, `y`, `G`) bit-identical to
        `train_inv=False`; an `inv_fits` hook that fits the configured knobs returns
        a net whose sha1 equals the plain `train_inv=True` path's.

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
    print("(iii) the AL window is in epoch units")
    window = InvariantGradientDescentERM.al_window
    upe = DOMNIST_CONFIG.erm_inv_updates_per_epoch
    for n in (60_000, 1_200_000):
        n_batches = int(np.ceil(n / 256))
        check(f"(iii) window at n={n:,} is n_batches // {upe}", window(n_batches, upe) == n_batches // upe)
    check("(iii) the window is at least 1", window(3, 20) == 1)

    rng = np.random.default_rng(0)
    for n_batches, epochs in ((10, 1), (100, 2)):
        n = 64 * n_batches
        X = rng.random((n, 3 * 14 * 14), dtype=np.float32)
        GX = np.clip(X + 0.2 * rng.standard_normal(X.shape).astype(np.float32), 0, 1)
        y = (X[:, :50].mean(axis=1) > 0.5).astype(np.float32)
        kw = erm_inv_fit_kwargs({**TRAIN_KW, "batch": 64}, 1e-4, epochs=epochs)
        net = InvariantGradientDescentERM("domnist-fast").fit(X, y, GX=GX, init_seed=0, **kw)
        w = window(n_batches, upe)
        steps = [row[0] for row in net.al_trace_]
        check(
            f"(iii) {n_batches} batches x {epochs} epochs: an update every {w} steps",
            steps == list(range(w, n_batches * epochs + 1, w)),
            str(steps[:5]),
        )
        if n_batches % upe == 0:
            check(f"(iii) {n_batches} batches: {upe} updates per epoch", len(steps) == upe * epochs, str(len(steps)))
        mus = [row[3] for row in net.al_trace_]
        check(
            "(iii) mu never shrinks and stays capped", mus == sorted(mus) and max(mus) <= DOMNIST_CONFIG.erm_inv_mu_max
        )
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
    expected = TRAIN_KW["epochs"] if DOMNIST_CONFIG.erm_inv_epochs is None else DOMNIST_CONFIG.erm_inv_epochs
    check("(iv) the epoch count is erm_inv_epochs, else the pair's", kw["epochs"] == expected)
    check("(iv) an explicit epoch count wins", erm_inv_fit_kwargs(TRAIN_KW, 4e-4, epochs=2)["epochs"] == 2)
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

    def replicate(**kwargs):
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
            net=orchestrator.net,
            split=orchestrator.split,
            split_seed=orchestrator.split_seed,
            **kwargs,
        )

    plain = replicate()
    with_inv = replicate(train_inv=True, erm_inv_tau=tau)
    fitted = {}

    def hook(X, GX, y, init_seed, train_kw):
        # the diagnostic pattern: another point first, then the configured one
        train_erm_inv(X, GX, y, init_seed=init_seed, tau=tau * 10, **train_kw)
        fitted["net"] = train_erm_inv(X, GX, y, init_seed=init_seed, tau=tau, **train_kw)
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
    sha = with_inv.nets["INV"].state_sha1()
    check("(v) the inv_fits hook trains the run's net (sha1)", hooked.nets["INV"].state_sha1() == sha)
    check("(v) run.json provenance carries the sha1", with_inv.diagnostics["erm_inv_state_sha1"] == sha)

    X, GX, y = fitted["pairs"]
    again = train_erm_inv(X, GX, y, init_seed=block["seed"], tau=tau, **TRAIN_KW)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nets", action="store_true", help="add the GPU replicate leg (60k draw)")
    args = parser.parse_args()
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    leg_i()
    leg_ii()
    leg_iii()
    leg_iv()
    if args.nets:
        leg_v()
    print(f"\nA78 {'PASS' if not FAIL else 'FAIL'}")
    for name in FAIL:
        print(f"  FAILED: {name}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
