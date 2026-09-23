"""A76: the ported copsens backend must reproduce the reference checkout's do-MNIST
bounds on frozen inputs.

Frozen against a SHARED ARTIFACT, not a shared seed: the nets differ across torch
versions and FactorAnalysis is reproducible only if `random_state` AND the input
matrix match, so the source stage dumps the fitted nets and the exact fit arrays
rather than the seed that produced them.

    # source stage, under the reference checkout and ITS env, on the CPU
    SRC=~/close-this/copsens/symmetry4CausalBounds
    (cd $SRC && CUDA_VISIBLE_DEVICES= PYTHONPATH=$SRC \\
        ~/scratch/uv_envs/symmetry4CausalBounds/bin/python \\
        <repo>/scripts/a76_domnist_parity.py --source ~/scratch/a76/fix)
    # check stage, under the repo env (the nets on the CPU too, so the only thing
    # that differs between the stages is the PI machinery)
    CUDA_VISIBLE_DEVICES= uv run python scripts/a76_domnist_parity.py --check ~/scratch/a76/fix

Method keys stay reference-spelled (PI_INV, DA+PI_IV) on purpose: `check` indexes
the source npz by them.

Two conventions differ by design and are handled, not hidden: the port returns
BOTH sides NaN when either side of a query has no accepted start (the reference
keeps the other side), and the port caches the floor on the radius. A row where
the reference has exactly one NaN side and the port has two counts as matching
and is reported.
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np

SEED, N_SAMPLES, N_PI, N_POP, N_QUERIES = 42, 100_000, 8_000, 2_000, 64
SPLIT, SPLIT_SEED, POP_SEED, EXEMPLAR_SEED = {"A": 40_000, "B": 10_000, "C": 10_000}, 420, 44, 420
AUGMENTATION = "translate > rotation > contrast > saturation > hue"
MIX_IN = 0.05
SEM_KW = dict(alpha=0.0, beta=0.4, eta=0.25, subsample=2)
TRAIN_KW = dict(
    lr=0.01,
    batch=256,
    epochs=1,
    optimizer="adam",
    loss="mse",
    betas=(0.7, 0.9),
    onecycle=True,
    weight_decay=0.0,
    label_smoothing=0.0,
    val_frac=0.0,
)
GAMMA = 0.08505258154439962
EPSILONS = (0.04, 0.06)
N_COMPONENTS, N_ANCHORS, N_ANCHORS_C, N_CONSTRAINT_INV, N_CONSTRAINT_IV = 32, 128, 48, 192, 384
TOL_BOUNDS, TOL_MID, TOL_FLOOR = 5e-4, 1e-6, 1e-4
METHODS = ("PI", "DA+PI", "DA+PI_IV", "PI_INV_off", "PI_INV_on")


def _flat(A):
    return np.ascontiguousarray(np.asarray(A).reshape(len(A), -1), dtype=np.float32)


def _digest(anchors) -> str:
    return hashlib.sha256(np.ascontiguousarray(anchors, dtype=np.float64).tobytes()).hexdigest()


def _key(name, eps):
    return f"{name}@{eps}"


# ------------------------------------------------------------------------ SOURCE


def dump(stem):
    """Runs under the reference checkout (cwd and PYTHONPATH) and its env, CPU only."""
    import src.methods.utils as reference_utils  # noqa: PLC0415

    reference_utils.CPU_ONLY = True  # before any net, DA or PI call
    from src.data_augmentors.domnist import DoMNISTDA  # noqa: PLC0415
    from src.sem.domnist import DoMNISTSEM  # noqa: PLC0415

    from src.experiments.utils import set_seed  # noqa: PLC0415
    from src.methods.copsens import CopSensPI  # noqa: PLC0415
    from src.methods.copsens import InvarianceConstrainedCopSens as InvCopSens  # noqa: PLC0415
    from src.methods.copsens import IVConstrainedCopSens as IVCopSens  # noqa: PLC0415
    from src.methods.regression import GradientDescentERM  # noqa: PLC0415

    os.makedirs(os.path.dirname(os.path.abspath(stem)), exist_ok=True)
    set_seed(SEED)
    sem = DoMNISTSEM(train=True, **SEM_KW)
    parts = sem.split(SPLIT, SPLIT_SEED)
    da = DoMNISTDA(AUGMENTATION, spread="match", amounts=None)

    # the nets' draw from A, the ERM on X, the in-place mix-in, the DA+ERM on the mixture
    X_img, y = parts["A"].sample(N_SAMPLES, mode="obs", seed=SEED)
    GX_img, _ = da(X_img)
    X, GX = _flat(X_img), _flat(GX_img)
    del X_img, GX_img
    erm = GradientDescentERM("domnist-fast").fit(X, y, init_seed=SEED, **TRAIN_KW)
    GX, _, mask_a = da.mix_in(X, GX, None, MIX_IN, seed=[SEED, 1], inplace=True)
    da_erm = GradientDescentERM("domnist-fast").fit(GX, y, init_seed=SEED, **TRAIN_KW)
    erm.save_state(f"{stem}_erm.pt")
    da_erm.save_state(f"{stem}_da_erm.pt")
    del X, GX

    # the PI rows from B, augmented, and the mixed copy for the DA+ family
    Xb_img, y_pi = parts["B"].sample(N_PI, mode="obs", seed=SEED + 1)
    GXb_img, G_pi = da(Xb_img)
    X_pi, GX_pi = _flat(Xb_img), _flat(GXb_img)
    del Xb_img, GXb_img
    GX_pi_mixed, G_pi_mixed, mask_b = da.mix_in(X_pi, GX_pi, G_pi, MIX_IN, seed=[SEED, 2])
    y_pi = np.asarray(y_pi)

    # the queries: the head of the evaluation population plus the exemplars
    te = DoMNISTSEM(train=False, **SEM_KW)
    P_img, _ = te(N=N_POP, seed=POP_SEED)
    h_star = np.asarray(te.h_star(te.last_["f"])).ravel()[:N_QUERIES]
    Q_pop = _flat(P_img)[:N_QUERIES]
    exemplars, digits = sem.exemplars(EXEMPLAR_SEED, colors="alternating")
    Q_exemplar = _flat(exemplars)
    Q = np.concatenate([Q_pop, Q_exemplar], axis=0)

    base = dict(
        gamma=GAMMA,
        gamma0=1.0,
        n_components=N_COMPONENTS,
        latent="fa",
        link="probit",
        calibrate_sigma=True,
        n_anchors=N_ANCHORS,
        n_anchors_c=N_ANCHORS_C,
        n_dir=128,
        n_radii=6,
        solver="cvxpy",
        jax_grad=True,
        pad=False,
        n_jobs=1,
        mu_clip=sem.attainable,
    )

    def build(name, eps):
        if name == "PI":
            return CopSensPI(outcome_model=erm, **base).fit(X_pi, y_pi)
        if name == "DA+PI":
            return CopSensPI(outcome_model=da_erm, **base).fit(GX_pi_mixed, y_pi)
        if name == "DA+PI_IV":
            return IVCopSens(
                outcome_model=da_erm,
                epsilon=eps,
                delta="from_eps",
                rho=1.0,
                gamma_z_star=0.0,
                n_constraint=N_CONSTRAINT_IV,
                **base,
            ).fit(GX_pi_mixed, y_pi, Z=G_pi_mixed)
        if name == "PI_INV_off":
            return InvCopSens(outcome_model=erm, epsilon=eps, n_constraint=N_CONSTRAINT_INV, **base).fit(
                X_pi, y_pi, GX=GX_pi
            )
        if name == "PI_INV_on":
            return InvCopSens(outcome_model=da_erm, epsilon=eps, n_constraint=N_CONSTRAINT_INV, **base).fit(
                GX_pi, y_pi, GX=X_pi
            )
        raise ValueError(name)

    bounds, intermediates = {}, {}
    for name in METHODS:
        for eps in EPSILONS if name not in ("PI", "DA+PI") else (None,):
            model = build(name, eps)
            key = _key(name, eps)
            print(f"source: {key}", flush=True)
            bounds[key] = np.asarray(model.predict(Q, gamma=GAMMA), dtype=float)
            budget = model._budget()
            intermediates[key] = {
                "sigma2": float(model.sigma2_),
                "scale": float(model._scale(GAMMA)),
                "sigma_eigs": np.linalg.eigvalsh(model.latent_.Sigma_).tolist(),
                "anchors_digest": _digest(model.anchors_),
                "anchors_abs_sum": float(np.abs(model.anchors_).sum()),
                "budget": None if budget is None else float(budget),
                "floor": None if budget is None else float(model.constraint_floor()),
                "nan_rows": int(np.isnan(bounds[key]).any(axis=1).sum()),
            }
            print(f"  {intermediates[key]}", flush=True)

    np.savez_compressed(
        f"{stem}.npz",
        X_pi=X_pi,
        GX_pi=GX_pi,
        GX_pi_mixed=GX_pi_mixed,
        y_pi=y_pi,
        G_pi=np.asarray(G_pi, dtype=float),
        G_pi_mixed=np.asarray(G_pi_mixed, dtype=float),
        Q=Q,
        Q_exemplar=Q_exemplar,
        h_star=h_star,
        **{f"b_{k}": v for k, v in bounds.items()},
    )
    with open(f"{stem}.json", "w") as fh:
        json.dump(
            {
                "attainable": list(sem.attainable),
                "digits": np.asarray(digits).tolist(),
                "mix_in_n_A": int(mask_a.sum()),
                "mix_in_n_B": int(mask_b.sum()),
                "intermediates": intermediates,
            },
            fh,
            indent=1,
        )
    print(f"wrote {stem}.npz / .json / _erm.pt / _da_erm.pt")


# ------------------------------------------------------------------------ TARGET


def _nan_match(got, want):
    """(pattern ok, rows where the reference has one NaN side and the port two)."""
    g, w = np.isnan(got), np.isnan(want)
    one_sided = (w.sum(axis=1) == 1) & g.all(axis=1)
    same = (g == w).all(axis=1) | one_sided
    return bool(same.all()), int(one_sided.sum())


def check(stem):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import src.methods.nets as nets  # noqa: PLC0415

    nets.CPU_ONLY = True  # the reference nets ran on the CPU; keep the centres bit-comparable
    from src.experiments.configs import DOMNIST_CONFIG  # noqa: PLC0415
    from src.methods.copsens import (  # noqa: PLC0415
        CopSensPI,
        InvarianceConstrainedCopSens,
        IVConstrainedCopSens,
        RecentredInvCopSens,
    )
    from src.methods.regression import GradientDescentERM  # noqa: PLC0415

    blob = np.load(f"{stem}.npz")
    with open(f"{stem}.json") as fh:
        ref = json.load(fh)
    attainable = tuple(ref["attainable"])
    if tuple(DOMNIST_CONFIG.attainable) != attainable:
        print(f"[WARN] attainable differs: {DOMNIST_CONFIG.attainable} vs {attainable}; using the reference's")

    erm = GradientDescentERM.load_state(f"{stem}_erm.pt")
    da_erm = GradientDescentERM.load_state(f"{stem}_da_erm.pt")
    X_pi, GX_pi, GX_pi_mixed = blob["X_pi"], blob["GX_pi"], blob["GX_pi_mixed"]
    y_pi, G_pi_mixed, Q = blob["y_pi"], blob["G_pi_mixed"], blob["Q"]

    # clipy=False: the reference never clips CopSens bounds. The clipped run is
    # compared separately below, so that delta stays a KNOWN one.
    base = dict(
        gamma=GAMMA,
        n_components=N_COMPONENTS,
        link="probit",
        calibrate_sigma=True,
        n_anchors=N_ANCHORS,
        n_anchors_c=N_ANCHORS_C,
        jax_grad=True,
        pad=False,
        recalibrate=False,
        clipy=False,
        n_jobs=1,
        mu_clip=attainable,
    )

    def build(name, eps):
        if name == "PI":
            return CopSensPI(outcome_model=erm, **base).fit(X_pi, y_pi)
        if name == "DA+PI":
            return CopSensPI(outcome_model=da_erm, **base).fit(GX_pi_mixed, y_pi)
        if name == "DA+PI_IV":
            return IVConstrainedCopSens(
                outcome_model=da_erm, epsilon_iv=eps, gamma_z_star=0.0, n_constraint=N_CONSTRAINT_IV, **base
            ).fit(GX_pi_mixed, y_pi, T=G_pi_mixed)
        if name == "PI_INV_off":
            return InvarianceConstrainedCopSens(
                outcome_model=erm, epsilon=eps, n_constraint=N_CONSTRAINT_INV, **base
            ).fit(X_pi, y_pi, GX=GX_pi)
        if name == "PI_INV_on":
            # the recentred class swaps the roles itself: ball on GX, pairs (GX, X)
            return RecentredInvCopSens(outcome_model=da_erm, epsilon=eps, n_constraint=N_CONSTRAINT_INV, **base).fit(
                X_pi, y_pi, GX=GX_pi
            )
        raise ValueError(name)

    failed, one_sided_total = [], {}
    for name in METHODS:
        for eps in EPSILONS if name not in ("PI", "DA+PI") else (None,):
            key = _key(name, eps)
            model = build(name, eps)
            got = np.asarray(model.predict(Q, gamma=GAMMA), dtype=float)
            want = blob[f"b_{key}"]
            pattern_ok, one_sided = _nan_match(got, want)
            one_sided_total[key] = one_sided
            finite = np.isfinite(got) & np.isfinite(want)
            delta = float(np.abs(got - want)[finite].max()) if finite.any() else 0.0
            ok = pattern_ok and delta <= TOL_BOUNDS
            print(
                f"[{'PASS' if ok else 'FAIL'}] {key:16s} max|dbound| {delta:.3g} over {int(finite.sum())} entries "
                f"(NaN pattern {'matches' if pattern_ok else 'DIFFERS'}; {one_sided} one-sided rows; "
                f"{int(np.isnan(want).any(axis=1).sum())} reference NaN rows)"
            )
            if not ok:
                failed.append(key)

            mid = ref["intermediates"][key]
            budget = model._budget()
            radius = model._radius(GAMMA)
            pairs = [
                ("sigma2", float(model.sigma2_), mid["sigma2"], TOL_MID),
                ("scale", float(radius), mid["scale"], TOL_MID),
                (
                    "Sigma eigs",
                    float(np.abs(np.linalg.eigvalsh(model.latent_.Sigma_)).max()),
                    float(np.abs(mid["sigma_eigs"]).max()),
                    TOL_MID,
                ),
                ("anchors", float(np.abs(model.anchors_).sum()), mid["anchors_abs_sum"], TOL_MID),
            ]
            if mid["budget"] is not None:
                pairs.append(("budget", float(budget), mid["budget"], TOL_MID))
                pairs.append(("floor", float(model.constraint_floor(radius)), mid["floor"], TOL_FLOOR))
            for label, mine, theirs, tol in pairs:
                rel = abs(mine - theirs) / max(abs(theirs), 1e-12)
                flag = "PASS" if rel <= tol else "FAIL"
                print(f"  [{flag}] {key:16s} {label:11s} {mine:.8g} vs {theirs:.8g} (rel {rel:.2g})")
                if rel > tol:
                    failed.append(f"{key}/{label}")
            same_digest = _digest(model.anchors_) == mid["anchors_digest"]
            note = "identical" if same_digest else "differs (abs-sum compared above)"
            print(f"  [INFO] {key:16s} anchors sha256 {note}")

    # the clipy delta must be exactly a clip to [y_min, y_max], nothing else
    clipped = CopSensPI(outcome_model=erm, **{**base, "clipy": True}).fit(X_pi, y_pi)
    got = np.asarray(clipped.predict(Q, gamma=GAMMA), dtype=float)
    expected = np.clip(blob[f"b_{_key('PI', None)}"], clipped.y_min, clipped.y_max)
    ok = np.allclose(got, expected, atol=TOL_BOUNDS, equal_nan=True)
    print(f"[{'PASS' if ok else 'FAIL'}] clipy delta is exactly a clip to [{clipped.y_min:g}, {clipped.y_max:g}]")
    if not ok:
        failed.append("clipy")

    print(f"\none-sided NaN rows (reference one side, port both): {one_sided_total}")
    print("A76 PASS" if not failed else f"A76 FAILURES: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", help="dump the reference fixture to this stem (reference checkout and env)")
    parser.add_argument("--check", help="compare the port against the fixture at this stem (repo env)")
    args = parser.parse_args()
    if args.source:
        dump(args.source)
    elif args.check:
        sys.exit(check(args.check))
    else:
        parser.error("one of --source or --check")
