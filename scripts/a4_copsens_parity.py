"""A4: the ported CopSens must reproduce SOURCE's bounds numerically.

Frozen against a SHARED ARTIFACT, not a shared seed: FactorAnalysis is reproducible
only if `random_state` AND the input matrix match, so stage 1 dumps the fitted nets
and the exact arrays rather than the seed that produced them.

    <source-venv>/bin/python scripts/a4_copsens_parity.py --source /tmp/a4
    <s4cb>/bin/python       scripts/a4_copsens_parity.py --check  /tmp/a4
"""

import argparse
import json
import os
import sys

import numpy as np

SOURCE = os.path.expanduser("~/neurips26/ZULFI/doMNIST/symmetry4CausalBoundsDoMNIST")

SEED, N_SAMPLES, N_PI, N_QUERIES = 42, 100_000, 8_000, 64
AUGMENTATION = "translate > rotation > contrast > saturation > hue"
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
)
# SOURCE's effective values: DoMNISTExperiment's own defaults, which config.yaml
# does not override. n_anchors is 128 there, NOT CopSensPI's own default of 256.
GAMMA, EPSILON, N_COMPONENTS, N_ANCHORS = 0.1, 0.05, 32, 128
TOL = 5e-4


def _flat(A):
    return np.asarray(A).reshape(len(A), -1)


# ------------------------------------------------------------------------ SOURCE


def dump(stem):
    sys.path.insert(0, SOURCE)
    os.chdir(SOURCE)
    from src.data_augmentors.domnist import DoMNISTDA  # noqa: PLC0415
    from src.sem.domnist import DoMNISTSEM  # noqa: PLC0415

    from src.methods.copsens import (
        CopSensPI,  # noqa: PLC0415
    )
    from src.methods.copsens import (
        InvarianceConstrainedCopSens as InvCopSens,
    )
    from src.methods.copsens import (
        IVConstrainedCopSens as IVCopSens,
    )
    from src.methods.regression import GradientDescentERM  # noqa: PLC0415

    sem = DoMNISTSEM(train=True, **SEM_KW)
    X_img, y, _ = sem.sample_paired(N_SAMPLES, seed=SEED)
    GX_img, G = DoMNISTDA(AUGMENTATION)(X_img)
    X, GX = _flat(X_img).astype(np.float32), _flat(GX_img).astype(np.float32)

    erm = GradientDescentERM("domnist-fast").fit(X, y, init_seed=SEED, **TRAIN_KW)
    da_erm = GradientDescentERM("domnist-fast").fit(GX, y, init_seed=SEED, **TRAIN_KW)
    erm.save_state(f"{stem}_erm.pt")
    da_erm.save_state(f"{stem}_da_erm.pt")

    j = np.random.default_rng(SEED).choice(len(X), N_PI, replace=False)
    X_pi, GX_pi, y_pi, G_pi = X[j], GX[j], np.asarray(y)[j], np.asarray(G)[j]
    Q = X[np.random.default_rng(SEED + 1).choice(len(X), N_QUERIES, replace=False)]

    base = dict(
        gamma=GAMMA,
        gamma0=1.0,
        n_components=N_COMPONENTS,
        latent="fa",
        link="probit",
        calibrate_sigma=True,
        n_anchors=N_ANCHORS,
        n_dir=128,
        n_radii=6,
        solver="cvxpy",
        jax_grad=True,
        pad=False,
        n_jobs=1,
        mu_clip=sem.attainable,
    )

    models = {
        "PI": CopSensPI(outcome_model=erm, **base).fit(X_pi, y_pi),
        "DA+PI": CopSensPI(outcome_model=da_erm, **base).fit(GX_pi, y_pi),
        # inv_recenter: on -- fit on (GX, y, GX=X)
        "PI_INV": InvCopSens(outcome_model=da_erm, epsilon=EPSILON, **base).fit(GX_pi, y_pi, GX=X_pi),
        "DA+PI_IV": IVCopSens(
            outcome_model=da_erm, epsilon=EPSILON, delta="from_eps", rho=None, gamma_z_star=0.0, **base
        ).fit(GX_pi, y_pi, Z=G_pi),
    }

    bounds, intermediates = {}, {}
    for name, model in models.items():
        bounds[name] = np.asarray(model.predict(Q, gamma=GAMMA))
        intermediates[name] = {
            "sigma2": float(model.sigma2_),
            "scale": float(model._scale(GAMMA)),
            "sigma_eigs": np.linalg.eigvalsh(model.latent_.Sigma_).tolist(),
            "anchors_digest": float(np.abs(model.anchors_).sum()),
            "budget": (None if model._budget() is None else float(model._budget())),
            "floor": (None if model._budget() is None else float(model.constraint_floor())),
        }

    np.savez_compressed(
        f"{stem}.npz", X_pi=X_pi, GX_pi=GX_pi, y_pi=y_pi, G_pi=G_pi, Q=Q, **{f"b_{k}": v for k, v in bounds.items()}
    )
    with open(f"{stem}.json", "w") as fh:
        json.dump({"attainable": list(sem.attainable), "intermediates": intermediates}, fh, indent=1)
    print(f"wrote {stem}.npz / .json / _erm.pt / _da_erm.pt")


# ------------------------------------------------------------------------ TARGET


def check(stem):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.methods.copsens import (
        CopSensPI,
        IVConstrainedCopSens,
        RecentredInvCopSens,  # noqa: PLC0415
    )
    from src.methods.regression import GradientDescentERM  # noqa: PLC0415

    blob = np.load(f"{stem}.npz")
    with open(f"{stem}.json") as fh:
        ref = json.load(fh)
    attainable = tuple(ref["attainable"])

    erm = GradientDescentERM.load_state(f"{stem}_erm.pt")
    da_erm = GradientDescentERM.load_state(f"{stem}_da_erm.pt")
    X_pi, GX_pi, y_pi, G_pi, Q = (blob["X_pi"], blob["GX_pi"], blob["y_pi"], blob["G_pi"], blob["Q"])

    # clipy=False: SOURCE never clips CopSens bounds. The clipped run is compared
    # separately below, so the delta stays a KNOWN one.
    base = dict(
        gamma=GAMMA,
        n_components=N_COMPONENTS,
        link="probit",
        calibrate=True,
        n_anchors=N_ANCHORS,
        jax_grad=True,
        pad=False,
        n_jobs=1,
        mu_clip=attainable,
        clipy=False,
    )

    models = {
        "PI": CopSensPI(outcome_model=erm, **base).fit(X_pi, y_pi),
        "DA+PI": CopSensPI(outcome_model=da_erm, **base).fit(GX_pi, y_pi),
        "PI_INV": RecentredInvCopSens(outcome_model=da_erm, epsilon=EPSILON, **base).fit(X_pi, y_pi, GX=GX_pi),
        "DA+PI_IV": IVConstrainedCopSens(outcome_model=da_erm, epsilon_iv=EPSILON, **base).fit(GX_pi, y_pi, Z=G_pi),
    }

    failed = []
    for name, model in models.items():
        got, want = np.asarray(model.predict(Q, gamma=GAMMA)), blob[f"b_{name}"]
        both_nan = np.isnan(got) & np.isnan(want)
        delta = float(np.nanmax(np.where(both_nan, 0.0, np.abs(got - want))))
        shape_ok = np.array_equal(np.isnan(got), np.isnan(want))
        ok = shape_ok and (np.isnan(delta) or delta <= TOL)
        print(
            f"[{'PASS' if ok else 'FAIL'}] {name:9s} max|dbound| {delta:.3g} "
            f"(NaN pattern {'matches' if shape_ok else 'DIFFERS'})"
        )
        if not ok:
            failed.append(name)

        want_mid = ref["intermediates"][name]
        eigs = np.linalg.eigvalsh(model.latent_.Sigma_)
        for label, mine, theirs in (
            ("sigma2", float(model.sigma2_), want_mid["sigma2"]),
            ("scale", float(model._radius(GAMMA)), want_mid["scale"]),
            ("Sigma eigs", float(np.abs(eigs).max()), float(np.abs(want_mid["sigma_eigs"]).max())),
            ("anchors", float(np.abs(model.anchors_).sum()), want_mid["anchors_digest"]),
            ("budget", model._budget(), want_mid["budget"]),
        ):
            if theirs is None and mine is None:
                continue
            rel = abs(float(mine) - float(theirs)) / max(abs(float(theirs)), 1e-12)
            flag = "PASS" if rel < 1e-6 else "FAIL"
            print(f"  [{flag}] {name:9s} {label:11s} {float(mine):.8g} vs {float(theirs):.8g}")
            if rel >= 1e-6:
                failed.append(f"{name}/{label}")

    # the clipy delta must be exactly a clip to [y_min, y_max], nothing else
    clipped = CopSensPI(outcome_model=erm, **{**base, "clipy": True}).fit(X_pi, y_pi)
    got = clipped.predict(Q, gamma=GAMMA)
    expected = np.clip(blob["b_PI"], clipped.y_min, clipped.y_max)
    ok = np.allclose(got, expected, atol=TOL, equal_nan=True)
    print(f"[{'PASS' if ok else 'FAIL'}] clipy delta is exactly a clip to [{clipped.y_min:g}, {clipped.y_max:g}]")
    if not ok:
        failed.append("clipy")

    print("\n" + ("A4 PASS" if not failed else f"A4 FAILURES: {failed}"))
    return 1 if failed else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source")
    parser.add_argument("--check")
    args = parser.parse_args()
    if args.source:
        dump(args.source)
    else:
        sys.exit(check(args.check))
