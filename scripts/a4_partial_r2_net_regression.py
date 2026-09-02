"""A4: the partial_r2_net backend must stay numerically frozen.

Repurposed. A4 used to pin the ported latent-factor model against SOURCE; it is
gone, so the same two-stage shape now pins `partial_r2_net` against a dump of ITSELF:
fit the four models on one fixture, freeze bounds + intermediates, re-check later.

Frozen against a SHARED ARTIFACT, not a shared seed: the nets and the exact arrays
are dumped, so the check never depends on torch reproducing a training run.

    python scripts/a4_partial_r2_net_regression.py --dump  ~/scratch/a4
    python scripts/a4_partial_r2_net_regression.py --check ~/scratch/a4
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_augmentors.do_mnist import DoMNISTDA  # noqa: E402
from src.experiments.configs import DOMNIST_CONFIG  # noqa: E402
from src.methods.partial_r2_net import (  # noqa: E402
    IVConstrainedPartialR2Net,
    PartialR2Net,
    RecentredInvPartialR2Net,
)
from src.methods.regression import GradientDescentERM  # noqa: E402
from src.sem.do_mnist import DoMNISTSEM  # noqa: E402

SEED, N_SAMPLES, N_PI, N_QUERIES, N_JOBS = 42, 40_000, 2_000, 16, 8
AUGMENTATION = "translate > rotation > contrast > saturation > hue"
TRAIN_KW = dict(lr=0.01, batch=256, epochs=1, optimizer="adam", loss="mse", betas=(0.7, 0.9), onecycle=True)
GAMMA, EPSILON, EPSILON_IV = 0.067, 0.2, 0.12
TOL = 5e-4  # bound scale is [0, 1]; the solves are multi-start, not closed form


def _flat(A):
    return np.asarray(A).reshape(len(A), -1)


def fixture(stem, dump):
    """(nets, X_pi, GX_pi, y_pi, G_pi, Q) -- drawn on `--dump`, reloaded on `--check`."""
    if not dump:
        blob = np.load(f"{stem}.npz")
        nets = {k: GradientDescentERM.load_state(f"{stem}_{k}.pt") for k in ("X", "GX")}
        return nets, blob["X_pi"], blob["GX_pi"], blob["y_pi"], blob["G_pi"], blob["Q"]

    sem = DoMNISTSEM(
        seed=SEED,
        train=True,
        target_samples=1,
        alpha=DOMNIST_CONFIG.alpha,
        beta=DOMNIST_CONFIG.beta,
        eta=DOMNIST_CONFIG.eta,
        subsample=DOMNIST_CONFIG.subsample,
    )
    X_img, y, _ = sem.sample_paired(N_SAMPLES, seed=SEED)
    GX_img, G = DoMNISTDA(AUGMENTATION)(X_img)
    X, GX = _flat(X_img).astype(np.float32), _flat(GX_img).astype(np.float32)

    nets = {
        "X": GradientDescentERM().fit(X, y, init_seed=SEED, **TRAIN_KW),
        "GX": GradientDescentERM().fit(GX, y, init_seed=SEED, **TRAIN_KW),
    }
    for key, net in nets.items():
        net.save_state(f"{stem}_{key}.pt")

    j = np.random.default_rng(SEED).choice(len(X), N_PI, replace=False)
    X_pi, GX_pi, y_pi, G_pi = X[j], GX[j], np.asarray(y)[j], np.asarray(G)[j]
    Q = X[np.random.default_rng(SEED + 1).choice(len(X), N_QUERIES, replace=False)]
    np.savez_compressed(f"{stem}.npz", X_pi=X_pi, GX_pi=GX_pi, y_pi=y_pi, G_pi=G_pi, Q=Q)
    return nets, X_pi, GX_pi, y_pi, G_pi, Q


def build(nets, X_pi, GX_pi, y_pi, G_pi):
    """The four constraint sets, on the fixture. clipy=False: raw bounds, so a clip
    can never mask a moved bound. The clipped run is compared separately."""
    common = dict(
        gamma=GAMMA,
        calibrate=True,
        clipy=False,
        pad=False,
        n_jobs=N_JOBS,
        link=DOMNIST_CONFIG.link,
        unfrozen_layers=DOMNIST_CONFIG.unfrozen_layers,
    )
    return {
        "PI": PartialR2Net(outcome_model=nets["X"], **common).fit(X_pi, y_pi),
        "DA+PI": PartialR2Net(outcome_model=nets["GX"], **common).fit(GX_pi, y_pi),
        "PI+INV": RecentredInvPartialR2Net(outcome_model=nets["GX"], epsilon=EPSILON, **common).fit(
            X_pi, y_pi, GX=GX_pi
        ),
        "DA+PI+IV": IVConstrainedPartialR2Net(outcome_model=nets["GX"], epsilon_iv=EPSILON_IV, **common).fit(
            GX_pi, y_pi, Z=G_pi
        ),
    }


def intermediates(model):
    budget = model._budget()
    return {
        "sigma2": float(model.sigma2_),
        "scale": float(model.scale),
        "r2_floor": float(model.r2_floor_),
        "budget": None if budget is None else float(budget),
        "floor": None if budget is None else float(model.constraint_floor(GAMMA)),
    }


def dump(stem):
    nets, X_pi, GX_pi, y_pi, G_pi, Q = fixture(stem, dump=True)
    models = build(nets, X_pi, GX_pi, y_pi, G_pi)
    bounds = {name: np.asarray(model.predict(Q, gamma=GAMMA)) for name, model in models.items()}
    clipped = PartialR2Net(
        gamma=GAMMA,
        calibrate=True,
        clipy=True,
        n_jobs=N_JOBS,
        link=DOMNIST_CONFIG.link,
        unfrozen_layers=DOMNIST_CONFIG.unfrozen_layers,
        outcome_model=nets["X"],
    ).fit(X_pi, y_pi)

    blob = dict(np.load(f"{stem}.npz"))
    np.savez_compressed(f"{stem}.npz", **blob, **{f"b_{k}": v for k, v in bounds.items()})
    with open(f"{stem}.json", "w") as fh:
        json.dump(
            {
                "y_range": [clipped.y_min, clipped.y_max],
                "intermediates": {name: intermediates(model) for name, model in models.items()},
            },
            fh,
            indent=1,
        )
    print(f"wrote {stem}.npz / .json / _X.pt / _GX.pt")


def check(stem):
    nets, X_pi, GX_pi, y_pi, G_pi, Q = fixture(stem, dump=False)
    blob = np.load(f"{stem}.npz")
    with open(f"{stem}.json") as fh:
        ref = json.load(fh)

    models = build(nets, X_pi, GX_pi, y_pi, G_pi)
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

        want_mid, mine = ref["intermediates"][name], intermediates(model)
        for label, theirs in want_mid.items():
            if theirs is None and mine[label] is None:
                continue
            rel = abs(mine[label] - theirs) / max(abs(theirs), 1e-12)
            flag = "PASS" if rel < 1e-6 else "FAIL"
            print(f"  [{flag}] {name:9s} {label:9s} {mine[label]:.8g} vs {theirs:.8g}")
            if rel >= 1e-6:
                failed.append(f"{name}/{label}")

    # the clipy delta must be exactly a clip to [y_min, y_max], nothing else
    clipped = PartialR2Net(
        gamma=GAMMA,
        calibrate=True,
        clipy=True,
        n_jobs=N_JOBS,
        link=DOMNIST_CONFIG.link,
        unfrozen_layers=DOMNIST_CONFIG.unfrozen_layers,
        outcome_model=nets["X"],
    ).fit(X_pi, y_pi)
    y_min, y_max = ref["y_range"]
    got = clipped.predict(Q, gamma=GAMMA)
    expected = np.clip(blob["b_PI"], y_min, y_max)
    ok = np.allclose(got, expected, atol=TOL, equal_nan=True)
    print(f"[{'PASS' if ok else 'FAIL'}] clipy delta is exactly a clip to [{y_min:g}, {y_max:g}]")
    if not ok:
        failed.append("clipy")

    print("\n" + ("A4 PASS" if not failed else f"A4 FAILURES: {failed}"))
    return 1 if failed else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump")
    parser.add_argument("--check")
    args = parser.parse_args()
    if args.dump:
        dump(args.dump)
    else:
        sys.exit(check(args.check))
