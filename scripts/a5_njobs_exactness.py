"""A5: query parallelism must change nothing but the wall clock.

Queries are independent and nothing warm-starts across them, so serial and
parallel bounds are bit-identical -- not merely close.

    python scripts/a5_njobs_exactness.py
    python scripts/a5_njobs_exactness.py --micro   # 4k/512/8, minutes (PLAN v2 C4)
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_augmentors.do_mnist import DoMNISTDA  # noqa: E402
from src.experiments.configs import DOMNIST_CONFIG  # noqa: E402
from src.experiments.do_mnist import Flatten  # noqa: E402
from src.experiments.utils import set_seed  # noqa: E402
from src.methods.partial_r2_net import (  # noqa: E402
    IVConstrainedPartialR2Net,
    PartialR2Net,
    RecentredInvPartialR2Net,
)
from src.methods.regression import GradientDescentERM  # noqa: E402
from src.sem.do_mnist import DoMNISTSEM  # noqa: E402

N_SAMPLES, N_PI, N_QUERIES, N_JOBS = 40_000, 5_000, 128, 16
MICRO = (4_000, 512, 8, 4)
FAIL = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAIL.append(name)


def main(micro=False):
    n_samples, n_pi, n_queries, n_jobs = MICRO if micro else (N_SAMPLES, N_PI, N_QUERIES, N_JOBS)
    set_seed(42)
    sem = DoMNISTSEM(
        seed=42,
        train=True,
        target_samples=1,
        alpha=DOMNIST_CONFIG.alpha,
        beta=DOMNIST_CONFIG.beta,
        eta=DOMNIST_CONFIG.eta,
    )
    flat = Flatten()
    X_img, y, _ = sem.sample_paired(n_samples, seed=42)
    GX_img, G = DoMNISTDA()(X_img)
    X, GX = flat.fit_transform(X_img), flat.fit_transform(GX_img)

    nets = {
        "X": GradientDescentERM().fit(X, y, init_seed=42, epochs=1),
        "GX": GradientDescentERM().fit(GX, y, init_seed=42, epochs=1),
    }
    keep = np.random.default_rng(42).choice(len(X), n_pi, replace=False)
    X, GX, y, G = X[keep], GX[keep], y[keep], G[keep]
    Q = X[:n_queries]

    common = dict(gamma=0.1, calibrate=True, clipy=True, unfrozen_layers=1)

    cases = {
        "PI": lambda nj: PartialR2Net(outcome_model=nets["X"], n_jobs=nj, **common).fit(X, y),
        "DA+PI": lambda nj: PartialR2Net(outcome_model=nets["GX"], n_jobs=nj, **common).fit(GX, y),
        # eps large enough to clear the floor, so this exercises a FEASIBLE
        # constrained solve rather than the all-INFEASIBLE gate
        "PI+INV": lambda nj: RecentredInvPartialR2Net(outcome_model=nets["GX"], epsilon=0.2, n_jobs=nj, **common).fit(
            X, y, GX=GX
        ),
        "DA+PI+IV": lambda nj: IVConstrainedPartialR2Net(
            outcome_model=nets["GX"], epsilon_iv=0.12, n_jobs=nj, **common
        ).fit(GX, y, Z=G),
    }

    for name, build in cases.items():
        serial = build(1)
        start = time.perf_counter()
        bounds_serial, status_serial = serial.predict(Q), serial.query_status.copy()
        t_serial = time.perf_counter() - start

        parallel = build(n_jobs)
        start = time.perf_counter()
        bounds_parallel, status_parallel = (parallel.predict(Q), parallel.query_status.copy())
        t_parallel = time.perf_counter() - start

        check(
            f"A5 {name:11s} bounds bit-identical",
            np.array_equal(bounds_serial, bounds_parallel, equal_nan=True),
            f"{t_serial:.1f}s -> {t_parallel:.1f}s ({t_serial / t_parallel:.1f}x)",
        )
        check(f"A5 {name:11s} query_status identical", np.array_equal(status_serial, status_parallel))

    print("\n" + ("ALL PASS" if not FAIL else f"FAILURES: {FAIL}"))
    return 1 if FAIL else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--micro", action="store_true", help="4k/512/8 fixture (PLAN v2 C4)")
    sys.exit(main(micro=parser.parse_args().micro))
