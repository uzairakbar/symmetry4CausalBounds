"""A10: PartialR2 must be bit-identical across the BoundedSA hoist.

Emits a digest of a small optical sweep + query solve. Run on both sides of the
change and diff the json.

    python scripts/a10_partial_r2_regression.py > /tmp/after.json
"""

import os
import sys
import json
import hashlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiments.utils import set_seed  # noqa: E402
from src.experiments.optical_device import OpticalOrchestrator  # noqa: E402
import src.methods.sensitivity_models as sm  # noqa: E402


def _digest(a):
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest()[:16]


def direct_solves():
    """Every PI class, serial and parallel, on a fixed fixture."""
    set_seed(3)
    X = np.random.randn(200, 6)
    y = X @ np.random.randn(6, 1) + 0.3 * np.random.randn(200, 1)
    GX = X + 0.2 * np.random.randn(*X.shape)
    Q = np.random.randn(60, 6)

    cases = {
        "PI": lambda nj: sm.PartialR2(gamma=0.5, epsilon=0.3, n_jobs=nj).fit(X, y),
        "PI_INV": lambda nj: sm.InvarianceConstrainedPartialR2(gamma=0.5, epsilon=0.3, n_jobs=nj).fit(X, y, GX=GX),
        "DA+PI_IV": lambda nj: sm.InstrumentalVariablePartialR2(gamma=0.5, epsilon=0.3, epsilon_iv=0.2, n_jobs=nj).fit(
            GX, y, Z=GX
        ),
        "PI&DA+PI": lambda nj: sm.IntersectedPartialR2(gamma=0.5, epsilon=0.3, n_jobs=nj).fit(X, y, GX=GX, G=GX),
    }
    out = {}
    for name, build in cases.items():
        for n_jobs in (1, 4):
            model = build(n_jobs)
            out[f"{name}|nj={n_jobs}"] = {
                "bounds": _digest(model.predict(Q)),
                "status": _digest(model.query_status),
            }
    return out


def optical_sweep():
    """A real sweep record through the orchestrator, at a reduced grid."""
    set_seed(69)
    orchestrator = OpticalOrchestrator(
        seed=69,
        n_samples=200,
        n_experiments=2,
        sweep_samples=3,
        methods=["PI", "PI_INV", "DA+PI", "PI&DA+PI"],
        hyperparameters={},
        n_jobs=1,
        augmentation="rotation > hflip > vflip > gaussian-noise",
        calibrate=False,
        pad=False,
        clipy=True,
    )
    _, results, statuses = orchestrator.sweep_record("gamma")
    return {
        name: {
            # wall_clock is a timing, never reproducible; every other field must be
            **{metric: _digest(values) for metric, values in record.items() if metric != "wall_clock"},
            "status_counts": _digest(statuses[name]),
        }
        for name, record in results.items()
    }


if __name__ == "__main__":
    print(json.dumps({"direct": direct_solves(), "optical": optical_sweep()}, indent=1, sort_keys=True))
