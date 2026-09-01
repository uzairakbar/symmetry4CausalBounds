"""A10: PartialR2 must be bit-identical across the BoundedSA hoist.

Emits a digest of a small optical sweep + query solve. Run on both sides of the
change and diff the json.

    python scripts/a10_partial_r2_regression.py > /tmp/after.json

`--dump out.npz` writes the raw arrays instead of digests, for cross-env
comparison at a tolerance (digests only prove bit-equality).
"""

import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.methods.sensitivity_models as sm  # noqa: E402
from src.experiments.optical_device import OpticalOrchestrator  # noqa: E402
from src.experiments.utils import set_seed  # noqa: E402

DUMP = None  # --dump swaps this for a dict; _digest then also stashes the raw arrays


def _digest(a, key=None):
    a = np.ascontiguousarray(a, dtype=np.float64)
    if DUMP is not None and key is not None:
        DUMP[key] = a
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]


def direct_solves():
    """Every PI class, serial and parallel, on a fixed fixture."""
    set_seed(3)
    X = np.random.randn(200, 6)
    y = X @ np.random.randn(6, 1) + 0.3 * np.random.randn(200, 1)
    GX = X + 0.2 * np.random.randn(*X.shape)
    Q = np.random.randn(60, 6)

    cases = {
        "PI": lambda nj: sm.PartialR2(gamma=0.5, epsilon=0.3, n_jobs=nj).fit(X, y),
        "PI+INV": lambda nj: sm.InvarianceConstrainedPartialR2(gamma=0.5, epsilon=0.3, n_jobs=nj).fit(X, y, GX=GX),
        "DA+PI+IV": lambda nj: sm.InstrumentalVariablePartialR2(gamma=0.5, epsilon=0.3, epsilon_iv=0.2, n_jobs=nj).fit(
            GX, y, Z=GX
        ),
        "PI&DA+PI": lambda nj: sm.IntersectedPartialR2(gamma=0.5, epsilon=0.3, n_jobs=nj).fit(X, y, GX=GX, G=GX),
    }
    out = {}
    for name, build in cases.items():
        for n_jobs in (1, 4):
            model = build(n_jobs)
            out[f"{name}|nj={n_jobs}"] = {
                "bounds": _digest(model.predict(Q), f"direct|{name}|nj={n_jobs}|bounds"),
                "status": _digest(model.query_status, f"direct|{name}|nj={n_jobs}|status"),
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
        methods=["PI", "PI+INV", "DA+PI", "PI&DA+PI"],
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
            **{
                metric: _digest(values, f"optical|{name}|{metric}")
                for metric, values in record.items()
                if metric != "wall_clock"
            },
            "status_counts": _digest(statuses[name], f"optical|{name}|status_counts"),
        }
        for name, record in results.items()
    }


if __name__ == "__main__":
    if "--dump" in sys.argv:
        DUMP = {}
    digests = json.dumps({"direct": direct_solves(), "optical": optical_sweep()}, indent=1, sort_keys=True)
    if DUMP is not None:
        np.savez(sys.argv[sys.argv.index("--dump") + 1], **DUMP)
    print(digests)
