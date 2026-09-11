"""A36: the optical trS knob after random-permutation honours p.

The optical knob s is the permutation probability of every component
(`_knob_to_augment_kwargs`: p = s), the grid is `linspace(0.2, 0.99)`, and
`RandomPermutation` permutes each row with probability p (default 1.0, which
keeps every other optical experiment bit-identical). Four legs:

  (i)   bit-identity at p = 1: the pre-change `__call__` (an inline copy, the
        reference lives here) and the new class give the same (GX, G) under the
        same seed AND leave the global RNG in the same state, so no mask is drawn
        at p = 1;
  (ii)  p = 0 returns X unchanged with identity rows in G; p = 0.5 on 4000 rows
        permutes a fraction within 0.5 +- 0.03 (4 SD);
  (iii) the knob map is p = s on the grid; the optical grid is
        `linspace(0.2, 0.99, n)` with exact endpoints; the sim grid is untouched;
  (iv)  fixture pin, the production `ExpansionStrategy` on the shipped chain
        (seed 42, experiment 0, grid [0.2, 0.4, 0.6, 0.8, 0.99], DA only, no
        solves): tr(S)/k strictly increasing, pinned to the measured values,
        span >= 0.2, rho at s = 0.2 <= 1.6. Fails on the pre-change code
        (span 0.06, no order).

    python scripts/a36_trs_knob.py
"""

import os
import sys

import numpy as np
from loguru import logger

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from src.data_augmentors.optical_device import RandomPermutation  # noqa: E402
from src.experiments.configs import PARAM_SPECS  # noqa: E402
from src.experiments.optical_device import OpticalOrchestrator, _knob_to_augment_kwargs  # noqa: E402
from src.experiments.utils import set_seed  # noqa: E402

CHAIN = "rotation > hflip > vflip > random-permutation"
FIXTURE_GRID = [0.2, 0.4, 0.6, 0.8, 0.99]
# measured 2026-09-11 on this fixture (seed 42, experiment 0); the reviewer's
# probe read 0.7944, 0.8493, 0.9134, 1.0099, 1.0924 at four decimals
EXPECT_TRS = [0.794393, 0.849343, 0.913383, 1.009853, 1.092403]
PIN_RTOL = 1e-5
FAIL = []


def check(tag, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {tag} {detail}")
    if not ok:
        FAIL.append(tag)


def reference_call(self, X, **kwargs):
    """`RandomPermutation.__call__` as shipped before p was honoured: every row permuted."""
    N, M = X.shape
    permutation_vector = np.arange(M, dtype=int)
    GX = np.zeros_like(X)
    G = np.zeros_like(X, dtype=int)
    for i in range(N):
        np.random.shuffle(permutation_vector)
        G[i, :] = permutation_vector
        x, g = X[i, :], G[i, :]
        GX[i, :] = self.augment(x, g)
    G = (G - G.mean(axis=1)[:, np.newaxis]) / G.std(axis=1)[:, np.newaxis]
    return GX, G


def identity_row(M):
    a = np.arange(M, dtype=float)
    return (a - a.mean()) / a.std()


def leg_i():
    print("(i) bit-identity at p = 1")
    X = np.random.default_rng(1).normal(size=(500, 9))
    np.random.seed(0)
    GX_ref, G_ref = reference_call(RandomPermutation(), X)
    state_ref = np.random.get_state()
    np.random.seed(0)
    GX_new, G_new = RandomPermutation()(X)
    state_new = np.random.get_state()
    check("(i) default p is 1.0", RandomPermutation().p == 1.0)
    check("(i) GX bit-identical at p = 1", np.array_equal(GX_ref, GX_new))
    check("(i) G bit-identical at p = 1", np.array_equal(G_ref, G_new))
    check(
        "(i) global RNG state identical after the call",
        np.array_equal(state_ref[1], state_new[1]) and state_ref[2:] == state_new[2:],
    )
    np.random.seed(0)
    GX_kw, G_kw = RandomPermutation()(X, p=1.0)
    check(
        "(i) p=1.0 passed at call time is the same path", np.array_equal(GX_kw, GX_new) and np.array_equal(G_kw, G_new)
    )


def leg_ii():
    print("(ii) p = 0 and p = 0.5")
    X = np.random.default_rng(2).normal(size=(4000, 9))
    np.random.seed(0)
    GX0, G0 = RandomPermutation(p=0.0)(X)
    check("(ii) p = 0 returns X", np.array_equal(GX0, X))
    check("(ii) p = 0 G rows are the identity permutation", np.allclose(G0, identity_row(9)[None, :]))
    np.random.seed(0)
    GX5, G5 = RandomPermutation(p=0.5)(X)
    permuted = ~np.all(np.isclose(G5, identity_row(9)[None, :]), axis=1)
    frac = permuted.mean()
    check("(ii) p = 0.5 permutes 0.5 +- 0.03 of 4000 rows", abs(frac - 0.5) <= 0.03, f"{frac:.4f}")
    check("(ii) p = 0.5 kept rows equal X", np.array_equal(GX5[~permuted], X[~permuted]))
    check("(ii) p = 0.5 permuted rows differ from X", not np.any(np.all(GX5[permuted] == X[permuted], axis=1)))


def leg_iii():
    print("(iii) knob map and grids")
    grid = PARAM_SPECS["trS"].grid_fn("optical_device", 7)
    want = np.linspace(0.2, 0.99, 7)
    check("(iii) optical grid == linspace(0.2, 0.99, 7)", np.array_equal(grid, want), f"{np.round(grid, 4).tolist()}")
    check("(iii) optical grid endpoints exact", grid[0] == 0.2 and grid[-1] == 0.99)
    check(
        "(iii) _knob_to_augment_kwargs(s)['p'] == s on the grid",
        all(_knob_to_augment_kwargs(s)["p"] == s for s in grid),
    )
    sim = PARAM_SPECS["trS"].grid_fn("simulation", 7)
    check("(iii) sim grid unchanged (logspace(-1.5, 1.0, 7))", np.array_equal(sim, np.logspace(-1.5, 1.0, num=7)))


def leg_iv():
    print("(iv) fixture pin on the production strategy")
    set_seed(42)
    orch = OpticalOrchestrator(
        seed=42,
        n_samples=1000,
        n_experiments=1,
        sweep_samples=len(FIXTURE_GRID),
        methods=["PI", "DA+PI"],
        hyperparameters={},
        n_jobs=1,
        recalibrate=False,
        pad=False,
        clipy=False,
        mean_match=True,
        augmentation=CHAIN,
    )
    runner = orch.get_sweep_runner_cls("trS")(
        methods=orch.methods,
        method_factory=orch.build_methods,
        param_grid_override=FIXTURE_GRID,
        **orch._get_clean_kwargs(),
    )
    check(
        "(iv) the runner maps the knob through _knob_to_augment_kwargs",
        runner.augment_kwargs_fn is _knob_to_augment_kwargs,
    )
    for s in FIXTURE_GRID:
        runner.generate_data(0, s)
    factors = np.array([runner._factors[(0, float(s))] for s in FIXTURE_GRID])
    rho, trs = factors[:, 0], factors[:, 1]
    print(f"      trS {np.round(trs, 6).tolist()}")
    print(f"      rho {np.round(rho, 4).tolist()}; rho trS {np.round(rho * trs, 4).tolist()}")
    check("(iv) tr(S)/k strictly increasing along the grid", bool(np.all(np.diff(trs) > 0)))
    check(
        "(iv) tr(S)/k pinned to the measured values",
        np.allclose(trs, EXPECT_TRS, rtol=PIN_RTOL, atol=0),
        f"vs {EXPECT_TRS}",
    )
    check("(iv) span max - min >= 0.2", trs.max() - trs.min() >= 0.2, f"{trs.max() - trs.min():.4f}")
    check("(iv) rho at s = 0.2 <= 1.6", rho[0] <= 1.6, f"{rho[0]:.4f}")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    os.chdir(REPO)  # the optical loader reads data/ relative to the cwd; nothing is written
    leg_i()
    leg_ii()
    leg_iii()
    leg_iv()
    print(f"\n{'A36 ALL PASS' if not FAIL else 'A36 FAILURES: ' + ', '.join(FAIL)}")
    sys.exit(bool(FAIL))
