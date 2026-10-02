"""A85: the optical sweeps run `rotation > hflip > vflip > translate` with the
rotation / flip probability 0.25, and the query panel keeps
`rotation > hflip > vflip > random-permutation` at the default 0.5 (a few minutes
on 8 cores, most of it the three two-step sweeps of (v)).

translate moves brightness along the sum-zero subspace of the 9-pixel space,
x -> x + B t, the linearisation of the pixel-permutation symmetry it replaces in
the sweeps' chain. The rotation / flip probability is the dataset block's
`augmentation_p` (`OpticalDeviceDA(p=...)`, default `P` = 0.5). Legs:

  (i)   the basis: B is 9 x 8, orthonormal (B'B = I), sum-zero (1'B = 0) and spans
        all of 1-perp (B B' = I - 11'/9).
        Catches: a basis that leaks into the pixel sum or misses a direction.
  (ii)  the chains: config.yaml and every optical recipe block but the query's
        name `CHAIN` with `augmentation_p` `SWEEP_P` (0.25), and the resolved
        block carries it to the orchestrator; the query recipe (opticalDeviceFig6)
        names `QUERY_CHAIN` and no `augmentation_p`. `OpticalDeviceDA(CHAIN,
        p=SWEEP_P)` parses to rotation, hflip, vflip at 0.25 and translate at
        `TRANSLATION_SCALE` 0.5; `OpticalDeviceDA(QUERY_CHAIN)` to rotation,
        hflip, vflip at `P` 0.5 and random-permutation at its own 1.0
        (`P_RANDOM_PERMUTATION`, which `p` never reaches). translate is not in
        "all" and carries no invariance error (the robustness sweep's appended
        gaussian-noise is the only knob). `augmentation_p` outside (0, 1], a bool
        or a string is a config error, and the key is unknown on every other
        block.
        Catches: a recipe left on the old chain, a probability leaking into the
        query (a global one), a drifted p or scale.
  (iii) on device 8's pool, for both chains: every augmented row keeps its pixel
        sum; T has 11 columns (three flip labels, then the 8 translation draws)
        and 12 for the query chain (three flip labels, the 9 permutation codes);
        each flip fires on a share of rows within `RATE_ATOL` of its p; taking
        `TRANSLATION_SCALE` std(X) T[:, 3:] B' off GX leaves a pixel permutation
        of each row of X (the query chain's GX is one outright). Under the omega
        knob's p the step is p times the scale.
        Catches: T without the translation amounts, a step off its scale, a
        probability that does not reach the draw.
  (iv)  E[GX | t] = B t: over `N_DRAWS` augmentations of the pool, the least
        squares slope of GX on t = s std(X) T[:, 3:] is B' within `SLOPE_ATOL`
        and the intercept is the pool's mean (zero, centred) within the same
        multiple of std(X).
        Catches: a T that does not inform GX (the reason it is an instrument).
  (v)   bit for bit against the experimental implementation the full-scale
        optical run used (scratch o12tiv at 5ae6726, env TIV_FLIP_P=0.25
        TIV_TR_SCALE=0.5): two-experiment, two-step gamma, omega and epsilon
        sweeps on `CHAIN` at `SWEEP_P` (`im-ci` off), every metric array but the
        wall clock, the x grids (the epsilon sweep's radii), each experiment's
        oracle gamma* / eps* and the robustness DA's tuned strength, hashed to
        `DIGEST`; the orchestrator's DA factory (robustness append included)
        builds the rotation / flips at `SWEEP_P`. The experimental tree
        reads `--src DIR --record` (with its env knobs) to print its digest, and
        `--dump FILE` saves the hashed arrays for a direct comparison. One BLAS
        thread, as in `digest_leg`: the thread count moves gamma* at the last
        ulp. Record and compare on the same node (`DIGEST_NODE`): CPU models
        differ at the last ulp too.
        Catches: any change of the RNG draw order or streams.

    uv run python scripts/a85_sum_zero_translation.py [--only LEG]
"""

import argparse
import hashlib
import os
import platform
import sys

# one BLAS thread in this process and every worker, before numpy loads; see (v)
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from loguru import logger  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHAIN = "rotation > hflip > vflip > translate"
SWEEP_P = 0.25
QUERY_CHAIN = "rotation > hflip > vflip > random-permutation"
QUERY_RECIPE = "opticalDeviceFig6"
OPTICAL_RECIPES = (
    "validityFig9",
    "sharpnessInformativenessFig10",
    "robustnessFig11",
    "recalibrationFig12",
    "nEfficiencyFig13",
    "mEfficiencyFig14",
    "latencyFig15",
    "stabilityFig16",
)
SEED = 42
N_JOBS = 8
N_DRAWS = 40
SLOPE_ATOL = 0.05
RATE_ATOL = 0.03  # ~ 3.5 binomial sd on the 1000-row pool at p 0.5
PARAMS = ("gamma", "omega", "epsilon")
# recorded off the experimental tree (scratch o12tiv at 5ae6726, TIV_FLIP_P=0.25
# TIV_TR_SCALE=0.5) with `--src <tree> --record`, on DIGEST_NODE
DIGEST = "510fa1321fdb41eb8c0a97590184e558890606c4"
DIGEST_NODE = "atl1-1-01-005-11-0"
FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAIL.append(name)


def leg_i():
    from src.data_augmentors.optical_device import SumZeroTranslation

    print("(i) the sum-zero basis")
    B = SumZeroTranslation().B
    one = np.ones(9)
    check("(i) B is 9 x 8", B.shape == (9, 8), f"{B.shape}")
    check("(i) orthonormal", np.allclose(B.T @ B, np.eye(8), atol=1e-12))
    check("(i) sum-zero", np.allclose(one @ B, 0.0, atol=1e-12))
    check("(i) spans 1-perp", np.allclose(B @ B.T, np.eye(9) - np.outer(one, one) / 9, atol=1e-12))


def _block(path):
    with open(path) as handle:
        return yaml.safe_load(handle).get("optical_device") or {}


def _rejects(block, name="optical_device"):
    from src.experiments.configs import resolve_dataset_block

    try:
        resolve_dataset_block(name, block)
    except ValueError:
        return True
    return False


def leg_ii():
    from src.data_augmentors import optical_device as od
    from src.experiments.configs import ROBUSTNESS_AUGMENTATION, resolve_dataset_block
    from src.experiments.optical_device import _with_component

    print("(ii) the chains")
    blocks = {"config.yaml": _block(os.path.join(REPO, "config.yaml"))}
    blocks |= {recipe: _block(os.path.join(REPO, "recipes", f"{recipe}.yaml")) for recipe in OPTICAL_RECIPES}
    for name, block in blocks.items():
        chain, p = block.get("augmentation"), block.get("augmentation_p")
        check(f"(ii) {name}: {CHAIN!r} at p {SWEEP_P}", chain == CHAIN and p == SWEEP_P, f"{chain!r} p {p!r}")
        resolved = resolve_dataset_block("optical_device", {k: v for k, v in block.items() if k != "experiment"})
        check(f"(ii) {name}: the resolved block carries it", resolved.get("augmentation_p") == SWEEP_P)
    query = _block(os.path.join(REPO, "recipes", f"{QUERY_RECIPE}.yaml"))
    check(
        f"(ii) {QUERY_RECIPE}: {QUERY_CHAIN!r}, no augmentation_p",
        query.get("augmentation") == QUERY_CHAIN and "augmentation_p" not in query,
        f"{query.get('augmentation')!r} p {query.get('augmentation_p', '-')!r}",
    )

    da = od.OpticalDeviceDA(CHAIN, p=SWEEP_P)
    names = [a.augmentation for a in da._augmentations]
    check("(ii) components", names == ["rotation", "hflip", "vflip", "translate"], f"{names}")
    ps = [a.p for a in da._augmentations[:3]]
    check(f"(ii) rotation / flip p {SWEEP_P}", ps == [SWEEP_P] * 3, f"{ps}")
    scale = da._augmentations[3].scale
    check("(ii) translate's scale 0.5", od.TRANSLATION_SCALE == 0.5 and scale == 0.5, f"{scale}")
    check("(ii) translate not in all", "translate" not in od.ALL_AUGMENTATIONS)
    check("(ii) no invariance error in the chain", da._inexact == [] and da.strength is None)
    robust = _with_component(CHAIN, ROBUSTNESS_AUGMENTATION["optical_device"])
    check("(ii) robustness chain", robust == f"{CHAIN} > gaussian-noise", f"{robust!r}")
    check(
        "(ii) its only knob is gaussian-noise",
        [a.augmentation for a in od.OpticalDeviceDA(robust, p=SWEEP_P)._inexact] == ["gaussian-noise"],
    )

    da = od.OpticalDeviceDA(QUERY_CHAIN)
    names = [a.augmentation for a in da._augmentations]
    want = ["rotation", "hflip", "vflip", "random-permutation"]
    check("(ii) query chain components", names == want, f"{names}")
    ps = [a.p for a in da._augmentations]
    check(
        "(ii) query chain p 0.5 x 3 (`P`), then random-permutation's own 1.0",
        od.P == 0.5 and od.P_RANDOM_PERMUTATION == 1.0 and ps == [0.5] * 3 + [1.0],
        f"{ps}",
    )
    ps = [a.p for a in od.OpticalDeviceDA(QUERY_CHAIN, p=SWEEP_P)._augmentations]
    check("(ii) p never reaches random-permutation", ps == [SWEEP_P] * 3 + [1.0], f"{ps}")

    base = {"seed": 42, "augmentation": CHAIN}
    for bad in (0, 0.0, 1.5, -0.1, True, "0.25", None):
        check(f"(ii) augmentation_p {bad!r} rejected", _rejects({**base, "augmentation_p": bad}))
    for good in (1, 0.25):
        check(f"(ii) augmentation_p {good!r} resolves", not _rejects({**base, "augmentation_p": good}))
    check("(ii) absent resolves", "augmentation_p" not in resolve_dataset_block("optical_device", base))
    check(
        "(ii) unknown on the cigarette block",
        _rejects(
            {"seed": 42, "augmentation": "translate", "target": "iv", "spec": "t3", "augmentation_p": 0.25},
            "cigarettes",
        ),
    )


def _pool():
    from src.experiments.configs import OPTICAL_CONFIG
    from src.sem.optical_device import OpticalDeviceSEM

    return OpticalDeviceSEM(experiment=OPTICAL_CONFIG.dataset_index).X


def _unshift(X, GX, T, s):
    """GX less the translation T's last 8 columns encode."""
    from src.data_augmentors.optical_device import SumZeroTranslation

    return GX - s * np.std(X) * T[:, -8:] @ SumZeroTranslation().B.T


def leg_iii():
    from src.data_augmentors.optical_device import TRANSLATION_SCALE, OpticalDeviceDA

    print("(iii) the augmented rows and T")
    X = _pool()
    for tag, chain, p, width in (("chain", CHAIN, SWEEP_P, 11), ("query chain", QUERY_CHAIN, 0.5, 12)):
        kwargs = {"p": p} if chain == CHAIN else {}
        np.random.seed(SEED)
        GX, T = OpticalDeviceDA(chain, **kwargs).augment(X)
        check(f"(iii) {tag}: pixel sums kept", np.allclose(GX.sum(axis=1), X.sum(axis=1), atol=1e-10))
        check(f"(iii) {tag}: T has {width} columns", T.shape == (len(X), width), f"{T.shape}")
        rates = [float(np.mean(T[:, k] > 0)) for k in range(3)]
        flips = all(len(np.unique(T[:, k])) == 2 for k in range(3))
        near = all(abs(rate - p) < RATE_ATOL for rate in rates)
        check(f"(iii) {tag}: the first three are flip labels firing at p {p}", flips and near, f"{np.round(rates, 3)}")
        for knob in (None, 0.4):
            np.random.seed(SEED)
            GX, T = OpticalDeviceDA(chain, **kwargs).augment(X, **({} if knob is None else {"p": knob}))
            rest = _unshift(X, GX, T, TRANSLATION_SCALE * (1.0 if knob is None else knob)) if chain == CHAIN else GX
            same = np.allclose(np.sort(rest, axis=1), np.sort(X, axis=1), atol=1e-10)
            what = "GX less B t" if chain == CHAIN else "GX"
            check(f"(iii) {tag}: {what} is a pixel permutation of X (knob p {knob})", same)


def leg_iv():
    from src.data_augmentors.optical_device import TRANSLATION_SCALE, OpticalDeviceDA, SumZeroTranslation

    print(f"(iv) E[GX | t] = B t over {N_DRAWS} draws")
    X = _pool()
    np.random.seed(SEED)
    da, rows, ts = OpticalDeviceDA(CHAIN, p=SWEEP_P), [], []
    for _ in range(N_DRAWS):
        GX, T = da.augment(X)
        rows.append(GX)
        ts.append(TRANSLATION_SCALE * np.std(X) * T[:, 3:])
    GX, t = np.vstack(rows), np.vstack(ts)
    design = np.hstack([np.ones((len(t), 1)), t])
    coef = np.linalg.lstsq(design, GX, rcond=None)[0]
    B = SumZeroTranslation().B
    gap = float(np.abs(coef[1:] - B.T).max())
    check(f"(iv) slope B' within {SLOPE_ATOL}", gap < SLOPE_ATOL, f"max |slope - B'| {gap:.4f}")
    off = float(np.abs(coef[0] - X.mean(axis=0)).max() / np.std(X))
    check(f"(iv) intercept the pool's mean within {SLOPE_ATOL} std(X)", off < SLOPE_ATOL, f"{off:.4f}")
    print(f"      pool mean max |.| {np.abs(X.mean(axis=0)).max():.2e}")


def sweeps(experimental=False):
    """The arrays (v) hashes, in order: [(name, array)]. The experimental tree takes
    its probability from the env, not the `augmentation_p` it predates."""
    from src.experiments.optical_device import OpticalOrchestrator
    from src.experiments.utils import set_seed

    arrays = []
    for param in PARAMS:
        set_seed(SEED)
        orch = OpticalOrchestrator(
            seed=SEED,
            n_samples=1000,
            n_experiments=2,
            sweep_samples=2,
            methods=["PI+INV", "PI", "DA+PI", "DA+PI+IV(T)", "PI&DA+PI", "PI&DA+PI+IV(T)"],
            hyperparameters={},
            n_jobs=N_JOBS,
            augmentation=CHAIN,
            **({} if experimental else {"augmentation_p": SWEEP_P}),
        )
        runner = orch.get_sweep_runner_cls(param)(
            methods=orch.methods, method_factory=orch.build_methods, **orch._get_clean_kwargs()
        )
        x, results, _ = runner.run(f"a85 {param}")
        oracles = [(runner.get_oracle(j).gamma_star, runner.get_oracle(j).epsilon_star) for j in range(2)]
        strengths = [runner.das[j].strength for j in range(2)]
        if not experimental:
            # the factory's DAs: the omega knob overrides p at call time, so the
            # runner's DAs hold its last step
            das = (orch._da_factory(), orch._da_factory(append="gaussian-noise"))
            ps = {a.p for da in das for a in da._augmentations if a.augmentation in ("rotation", "hflip", "vflip")}
            check(f"(v) {param}: the sweep DA factory's rotation / flip p {SWEEP_P}", ps == {SWEEP_P}, f"{ps}")
        print(
            f"      {param}: x {np.round(np.asarray(x, dtype=float), 6)} (gamma*, eps*) {oracles} strength {strengths}"
        )
        arrays.append((f"{param}|x", np.asarray(x, dtype=float)))
        arrays.append((f"{param}|oracles", np.asarray(oracles, dtype=float)))
        arrays.append((f"{param}|strength", np.asarray([np.nan if s is None else s for s in strengths], dtype=float)))
        for method in sorted(results):
            for metric in sorted(results[method]):
                if metric != "wall_clock":
                    arrays.append((f"{param}|{method}|{metric}", np.asarray(results[method][metric], dtype=float)))
    return arrays


def digest(arrays):
    """sha1 over (v)'s arrays: each param's name, then every array's bytes (the
    metric arrays behind their `method|metric` key)."""
    h = hashlib.sha1(usedforsecurity=False)
    for name, array in arrays:
        param, _, rest = name.partition("|")
        if rest == "x":
            h.update(param.encode())
        elif "|" in rest:
            h.update(rest.encode())
        h.update(np.ascontiguousarray(array).tobytes())
    return h.hexdigest()


def leg_v(dump=None):
    print("(v) bit for bit against the experimental implementation")
    node = platform.node().split(".")[0]
    if node != DIGEST_NODE:
        print(f"  [WARN] (v) on {node}, the digest was recorded on {DIGEST_NODE}: a mismatch means nothing here")
    arrays = sweeps()
    if dump:
        np.savez(dump, **dict(arrays))
    got = digest(arrays)
    check("(v) digest", got == DIGEST, f"{got} (recorded {DIGEST})")


LEGS = {"i": leg_i, "ii": leg_ii, "iii": leg_iii, "iv": leg_iv, "v": leg_v}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=sorted(LEGS), default=None)
    parser.add_argument("--src", default=REPO, help="the tree to import `src` from")
    parser.add_argument("--record", action="store_true", help="print (v)'s digest of --src and exit")
    parser.add_argument("--dump", default=None, help="save (v)'s arrays to this .npz")
    args = parser.parse_args()
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    sys.path.insert(0, os.path.abspath(args.src))
    os.chdir(REPO)
    if args.record:
        arrays = sweeps(experimental=True)
        if args.dump:
            np.savez(args.dump, **dict(arrays))
        print(digest(arrays))
        sys.exit(0)
    for name, leg in LEGS.items():
        if args.only in (None, name):
            leg(args.dump) if name == "v" else leg()
    print(f"\n{'A85 ALL PASS' if not FAIL else 'A85 FAILURES: ' + ', '.join(FAIL)}")
    sys.exit(bool(FAIL))
