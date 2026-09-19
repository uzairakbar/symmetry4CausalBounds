"""A59: the simulation SEM generates a rank-m instrument, guarded so m = 0 is today's run.

refactor4 touches `src/sem/simulation.py` (the `iv_dim` argument, the guarded
`U` / `A` / `root` draw at construction, the guarded `sample` branch returning
[X | Z] on both the observational and the interventional draw) and
`src/experiments/simulation.py` (the `iv` key forwarded to the SEM factory). Legs:

  (D)   the digest leg (scripts/digest_leg.py), as a56 to a58: one query panel and
        one gamma sweep step per dataset at the shipped configuration, no `iv` key,
        against the reference recorded on 14509db, no tolerance. Catches: any moved
        number on the shipped path, in particular an unconditional draw of `U` at
        construction, which shifts every later SEM's f off the shared stream.
        Misses: every path with `iv > 0`.
  (i)   `iv: 0` written out reproduces the simulation digests exactly (the key absent
        is leg (D) itself), and the guard directly: at one seed the SEM built with
        `iv_dim=0` has the same `W_XXi` and `W_XY` as one built without the argument
        and leaves the stream at the same point, while `iv_dim=4` draws its `U` after
        both. Catches: `U` drawn unconditionally (SS5.1), a draw moved before `W_XY`.
        Misses: the non-digest half compares two constructions that share any
        unconditional draw, so it is blind to one; the digest half is the pin, and
        it sees the draw only when it consumes the stream (a `randn(d, 0)` basis
        consumes nothing and is harmless). Nothing about the instrument itself,
        which (ii) to (v) cover.
  (ii)  deterministic construction, no sampling: `A A' + root root' == I` and
        `U'U == I` to 1e-12, at the shipped `alpha` 1 and at 0.5. At `alpha` 1 the
        symmetric root coincides with `I - U U'`, so the 0.5 case is what separates
        the exact root from that shortcut. Catches: `randn @ (I - U U')` in place of
        the symmetric root, a non-orthonormal `U`. Misses: the draw.
  (iii) validity: `U' W_XXi` is 0 to 1e-14, and on the seed-42 draw of n 2048 the IV
        slack at the ERM centre `||Q_Z'(y - X h_erm)||/sqrt(n)` is under 1e-12
        (MEASURED p6: 0.00000; with alpha 1, X U = Z exactly, so the OLS residual is
        orthogonal to Z); the honest moment `||Q_Z'(y - X f)||/sqrt(n)` is recorded
        (p6: 0.0552). Catches: the projection off `W_XXi` dropped (the instrument
        then correlates with xi and the slack at the centre is order 0.1). Misses:
        a wrong loading at alpha < 1, which (ii) covers.
  (iv)  `gamma* == gamma_true` to 1e-12 with and without the instrument, read off
        the oracle. Catches: an `iv` branch that touches kappa, bias_sq or
        sigma_sq. Misses: nothing else.
  (v)   the SS5.2 ordering on one tiny draw, through the registry and `fit_model`:
        on `u in range(A)` PI+INV+IV is no wider than PI+IV and than PI+INV (within
        1e-6), and the whole width table is recorded beside p6's. Catches: the IV
        cone not reaching PI+INV+IV, an instrument that identifies nothing. Misses:
        the exact p6 widths, which depend on the DA draw and are recorded, not pinned.
  (vi)  the carrier on the shipped SEM (batch B hand-off): the interventional draw
        has k + m columns, `f` sees k columns through the sweep runner's own
        `_draw_base` and the query runner's `_load_data`, the recipe `validityFig9.yaml`
        resolves and its orchestrator hands the SEM factory `iv_dim` 4 (the sweep
        SEMs and the query SEM carry `iv_width` 4, Z_train and Z are (n, 4)), and on
        this SEM the two IV terms coincide to the bit (h* is exactly T-invariant, so
        `eps_iv_star` is 0 and the joint budget IS the Z piece). Catches: an
        interventional draw without its Z (the runner mis-slices), the key not
        forwarded, a Z that never reaches the runner. Misses: the solves at the
        recipe's full scale, which the recipe run is for.

    MPLBACKEND=Agg python scripts/a59_sim_iv.py [--seed 42] [--reference JSON] [--skip-digest]

`--skip-digest` drops leg (D) and exists for the break-it runs of the other legs;
the committed state is always gated with it on.
"""

import argparse
import json
import os
import sys

import numpy as np
import yaml
from loguru import logger

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import digest_leg  # noqa: E402
from munch import munchify  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

import src.sem.simulation as sim_module  # noqa: E402
from src.data_augmentors.simulation import NullSpaceTranslation  # noqa: E402
from src.experiments.configs import (  # noqa: E402
    EPS_TOL,
    FLOOR_GUARD_R,
    SIMULATION_CONFIG,
    MethodRegistry,
    resolve_dataset_block,
)
from src.experiments.simulation import SimulationOrchestrator  # noqa: E402
from src.experiments.utils import set_seed  # noqa: E402
from src.experiments.utils.constants import iv_mode, parse_method  # noqa: E402
from src.experiments.utils.metrics import rho_hat  # noqa: E402
from src.experiments.utils.model_fitting import fit_model  # noqa: E402
from src.methods.sensitivity_models import constraint_floor  # noqa: E402
from src.oracle import gamma_star  # noqa: E402
from src.sem.simulation import IV_ALPHA, LinearSimulationSEM  # noqa: E402

D, M, N = 32, 4, 2048
# the simulation block with `iv: 4` that owns the gamma sweep this leg drives;
# the recipes are split by experiment type, so the sweep lives in its own file
RECIPE = "validityFig9.yaml"
TOGGLES = dict(recalibrate=True, pad=False, clipy=False, mean_match=True, n_jobs=1)
IV_BOUND = 0.05  # p6's evi
NAMES = ["PI", "PI+IV", "PI+INV", "PI+INV+IV", "DA+PI+IV"]
# p6, seed 42, d 32, m 4, n 2048, kernel_dim 0: widths (ratio to PI) on the four
# query directions, recorded beside the reproduction, never pinned
P6_WIDTHS = {
    "PI": (1.3847, 1.3991, 1.4183, 1.4672),
    "PI+IV": (0.2792, 1.3991, 1.3675, 1.4040),
    "PI+INV": (0.3152, 0.1252, 0.7504, 0.3582),
    "PI+INV+IV": (0.1125, 0.0824, 0.6156, 0.3264),
    "DA+PI+IV": (0.1351, 0.1221, 1.0198, 0.5939),
}
FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAIL.append(name)


# the +IV classes this leg can fixture on, best first. The recipe decides which of
# them run, so the gate takes the first the block lists rather than pinning one; the
# intersection (`PI&DA+PI+IV`) is deliberately absent, it is not a bare IV class
IV_FIXTURES = ("PI+IV", "PI+INV+IV", "DA+PI+IV", "DA+PI+IV(Z)")


def iv_fixture(methods, label):
    """The first `IV_FIXTURES` entry the block lists, asserted present."""
    folded = {parse_method(name)[0] if iv_mode(name) == "T,Z" else name for name in methods}
    name = next((n for n in IV_FIXTURES if n in folded), None)
    check(f"{label}: the block lists a +IV class to fixture on", name is not None, f"{sorted(folded)}")
    return name


def seeded_sem(seed, **kwargs):
    np.random.seed(seed)
    return LinearSimulationSEM(treatment_dimension=D, gamma=SIMULATION_CONFIG.gamma_true, **kwargs)


def recipe_block():
    with open(os.path.join(REPO, "recipes", RECIPE)) as handle:
        config = yaml.safe_load(handle)
    defaults = config.pop("defaults", {}) or {}
    return {**defaults, **config["simulation"]}


# ------------------------------------------------------------------ legs


def leg_d(reference):
    print("(D) the shipped configuration does not move")
    for label, ok, detail in digest_leg.compare(REPO, reference):
        check(f"(D) {label}", ok, detail)


def leg_i(seed, reference=None):
    """`reference` None skips the digest half (a break-it run)."""
    print("(i) iv: 0 is today's run, and the U draw is guarded")
    plain, explicit, wide = seeded_sem(seed), seeded_sem(seed, iv_dim=0), seeded_sem(seed, iv_dim=M)
    same = np.array_equal(plain.W_XXi, explicit.W_XXi) and np.array_equal(plain.W_XY, explicit.W_XY)
    check("(i) iv_dim=0 draws the same W_XXi and W_XY as no argument", same)
    np.random.seed(seed)
    LinearSimulationSEM(treatment_dimension=D, gamma=SIMULATION_CONFIG.gamma_true)
    after_plain = np.random.randn()
    np.random.seed(seed)
    LinearSimulationSEM(treatment_dimension=D, gamma=SIMULATION_CONFIG.gamma_true, iv_dim=0)
    after_explicit = np.random.randn()
    check("(i) iv_dim=0 leaves the stream where no argument leaves it", after_plain == after_explicit)
    same_head = np.array_equal(plain.W_XXi, wide.W_XXi) and np.array_equal(plain.W_XY, wide.W_XY)
    check("(i) iv_dim=4 draws W_XXi and W_XY first, U after", same_head and hasattr(wide, "U"))
    check("(i) iv_dim=0 has no U and iv_width 0", not hasattr(explicit, "U") and explicit.iv_width == 0)
    if reference is None:
        print("      (i) iv: 0 digest half SKIPPED by --skip-digest")
        return

    with open(reference) as handle:
        want = json.load(handle)["digests"]["simulation"]
    shipped = digest_leg.BLOCKS["simulation"]
    digest_leg.BLOCKS["simulation"] = {**shipped, "iv": 0}
    try:
        digest_leg._enter(REPO)
        with threadpool_limits(limits=1):
            got = digest_leg.run_all(datasets=("simulation",), quiet=True)["simulation"]
    finally:
        digest_leg.BLOCKS["simulation"] = shipped
    for artifact in sorted(set(want) | set(got)):
        ok = want.get(artifact) is not None and want.get(artifact) == got.get(artifact)
        check(f"(i) iv: 0 written out: simulation/{artifact}", ok, f"{want.get(artifact)} vs {got.get(artifact)}")


def leg_ii(seed):
    print("(ii) deterministic construction")
    for alpha in (IV_ALPHA, 0.5):
        sem = seeded_sem(seed, iv_dim=M, alpha=alpha)
        identity = np.abs(sem.A @ sem.A.T + sem.root @ sem.root.T - np.eye(D)).max()
        check(f"(ii) alpha {alpha}: A A' + root root' == I to 1e-12", identity < 1e-12, f"{identity:.2e}")
        orthonormal = np.abs(sem.U.T @ sem.U - np.eye(M)).max()
        check(f"(ii) alpha {alpha}: U'U == I to 1e-12", orthonormal < 1e-12, f"{orthonormal:.2e}")
        check(f"(ii) alpha {alpha}: rank(A) = {M}", np.linalg.matrix_rank(sem.A) == M)
    check("(ii) the shipped alpha is 1", IV_ALPHA == 1.0 and seeded_sem(seed, iv_dim=M).alpha == 1.0)


def leg_iii(seed):
    print("(iii) validity: U orthogonal to the confounding, exact slack at the ERM centre")
    sem = seeded_sem(seed, iv_dim=M)
    projection = np.abs(sem.U.T @ sem.W_XXi).max()
    check("(iii) U' W_XXi is 0 to 1e-14", projection < 1e-14, f"{projection:.2e}")
    XZ, y = sem(N=N)
    X, Z = sem.split_instruments(XZ)
    f = sem.W_XY.ravel()
    h_erm = np.linalg.lstsq(X, y.ravel(), rcond=None)[0]
    Q = np.linalg.qr(Z)[0]
    slack = float(np.linalg.norm(Q.T @ (y.ravel() - X @ h_erm)) / np.sqrt(N))
    moment = float(np.linalg.norm(Q.T @ (y.ravel() - X @ f)) / np.sqrt(N))
    cross = float(np.abs(Z.T @ (y.ravel() - X @ f)).max() / N)
    check("(iii) ||Q_Z'(y - X h_erm)||/sqrt(n) < 1e-12 at seed 42, n 2048", slack < 1e-12, f"{slack:.2e}")
    print(f"      RECORDED: ||Q_Z'(y - X f)||/sqrt(n) = {moment:.5f} (p6 0.05523), max|Z'(y - Xf)|/n = {cross:.5f}")
    check("(iii) the honest moment is sampling noise around 0, not 0", 0.01 < moment < 0.2, f"{moment:.5f}")


def leg_iv(seed):
    print("(iv) gamma* == gamma_true")
    for iv_dim in (0, M):
        sem = seeded_sem(seed, iv_dim=iv_dim)
        gap = abs(gamma_star(sem) - SIMULATION_CONFIG.gamma_true)
        check(f"(iv) iv_dim={iv_dim}: gamma* == gamma_true to 1e-12", gap < 1e-12, f"gap {gap:.2e}")


def leg_v(seed):
    print("(v) the SS5.2 ordering on one draw, through the registry and fit_model")
    sem = seeded_sem(seed, iv_dim=M)
    XZ, y = sem(N=N)
    X, Z = sem.split_instruments(XZ)
    da = NullSpaceTranslation(sem.W_XY, kernel_dim=0)
    GX, G = da(X)
    gamma = float(sem.bias_sq / sem.sigma_sq)
    rho = float(rho_hat(X, GX, y, intercept=True))
    builders = MethodRegistry.build_methods(
        NAMES, gamma=gamma, epsilon=EPS_TOL, epsilon_iv=IV_BOUND, epsilon_iv_z=IV_BOUND, rho=rho, **TOGGLES
    )
    f = sem.W_XY.ravel()
    u1 = sem.U[:, 0]
    kernel = da.W_ZXtilde
    u2 = kernel[0] - sem.U @ (sem.U.T @ kernel[0])
    u2 /= np.linalg.norm(u2)
    u3 = np.random.randn(D)
    u3 -= sem.U @ (sem.U.T @ u3)
    u3 -= kernel.T @ np.linalg.lstsq(kernel.T, u3, rcond=None)[0]
    u3 /= np.linalg.norm(u3)
    queries = np.vstack([u1, u2, u3, np.eye(D)[0]])
    labels = ("u in range(A)", "u in ker(DA), IV-blind", "blind to both", "e_0")
    widths, covered = {}, {}
    for name in NAMES:
        model = builders[name]()
        fit_model(model=model, method_name=name, X=X, y=y, GX=GX, G=G, Z=Z, da=da)
        out = model.predict(queries, gamma=gamma)
        widths[name] = out[:, 1] - out[:, 0]
        covered[name] = [bool(out[j, 0] <= queries[j] @ f <= out[j, 1]) for j in range(4)]
        check(f"(v) {name} solves OK on the four directions", np.all(np.isfinite(out)))
    print(f"      RECORDED width (ratio to PI) [p6]: {' | '.join(labels)}")
    for name in NAMES:
        cells = [
            f"{widths[name][j]:.4f} ({widths[name][j] / widths['PI'][j]:.3f}) [{P6_WIDTHS[name][j]:.4f}]"
            for j in range(4)
        ]
        print(f"      {name:10s} " + " | ".join(cells) + f"  covered {covered[name]}")
    w = {name: widths[name][0] for name in NAMES}
    check("(v) on range(A): PI+INV+IV <= PI+IV within 1e-6", w["PI+INV+IV"] <= w["PI+IV"] + 1e-6)
    check("(v) on range(A): PI+INV+IV <= PI+INV within 1e-6", w["PI+INV+IV"] <= w["PI+INV"] + 1e-6)
    check("(v) on range(A): PI+IV is under a third of PI", w["PI+IV"] < w["PI"] / 3, f"{w['PI+IV'] / w['PI']:.3f}")
    blind = widths["PI+IV"][1]
    check("(v) IV-blind direction: PI+IV equals PI to 1e-6", abs(blind - widths["PI"][1]) < 1e-6)
    check("(v) f is covered by every method on every direction", all(all(c) for c in covered.values()))


def leg_vi(seed):
    print("(vi) the carrier on the shipped SEM and the recipe's orchestrator")
    sem = seeded_sem(seed, iv_dim=M)
    XZ_test, _ = sem(N=64, intervention=True)
    check("(vi) sample(intervention=True) has k + m columns", XZ_test.shape[1] == D + M, f"{XZ_test.shape}")
    XZ_obs, _ = sem(N=64)
    check("(vi) the observational draw has k + m columns too", XZ_obs.shape[1] == D + M)

    block = resolve_dataset_block("simulation", recipe_block())
    check(f"(vi) recipes/{RECIPE} resolves with iv 4", block.get("iv") == 4)
    reduced = {**block, "n_experiments": 1, "n_samples": 512, "sweep_samples": 4, "n_jobs": 1}
    set_seed(reduced["seed"])
    orchestrator = SimulationOrchestrator(**reduced, hyperparameters=munchify(digest_leg.HYPERPARAMETERS))
    check("(vi) the orchestrator carries iv_dim 4", orchestrator.iv_dim == 4)
    check("(vi) _sem_factory hands the SEM iv_dim 4", orchestrator._sem_factory().iv_width == 4)

    runner = orchestrator.get_sweep_runner_cls("gamma")(
        methods=orchestrator.methods, method_factory=orchestrator.build_methods, **orchestrator._get_clean_kwargs()
    )
    check("(vi) the sweep SEM carries iv_width 4", runner.sems[0].iv_width == 4)
    X_raw, X, y, X_test, estimand, Z = runner._base_data(0)
    check("(vi) _draw_base: X_train has k columns, Z_train m", X.shape[1] == D and Z.shape == (len(X), M))
    check(
        "(vi) _draw_base: X_test has k columns, f sees k", X_test.shape[1] == D and estimand.shape == (len(X_test), 1)
    )
    data = runner.generate_data(0, 1.0)
    check("(vi) SweepData.Z is (n, 4)", data.Z.shape == (len(data.X), M))
    eps_da, eps_z = runner.fit_epsilon_iv(0, 0, data), runner.fit_epsilon_iv_z(0, data)
    oracle = runner.get_oracle(0)
    check("(vi) declared_iv is False on the sim sweep runner", runner.declared_iv is False)
    check(
        "(vi) eps_iv_star is 0 on the sim (h* is T-invariant)",
        abs(oracle.eps_iv_star) < 1e-12,
        f"{oracle.eps_iv_star:.2e}",
    )
    check(
        "(vi) eps_iv_z_star > 0 on the sim (a real Z, sampling noise)",
        oracle.eps_iv_z_star > 0.01,
        f"{oracle.eps_iv_z_star:.5f}",
    )
    print(f"      RECORDED: epsilon_iv (T budget, guarded) {eps_da:.6f}, epsilon_iv_z (Z budget) {eps_z:.6f}")
    # the two no longer coincide: the T budget is `eps_iv_star` alone and the Z one
    # is `eps_iv_z_star`, each the radius of its own constraint
    floor_t = constraint_floor(
        data.GX,
        data.y,
        runner.fit_gamma(0),
        kind="iv",
        Z=np.asarray(data.G).reshape(len(data.GX), -1),
        mean_match=runner.mean_match,
        rho=runner.fit_rho(0, data),
        recalibrate=runner.recalibrate,
    )
    raw_t = float(oracle.eps_iv_star) + EPS_TOL
    check(
        "(vi) epsilon_iv is the T piece + EPS_TOL, or the guard's sqrt(9 floor_T)",
        eps_da == raw_t or abs(eps_da - np.sqrt(FLOOR_GUARD_R * floor_t)) < 1e-12,
        f"{eps_da:.6f} vs raw {raw_t:.6f}",
    )
    check(
        "(vi) epsilon_iv_z is the Z piece + EPS_TOL",
        abs(eps_z - (float(oracle.eps_iv_z_star) + EPS_TOL)) < 1e-12,
        f"{eps_z:.6f}",
    )

    query = orchestrator.get_query_runner_cls()(
        methods=orchestrator.methods, **{**orchestrator._get_clean_kwargs(), "n_experiments": 1}
    )
    check(
        "(vi) the query runner's Z is (n, 4) and X_raw has k columns",
        query.Z.shape == (len(query.X_raw), M) and query.X_raw.shape[1] == D,
    )
    fixture = iv_fixture(block["methods"], "(vi)")
    if fixture is None:
        return
    check(
        f"(vi) the query runner's {fixture} read the instrument",
        query.methods[fixture]().__class__.__name__ == "InstrumentalVariablePartialR2",
    )
    context = query.setup_data()
    model = query.methods[fixture]()
    fit_model(
        model=model,
        method_name=fixture,
        X=context.X,
        y=context.y,
        GX=context.GX,
        G=context.G,
        Z=context.Z,
        da=context.da,
    )
    check(
        f"(vi) {fixture} fitted through the query context sees a 4-column Z",
        model._has_iv and model.Z_projector_R.shape[0] == M + D,
    )


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    os.chdir(REPO)
    os.environ.setdefault("MPLBACKEND", "Agg")
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reference", default=digest_leg.DEFAULT_REFERENCE)
    parser.add_argument("--skip-digest", action="store_true")
    args = parser.parse_args()
    print(f"tree: {sim_module.__file__}")
    legs = [
        ("(ii)", leg_ii, (args.seed,)),
        ("(iii)", leg_iii, (args.seed,)),
        ("(iv)", leg_iv, (args.seed,)),
        ("(v)", leg_v, (args.seed,)),
        ("(vi)", leg_vi, (args.seed,)),
    ]
    legs.append(("(i)", leg_i, (args.seed, None if args.skip_digest else args.reference)))
    if not args.skip_digest:
        legs.append(("(D)", leg_d, (args.reference,)))
    for tag, leg, leg_args in legs:
        try:
            leg(*leg_args)
        except Exception as error:  # a raise is a FAIL line, and the later legs still report
            check(f"{tag} ran without raising", False, f"{type(error).__name__}: {error}")
    if args.skip_digest:
        print("(D) SKIPPED by --skip-digest: a break-it run, not the committed state")
    if not FAIL:
        print("A59 PASS")
    else:
        print(f"A59 FAIL: {FAIL}")
        sys.exit(1)
