"""A60: the cigarette SEM carries a configured instrument set, and the target follows it.

refactor5 touches `src/sem/cigarettes.py` (`instrument_set`, `restricted_fit` against
a configured set, `iv_columns` / `iv_width` / `iv_pool` on the SEM, `[X | Z]` draws
in both replicate modes) and `src/experiments/cigarettes.py` (`iv` and `gamma_z`
from the block, `declared_iv` on both runners, `gamma_z` forwarded into the
registry, the replicate rule of decision 9), plus `gamma_z > 0` under a non-empty
set in `configs.py` (a56 gains the case). Legs:

  (D)   the digest leg (scripts/digest_leg.py), as a56 to a59: the shipped blocks,
        no `iv` key, against the reference recorded on 14509db, no tolerance.
        Catches: a moved number on the anchor-set path (the target, the draw, the
        budgets, the replicate). Misses: every path with a configured set.
  (i)   with `iv: []` (the shipped block) the target is b_r, the anchor-set
        restricted fit: raw (-1.992, 0.507, 1.237, 0.249) to 5e-4 (SS1 fact 5, the
        plan's digits) and an independent anchor-set fit to 1e-10; the SEM has
        `iv_width` 0, no `iv_pool`, a k-column draw; both runners read
        `declared_iv` False and the registry hands `gamma_z` 0. Catches: a
        fallback that reads the configured-set code with an empty set, a
        `declared_iv` or `gamma_z` that leaks onto the shipped path. Misses: the
        numbers downstream, which (D) has.
  (ii)  the shipped set [tax_s, y, cpi] at t3 / own-tax gives p1's singular values
        (3.142, 2.475, 1.504) to 1e-3 and abs(v'd) = 0.407190 to 1e-6 (never the
        signed value, SS3.2), d_pn 0.940 to 1e-3; the SEM's `iv_pool` is that matrix
        bit for bit and both replicate modes return [X | Z] with Z's rows aligned to
        X's. Catches: a raw (non-FWL) column (SS2.1, p7: the geometry moves), a
        column in the wrong order, Z split before the rows are resampled. Misses:
        another anchor or spec, which (vi) covers for the spec.
  (iii) the phase-b target satisfies v'b = 0 and the three moments to 1e-12
        (MEASURED p5: 2.3e-16). Catches: a target off the wrong set or off a raw
        column. Misses: nothing about its coverage, which (v) has.
  (iv)  gamma*(b) = 0.3705 to 1e-4, on the SEM and through the runner's oracle
        (MEASURED p4, p5: 0.37052). Catches: the target built from the anchor set
        under a configured one (gamma* then reads 0.1922). Misses: nothing else.
  (v)   the replicate mechanism follows decision 9: with a non-empty `iv:` the
        sweep runner's SEMs carry `bootstrap` False (90% row splits) and both
        runners `declared_iv` True, the solver reads `gamma_z` 2^-8 with
        `epsilon_iv` 0 on PI+IV (bound exactly s sqrt(gamma_z)) and r_T on
        DA+PI+IV; on the runner's own 8 seeded row splits at gamma*(b) the
        coverage of b* on the four coefficient queries is 1.000 for PI and 1.000
        for PI+IV; the same runner class forced onto the state-cluster bootstrap
        reads PI+IV coverage under 1 (RECORDED beside p10's 0.500 / 0.375); with
        `iv: []` the runner is built with `bootstrap` True. Catches: `bootstrap`
        forced True under a configured set (the coverage then reads a resampling
        rate), `declared_iv` or `gamma_z` not forwarded. Misses: the production
        8-experiment mean over a full sweep (SS14).
  (vi)  the SS6.3 spec table: abs(v'd) and gamma*(b) to 1e-3 at every spec, the
        signed v'd at `s` to 1e-3 (-0.0243, under d_pn > 0), and the PI+IV
        feasibility floor at bound 0.0625 to 1e-3 (p8's column), so `s` is seen
        to degenerate. Catches: a spec-dependent slip in the controls or the set.
        Misses: the other anchor (SS14).
  Two hand-off items ride along: the query panel's fitted +IV models see the real
  Z (`PanelBuilder` forwards it, batch B ruling 2), and the overlap number of the
  batch A ruling, the oracle's `eps_iv_z_star` at the phase-b target, is RECORDED
  as the RMS over the 8 pooled DA draws beside p10's one-draw 0.000460.

    MPLBACKEND=Agg python scripts/a60_cigarettes_iv.py [--reference JSON] [--skip-digest]

`--skip-digest` drops leg (D) and exists for the break-it runs of the other legs;
the committed state is always gated with it on.
"""

import argparse
import os
import sys
from functools import partial

import numpy as np
import yaml
from loguru import logger

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import digest_leg  # noqa: E402
from munch import munchify  # noqa: E402

import src.sem.cigarettes as sem_module  # noqa: E402
from src.experiments.cigarettes import CigaretteOrchestrator  # noqa: E402
from src.experiments.configs import EPS_TOL, GAMMA_Z_DEFAULT, resolve_dataset_block  # noqa: E402
from src.experiments.generic_runner import STRATEGIES  # noqa: E402
from src.experiments.utils import PanelBuilder, set_seed  # noqa: E402
from src.experiments.utils.constants import iv_mode, parse_method  # noqa: E402
from src.methods.sensitivity_models import SolveStatus, constraint_floor  # noqa: E402
from src.sem.cigarettes import TREATMENTS, CigaretteSEM, build_design, instrument_set, null_basis  # noqa: E402

PHASE_B = ("tax_s", "y", "cpi")
IV_BOUND = 0.0625  # s sqrt(2^-8) on the sigma-normalised panel
B_R_RAW = (-1.992, 0.507, 1.237, 0.249)  # SS1 fact 5
SINGULAR_VALUES = (3.1421, 2.4750, 1.5042)  # p1
ABS_VD, D_PN = 0.407190, 0.9399  # p1
GAMMA_STAR_B = 0.37052  # p4, p5
# p8 M6: spec -> (signed v'd under d_pn > 0, gamma*(b), PI+IV floor at bound 0.0625)
SPEC_TABLE = {
    "s": (-0.0243, 2.9301, 0.1104),
    "t1": (0.3536, 0.3494, 0.1397),
    "t2": (1.0349, 0.4033, 0.1040),
    "t3": (0.4072, 0.3705, 0.1107),
    "t4": (0.4108, 0.4343, 0.1330),
}
P10_BOOTSTRAP_COVERAGE = {"PI": 0.500, "PI+IV": 0.375}
REPLICATES = 8
FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAIL.append(name)


def fold_default_mode(name):
    """`DA+PI+IV(T,Z)` and the bare `DA+PI+IV` are the SAME estimator, spelled two
    ways; the recipes spell it out and the headline figures key on the bare name
    (`src/experiments/cigarettes.py` folds it the same way before the lookup). Fold
    before keying on a method name, never respell the recipe."""
    return parse_method(name)[0] if iv_mode(name) == "T,Z" else name


def fold_keys(mapping):
    """`mapping` re-keyed through `fold_default_mode`.

    A block listing BOTH `DA+PI+IV` and `DA+PI+IV(T,Z)` folds them onto one key, so
    one of the two would vanish without a word. Production has the same collision
    (`src/experiments/cigarettes.py`), so this is a pre-existing hole the fold
    inherits rather than a new one -- but silent is what makes it a hole, and `a63`'s
    duplicate check is on the RAW list and cannot see it. Say so instead."""
    folded = {}
    for name, value in mapping.items():
        key = fold_default_mode(name)
        if key in folded:
            check(f"fold_keys: {name!r} and another spelling both fold onto {key!r}", False, f"{sorted(mapping)}")
        folded[key] = value
    return folded


def recipe_methods(**overrides):
    """The block's own method list, folded. The EXPECTATION is read from here, never
    from what production happened to build: intersecting production with itself
    cannot notice a method that stopped being built, it just drops the check."""
    return {fold_default_mode(name) for name in recipe_block(**overrides)["methods"]}


def listed(names, block_methods, produced, label):
    """`names` restricted to what the RECIPE lists, asserted non-empty, and asserted
    to have actually been produced.

    The recipe decides which methods run and the owner changes it freely, so the gate
    derives its expectation from the block; the non-emptiness check stops the
    derivation going vacuous, and the `produced` check stops a method the recipe lists
    disappearing from the run without a word."""
    kept = tuple(name for name in names if name in block_methods)
    check(f"{label}: the block lists at least one of {list(names)}", bool(kept), f"lists {sorted(block_methods)}")
    missing = [name for name in kept if name not in produced]
    check(f"{label}: every listed one was built", not missing, f"missing {missing} from {sorted(produced)}")
    return tuple(name for name in kept if name in produced)


def recipe_block(**overrides):
    with open(os.path.join(REPO, "recipes", "cigarettesFig7.yaml")) as handle:
        config = yaml.safe_load(handle)
    defaults = config.pop("defaults", {}) or {}
    block = {**defaults, **config["cigarettes"], "im-ci": 0}
    block.pop("experiment", None)
    return resolve_dataset_block("cigarettes", {**block, **overrides})


def shipped_block(**overrides):
    block = {**digest_leg.TOGGLES, **digest_leg.BLOCKS["cigarettes"], **overrides}
    return resolve_dataset_block("cigarettes", block)


def orchestrator(block):
    set_seed(block["seed"])
    return CigaretteOrchestrator(**block, hyperparameters=munchify(digest_leg.HYPERPARAMETERS))


def sweep_runner(orch, param="gamma", **extra):
    return orch.get_sweep_runner_cls(param)(
        methods=orch.methods, method_factory=orch.build_methods, **{**orch._get_clean_kwargs(), **extra}
    )


def query_runner(orch):
    return orch.get_query_runner_cls()(methods=orch.methods, **{**orch._get_clean_kwargs(), "n_experiments": 1})


def free_direction(design, Z):
    """The unit null direction of Q_Z'X, signed so d_pn > 0 (p8's convention)."""
    A = np.linalg.qr(Z)[0].T @ design.X
    d = np.linalg.svd(A)[2][Z.shape[1]]
    return d * np.sign(d[TREATMENTS.index("pn")]), np.linalg.svd(A, compute_uv=False)


def restricted_against(design, Z):
    """The restricted 2SLS point against Z, written out independently of the SEM."""
    N = null_basis()
    Q = np.linalg.qr(Z)[0]
    return N @ np.linalg.lstsq(Q.T @ (design.X @ N), Q.T @ design.y, rcond=None)[0]


def min_feasible_gamma(design, Z, bound=IV_BOUND):
    lo, hi = 1e-6, 4.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if np.sqrt(constraint_floor(design.X, design.y, mid, kind="iv", Z=Z, mean_match=True)) <= bound:
            hi = mid
        else:
            lo = mid
    return hi


def covers(model, queries, target, gamma):
    out = model.predict(queries, gamma=gamma)
    inside = np.all((out[:, 0] <= target + 1e-12) & (target <= out[:, 1] + 1e-12))
    return bool(inside and np.all(np.asarray(model.query_status) == SolveStatus.OK))


def coverage_over_replicates(runner, target):
    """Coverage of `target` on the coefficient queries at gamma*(b), one draw per
    experiment through the runner's own `generate_data` and `build_models`.

    Over whichever of the two uncovered baselines the recipe lists: the block is the
    owner's to change, so the gate follows it rather than pinning both. `wanted` comes
    from the RECIPE, and a name the recipe lists but the runner has not got is a FAIL,
    not a silently smaller loop."""
    wanted = [name for name in ("PI", "PI+IV") if name in recipe_methods()]
    check(
        "(v) coverage replicates: the runner carries every baseline the block lists",
        all(name in runner.methods for name in wanted),
        f"wants {wanted}, has {sorted(runner.methods)}",
    )
    keep = [name for name in wanted if name in runner.methods]
    runner.methods = {name: runner.methods[name] for name in keep}
    hits = {name: [] for name in keep}
    queries = np.eye(len(target))
    for j in range(runner.n_experiments):
        data = runner.generate_data(j, 1.0)
        models = fold_keys(runner.build_models(j, 0, data))
        gamma = runner.fit_gamma(j)
        for name in hits:
            hits[name].append(covers(models[name], queries, target, gamma))
    return {name: float(np.mean(values)) for name, values in hits.items()}


# ------------------------------------------------------------------ legs


def leg_d(reference):
    print("(D) the shipped configuration does not move")
    for label, ok, detail in digest_leg.compare(REPO, reference):
        check(f"(D) {label}", ok, detail)


def leg_i():
    print("(i) iv: [] falls back to the anchor set and b_r")
    orch = orchestrator(shipped_block(n_experiments=1, sweep_samples=4))
    sem = orch._sem_factory()
    design = sem.design
    raw = design.sigma * sem.solution.ravel()
    check("(i) b_r raw is (-1.992, 0.507, 1.237, 0.249) to 5e-4", np.abs(raw - B_R_RAW).max() < 5e-4, f"{raw}")
    independent = restricted_against(design, design.instruments)
    gap = np.abs(sem.solution.ravel() - independent).max()
    check("(i) b_r equals the independent anchor-set fit to 1e-10", gap < 1e-10, f"gap {gap:.2e}")
    check("(i) the SEM has iv_width 0 and no iv_pool", sem.iv_width == 0 and sem.iv_pool is None)
    X, _ = sem(N=design.n)
    check("(i) the draw has k columns", X.shape == (design.n, design.k), f"{X.shape}")
    check("(i) iv_columns is empty and gamma_z 0 on the orchestrator", orch.iv_columns == () and orch.gamma_z == 0.0)
    runner = sweep_runner(orch)
    check("(i) the sweep runner reads declared_iv False", runner.declared_iv is False)
    check("(i) the sweep runner's SEM is on the cluster bootstrap", runner.sems[0].bootstrap is True)
    data = runner.generate_data(0, 1.0)
    models = runner.build_models(0, 0, data)
    da_iv = models["DA+PI+IV"]
    on_t_only = da_iv.gamma_z == 0.0 and not da_iv._has_z and da_iv.T_projector_R.shape[0] == 1 + design.k
    check("(i) DA+PI+IV carries gamma_z 0 and one T constraint, no Z one", on_t_only, f"{da_iv.T_projector_R.shape}")
    query = query_runner(orch)
    check(
        "(i) the query runner reads declared_iv False and Z (n, 0)",
        query.declared_iv is False and query.Z.shape[1] == 0,
    )


def leg_ii():
    print("(ii) the shipped set's geometry, and the carrier")
    design = build_design(CigaretteSEM.panel(), spec="t3", anchor="own-tax")
    Z = instrument_set(design, PHASE_B)
    check("(ii) instrument_set is (n, 3)", Z.shape == (design.n, 3))
    d, singular = free_direction(design, Z)
    check(
        "(ii) singular values (3.142, 2.475, 1.504) to 1e-3",
        np.abs(singular - SINGULAR_VALUES).max() < 1e-3,
        f"{singular}",
    )
    check("(ii) abs(v'd) = 0.407190 to 1e-6", abs(abs(d.sum()) - ABS_VD) < 1e-6, f"{abs(d.sum()):.6f}")
    check("(ii) d_pn = 0.940 to 1e-3", abs(d[2] - D_PN) < 1e-3, f"{d[2]:.4f}")
    check(
        "(ii) the columns are the FWL'd tax_s, y, cpi in that order",
        np.array_equal(Z[:, 0], design.Z[:, 0])
        and np.array_equal(Z[:, 1], design.X[:, 1])
        and np.array_equal(Z[:, 2], design.X[:, 3]),
    )
    check("(ii) each column is mean zero (FWL through the state dummies)", np.abs(Z.mean(axis=0)).max() < 1e-10)
    try:
        instrument_set(design, ("tax_sn",))
        check("(ii) an excise the anchor lacks raises", False)
    except ValueError as error:
        check("(ii) an excise the anchor lacks raises", "tax_sn" in str(error), str(error))

    sem = CigaretteSEM(spec="t3", target="iv", anchor="own-tax", iv_columns=PHASE_B)
    check("(ii) the SEM's iv_width is 3", sem.iv_width == 3)
    check("(ii) iv_pool is the instrument matrix bit for bit", np.array_equal(sem.iv_pool, Z))
    XZ, y = sem(N=design.n)
    X_part, Z_part = sem.split_instruments(XZ)
    check("(ii) the panel draw is [X | Z]", np.array_equal(X_part, sem.X) and np.array_equal(Z_part, Z))
    boot = CigaretteSEM(spec="t3", target="iv", anchor="own-tax", bootstrap=True, iv_columns=PHASE_B)
    np.random.seed(0)
    XZ_b, y_b = boot(N=design.n)
    X_b, Z_b = boot.split_instruments(XZ_b)
    # every resampled row's Z is the Z of the row its X came from
    lookup = {tuple(row): index for index, row in enumerate(sem.X)}
    rows = np.array([lookup[tuple(row)] for row in X_b])
    aligned = np.array_equal(Z_b, Z[rows]) and np.array_equal(y_b, sem.y[rows])
    check("(ii) the cluster bootstrap keeps Z and y aligned with X row for row", aligned, f"{XZ_b.shape}")
    check("(ii) the bootstrap draw resamples (repeated rows)", len(np.unique(rows)) < len(rows))


def leg_iii():
    print("(iii) the phase-b target is homogeneous and on the moment line")
    sem = CigaretteSEM(spec="t3", target="iv", anchor="own-tax", iv_columns=PHASE_B)
    b = sem.solution.ravel()
    Q = np.linalg.qr(sem.iv_pool)[0]
    moment = float(np.linalg.norm(Q.T @ (sem.y.ravel() - sem.X @ b)) / np.sqrt(sem.design.n))
    check("(iii) v'b = 0 to 1e-12", abs(b.sum()) < 1e-12, f"{b.sum():.2e}")
    check("(iii) the three moments hold to 1e-12", moment < 1e-12, f"{moment:.2e}")
    raw = sem.design.sigma * b
    print(f"      RECORDED: b* raw {np.round(raw, 5)} (p4: -2.08874 0.38917 1.85356 -0.15399)")
    check("(iii) beta_pn raw is 1.854 to 1e-3", abs(raw[2] - 1.8536) < 1e-3)


def leg_iv():
    print("(iv) gamma*(b)")
    sem = CigaretteSEM(spec="t3", target="iv", anchor="own-tax", iv_columns=PHASE_B)
    gamma = sem.bias_sq / sem.sigma_sq
    check("(iv) gamma*(b) = 0.3705 to 1e-4 on the SEM", abs(gamma - GAMMA_STAR_B) < 1e-4, f"{gamma:.6f}")
    orch = orchestrator(recipe_block(n_experiments=1, sweep_samples=4, n_jobs=1))
    runner = sweep_runner(orch)
    oracle = runner.get_oracle(0)
    check(
        "(iv) gamma* through the runner's oracle to 1e-4",
        abs(oracle.gamma_star - GAMMA_STAR_B) < 1e-4,
        f"{oracle.gamma_star:.6f}",
    )
    print(
        f"      RECORDED overlap (batch A ruling 1): eps_iv_z_star at the phase-b target, RMS over the "
        f"{runner_draws()} pooled DA draws = {oracle.eps_iv_z_star:.6f} (p10, one draw at seed 0: 0.000460); "
        f"eps_iv_star = {oracle.eps_iv_star:.2e}"
    )
    check(
        "(iv) the overlap is a small fraction of the declared radius 0.0625 (recorded, not pinned)",
        oracle.eps_iv_z_star < 0.1 * IV_BOUND,
    )


def runner_draws():
    from src.experiments.generic_runner import ORACLE_POOL_DRAWS

    return ORACLE_POOL_DRAWS


def leg_v():
    print("(v) the replicate mechanism of decision 9, and the declared budget in the solver")
    block_methods = recipe_methods()
    orch = orchestrator(recipe_block(n_experiments=REPLICATES, sweep_samples=4, n_jobs=1))
    check(
        "(v) the orchestrator reads iv and gamma_z 2^-8", orch.iv_columns == PHASE_B and orch.gamma_z == GAMMA_Z_DEFAULT
    )
    runner = sweep_runner(orch)
    check(
        "(v) every sweep SEM is on 90% row splits (bootstrap False)", all(sem.bootstrap is False for sem in runner.sems)
    )
    check("(v) the sweep runner reads declared_iv True", runner.declared_iv is True)
    target = runner.sems[0].solution.ravel()
    data = runner.generate_data(0, 1.0)
    n_train = int(round(0.9 * runner.sems[0].design.n))
    check(
        "(v) a split carries 90% of the panel rows and a 3-column Z",
        len(data.X) == n_train and data.Z.shape == (n_train, 3),
    )
    models = fold_keys(runner.build_models(0, 0, data))
    r_t = float(runner.get_oracle(0).eps_iv_star) + EPS_TOL
    gated = listed(("PI+IV", "DA+PI+IV", "PI&DA+PI+IV"), block_methods, models, "(v) the declared budget")
    if "PI+IV" in gated:
        pi_iv = models["PI+IV"]
        r_z = np.sqrt(pi_iv.sigma_sq / pi_iv.rho * GAMMA_Z_DEFAULT)
        check(
            "(v) PI+IV: gamma_z declared, no T block, Z radius exactly s sqrt(gamma_z)",
            pi_iv.gamma_z == GAMMA_Z_DEFAULT and not pi_iv._has_t and pi_iv.z_bound == r_z,
            f"{pi_iv.z_bound:.6f}",
        )
    if "DA+PI+IV" in gated:
        da_pi_iv = models["DA+PI+IV"]
        r_z_own = np.sqrt(da_pi_iv.sigma_sq / da_pi_iv.rho * GAMMA_Z_DEFAULT) + da_pi_iv._z_allowance
        check(
            "(v) DA+PI+IV: two constraints, r_T and r_Z",
            abs(da_pi_iv.t_bound - r_t) < 1e-12 and abs(da_pi_iv.z_bound - r_z_own) < 1e-12,
            f"r_T {da_pi_iv.t_bound:.6f}, r_Z {da_pi_iv.z_bound:.6f}",
        )
    check("(v) and r_T is eps_iv_star + EPS_TOL, never raised", abs(r_t - EPS_TOL) < 1e-12, f"{r_t!r}")
    if "PI&DA+PI+IV" in gated:
        check(
            "(v) the intersection's baseline carries no T block, its DA branch r_T",
            not models["PI&DA+PI+IV"].baseline._has_t and models["PI&DA+PI+IV"].augmented.t_bound == r_t,
        )

    split = coverage_over_replicates(runner, target)
    for name in listed(("PI", "PI+IV"), block_methods, split, "(v) row-split coverage"):
        check(f"(v) row splits: {name} coverage 1.000 over {REPLICATES}", split[name] == 1.0, f"{split[name]:.3f}")

    # the same runner class on the state-cluster bootstrap: the flip refactor7 makes
    forced = STRATEGIES["gamma"](
        sem_factory=partial(orch._sem_factory, bootstrap=True),
        da_factory=orch._da_factory,
        poly_transform=None,
        test_fraction=0.1,
        default_gamma=orch.gamma,
        default_epsilon=orch._epsilon_budget(None),
        experiment_name="cigarettes",
        declared_iv=True,
        methods=orch.methods,
        method_factory=orch.build_methods,
        **orch._get_clean_kwargs(),
    )
    boot = coverage_over_replicates(forced, target)
    reported = listed(("PI", "PI+IV"), block_methods, boot, "(v) cluster-bootstrap coverage")
    print(
        f"      RECORDED cluster-bootstrap coverage at gamma*(b) over {REPLICATES}: "
        + ", ".join(f"{name} {boot[name]:.3f} (p10 {P10_BOOTSTRAP_COVERAGE[name]:.3f})" for name in reported)
    )
    if "PI+IV" in reported:
        check(
            "(v) cluster bootstrap: PI+IV coverage reads a resampling rate, under 1",
            boot["PI+IV"] < 1.0,
            f"{boot['PI+IV']:.3f}",
        )

    query = query_runner(orch)
    check(
        "(v) the query runner reads declared_iv True and a 3-column Z",
        query.declared_iv is True and query.Z.shape[1] == 3,
    )
    panel = PanelBuilder(query, "cigarettes", False, has_z=orch.has_z)
    panel._fit_all_models()
    fitted = fold_keys(panel.fitted_models)
    plain_iv = listed(("PI+IV", "PI+INV+IV", "DA+PI+IV"), block_methods, fitted, "(v) the query panel's +IV models")
    on_z = all(fitted[name]._has_iv for name in plain_iv)
    if "PI&DA+PI+IV" in fitted:
        on_z = on_z and fitted["PI&DA+PI+IV"].baseline._has_iv
    check("(v) the query panel's +IV models are fitted on the real Z (batch B ruling 2)", on_z, f"{plain_iv}")
    # 3 declared Z moments and 1 T moment, each beside the 4 mean-match rows
    want = {"PI+IV": (3 + 4,), "PI+INV+IV": (3 + 4,), "DA+PI+IV": (3 + 4, 1 + 4)}
    for name in plain_iv:
        rows = (fitted[name].Z_projector_R.shape[0],)
        if name == "DA+PI+IV":
            rows = (*rows, fitted[name].T_projector_R.shape[0])
        check(f"(v) {name} projects on {want[name]} moment rows", rows == want[name], f"{rows}")
    print(
        "      RECORDED query-path radii: "
        + ", ".join(
            f"{name} r_Z {fitted[name].z_bound:.6f}"
            + (f" r_T {fitted[name].t_bound:.6f}" if name == "DA+PI+IV" else "")
            for name in plain_iv
        )
        + f" (r_T there is the query tolerance {query.eps_tol:g})"
    )

    plain = orchestrator(shipped_block(n_experiments=1, sweep_samples=4))
    check("(v) iv: []: the sweep runner is built with bootstrap True", sweep_runner(plain).sems[0].bootstrap is True)


def leg_vi():
    print("(vi) the SS6.3 spec table")
    panel = CigaretteSEM.panel()
    print(f"      {'spec':5s} {'v.d':>8s} {'d_pn':>7s} {'gamma*(b)':>10s} {'floor@.0625':>12s}   [p8]")
    for spec, (signed, gamma_b, floor) in SPEC_TABLE.items():
        design = build_design(panel, spec=spec, anchor="own-tax")
        Z = instrument_set(design, PHASE_B)
        d, _ = free_direction(design, Z)
        sem = CigaretteSEM(spec=spec, target="iv", anchor="own-tax", iv_columns=PHASE_B)
        gamma = sem.bias_sq / sem.sigma_sq
        got_floor = min_feasible_gamma(design, Z)
        reference = f"[{signed:.4f} {gamma_b:.4f} {floor:.4f}]"
        print(f"      {spec:5s} {d.sum():8.4f} {d[2]:7.4f} {gamma:10.4f} {got_floor:12.4f}   {reference}")
        check(f"(vi) {spec}: abs(v'd) to 1e-3", abs(abs(d.sum()) - abs(signed)) < 1e-3, f"{abs(d.sum()):.4f}")
        check(f"(vi) {spec}: gamma*(b) to 1e-3", abs(gamma - gamma_b) < 1e-3, f"{gamma:.4f}")
        check(f"(vi) {spec}: PI+IV floor at 0.0625 to 1e-3", abs(got_floor - floor) < 1e-3, f"{got_floor:.4f}")
        if spec == "s":
            check("(vi) s: signed v'd = -0.0243 to 1e-3 under d_pn > 0", abs(d.sum() - signed) < 1e-3, f"{d.sum():.4f}")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    os.chdir(REPO)
    os.environ.setdefault("MPLBACKEND", "Agg")
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", default=digest_leg.DEFAULT_REFERENCE)
    parser.add_argument("--skip-digest", action="store_true")
    args = parser.parse_args()
    print(f"tree: {sem_module.__file__}")
    legs = [("(i)", leg_i), ("(ii)", leg_ii), ("(iii)", leg_iii), ("(iv)", leg_iv), ("(v)", leg_v), ("(vi)", leg_vi)]
    if not args.skip_digest:
        legs.append(("(D)", partial(leg_d, args.reference)))
    for tag, leg in legs:
        try:
            leg()
        except Exception as error:  # a raise is a FAIL line, and the later legs still report
            check(f"{tag} ran without raising", False, f"{type(error).__name__}: {error}")
    if args.skip_digest:
        print("(D) SKIPPED by --skip-digest: a break-it run, not the committed state")
    if not FAIL:
        print("A60 PASS")
    else:
        print(f"A60 FAIL: {FAIL}")
        sys.exit(1)
