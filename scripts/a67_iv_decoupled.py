"""A67: one constraint per instrument, and the DA-side allowance that keeps it valid.

refactor10 splits the single IV constraint in two: the DA translation amounts reach
`fit` as `T` and the observed instrument as `Z`, each block getting its own SOC
constraint at its own radius (`t_bound` from `epsilon_iv`, `z_bound` from
`epsilon_iv_z`, `gamma_z` and the DA-side allowance). Nothing is added in
quadrature across instruments any more. Legs:

  (D)   the digest leg (scripts/digest_leg.py): the shipped no-instrument
        configurations, every pkl and tex byte-identical to the reference recorded
        on 14509db. Catches: an empty Z that is not exactly today's program.
  (i)   blocks and radii through the registry and `fit_model`, plus the DA-side
        allowance of SS2.6: which blocks each spelling carries, the radii they
        read, the constraint count, and kappa recomputed in the leg against
        `_declared_allowance`. Catches: a baseline handed T, a radius that pools
        the two budgets, the allowance's operator loosened.
  (ii)  the empty-instrument program at the model level: the T block is exactly
        `iv_constraint_terms(GX, y, G)`, the threshold is sqrt(N) r_T bit for bit,
        there is no Z block, and `DA+PI+IV(T)` predicts identically to bare
        `DA+PI+IV`. Catches: a double jitter, a bare mode that loses its T block.
  (iii) decoupled against joint on the simulation `iv: 4`, the joint one built from
        the SAME class by handing the stacked matrix as one T block: the width
        ratios, and the fact that the decoupled bounds are NOT inside the joint
        ones. Catches: a silent re-pooling.
  (iv)  the cigarette query numbers under `target: iv`, RECORDED at the adopted
        leak budget with the DA-side allowance in force. Catches: the two radii
        swapped, the allowance dropped.
  (v)   floors and budgets: the T constraint's own floor differs from the stacked
        one, `fit_epsilon_iv` is the guarded T piece, and the declared path logs
        "the T constraint's own floor" and never raises. Catches: the guard
        measuring the stacked instrument again.
  (vi)  the oracle: `eps_iv_z_star` is the larger of the two residual moments, it
        uses the same `W#` as `eps_iv_star`, and `OracleParameters` has no
        `iv_budget`. Catches: the span(Z|G) orthogonalisation put back, the sign of
        the DA-side residual.
  (vii) text hygiene: no `iv_bound`, `iv_budget` or pooled-budget prose under
        `src/`, `config.yaml` or `recipes/`, `Z-tilde` only where it is named to
        say the PI classes do NOT pool it, ruff clean, ASCII only.
  (viii) the plasmode confounder (R8), SEM only, no solving: the projection, the
        calibration that survives it, and the instrument the projection rescues.
  (ix)  the two plasmode guards fire.

    MPLBACKEND=Agg python scripts/a67_iv_decoupled.py [--seed 42] [--only LEG] [--skip-digest]
"""

import argparse
import os
import subprocess
import sys
from functools import partial

import numpy as np
from loguru import logger

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import a59_sim_iv as a59  # noqa: E402
import a60_cigarettes_iv as a60  # noqa: E402
import digest_leg  # noqa: E402
from munch import munchify  # noqa: E402

from src.data_augmentors.cigarettes import ScaleTranslation  # noqa: E402
from src.experiments.configs import (  # noqa: E402
    EPS_TOL,
    GAMMA_Z_DEFAULT,
    MethodRegistry,
    resolve_dataset_block,
)
from src.experiments.simulation import SimulationOrchestrator  # noqa: E402
from src.experiments.utils import PanelBuilder, set_seed  # noqa: E402
from src.experiments.utils.metrics import rho_hat  # noqa: E402
from src.experiments.utils.model_fitting import fit_model  # noqa: E402
from src.methods.sensitivity_models import (  # noqa: E402
    InstrumentalVariablePartialR2,
    SolveStatus,
    constraint_floor,
    iv_constraint_terms,
)
from src.oracle import OracleParameters, _invariance_signal, eps_iv_star, eps_iv_z_star  # noqa: E402
from src.sem.cigarettes import (  # noqa: E402
    CigaretteSEM,
    V,
    build_design,
)

# the commit this round branched from: the ASCII rule in leg (vii) is on the lines
# ADDED since then, not on prose the tree already carried
BASE = "c566b76"
GAMMA = 0.25
EPS_IV = 2**-5
EPS_IV_Z = 2**-6
GAMMA_Z = 2**-8
TOGGLES = dict(clipy=False, mean_match=True, n_jobs=1, recalibrate=True)
SPELLINGS = ("PI+IV", "PI+INV+IV", "DA+PI+IV", "DA+PI+IV(Z)", "DA+PI+IV(T)", "PI&DA+PI+IV")
# RECORDED on this tree at the adopted leak budget (gamma_z 0.0177), the query
# panel of recipes/neighbour-price_fig12.yaml at n_experiments 1, sweep_samples 8
QUERY_ROWS = {
    "PI+IV": (0.108698, 1.652794),
    "DA+PI+IV(Z)": (0.163583, 1.655723),
    "DA+PI+IV": (0.379622, 1.655723),
}
QUERY_TOL = 1e-3
FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAIL.append(name)


# ------------------------------------------------------------------ fixture


def fixture(seed=42):
    """The cigarette design, one seed-42 DA draw and a 1-column observed Z."""
    design = build_design(CigaretteSEM.panel(), spec="t3", anchor="own-tax")
    np.random.seed(seed)
    GX, G = ScaleTranslation(V, std=float(np.std(design.X @ (V / np.linalg.norm(V)))))(design.X)
    return design, design.X, design.y, GX, np.reshape(G, (len(design.X), -1)), design.Z[:, :1]


def built(names, X, y, GX, G, Z, **overrides):
    rho = float(rho_hat(X, GX, y, intercept=True))
    kwargs = dict(
        gamma=GAMMA,
        epsilon=EPS_TOL,
        epsilon_iv=EPS_IV,
        epsilon_iv_z=EPS_IV_Z,
        gamma_z=GAMMA_Z,
        rho=rho,
        pad=False,
        **TOGGLES,
    )
    builders = MethodRegistry.build_methods(list(names), **{**kwargs, **overrides})
    models = {}
    for name in names:
        model = builders[name]()
        fit_model(model=model, method_name=name, X=X, y=y, GX=GX, G=G, Z=Z)
        models[name] = model
    return models


def kappa_by_hand(X, Z, X_pre):
    """The tight operator of SS2.6, recomputed here so it cannot drift from
    `InstrumentalVariablePartialR2._declared_allowance`: MUST mirror that body."""
    X = np.asarray(X, dtype=float)
    difference = X - np.asarray(X_pre, dtype=float).reshape(len(X), -1)
    left, singular, _ = np.linalg.svd(difference, full_matrices=False)
    keep = singular > max(float(singular[0]), 1.0) * 1e-12
    Q_Z, _ = np.linalg.qr(Z)
    Q_X, _ = np.linalg.qr(X)
    Q_D = left[:, keep]
    return float(np.linalg.svd(Q_Z.T @ (Q_D - Q_X @ (Q_X.T @ Q_D)), compute_uv=False)[0])


# ------------------------------------------------------------------ legs


def leg_d(reference):
    print("(D) the shipped no-instrument configurations do not move")
    for label, ok, detail in digest_leg.compare(REPO, reference):
        check(f"(D) {label}", ok, detail)


def leg_i(seed):
    print("(i) blocks, radii and the DA-side allowance")
    design, X, y, GX, G, Z = fixture(seed)
    models = built(SPELLINGS, X, y, GX, G, Z)
    M, p, m = design.k, G.shape[1], Z.shape[1]

    wants = {
        "PI+IV": (False, True),
        "PI+INV+IV": (False, True),
        "DA+PI+IV": (True, True),
        "DA+PI+IV(Z)": (False, True),
        "DA+PI+IV(T)": (True, False),
    }
    for name, (has_t, has_z) in wants.items():
        model = models[name]
        blocks = (model._has_t, model._has_z)
        check(f"(i) {name} blocks (T, Z) = {(has_t, has_z)}", blocks == (has_t, has_z), f"{blocks}")
        if has_t:
            check(f"(i) {name} T block is (p + M, M)", model.T_projector_R.shape == (p + M, M))
            check(f"(i) {name} t_bound is epsilon_iv EXACTLY", model.t_bound == EPS_IV, f"{model.t_bound!r}")
        else:
            check(f"(i) {name} has no T block", model.T_projector_R is None)
        if has_z:
            check(f"(i) {name} Z block is (m + M, M)", model.Z_projector_R.shape == (m + M, M))
        else:
            check(f"(i) {name} has no Z block", model.Z_projector_R is None)
        # one ball plus one constraint per non-empty block; PI+INV+IV adds its cone
        want_count = 1 + has_t + has_z + (name == "PI+INV+IV")
        model.predict(np.eye(M), gamma=GAMMA)
        got = len(model.min_problem.constraints)
        check(f"(i) {name} carries {want_count} constraints", got == want_count, f"{got}")

    inter = models["PI&DA+PI+IV"]
    check("(i) PI&DA+PI+IV baseline: Z only", not inter.baseline._has_t and inter.baseline._has_z)
    check("(i) PI&DA+PI+IV DA branch: both blocks", inter.augmented._has_t and inter.augmented._has_z)
    for mode, has_t, has_z in (("Z", False, True), ("T", True, False)):
        spelled = built([f"PI&DA+PI+IV({mode})"], X, y, GX, G, Z)[f"PI&DA+PI+IV({mode})"]
        blocks = (spelled.augmented._has_t, spelled.augmented._has_z)
        check(f"(i) PI&DA+PI+IV({mode}) DA branch blocks {(has_t, has_z)}", blocks == (has_t, has_z), f"{blocks}")
        check(f"(i) PI&DA+PI+IV({mode}) baseline is still Z only", not spelled.baseline._has_t)

    # the Z radius: the non-DA methods carry no allowance, the DA+ ones do
    for name in ("PI+IV", "PI+INV+IV"):
        model = models[name]
        want = float(np.hypot(EPS_IV_Z, np.sqrt(model.sigma_sq / model.rho * GAMMA_Z)))
        check(f"(i) {name} z_bound is the declared radius at its own s", abs(model.z_bound - want) < 1e-12)
        check(f"(i) {name} carries no DA-side allowance", model._z_allowance == 0.0, f"{model._z_allowance!r}")

    # `fit_model` hands a DA+ method `X_pre`, so the allowance IS in force there
    da = models["DA+PI+IV"]
    kappa = kappa_by_hand(GX - GX.mean(axis=0), Z, X - GX.mean(axis=0))
    check(
        "(i) DA+PI+IV's allowance is kappa * epsilon, kappa recomputed in the leg",
        abs(da._z_allowance - kappa * EPS_TOL) < 1e-12,
        f"{da._z_allowance:.9f}, kappa {kappa:.6f}",
    )
    check("(i) 0 < kappa <= 1", 0.0 < kappa <= 1.0 + 1e-12, f"{kappa:.6f}")
    Q_Z = np.linalg.qr(Z)[0]
    Q_X = np.linalg.qr(GX - GX.mean(axis=0))[0]
    loose = float(np.linalg.svd(Q_Z.T @ Q_X, compute_uv=False)[0])
    check("(i) and strictly below the loose operator ||Q_Z' Q_X||", kappa < loose - 1e-9, f"{kappa:.6f} < {loose:.6f}")
    want = float(np.hypot(EPS_IV_Z, np.sqrt(da.sigma_sq / da.rho * GAMMA_Z) + da._z_allowance))
    check("(i) and z_bound carries it", abs(da.z_bound - want) < 1e-12, f"{da.z_bound:.9f}")

    # the three zero cases
    plain = InstrumentalVariablePartialR2(
        gamma=GAMMA, epsilon=EPS_TOL, epsilon_iv=EPS_IV, epsilon_iv_z=EPS_IV_Z, gamma_z=GAMMA_Z, pad=False, **TOGGLES
    ).fit(GX, y, T=G, Z=Z)
    check("(i) no X_pre: the allowance is exactly 0.0", plain._z_allowance == 0.0, f"{plain._z_allowance!r}")
    unmoved = InstrumentalVariablePartialR2(
        gamma=GAMMA, epsilon=EPS_TOL, epsilon_iv=EPS_IV, epsilon_iv_z=EPS_IV_Z, gamma_z=GAMMA_Z, pad=False, **TOGGLES
    ).fit(GX, y, T=G, Z=Z, X_pre=GX)
    check("(i) an augmentation that moved nothing: 0.0", unmoved._z_allowance == 0.0, f"{unmoved._z_allowance!r}")
    no_leak = InstrumentalVariablePartialR2(
        gamma=GAMMA, epsilon=EPS_TOL, epsilon_iv=EPS_IV, epsilon_iv_z=EPS_IV_Z, gamma_z=0.0, pad=False, **TOGGLES
    ).fit(GX, y, T=G, Z=Z, X_pre=X)
    check("(i) gamma_z 0: 0.0, and z_bound is exactly epsilon_iv_z", no_leak._z_allowance == 0.0)
    check("(i) and z_bound == epsilon_iv_z bit for bit", no_leak.z_bound == EPS_IV_Z, f"{no_leak.z_bound!r}")
    non_da = InstrumentalVariablePartialR2(
        gamma=GAMMA, epsilon=EPS_TOL, epsilon_iv=EPS_IV, epsilon_iv_z=EPS_IV_Z, gamma_z=GAMMA_Z, pad=False, **TOGGLES
    ).fit(X, y, Z=Z)
    check("(i) a non-DA method refitted the same way still reads 0.0", non_da._z_allowance == 0.0)


def leg_ii(seed):
    print("(ii) the empty-instrument program is today's, to the bit")
    _, X, y, GX, G, _ = fixture(seed)
    models = built(("DA+PI+IV", "DA+PI+IV(T)", "PI+IV", "PI+INV+IV"), X, y, GX, G, None)
    bare = models["DA+PI+IV"]
    want_A, want_b = iv_constraint_terms(GX - GX.mean(axis=0), y - np.mean(y), G)
    check("(ii) the T block is iv_constraint_terms(GX, y, G)", np.array_equal(bare.T_projector_R, want_A))
    check("(ii) and its residual base too", np.array_equal(bare.t_residual_base, want_b))
    check("(ii) no Z block at all", not bare._has_z and bare.Z_projector_R is None)
    queries = np.eye(X.shape[1])
    bare.predict(queries, gamma=GAMMA)
    threshold = np.sqrt(bare.N_samples) * EPS_IV
    check(
        "(ii) the T threshold is sqrt(N) r_T bit for bit",
        bare.t_threshold_param.value == threshold,
        f"{bare.t_threshold_param.value!r}",
    )
    for name in ("PI+IV", "PI+INV+IV"):
        check(f"(ii) {name} carries no constraint at all", not models[name]._has_iv)
    t_only = models["DA+PI+IV(T)"]
    left = np.asarray(t_only.predict(queries, gamma=GAMMA))
    right = np.asarray(bare.predict(queries, gamma=GAMMA))
    gap = float(np.abs(left - right).max())
    same_status = np.array_equal(np.asarray(t_only.query_status), np.asarray(bare.query_status))
    check("(ii) DA+PI+IV(T) predicts identically to bare DA+PI+IV", gap == 0.0 and same_status, f"{gap!r}")


def leg_iii(seed):
    print("(iii) decoupled against joint on the simulation iv: 4")
    block = resolve_dataset_block("simulation", a59.recipe_block())
    reduced = {**block, "n_experiments": 1, "n_samples": 512, "sweep_samples": 8, "n_jobs": 1}
    set_seed(reduced["seed"])
    orch = SimulationOrchestrator(**reduced, hyperparameters=munchify(digest_leg.HYPERPARAMETERS))
    runner = orch.get_sweep_runner_cls("gamma")(
        methods=orch.methods, method_factory=orch.build_methods, **orch._get_clean_kwargs()
    )
    data = runner.generate_data(0, 1.0)
    G = np.asarray(data.G).reshape(len(data.X), -1)
    r_t, r_z = runner.fit_epsilon_iv(0, 0, data), runner.fit_epsilon_iv_z(0, data)
    pooled = float(np.hypot(r_t, r_z))
    common = dict(
        gamma=runner.fit_gamma(0),
        epsilon=runner.fit_epsilon(0, 0, data),
        pad=False,
        rho=runner.fit_rho(0, data),
        gamma_z=0.0,
        **TOGGLES,
    )
    # today's joint constraint expressed in the new class: the STACKED matrix as
    # one T block at the pooled radius
    joint = InstrumentalVariablePartialR2(epsilon_iv=pooled, epsilon_iv_z=0.0, **common).fit(
        data.GX, data.y, T=np.column_stack([G, data.Z])
    )
    decoupled = InstrumentalVariablePartialR2(epsilon_iv=r_t, epsilon_iv_z=r_z, **common).fit(
        data.GX, data.y, T=G, Z=data.Z
    )
    pi_iv = InstrumentalVariablePartialR2(epsilon_iv=r_t, epsilon_iv_z=r_z, **common).fit(data.X, data.y, Z=data.Z)
    out = {
        name: np.asarray(m.predict(data.X_test))
        for name, m in (("joint", joint), ("decoupled", decoupled), ("PI+IV", pi_iv))
    }
    widths = {name: float(np.nanmean(v[:, 1] - v[:, 0])) for name, v in out.items()}
    print(f"      RECORDED widths: { ({k: round(v, 6) for k, v in widths.items()}) }, pooled radius {pooled:.6f}")
    check(
        "(iii) every method solved OK",
        all(np.all(np.asarray(m.query_status) == SolveStatus.OK) for m in (joint, decoupled, pi_iv)),
    )
    check(
        "(iii) the decoupled bounds are NOT inside the joint ones",
        bool(np.any(out["decoupled"][:, 0] < out["joint"][:, 0] - 1e-9))
        or bool(np.any(out["decoupled"][:, 1] > out["joint"][:, 1] + 1e-9)),
    )
    check(
        "(iii) decoupled is sharper than PI+IV",
        widths["decoupled"] < widths["PI+IV"],
        f"{widths['decoupled'] / widths['PI+IV']:.4f}",
    )


def leg_iv():
    print("(iv) the cigarette query numbers under target: iv, at the adopted leak budget")
    block = a60.recipe_block(n_experiments=1, sweep_samples=8, n_jobs=1)
    orch = a60.orchestrator(block)
    query = a60.query_runner(orch)
    panel = PanelBuilder(query, "cigarettes", False)
    panel._fit_all_models()
    models = panel.fitted_models
    # the panel is sigma-normalised; the reported figures are in RAW log units
    scale = query.sem.design.sigma
    queries = np.eye(query.X.shape[1])
    for name, want in QUERY_ROWS.items():
        model = models[name]
        got = scale * np.asarray(model.predict(queries, gamma=GAMMA), dtype=float)[2]
        gap = float(np.abs(got - np.asarray(want)).max())
        check(
            f"(iv) {name} neighbour price [{want[0]:.3f}, {want[1]:.3f}] to 1e-3",
            gap < QUERY_TOL,
            f"got [{got[0]:.6f}, {got[1]:.6f}], gap {gap:.2e}",
        )
    da = models["DA+PI+IV"]
    check(
        "(iv) DA+PI+IV carries two radii and the allowance",
        da._has_t and da._has_z and da._z_allowance > 0.0,
        f"r_T {da.t_bound:.6f}, r_Z {da.z_bound:.6f}, allowance {da._z_allowance:.6f}",
    )


def leg_v(seed):
    print("(v) floors and budgets: the T constraint's own")
    block = resolve_dataset_block("simulation", a59.recipe_block())
    reduced = {**block, "n_experiments": 1, "n_samples": 512, "sweep_samples": 8, "n_jobs": 1}
    set_seed(reduced["seed"])
    orch = SimulationOrchestrator(**reduced, hyperparameters=munchify(digest_leg.HYPERPARAMETERS))
    runner = orch.get_sweep_runner_cls("epsilon")(
        methods=orch.methods, method_factory=orch.build_methods, **orch._get_clean_kwargs()
    )
    data = runner.generate_data(0, 1.0)
    G = np.asarray(data.G).reshape(len(data.X), -1)
    common = dict(mean_match=runner.mean_match, rho=runner.fit_rho(0, data), recalibrate=runner.recalibrate, kind="iv")
    floor_t = constraint_floor(data.GX, data.y, runner.fit_gamma(0), Z=G, **common)
    floor_stacked = constraint_floor(data.GX, data.y, runner.fit_gamma(0), Z=np.column_stack([G, data.Z]), **common)
    print(f"      RECORDED floors: T alone {floor_t:.6f}, stacked {floor_stacked:.6f}")
    check(
        "(v) the T constraint's own floor differs from the stacked one",
        abs(floor_t - floor_stacked) > 1e-3,
        f"{abs(floor_t - floor_stacked):.6f}",
    )
    raw = float(runner.get_oracle(0).eps_iv_star) + EPS_TOL
    guarded = raw if raw**2 >= floor_t else float(np.sqrt(9.0 * max(floor_t, 0.0)))
    got = runner.fit_epsilon_iv(0, 0, data)
    check("(v) fit_epsilon_iv is guard_T(eps_iv_star + EPS_TOL)", got == guarded, f"{got!r} vs {guarded!r}")

    records = []
    sink = logger.add(lambda message: records.append(message.record), level="DEBUG")
    try:
        cig = a60.orchestrator(a60.recipe_block(n_experiments=1, sweep_samples=8, n_jobs=1))
        cig_runner = a60.sweep_runner(cig, "epsilon")
        cig_data = cig_runner.generate_data(0, 1.0)
        declared = cig_runner.fit_epsilon_iv(0, 0, cig_data)
    finally:
        logger.remove(sink)
    want = float(cig_runner.get_oracle(0).eps_iv_star) + EPS_TOL
    check("(v) the declared path returns eps_iv_star + EPS_TOL unchanged", declared == want, f"{declared!r}")
    lines = [r["message"] for r in records if "declared path" in r["message"]]
    check("(v) and logs the T constraint's own floor", any("T constraint's own floor" in m for m in lines), lines[:1])


def leg_vi(seed):
    print("(vi) the oracle: two pieces, the larger residual moment, no iv_budget")
    check("(vi) OracleParameters has no iv_budget", not hasattr(OracleParameters, "iv_budget"))
    block = resolve_dataset_block("simulation", a59.recipe_block())
    reduced = {**block, "n_experiments": 1, "n_samples": 512, "sweep_samples": 8, "n_jobs": 1}
    set_seed(reduced["seed"])
    orch = SimulationOrchestrator(**reduced, hyperparameters=munchify(digest_leg.HYPERPARAMETERS))
    runner = orch.get_sweep_runner_cls("gamma")(
        methods=orch.methods, method_factory=orch.build_methods, **orch._get_clean_kwargs()
    )
    sem, da = runner.sems[0], runner.das[0]
    data = runner.generate_data(0, 1.0)
    X, y, Z = data.X, data.y, data.Z
    got = eps_iv_z_star(sem, da, X=X, y=y, Z=Z, mean_match=runner.mean_match)

    # the two moments, written out here with no oracle code
    GX = data.GX
    w = (np.asarray(sem.f(X)) - np.asarray(sem.f(GX))).ravel()
    Phi = GX - GX.mean(axis=0)
    w_centred = w - w.mean()
    sharp = w_centred - Phi @ np.linalg.lstsq(Phi, w_centred, rcond=None)[0]
    residual = np.asarray(y).ravel() - np.asarray(sem.f(X)).ravel()
    Q = np.linalg.qr(Z)[0]

    def moment(r):
        r = r - r.mean()
        return float(np.linalg.norm(Q.T @ r) / np.sqrt(len(r)))

    on_x, on_gx = moment(residual), moment(residual + sharp)
    print(f"      RECORDED: Z moment at h_* {on_x:.6f}, at h# {on_gx:.6f}, returned {got:.6f}")
    check("(vi) eps_iv_z_star is the LARGER of the two moments", abs(got - max(on_x, on_gx)) < 1e-9, f"{got:.9f}")
    # the h# moment is the one that uses W#: the identity the sign rests on
    # the identity the SIGN rests on: r_* + W# is the residual at h# on GX
    b = np.linalg.lstsq(np.asarray(X), np.asarray(sem.f(X)).ravel(), rcond=None)[0]
    h_sharp = b + np.linalg.lstsq(Phi, w_centred, rcond=None)[0]
    direct = np.asarray(y).ravel() - Phi @ h_sharp
    left = residual + sharp
    gap = float(np.abs((left - left.mean()) - (direct - direct.mean())).max())
    check("(vi) r_* + W# IS the residual at h# on the augmented design", gap < 1e-8, f"{gap:.2e}")
    check("(vi) an empty Z is exactly 0.0", eps_iv_z_star(sem, da, X=X, y=y, Z=None) == 0.0)

    # the shipped DA is exactly invariant, so the two moments coincide and the max
    # would pass vacuously. The epsilon strategy RETUNES the DA to a true eps*, and
    # there they come apart: the leg pins that the larger is what is returned
    tuned_runner = orch.get_sweep_runner_cls("epsilon")(
        methods=orch.methods, method_factory=orch.build_methods, **orch._get_clean_kwargs()
    )
    tuned = tuned_runner.generate_data(0, 1.0)
    tuned_sem, tuned_da = tuned_runner.sems[0], tuned_runner.das[0]
    tuned_got = eps_iv_z_star(tuned_sem, tuned_da, X=tuned.X, y=tuned.y, Z=tuned.Z, mean_match=tuned_runner.mean_match)
    # `_invariance_signal` is the SOLE definition of W, the one `eps_iv_star` uses,
    # and it makes its own seeded draw: reading it here is what pins that the two
    # budgets are measured on the same W#, not that the leg can redraw it
    w_t, Phi_t, G_t = _invariance_signal(tuned_sem, tuned_da, tuned.X, None, len(tuned.X))
    Phi_t = Phi_t - Phi_t.mean(axis=0)
    w_t = w_t - w_t.mean()
    sharp_t = w_t - Phi_t @ np.linalg.lstsq(Phi_t, w_t, rcond=None)[0]
    residual_t = np.asarray(tuned.y).ravel() - np.asarray(tuned_sem.f(tuned.X)).ravel()
    Q_t = np.linalg.qr(tuned.Z)[0]

    def tuned_moment(r):
        r = r - r.mean()
        return float(np.linalg.norm(Q_t.T @ r) / np.sqrt(len(r)))

    pair = tuned_moment(residual_t), tuned_moment(residual_t + sharp_t)
    print(f"      RECORDED under the retuned DA: h_* {pair[0]:.6f}, h# {pair[1]:.6f}, returned {tuned_got:.6f}")
    check("(vi) under the retuned DA the two moments DIFFER", abs(pair[0] - pair[1]) > 1e-6, f"{pair}")
    check("(vi) and the larger is returned", abs(tuned_got - max(pair)) < 1e-9, f"{tuned_got:.9f} vs {max(pair):.9f}")
    check("(vi) and it is not the smaller", abs(tuned_got - min(pair)) > 1e-6)
    # the same W#: projecting it on span(T) instead of span(Z) must reproduce
    # `eps_iv_star` on the SAME draw, so the two budgets cannot drift apart
    Q_G = np.linalg.qr(G_t)[0]
    t_piece = float(np.sqrt(np.mean((Q_G @ (Q_G.T @ sharp_t)) ** 2)))
    direct, _, _ = eps_iv_star(tuned_sem, tuned_da, X=tuned.X, mean_match=tuned_runner.mean_match)
    check(
        "(vi) the Z budget's W# is the one eps_iv_star projects",
        abs(t_piece - float(direct)) < 1e-9,
        f"{t_piece:.9f} vs {direct:.9f}",
    )


def leg_vii():
    print("(vii) text hygiene")
    roots = [os.path.join(REPO, "src"), os.path.join(REPO, "recipes")]
    files = [os.path.join(REPO, "config.yaml")]
    for root in roots:
        for base, _, names in os.walk(root):
            files += [os.path.join(base, n) for n in names if n.endswith((".py", ".yaml"))]
    # the ASCII rule is on the lines this round ADDS; the tree already carries the
    # paper's own glyphs (section signs, epsilons) in prose that predates it
    added = subprocess.run(
        ["git", "diff", "--unified=0", BASE, "--", "src", "config.yaml", "recipes", "scripts"],
        cwd=REPO,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    non_ascii = [
        line[:70] for line in added if line.startswith("+") and not line.startswith("+++") and not line.isascii()
    ]
    em_dash = [line[:70] for line in added if line.startswith("+") and "\u2014" in line]
    check("(vii) the added lines are ASCII, no em dash", not non_ascii and not em_dash, str((non_ascii + em_dash)[:2]))
    # `Z-tilde` survives ONLY where it is named to say the PI classes do not pool it
    allowed_tilde = (
        "Z-tilde = (T, Z) of Asm. 3 as ONE matrix",
        "NOT in general inside the single constraint on Z-tilde",
        "2SLS on the augmented data with ONE instrument matrix",
        "Z-tilde; `IV` under an empty set is a config error",
    )
    banned = ("iv_bound", "iv_budget", "root sum square", "root-sum-square", "joint bound")
    # `z_bound` combines two pieces of ONE instrument's radius, which is not the
    # pooling this grep is for; that one line is named rather than exempted wholesale
    allowed_rss = ("declared one in root sum square, s = sigma-hat of the PRE-DA data",)
    hits = []
    tilde_hits = []
    for path in files:
        with open(path, encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                where = f"{os.path.relpath(path, REPO)}:{number}"
                if any(word in line for word in banned) and not any(ok in line for ok in allowed_rss):
                    hits.append(f"{where} {line.strip()[:70]}")
                if "Z-tilde" in line and not any(ok in line for ok in allowed_tilde):
                    tilde_hits.append(f"{where} {line.strip()[:70]}")
    check("(vii) no pooled-budget prose under src/, config.yaml or recipes/", not hits, "; ".join(hits[:3]))
    check("(vii) Z-tilde only where the 2SLS stacker is named", not tilde_hits, "; ".join(tilde_hits[:3]))
    # the interpreter's own ruff: the venv may live beside a worktree, not in it
    ruff = os.path.join(os.path.dirname(sys.executable), "ruff")
    for command in (["check", "."], ["format", "--check", "."]):
        done = subprocess.run(
            [ruff] + command,
            cwd=REPO,
            capture_output=True,
            text=True,
        )
        check(f"(vii) ruff {' '.join(command)} clean", done.returncode == 0, str(done.stdout.strip().splitlines()[-1:]))


def leg_viii():
    print("(viii) the plasmode confounder: the projection, the calibration, the instrument")
    sets = {(): (1.0, 1.0), ("tax_s",): (0.9537, 0.9380), ("tax_s", "y", "cpi"): (0.8286, 0.9212)}
    for columns, (want_surviving, want_q) in sets.items():
        np.random.seed(42)
        sem = CigaretteSEM(spec="t3", target="plasmode", anchor="own-tax", iv_columns=columns)
        u, q, surviving = sem._confounder("v")
        block = sem._exogenous_block()
        label = f"iv={list(columns)}"
        if not columns:
            check(f"(viii) {label}: the exogenous block is (n, 0)", block.shape == (len(sem.X), 0), f"{block.shape}")
            raw = sem.X @ (V / np.linalg.norm(V))
            check(f"(viii) {label}: the confounder is the unprojected one", np.array_equal(u, raw / raw.std()))
            check(f"(viii) {label}: q and surviving are exactly 1.0", (q, surviving) == (1.0, 1.0))
            check(
                f"(viii) {label}: _calibrate(0.25, 1.0) == _calibrate(0.25) bit for bit",
                sem._calibrate(0.25, 1.0) == sem._calibrate(0.25),
            )
        else:
            check(
                f"(viii) {label}: surviving {want_surviving}",
                abs(surviving - want_surviving) < 1e-3,
                f"{surviving:.4f}",
            )
            check(f"(viii) {label}: q {want_q}", abs(q - want_q) < 1e-3, f"{q:.4f}")
            orth = float(np.max(np.abs(block.T @ u))) / len(u)
            check(f"(viii) {label}: the confounder is orthogonal to the block", orth < 1e-10, f"{orth:.2e}")
            twice = u - block @ np.linalg.lstsq(block, u, rcond=None)[0]
            idem = float(np.max(np.abs(twice / twice.std() - u)))
            check(f"(viii) {label}: projecting twice changes nothing", idem < 1e-12, f"{idem:.2e}")
        realised = sem.bias_sq / sem.sigma_sq
        check(f"(viii) {label}: realised gamma* is 0.250000", abs(realised - 0.25) < 1e-9, f"{realised:.9f}")

    # the projection rescues the instrument: the Z moment at the target falls
    np.random.seed(42)
    sem = CigaretteSEM(spec="t3", target="plasmode", anchor="own-tax", iv_columns=("tax_s", "y", "cpi"))
    Q = np.linalg.qr(sem._Z_iv)[0]
    residual = np.asarray(sem.y).ravel() - sem.X @ np.asarray(sem.solution).ravel()
    projected = float(np.linalg.norm(Q.T @ residual) / np.sqrt(len(residual)))
    raw = sem.X @ (V / np.linalg.norm(V))
    unprojected = raw / raw.std()
    kappa = np.sqrt(sem._kappa_sq)
    residual_unprojected = residual - kappa * np.asarray(sem._confounder("v")[0]).ravel() + kappa * unprojected
    before = float(np.linalg.norm(Q.T @ residual_unprojected) / np.sqrt(len(residual)))
    r_z = float(np.sqrt(sem.sigma_sq * GAMMA_Z_DEFAULT))
    print(
        f"      RECORDED: Z moment at the target {before:.4f} unprojected -> {projected:.4f} projected, r_Z {r_z:.4f}"
    )
    check(
        "(viii) the projection drops the Z moment at the target by 10x or more",
        before > 10 * projected,
        f"{before / max(projected, 1e-12):.1f}x",
    )
    check("(viii) and lands it below s sqrt(gamma_z)", projected < r_z, f"{projected:.4f} < {r_z:.4f}")


def leg_ix():
    print("(ix) the two plasmode guards")
    np.random.seed(42)
    sem = CigaretteSEM(spec="t3", target="plasmode", anchor="own-tax", iv_columns=("tax_s",))
    sem._exogenous_block = lambda: sem.X  # everything declared exogenous
    try:
        sem._confounder("v")
        raised = None
    except ValueError as error:
        raised = str(error)
    check(
        "(ix) _confounder raises naming the direction when nothing survives",
        raised is not None and "'v'" in raised and "declared-exogenous" in raised,
        (raised or "no error")[:80],
    )
    try:
        sem._calibrate(0.99, 0.2)
        raised = None
    except ValueError as error:
        raised = str(error)
    check(
        "(ix) _calibrate raises at kappa^2 > 1",
        raised is not None and "kappa^2" in raised,
        (raised or "no error")[:80],
    )
    try:
        sem._calibrate(0.25, 0.0)
        raised = None
    except ValueError as error:
        raised = str(error)
    check("(ix) and at q = 0", raised is not None, (raised or "no error")[:60])


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    os.chdir(REPO)
    os.environ.setdefault("MPLBACKEND", "Agg")
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reference", default=digest_leg.DEFAULT_REFERENCE)
    parser.add_argument("--skip-digest", action="store_true")
    parser.add_argument("--only", default=None)
    args = parser.parse_args()
    legs = [
        ("i", partial(leg_i, args.seed)),
        ("ii", partial(leg_ii, args.seed)),
        ("iii", partial(leg_iii, args.seed)),
        ("iv", leg_iv),
        ("v", partial(leg_v, args.seed)),
        ("vi", partial(leg_vi, args.seed)),
        ("vii", leg_vii),
        ("viii", leg_viii),
        ("ix", leg_ix),
    ]
    if not args.skip_digest and args.only in (None, "D"):
        legs.append(("D", partial(leg_d, args.reference)))
    if args.only:
        legs = [(tag, leg) for tag, leg in legs if tag.lower() == args.only.lower()]
        if not legs:
            sys.exit(f"unknown leg {args.only!r}")
    for tag, leg in legs:
        try:
            leg()
        except Exception as error:  # a raise is a FAIL line, and the later legs still report
            check(f"({tag}) ran without raising", False, f"{type(error).__name__}: {error}")
    if args.skip_digest:
        print("(D) SKIPPED by --skip-digest: a break-it run, not the committed state")
    if not FAIL:
        print("A67 PASS")
    else:
        print(f"A67 FAIL: {FAIL}")
        sys.exit(1)
