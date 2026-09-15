"""A56: the IV registry: the `iv` key, its validation, two new methods, their builders.

refactor1 exposes the instrument set through config.yaml and the registry and runs
nothing new: no block ships an active `iv` key, so every shipped number is today's.
Legs:

  (D)   the digest leg (scripts/digest_leg.py): one query panel and one gamma sweep
        step per dataset at the shipped configuration, every numeric artifact hashed
        against the reference recorded on the parent commit 14509db, no tolerance.
        Catches: any moved number on any shipped path, including one hiding behind
        a defaulted argument or a re-ordered draw. Misses: a change on a path the
        shipped config does not take (a non-empty `iv:`), and figure bytes, which
        carry a timestamp and are not hashed.
  (i)   the two recipes resolve through `resolve_dataset_block` with `iv: 4` and
        `iv: [tax_s, y, cpi]` (gamma_z 2^-8 written out) and neither lists the `IV`
        baseline; the shipped config.yaml blocks and leg (D)'s own blocks resolve
        with no `iv` key at all. Catches: a recipe carrying a retired key, a
        validator that rejects its own recipe, an `iv` key leaked into the shipped
        yaml. Misses: whether a run does anything with the key (refactor3 on).
  (ii)  `iv` (and `gamma_z`) on the optical and do-MNIST blocks raise a ValueError
        naming the key: rejected, not ignored (decision 2). Catches: `iv` added to
        their DATASET_KEYS. Misses: nothing about what the key would do there.
  (iii) the rejections: a duplicate name, an unknown name, a bare string, a
        non-string entry, a negative int, a bool, an int above treatment_dim, a
        gamma_z outside [0, 1) all raise; every legal set of SS7.1 resolves, and an
        absent key stays absent. Catches: the duplicate check dropped (a repeated
        column is a spurious moment, SS2.1), a bool passing as an int. Misses: a
        legal-looking name the panel does not carry, which the loader catches.
  (iv)  the registry builds exactly ALL_METHODS in the plan's order, and
        PARTIAL_R2_NET_METHODS is the unchanged nine. Catches: a builder missing or
        out of sync, a name added to the do-MNIST backend. Misses: what a builder
        builds; that is (vi) and a57.
  (v)   every ALL_METHODS name has a TEX_MAPPER, COLOR_MAP and ALPHA_MAP entry, the
        new TeX strings compose from the building blocks, IV is a point estimate
        and PI+INV+IV takes the one unused hue. Catches: a name that would KeyError
        at plot time. Misses: how the figure looks.
  (vi)  every `+IV` builder fits with a 1-column Z without raising, reads the
        instrument (`_has_iv`) and predicts finite bounds; `IV` requested under an
        empty set is a config error naming both `IV` and `iv`, and an omitted
        `methods` falls back to ALL_METHODS without `IV` unless there is an
        instrument set (a default is not a request); the four IV classes built
        with `gamma_z` 2^-8 carry it, and their bounds follow SS2.6 (the joint
        0.069877 on the non-DA classes at s = 1, the joint at its own s on a DA
        class, r_Z 0.0625 on the intersection's baseline branch). Catches: `PI+IV`
        built from `common` (no `epsilon_iv`, so the class raises on the first
        real Z), a silent 2SLS on no instrument, a fallback that trips its own
        error, `gamma_z` not forwarded by the registry. Misses: the numbers the
        fits produce (a57).

    MPLBACKEND=Agg python scripts/a56_iv_registry.py [--seed 42] [--reference JSON] [--skip-digest]

`--skip-digest` drops leg (D), about five minutes; it exists for the break-it runs
of the other legs and is never used on the committed state.
"""

import argparse
import os
import sys

import numpy as np
import yaml
from loguru import logger

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import digest_leg  # noqa: E402

import src.methods.sensitivity_models as solvers  # noqa: E402
from src.data_augmentors.cigarettes import ScaleTranslation  # noqa: E402
from src.experiments.configs import (  # noqa: E402
    ALL_METHODS,
    DATASET_KEYS,
    EPS_TOL,
    GAMMA_Z_DEFAULT,
    PARTIAL_R2_NET_METHODS,
    MethodRegistry,
    _partial_r2_net_builders,
    parse_experiment_plan,
    resolve_dataset_block,
)
from src.experiments.utils.constants import ALPHA_MAP, COLOR_MAP, INV, IV, PI, POINT_ESTIMATES, TEX_MAPPER  # noqa: E402
from src.sem.cigarettes import CigaretteSEM, V, build_design  # noqa: E402

PLAN_METHODS = (
    "ATE",
    "ERM",
    "DA+ERM",
    "IV",
    "DA+IV",
    "PI+INV",
    "PI",
    "PI+IV",
    "PI+INV+IV",
    "DA+PI",
    "DA+PI+IV",
    "PI&DA+PI",
    "PI&DA+PI+IV",
)
NET_METHODS = ("ATE", "ERM", "DA+ERM", "PI+INV", "PI", "DA+PI", "DA+PI+IV", "PI&DA+PI", "PI&DA+PI+IV")
RECIPES = {"iv_fig13": ("simulation", 4), "neighbour-price_fig12": ("cigarettes", ["tax_s", "y", "cpi"])}
LEGAL_SETS = ([], ["tax_s"], ["tax_sn"], ["tax_s", "tax_sn"], ["tax_s", "y", "cpi"])
IV_METHODS = ("IV", "DA+IV", "PI+IV", "PI+INV+IV", "DA+PI+IV", "PI&DA+PI+IV")
GAMMA = 0.25
FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAIL.append(name)


def base_block(name):
    """A block carrying its required keys and nothing else."""
    return {
        "simulation": dict(seed=42, kernel_dim=0),
        "optical_device": dict(seed=42, augmentation="all"),
        "cigarettes": dict(seed=42, augmentation="translate", target="iv", spec="t3"),
        "do_mnist": dict(seed=42, augmentation="translate", gamma=0.067, epsilon=0.1, methods=["PI"]),
    }[name]


def rejection(name, **extra):
    """The ValueError message `resolve_dataset_block` raises on the block, or None."""
    try:
        resolve_dataset_block(name, {**base_block(name), **extra})
    except ValueError as error:
        return str(error)
    return None


def load_recipe(fname):
    """Resolve a recipe exactly as src/main.py resolves config.yaml."""
    with open(os.path.join(REPO, "recipes", f"{fname}.yaml")) as handle:
        config = yaml.safe_load(handle)
    defaults = config.pop("defaults", {}) or {}
    config.pop("hyperparameters", None)
    ((name, block),) = config.items()
    block = {**defaults, **block}
    plan = parse_experiment_plan(block.get("experiment"))
    return name, resolve_dataset_block(name, block), plan


def leg_d(reference):
    print("(D) the shipped configuration does not move: query panel and gamma sweep step, three datasets")
    for label, ok, detail in digest_leg.compare(REPO, reference):
        check(f"(D) {label}", ok, detail)


def leg_i():
    print("(i) the recipes resolve with the key, the shipped yaml and leg (D)'s blocks without it")
    for fname, (want_name, want_iv) in RECIPES.items():
        try:
            name, block, plan = load_recipe(fname)
        except ValueError as error:
            check(f"(i) {fname} resolves", False, str(error))
            continue
        check(f"(i) {fname}: the {want_name} block", name == want_name, name)
        check(f"(i) {fname}: iv == {want_iv!r}", block.get("iv") == want_iv, repr(block.get("iv")))
        check(f"(i) {fname}: PI+IV and PI+INV+IV in methods", {"PI+IV", "PI+INV+IV"} <= set(block["methods"]))
        check(f"(i) {fname}: the IV baseline is not listed", "IV" not in block["methods"])
        check(f"(i) {fname}: query and sweep planned", plan.query and plan.sweep is not None)
    _, block, _ = load_recipe("neighbour-price_fig12")
    written_out = block.get("gamma_z") == GAMMA_Z_DEFAULT == 2**-8
    check("(i) neighbour-price: gamma_z written out as the default 2^-8", written_out, repr(block.get("gamma_z")))

    with open(os.path.join(REPO, "config.yaml")) as handle:
        config = yaml.safe_load(handle)
    defaults = config.pop("defaults", {}) or {}
    config.pop("hyperparameters", None)
    for name, raw in config.items():
        check(f"(i) config.yaml {name}: no active iv or gamma_z key", "iv" not in raw and "gamma_z" not in raw)
        check(f"(i) config.yaml {name}: resolves", rejection(name, **{**defaults, **raw}) is None)
    for name in digest_leg.DATASETS:
        block = {**digest_leg.TOGGLES, **digest_leg.BLOCKS[name]}
        check(f"(i) leg (D) block {name}: no iv key, resolves", "iv" not in block and rejection(name, **block) is None)


def leg_ii():
    print("(ii) `iv` on the optical and do-MNIST blocks is rejected by name")
    for name in ("optical_device", "do_mnist"):
        message = rejection(name, iv=4)
        check(f"(ii) {name}: iv raises and the error names the key", message is not None and "'iv'" in message, message)
        check(f"(ii) {name}: gamma_z raises too", (rejection(name, gamma_z=0.1) or "").count("gamma_z") > 0)
        check(f"(ii) {name}: iv not in DATASET_KEYS", "iv" not in DATASET_KEYS[name])


def leg_iii():
    print("(iii) the rejections, and every legal set")
    cases = [
        ("cigarettes", dict(iv=["tax_s", "tax_s"]), "a duplicate excise"),
        ("cigarettes", dict(iv=["tax_s", "y", "y"]), "a duplicate treatment"),
        ("cigarettes", dict(iv=["tax_x"]), "an unknown name"),
        ("cigarettes", dict(iv="tax_s"), "a bare string"),
        ("cigarettes", dict(iv=[1]), "a non-string entry"),
        ("cigarettes", dict(iv=["tax_s"], gamma_z=1.0), "gamma_z at 1"),
        ("cigarettes", dict(iv=["tax_s"], gamma_z=-0.1), "a negative gamma_z"),
        ("cigarettes", dict(iv=["tax_s"], gamma_z=True), "a bool gamma_z"),
        ("simulation", dict(iv=-1), "a negative int"),
        ("simulation", dict(iv=True), "a bool"),
        ("simulation", dict(iv=33), "an int above treatment_dim 32"),
        ("simulation", dict(iv="4"), "a string"),
        ("simulation", dict(iv=[4]), "a list"),
    ]
    for name, extra, why in cases:
        message = rejection(name, **extra)
        check(f"(iii) {name} {why} raises", message is not None, message or "no error")
    for iv in LEGAL_SETS:
        check(f"(iii) cigarettes iv={iv} resolves", rejection("cigarettes", iv=iv) is None)
    for iv in (0, 1, 4, 32):
        check(f"(iii) simulation iv={iv} resolves", rejection("simulation", iv=iv) is None)
    for gamma_z in (0, 0.5, 2**-8):
        message = rejection("cigarettes", iv=["tax_s"], gamma_z=gamma_z)
        check(f"(iii) cigarettes gamma_z={gamma_z} resolves", message is None, message or "")
    absent = all("iv" not in resolve_dataset_block(name, base_block(name)) for name in ("cigarettes", "simulation"))
    check("(iii) an absent iv stays absent (the consumers apply the defaults)", absent)


def leg_iv():
    print("(iv) the registry is ALL_METHODS and the do-MNIST backend is untouched")
    check("(iv) ALL_METHODS is the plan's tuple, in order", ALL_METHODS == PLAN_METHODS, repr(ALL_METHODS))
    built = MethodRegistry.build_methods(list(ALL_METHODS), gamma=GAMMA, epsilon=EPS_TOL, epsilon_iv=EPS_TOL)
    check("(iv) build_methods returns every name, in order", tuple(built) == ALL_METHODS, repr(tuple(built)))
    check("(iv) PARTIAL_R2_NET_METHODS unchanged", PARTIAL_R2_NET_METHODS == NET_METHODS, repr(PARTIAL_R2_NET_METHODS))
    net = _partial_r2_net_builders(
        list(PARTIAL_R2_NET_METHODS),
        gamma=0.067,
        epsilon=0.1,
        epsilon_iv=0.1,
        recalibrate=False,
        pad=False,
        clipy=True,
        n_jobs=1,
        mean_match=True,
        rho=1.0,
        outcome_models=None,
        unfrozen_layers=1,
    )
    check("(iv) the net backend still builds exactly its nine", tuple(net) == NET_METHODS)
    check("(iv) IV and PI+INV+IV are not net methods", not ({"IV", "PI+INV+IV"} & set(PARTIAL_R2_NET_METHODS)))


def leg_v():
    print("(v) every method has its display entries")
    for name in ALL_METHODS:
        present = name in TEX_MAPPER and name in COLOR_MAP and name in ALPHA_MAP
        check(f"(v) {name} in TEX_MAPPER, COLOR_MAP, ALPHA_MAP", present)
    check("(v) IV tex", TEX_MAPPER.get("IV") == rf"${IV}$", TEX_MAPPER.get("IV"))
    check("(v) PI+INV+IV tex composes PI, INV, IV", TEX_MAPPER.get("PI+INV+IV") == rf"${PI}+{INV}+{IV}$")
    check("(v) IV shares DA+IV's hue 2", COLOR_MAP.get("IV") == COLOR_MAP.get("DA+IV") == 2)
    hue = COLOR_MAP.get("PI+INV+IV")
    check("(v) PI+INV+IV takes the unused hue 5", hue == 5 and list(COLOR_MAP.values()).count(5) == 1, repr(hue))
    check("(v) alphas: IV solid, PI+INV+IV as PI+INV", ALPHA_MAP.get("IV") == 1.0 and ALPHA_MAP.get("PI+INV+IV") == 0.8)
    point = "IV" in POINT_ESTIMATES and "PI+INV+IV" not in POINT_ESTIMATES
    check("(v) IV is a point estimate, PI+INV+IV is not", point)


def leg_vi(seed):
    print("(vi) every +IV builder fits with a 1-column Z; IV under an empty set is a config error")
    design = build_design(CigaretteSEM.panel(), spec="t3", anchor="own-tax")
    X, y = design.X, design.y
    z = design.Z[:, :1]
    np.random.seed(seed)
    GX, G = ScaleTranslation(V, std=float(np.std(X @ (V / np.linalg.norm(V)))))(X)
    # two budgets (batch B's ruling): the DA classes carry `epsilon_iv`, the non-DA
    # ones and the intersection's baseline `epsilon_iv_z`; both positive here, or
    # the non-DA bound would be exactly 0 and INFEASIBLE (SS3.3)
    builders = MethodRegistry.build_methods(
        list(IV_METHODS),
        gamma=GAMMA,
        epsilon=EPS_TOL,
        epsilon_iv=EPS_TOL,
        epsilon_iv_z=EPS_TOL,
        recalibrate=True,
        pad=False,
        clipy=False,
        n_jobs=1,
        mean_match=True,
    )
    calls = {
        "IV": dict(X=X, y=y, Z=z),
        "DA+IV": dict(X=GX, y=y, Z=z),
        "PI+IV": dict(X=X, y=y, Z=z),
        "PI+INV+IV": dict(X=X, y=y, GX=GX, Z=z),
        "DA+PI+IV": dict(X=GX, y=y, Z=z),
        # the intersection's DA branch takes its instrument as G (a57 gives it Z too)
        "PI&DA+PI+IV": dict(X=X, y=y, GX=GX, G=z),
    }
    for name in IV_METHODS:
        model = builders[name]()
        try:
            model.fit(**calls[name])
            raised = None
        except Exception as error:  # the leg reports it rather than dying on it
            raised = error
        check(f"(vi) {name} fits with a 1-column Z", raised is None, repr(raised) if raised else "")
        if raised is not None:
            continue
        out = np.asarray(model.predict(np.eye(X.shape[1])), dtype=float)
        check(f"(vi) {name} predicts finite values on e_j", np.all(np.isfinite(out)), f"shape {out.shape}")
        if hasattr(model, "_has_iv"):
            check(f"(vi) {name} read the instrument", model._has_iv)
        if name == "PI&DA+PI+IV":
            check("(vi) PI&DA+PI+IV DA branch read the instrument", model.augmented._has_iv)
    # gamma_z reaches every IV class through the registry, and the bound follows
    # SS2.6 per row: the non-DA classes and the intersection's baseline carry
    # `epsilon_iv_z` (default 0.0, the declared path) so their bound is exactly
    # r_Z = s sqrt(gamma_z); the DA classes carry `epsilon_iv` and read the joint
    declared = MethodRegistry.build_methods(
        ["PI+IV", "PI+INV+IV", "DA+PI+IV", "PI&DA+PI+IV"],
        gamma=GAMMA,
        epsilon=EPS_TOL,
        epsilon_iv=EPS_TOL,
        gamma_z=2**-8,
        recalibrate=True,
        pad=False,
        clipy=False,
        n_jobs=1,
        mean_match=True,
    )
    fitted = {name: declared[name]().fit(**calls[name]) for name in declared}
    fitted["PI&DA+PI+IV baseline"] = fitted["PI&DA+PI+IV"].baseline
    fitted["PI&DA+PI+IV DA branch"] = fitted["PI&DA+PI+IV"].augmented
    for name, model in fitted.items():
        check(f"(vi) {name} built with gamma_z 2^-8 carries it", model.gamma_z == 2**-8, repr(model.gamma_z))
    for name in ("PI+IV", "PI+INV+IV", "PI&DA+PI+IV baseline"):
        model = fitted[name]
        r_z = float(np.sqrt(model.sigma_sq / model.rho * 2**-8))
        exact = model.epsilon_iv == 0.0 and model.iv_bound == r_z
        check(f"(vi) {name}: epsilon_iv 0.0 and bound exactly r_Z = s sqrt(gamma_z)", exact, f"{model.iv_bound!r}")
    for name in ("DA+PI+IV", "PI&DA+PI+IV DA branch"):
        model = fitted[name]
        want = float(np.hypot(EPS_TOL, np.sqrt(model.sigma_sq / model.rho) * np.sqrt(2**-8)))
        check(
            f"(vi) {name} bound is the joint at its own s", abs(model.iv_bound - want) < 1e-12, f"{model.iv_bound:.9f}"
        )
    got = fitted["PI&DA+PI+IV baseline"].iv_bound
    check("(vi) PI&DA+PI+IV baseline bound is r_Z = 0.0625 to 1e-6", abs(got - 0.0625) < 1e-6, f"{got:.9f}")
    # with a positive `epsilon_iv_z` the non-DA classes read the joint 0.069877
    # (s = 1 on this panel) and `epsilon_iv_z` never reaches the DA class
    both = MethodRegistry.build_methods(
        ["PI+IV", "PI+INV+IV", "DA+PI+IV", "PI&DA+PI+IV"],
        gamma=GAMMA,
        epsilon=EPS_TOL,
        epsilon_iv=EPS_TOL,
        epsilon_iv_z=EPS_TOL,
        gamma_z=2**-8,
        recalibrate=True,
        pad=False,
        clipy=False,
        n_jobs=1,
        mean_match=True,
    )
    fitted = {name: both[name]().fit(**calls[name]) for name in both}
    fitted["PI&DA+PI+IV baseline"] = fitted["PI&DA+PI+IV"].baseline
    joint = float(np.hypot(EPS_TOL, np.sqrt(2**-8)))
    for name in ("PI+IV", "PI+INV+IV", "PI&DA+PI+IV baseline"):
        got = fitted[name].iv_bound
        ok = fitted[name].epsilon_iv == EPS_TOL and abs(got - joint) < 1e-6
        check(f"(vi) {name} with epsilon_iv_z 2^-5: bound is the joint 0.069877 to 1e-6", ok, f"{got:.9f}")
    model = fitted["DA+PI+IV"]
    want = float(np.hypot(EPS_TOL, np.sqrt(model.sigma_sq / model.rho) * np.sqrt(2**-8)))
    check(
        "(vi) DA+PI+IV with epsilon_iv_z 2^-5: still the joint at its own s, epsilon_iv_z never reaches it",
        abs(model.iv_bound - want) < 1e-12 and model.epsilon_iv == EPS_TOL,
        f"{model.iv_bound:.9f}",
    )
    empty = (("cigarettes", {}), ("cigarettes", dict(iv=[])), ("simulation", {}), ("simulation", dict(iv=0)))
    for name, extra in empty:
        message = rejection(name, methods=["PI", "IV"], **extra) or ""
        label = f"(vi) {name} iv={extra.get('iv', 'absent')!r} with IV in methods raises, naming both"
        check(label, "'IV'" in message and "iv = " in message, message or "no error")
    ok = rejection("cigarettes", methods=["PI", "IV"], iv=["tax_s"]) is None
    ok = ok and rejection("simulation", methods=["PI", "IV"], iv=4) is None
    check("(vi) IV with a real instrument set resolves", ok)
    # an omitted `methods` is a default, not a request: it carries IV only with an instrument set
    without = resolve_dataset_block("cigarettes", base_block("cigarettes"))["methods"]
    with_iv = resolve_dataset_block("cigarettes", {**base_block("cigarettes"), "iv": ["tax_s"]})["methods"]
    check("(vi) omitted methods, no instrument: ALL_METHODS minus IV", without == [m for m in ALL_METHODS if m != "IV"])
    check("(vi) omitted methods, an instrument set: all of ALL_METHODS", with_iv == list(ALL_METHODS))


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    os.chdir(REPO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reference", default=digest_leg.DEFAULT_REFERENCE)
    parser.add_argument("--skip-digest", action="store_true")
    args = parser.parse_args()
    print(f"tree: {solvers.__file__}")
    leg_i()
    leg_ii()
    leg_iii()
    leg_iv()
    leg_v()
    leg_vi(args.seed)
    if not args.skip_digest:
        leg_d(args.reference)
    else:
        print("(D) SKIPPED by --skip-digest: a break-it run, not the committed state")
    if not FAIL:
        print("A56 PASS")
    else:
        print(f"A56 FAIL: {FAIL}")
        sys.exit(1)
