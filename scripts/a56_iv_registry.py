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
  (i)   EVERY dataset block of EVERY recipe resolves through `resolve_dataset_block`
        and carries EXACTLY ONE of `query:`, `sweep:` and `perf:`, and the blocks of
        one file agree on which and on its param -- derived per block off the glob,
        so a recipe added tomorrow is covered with no row to write. On top, the
        named recipes carry `iv: 2` / `iv: [tax_s, y, cpi]` (gamma_z written out),
        list no `IV` baseline and are the type `RECIPES` says; the shipped
        config.yaml blocks and leg (D)'s own blocks resolve with no `iv` key at all.
        A block declaring `iv` lists a method that READS the observed Z: an `IV_METHODS`
        base in a mode other than `(T)`, which reads the translation amounts alone.
        The `DGP_ONLY_IV` blocks invert it: their `iv` shapes the DGP only, so no
        method there may read Z.
        Catches: a recipe carrying a retired key or an illegal method spelling, a
        validator that rejects its own recipe, ANY block that grew a second
        experiment type or lost its only one, a file whose blocks disagree, an `iv`
        key leaked into the shipped yaml, a `PENDING` exemption left behind once it
        starts resolving, an `iv` that only `(T)` spellings "consume", a DGP-only
        block that grew a Z reader. Misses: whether a run does anything with the key
        (refactor3 on).
  (ii)  `iv` (and `gamma_z`) on the optical and do-MNIST blocks raise a ValueError
        naming the key: rejected, not ignored (decision 2). Catches: `iv` added to
        their DATASET_KEYS. Misses: nothing about what the key would do there.
  (iii) the rejections: a duplicate name, an unknown name, a bare string, a
        non-string entry, a negative int, a bool, an int above treatment_dim, a
        gamma_z outside [0, 1) and a gamma_z of 0 under a non-empty set (a bound of
        exactly 0 is INFEASIBLE everywhere, SS3.3) all raise; every legal set of
        SS7.1 resolves, gamma_z 0 resolves under an empty set (unread there), and
        an absent key stays absent. Catches: the duplicate check dropped (a repeated
        column is a spurious moment, SS2.1), a bool passing as an int. Misses: a
        legal-looking name the panel does not carry, which the loader catches.
  (iv)  the registry builds exactly ALL_METHODS in the plan's order, and
        COPSENS_METHODS is the pinned ten (the nine plus the do-MNIST-only
        ERM+INV). Catches: a builder missing or out of sync, a name added to the
        do-MNIST backend. Misses: what a builder builds; that is (vi) and a57.
  (v)   every ALL_METHODS name and the nine stored mode spellings (`base(Z)`,
        `base(T)`, `base(T,Z)` on the DA+ IV methods) have a style with and without
        a real Z (`method_style`); with one, the ERM+IV and PI+INV+IV labels compose
        from the building blocks, without one PI+INV+IV reads pi+inv; ERM+IV takes
        ERM's blue (deep 0) and DA+ERM+IV green (2), PI+INV+IV PI+INV's hue and PI+IV
        PI's, the alphas are unchanged; ERM+IV and the DA+ERM+IV spellings are point
        estimates and PI+INV+IV is not. Catches: a name that would raise at plot
        time. Misses: how the figure looks.
  (vi)  every `+IV` builder fits with a 1-column Z without raising, reads the
        instrument (`_has_iv`) and predicts finite bounds; `ERM+IV` requested under an
        empty set is a config error naming both `ERM+IV` and `iv`, and an omitted
        `methods` falls back to ALL_METHODS without the non-DA `+IV` methods
        (`ERM+IV`; `PI+IV` and `PI+INV+IV` are PI and PI+INV without a Z) unless there
        is an instrument set (a default is not a request); the four IV classes built
        with `gamma_z` 2^-8 carry it, and their bounds follow SS2.6 (the joint
        0.069877 on the non-DA classes at s = 1, the joint at its own s on a DA class,
        r_Z 0.0625 on the intersection's baseline branch). Catches: `PI+IV` built from
        `common` (no `epsilon_iv`, so the class raises on the first real Z), a silent
        ERM on no instrument, a fallback that trips its own error, `gamma_z` not
        forwarded by the registry. Misses: the numbers the fits produce (a57).

    MPLBACKEND=Agg python scripts/a56_iv_registry.py [--seed 42] [--reference JSON] [--skip-digest]

`--skip-digest` drops leg (D), about five minutes; it exists for the break-it runs
of the other legs and is never used on the committed state.
"""

import argparse
import glob
import os
import subprocess
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
    COPSENS_METHODS,
    DATASET_KEYS,
    EPS_TOL,
    GAMMA_Z_DEFAULT,
    MethodRegistry,
    _copsens_builders,
    parse_experiment_plan,
    resolve_dataset_block,
)
from src.experiments.utils.constants import (  # noqa: E402
    DEEP_INDEX,
    ERM,
    INV,
    IV,
    IV_MODE_METHODS,
    IV_MODES,
    PI,
    alpha,
    hue,
    is_point_estimate,
    method_style,
    parse_method,
)
from src.experiments.utils.constants import label as method_label  # noqa: E402
from src.sem.cigarettes import CigaretteSEM, V, build_design  # noqa: E402

PLAN_METHODS = (
    "ATE",
    "ERM",
    "ERM+IV",
    "DA+ERM",
    "DA+ERM+IV",
    "PI+INV",
    "PI",
    "PI+IV",
    "PI+INV+IV",
    "DA+PI",
    "DA+PI+IV",
    "PI&DA+PI",
    "PI&DA+PI+IV",
)
NET_METHODS = ("ATE", "ERM", "DA+ERM", "ERM+INV", "PI+INV", "PI", "DA+PI", "DA+PI+IV", "PI&DA+PI", "PI&DA+PI+IV")
CIGARETTE_IV = ["tax_s", "y", "cpi"]
EXPERIMENT_TYPES = ("query", "sweep", "perf")
# (recipe, dataset, iv, what the experiment plan must carry). This table pins what
# cannot be read off the tree: the instrument set of each block, and which of the
# three types the named recipes are. The SHAPE rule itself -- every block carries
# exactly one type, and the blocks of one file agree on it and on its param -- is
# derived per block in leg (i) and needs no row here, so a recipe added tomorrow is
# checked the day it lands. The cigarette experiment is split by target too: the
# restricted-2sls one reports the query figures, the plasmode carries every sweep
# and every perf metric.
# the simulation recipes' instrument width (`iv:`)
SIM_IV = 2
RECIPES = (
    ("simulationFig5", "simulation", SIM_IV, "query"),
    ("ivSimulationFig5", "simulation", SIM_IV, "query"),
    ("opticalDeviceFig6", "optical_device", None, "query"),
    ("cigarettesFig7", "cigarettes", CIGARETTE_IV, "query"),
    ("validityFig9", "simulation", SIM_IV, "sweep"),
    ("validityFig9", "optical_device", None, "sweep"),
    ("validityFig9", "cigarettes", CIGARETTE_IV, "sweep"),
    ("robustnessFig11", "simulation", SIM_IV, "sweep"),
    ("robustnessFig11", "cigarettes", CIGARETTE_IV, "sweep"),
    ("latencyFig15", "simulation", SIM_IV, "perf"),
    ("latencyFig15", "cigarettes", CIGARETTE_IV, "perf"),
    ("stabilityFig16", "optical_device", None, "perf"),
)
# the one shipped block that DOES list the observed-Z baseline: the ivSimulation
# panel exists to draw it beside the two DA spellings. Every other block must not,
# and this block must, so the check below runs both ways
ERM_IV_BLOCKS = frozenset({("ivSimulationFig5", "simulation")})
# blocks leg (i) does not require to resolve yet. Empty: `ivSimulationFig5` spelled
# the retired `IV(Z)`, which the grammar rejected; it now spells `ERM+IV` and
# resolves. Leg (i) FAILS on an entry here that has started resolving, so an
# exemption cannot be left behind.
PENDING: tuple[tuple[str, str], ...] = ()
LEGAL_SETS = ([], ["tax_s"], ["tax_sn"], ["tax_s", "tax_sn"], ["tax_s", "y", "cpi"])
IV_METHODS = ("ERM+IV", "DA+ERM+IV", "PI+IV", "PI+INV+IV", "DA+PI+IV", "PI&DA+PI+IV")
# blocks whose `iv` is read by NO method, by design: `iv: 2` shapes the simulation DGP
# (the instrument enters X through a rank-m map), so the headline panel shares the
# sweeps' draw, and its methods are the (T) spellings, which read G alone. Leg (i)
# inverts the consumer check on these: no method may read Z
DGP_ONLY_IV = frozenset({("simulationFig5", "simulation")})
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


def recipe_blocks(fname):
    """The dataset blocks of a recipe, merged over its defaults, unresolved."""
    with open(os.path.join(REPO, "recipes", f"{fname}.yaml")) as handle:
        config = yaml.safe_load(handle)
    defaults = config.pop("defaults", {}) or {}
    config.pop("hyperparameters", None)
    return {name: {**defaults, **block, "im-ci": 0} for name, block in config.items()}


def load_recipe(fname, dataset):
    """Resolve ONE dataset block of a recipe exactly as src/main.py resolves config.yaml.

    The recipes carry several dataset blocks each, so the caller names the one it
    wants; a missing block is a KeyError naming what the file does carry."""
    blocks = recipe_blocks(fname)
    if dataset not in blocks:
        raise KeyError(f"recipes/{fname}.yaml carries no `{dataset}:` block, only {sorted(blocks)}")
    block = blocks[dataset]
    plan = parse_experiment_plan(block.get("experiment"))
    return resolve_dataset_block(dataset, block), plan


def plan_shape(plan):
    """The experiment types a plan carries, in `EXPERIMENT_TYPES` order.

    A recipe block is meant to carry exactly one; the tuple is what leg (i) reports
    when it carries none or two."""
    carried = {"query": plan.query, "sweep": plan.sweep is not None, "perf": plan.perf is not None}
    return tuple(name for name in EXPERIMENT_TYPES if carried[name])


def plan_axis(plan):
    """What the one experiment type is swept or measured over, for the agreement check."""
    if plan.sweep is not None:
        return tuple(plan.sweep.param)
    if plan.perf is not None:
        return tuple(plan.perf.metric)
    return ()


def leg_d(reference):
    print("(D) the shipped configuration does not move: query panel and gamma sweep step, three datasets")
    for label, ok, detail in digest_leg.compare(REPO, reference):
        check(f"(D) {label}", ok, detail)


def leg_i():
    print("(i) every recipe block resolves and carries one experiment type, the shipped yaml without the key")
    # the SHAPE contract, derived per block: EVERY block of EVERY recipe, so a recipe
    # added tomorrow is covered without a row anywhere. A block carrying two types
    # would run twice under one name; a block carrying none would run nothing.
    for path in sorted(glob.glob(os.path.join(REPO, "recipes", "*.yaml"))):
        fname = os.path.splitext(os.path.basename(path))[0]
        shapes = {}
        for dataset in recipe_blocks(fname):
            if (fname, dataset) in PENDING:
                try:
                    load_recipe(fname, dataset)
                except Exception:  # noqa: BLE001 - the exemption is exactly this failure
                    print(f"      report: {fname}.{dataset} not required to resolve yet")
                    continue
                check(f"(i) {fname}.{dataset} resolves now: drop it from PENDING", False)
                continue
            try:
                _, plan = load_recipe(fname, dataset)
            except Exception as error:  # noqa: BLE001 - one FAIL line beats a traceback
                check(f"(i) {fname}.{dataset} resolves", False, f"{type(error).__name__}: {error}")
                continue
            check(f"(i) {fname}.{dataset} resolves", True)
            shape = plan_shape(plan)
            check(f"(i) {fname}.{dataset}: exactly one experiment type", len(shape) == 1, f"{shape}")
            shapes[dataset] = (shape, plan_axis(plan))
        # and the blocks of one file agree, so a `python -m src.aggregate` column can
        # be read off the file name
        if len(shapes) > 1:
            check(
                f"(i) {fname}: its blocks agree on the type and its param",
                len(set(shapes.values())) == 1,
                f"{shapes}",
            )
    # what the tree cannot tell us: the instrument set, and which type each named
    # recipe is meant to be
    for fname, dataset, want_iv, want_plan in RECIPES:
        try:
            block, plan = load_recipe(fname, dataset)
        except Exception as error:  # noqa: BLE001
            check(f"(i) {fname}.{dataset} resolves", False, f"{type(error).__name__}: {error}")
            continue
        check(f"(i) {fname}.{dataset}: iv == {want_iv!r}", block.get("iv") == want_iv, repr(block.get("iv")))
        want_erm_iv = (fname, dataset) in ERM_IV_BLOCKS
        check(
            f"(i) {fname}.{dataset}: lists the ERM+IV baseline: {want_erm_iv}",
            ("ERM+IV" in block["methods"]) is want_erm_iv,
            f"{block['methods']}",
        )
        check(
            f"(i) {fname}.{dataset}: the plan is {want_plan} alone",
            plan_shape(plan) == (want_plan,),
            f"{plan_shape(plan)}",
        )
    # a block that DECLARES an instrument set must list a method that can consume it:
    # an `iv:` no method reads is a configuration error, and this is the registry's
    # gate for it. Which +IV methods is the owner's call, so the check is on the
    # family rather than on two names (round 2 pinned `PI+IV` and `PI+INV+IV`, which
    # went red the moment a block was switched back to a pre-IV method list). A `(T)`
    # spelling reads the translation amounts alone, so it is no consumer of Z
    for fname, dataset, want_iv, _ in RECIPES:
        if not want_iv:
            continue
        block, _ = load_recipe(fname, dataset)
        consumers = sorted(
            {m for m in block["methods"] if parse_method(m)[0] in IV_METHODS and parse_method(m)[1] != "T"}
        )
        if (fname, dataset) in DGP_ONLY_IV:
            check(
                f"(i) {fname}.{dataset}: declares iv for the DGP only and lists no method that reads it",
                not consumers,
                f"Z readers {consumers}",
            )
            continue
        check(
            f"(i) {fname}.{dataset}: declares iv and lists a method that reads it",
            bool(consumers),
            f"{block['methods']}",
        )
    listed = {(fname, dataset) for fname, dataset, want_iv, _ in RECIPES if want_iv}
    check("(i) every DGP_ONLY_IV block is a RECIPES row declaring iv", listed >= DGP_ONLY_IV, f"{sorted(DGP_ONLY_IV)}")
    plasmode_block, _ = load_recipe("robustnessFig11", "cigarettes")
    check(
        "(i) the sweep recipes declare target plasmode with the same leak budget",
        plasmode_block.get("target") == "plasmode" and plasmode_block.get("gamma_z") == GAMMA_Z_DEFAULT,
        f"{plasmode_block.get('target')!r}, {plasmode_block.get('gamma_z')!r}",
    )
    block, _ = load_recipe("cigarettesFig7", "cigarettes")
    written_out = block.get("target") == "iv" and block.get("gamma_z") == GAMMA_Z_DEFAULT == 0.0177
    check(
        "(i) the cigarette query recipe is the restricted-2sls target, gamma_z written out (Conley delta 0.05)",
        written_out,
        f"{block.get('target')!r}, {block.get('gamma_z')!r}",
    )

    # "the shipped default does not silently enable IV, so a bare `python -m src.main`
    # reproduces the non-IV baseline" is an invariant about what SHIPS, and only
    # COMMITTED state ships. Read the committed blob for it: nothing runs this gate at
    # commit time (`.pre-commit-config.yaml` is ruff plus the hygiene hooks), so the
    # working-tree form policed nothing while going red in the owner's ordinary state,
    # and a permanently red gate stops being read -- which is how a65's 0.0236 sat
    # stale for five commits. The working tree still has to RESOLVE, and an active key
    # there is reported, not failed
    committed = yaml.safe_load(
        subprocess.run(["git", "show", "HEAD:config.yaml"], cwd=REPO, capture_output=True, text=True, check=True).stdout
    )
    committed.pop("defaults", None)
    committed.pop("hyperparameters", None)
    for name, raw in committed.items():
        check(
            f"(i) committed config.yaml {name}: no active iv or gamma_z key",
            "iv" not in raw and "gamma_z" not in raw,
            f"{sorted(set(raw) & {'iv', 'gamma_z'})}",
        )
    with open(os.path.join(REPO, "config.yaml")) as handle:
        config = yaml.safe_load(handle)
    defaults = config.pop("defaults", {}) or {}
    config.pop("hyperparameters", None)
    for name, raw in config.items():
        active = sorted(set(raw) & {"iv", "gamma_z"})
        if active:
            print(f"      report: the WORKING-tree config.yaml {name} carries {active}; uncommitted, so not a FAIL")
        check(f"(i) config.yaml {name}: resolves", rejection(name, **{**defaults, **raw, "im-ci": 0}) is None)
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
        ("cigarettes", dict(iv=["tax_s"], gamma_z=0), "gamma_z 0 under a non-empty set"),
        ("cigarettes", dict(iv=["tax_s", "y", "cpi"], gamma_z=0.0), "gamma_z 0.0 under the phase-b set"),
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
    for gamma_z in (0.5, 2**-8, 1e-9):
        message = rejection("cigarettes", iv=["tax_s"], gamma_z=gamma_z)
        check(f"(iii) cigarettes gamma_z={gamma_z} resolves under a non-empty set", message is None, message or "")
    message = rejection("cigarettes", iv=[], gamma_z=0)
    check("(iii) cigarettes gamma_z=0 resolves under an empty set (unread there)", message is None, message or "")
    absent = all("iv" not in resolve_dataset_block(name, base_block(name)) for name in ("cigarettes", "simulation"))
    check("(iii) an absent iv stays absent (the consumers apply the defaults)", absent)


def leg_iv():
    print("(iv) the registry is ALL_METHODS and the do-MNIST backend is untouched")
    check("(iv) ALL_METHODS is the plan's tuple, in order", ALL_METHODS == PLAN_METHODS, repr(ALL_METHODS))
    built = MethodRegistry.build_methods(list(ALL_METHODS), gamma=GAMMA, epsilon=EPS_TOL, epsilon_iv=EPS_TOL)
    check("(iv) build_methods returns every name, in order", tuple(built) == ALL_METHODS, repr(tuple(built)))
    check("(iv) COPSENS_METHODS is the pinned ten", COPSENS_METHODS == NET_METHODS, repr(COPSENS_METHODS))
    net = _copsens_builders(
        list(COPSENS_METHODS),
        gamma=0.085,
        epsilon=0.04,
        epsilon_iv=0.04,
        recalibrate=False,
        pad=False,
        clipy=True,
        n_jobs=1,
        mean_match=True,
        rho=1.0,
        outcome_models=None,
        n_components=32,
        inv_recenter="off",
        calibrate_sigma=True,
        gamma_z_star=0.0,
    )
    check("(iv) the copsens backend builds exactly its ten", tuple(net) == NET_METHODS)
    check(
        "(iv) ERM+IV and PI+INV+IV are not copsens methods",
        not ({"ERM+IV", "DA+ERM+IV", "PI+INV+IV"} & set(COPSENS_METHODS)),
    )


def leg_v():
    print("(v) every method and every stored mode spelling has its display style")
    spelled = [f"{base}({mode})" for base in IV_MODE_METHODS for mode in IV_MODES]
    for name in ALL_METHODS + tuple(spelled):
        styled = []
        for has_z in (True, False):
            try:
                method_style(name, has_z)
                styled.append(has_z)
            except ValueError:
                pass
        check(f"(v) {name} has a style with and without a real Z", styled == [True, False], f"{styled}")
    for name in ("DA+ERM+IV(Z)", "DA+ERM+IV(T)", "DA+ERM+IV(T,Z)", "ERM+IV"):
        check(f"(v) {name} is a point estimate", is_point_estimate(name))
    check("(v) PI+INV+IV is not a point estimate", not is_point_estimate("PI+INV+IV"))
    got = method_label("ERM+IV", True)
    check("(v) ERM+IV tex composes ERM and IV", got == rf"${ERM}+{IV}$", got)
    check("(v) PI+INV+IV tex composes PI, INV, IV", method_label("PI+INV+IV", True) == rf"${PI}+{INV}+{IV}$")
    check("(v) without a Z, PI+INV+IV reads pi+inv", method_label("PI+INV+IV", False) == rf"${PI}+{INV}$")
    check(
        "(v) ERM+IV takes ERM's blue 0, DA+ERM+IV green 2",
        DEEP_INDEX[hue("ERM+IV", True)] == 0 and DEEP_INDEX[hue("DA+ERM+IV", True)] == 2,
    )
    check(
        "(v) PI+INV+IV shares PI+INV's hue, PI+IV shares PI's",
        hue("PI+INV+IV", True) == hue("PI+INV", True) and hue("PI+IV", True) == hue("PI", True),
        hue("PI+INV+IV", True),
    )
    check(
        "(v) alphas: ERM+IV solid, PI+INV+IV as PI+INV",
        alpha("ERM+IV", True) == 1.0 and alpha("PI+INV+IV", True) == 0.8,
    )


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
        "ERM+IV": dict(X=X, y=y, Z=z),
        "DA+ERM+IV": dict(X=GX, y=y, Z=z),
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
        exact = model.t_bound == EPS_TOL and model.z_bound == r_z
        check(f"(vi) {name}: epsilon_iv inert and the Z radius exactly r_Z", exact, f"{model.z_bound!r}")
    for name in ("DA+PI+IV", "PI&DA+PI+IV DA branch"):
        model = fitted[name]
        want = float(np.sqrt(model.sigma_sq / model.rho * 2**-8))
        check(f"(vi) {name} Z radius is r_Z at its own s", abs(model.z_bound - want) < 1e-12, f"{model.z_bound:.9f}")
        check(f"(vi) {name} T radius is epsilon_iv alone", model.t_bound == EPS_TOL, f"{model.t_bound!r}")
    got = fitted["PI&DA+PI+IV baseline"].z_bound
    check("(vi) PI&DA+PI+IV baseline bound is r_Z = 0.0625 to 1e-6", abs(got - 0.0625) < 1e-6, f"{got:.9f}")
    # with a positive `epsilon_iv_z` every Z constraint reads the same radius
    # 0.069877 on this panel (s = 1), and the T radius never moves
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
        got = fitted[name].z_bound
        ok = abs(got - joint) < 1e-6
        check(f"(vi) {name} with epsilon_iv_z 2^-5: the Z radius is 0.069877 to 1e-6", ok, f"{got:.9f}")
    model = fitted["DA+PI+IV"]
    want = float(np.hypot(EPS_TOL, np.sqrt(model.sigma_sq / model.rho) * np.sqrt(2**-8)))
    check(
        "(vi) DA+PI+IV with epsilon_iv_z 2^-5: the same Z radius at its own s, the T radius unmoved",
        abs(model.z_bound - want) < 1e-12 and model.t_bound == EPS_TOL,
        f"{model.z_bound:.9f}",
    )
    empty = (("cigarettes", {}), ("cigarettes", dict(iv=[])), ("simulation", {}), ("simulation", dict(iv=0)))
    for name, extra in empty:
        message = rejection(name, methods=["PI", "ERM+IV"], **extra) or ""
        label = f"(vi) {name} iv={extra.get('iv', 'absent')!r} with ERM+IV in methods raises, naming both"
        check(label, "'ERM+IV'" in message and "iv = " in message, message or "no error")
    ok = rejection("cigarettes", methods=["PI", "ERM+IV"], iv=["tax_s"]) is None
    ok = ok and rejection("simulation", methods=["PI", "ERM+IV"], iv=4) is None
    check("(vi) ERM+IV with a real instrument set resolves", ok)
    # an omitted `methods` is a default, not a request: it carries the non-DA +IV methods only with an
    # instrument set, since without one PI+IV and PI+INV+IV are PI and PI+INV (a57 (i), (ii))
    without = resolve_dataset_block("cigarettes", base_block("cigarettes"))["methods"]
    with_iv = resolve_dataset_block("cigarettes", {**base_block("cigarettes"), "iv": ["tax_s"]})["methods"]
    check(
        "(vi) omitted methods, no instrument: ALL_METHODS minus ERM+IV, PI+IV and PI+INV+IV",
        without == [m for m in ALL_METHODS if m not in ("ERM+IV", "PI+IV", "PI+INV+IV")],
    )
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
