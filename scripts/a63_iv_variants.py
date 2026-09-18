"""A63: the instrument modes on the DA+ IV methods, the approx_error rollback, the clamp.

Four commits on refactor6/iv-figures: `_approx_error` leaves the normalised ids
(`constants.py`, `plotting.py`, a61-vi); every `_coverage` figure and every figure
drawn normalised is clamped to a linear axis on `CLAMP_YLIM` (`create_sweep_plot`);
a DA+ IV method may spell its instrument mode, `DA+PI+IV(Z)` for the configured Z
alone and `DA+PI+IV(T,Z)` for both constraints
(`parse_method` in `constants.py`, `resolve_dataset_block`, `build_methods`,
`fit_model`, the intersection's `instrument`, `_plot_headline`); both recipes list
`DA+PI+IV(Z)` beside `DA+PI+IV`. Legs:

  (D)   the digest leg (scripts/digest_leg.py), as a56 to a61: no shipped block
        spells a mode, the toggle is off there, and figures are not hashed.
        Catches: a bare name parsed as the Z mode (every DA+PI+IV pkl moves), a
        moved number anywhere on the shipped path. Misses: the configured-set
        path and the figure bytes, which (i) to (v) cover.
  (i)   the grammar: the 13 bare names parse as (base, "T,Z"), the eight legal
        spellings (whitespace inside the parentheses tolerated; `(T)` since
        round 12, a64 pins what it computes) and the twelve rejections, through
        `parse_method` and through `resolve_dataset_block`, where the stored
        spelling is `base`, `base(Z)`, `base(T)` or `base(T,Z)`; a duplicate
        (base, mode) pair raises naming both entries (`DA+PI+IV` beside
        `DA+PI+IV(T,Z)`, `PI` twice), `DA+PI+IV` beside `DA+PI+IV(Z)` resolves; a
        suffix on a non-DA method raises naming it; `DA+IV(Z)` under an empty set
        raises naming it and `iv`, `DA+PI+IV(Z)` resolves there and on the optical
        block, any suffix on the do_mnist block raises; an unquoted `(T,Z)` in a
        YAML flow list raises with the quoting hint and the quoted form resolves;
        `IV_MODE_METHODS` is the derived set, every stored spelling has its display
        entries (`(T,Z)` equal to the bare, `(Z)` on the same hue), `DA+IV(Z)` is
        a point estimate, `build_methods` returns the requested spellings in order.
        Catches: a parser accepting a suffix on a non-DA method, a duplicate check
        keyed on the spelling (it would let R2's case through), a missing display
        entry (a KeyError at plot time), a registry that reorders. Misses: what a
        spelled method computes, which (ii) pins.
  (ii)  the three modes on fitted attributes, on the cigarette design and the
        seed-42 DA draw with a 1-column Z, `epsilon_iv` 2^-5 and `epsilon_iv_z`
        2^-6 (distinct on purpose), `gamma_z` 2^-8 and `rho = rho_hat` of the draw
        (the bound pin depends on it: `s^2` is `sigma_sq(GX) / rho` and with the
        registry default 1.0 the two bounds differ by 4e-5), through the registry
        and `fit_model`: instrument width 2 on bare and `(T,Z)`, 1 on `(Z)` and
        `PI+IV`, the intersection's DA branch the same and its baseline 1 in both
        modes; `epsilon_iv` 2^-5 on bare and `(T,Z)`, 2^-6 on `(Z)`, `(Z)`'s bound
        equal to `PI+IV`'s to 1e-12; `(T,Z)` predicts as bare to 1e-9, `DA+IV(T,Z)`
        as `DA+IV` to 1e-12, `DA+IV(Z)` as a fresh 2SLS on (GX, y, Z); under an
        empty Z `DA+PI+IV(Z)` is `DA+PI` and `PI&DA+PI+IV(Z)` is `PI&DA+PI` to
        exactly 0.0, and `fit_model` on `DA+IV(Z)` raises. Catches: G stacked in
        the Z mode (width 2), the Z variant built from `da_iv_common` (the T-side
        term in its bound), the intersection ignoring `instrument`. Misses: a wrong
        `gamma_z` compensated by a wrong `epsilon_iv_z`, which nothing produces.
  (iii) the rollback: `NORMALIZED_SWEEP_SUFFIXES` is width and worst_error,
        `normalize_sweep` returns an `_approx_error` input untouched with one
        warning, `validate_plot_keys` raises on `normalize` under `_approx_error`
        and `_coverage` and accepts it under `_width` and `_worst_error`, and an
        approx-error figure drawn with `normalize=True` keeps its asinh axis and a
        y-label without the baseline. Catches: `_approx_error` back in the tuple.
        Misses: nothing on the rule; the figure bytes.
  (iv)  the clamp on rendered axes, synthetic sweeps read off `plt.gca()`: a
        coverage figure under a `PLOT_CONFIGS` log scale and (0.5, 2) limits reads
        linear and exactly (-0.05, 1.05) with two warnings; a coverage series
        spanning 0.005 to 1 that `_resolve_scale` would promote to log reads the
        same; a normalised width figure with DA+PI at 1.3 reads the same and the
        line is drawn to 1.3 (the frame clips it); the same figure un-normalised,
        a normalised worst-error figure with no baseline among the methods, and an
        approx-error figure under the toggle are not clamped; on a sweep drawn
        with `DA+PI+IV(Z)` beside `DA+PI+IV` the `(Z)` line carries the
        `INSTRUMENT_Z_STYLE` dash pattern and the base line is solid. Catches: the
        hook above `_rescale` (the pad survives), keyed on the toggle rather than
        on a baseline (the no-baseline and approx-error cases clamp), a dropped
        `set_yscale` (the log config survives), the `(Z)` sibling drawn solid.
        Misses: a hook after the tick helpers, which leaves the limits right and
        the minor labels re-blanked.
  (v)   both recipes resolve with `DA+PI+IV` and `DA+PI+IV(Z)` and no stored
        duplicate, and run at reduced scale (1 experiment, 4 sweep samples,
        n_jobs 1) through the production path: on the cigarette gamma sweep
        both keys are present, the `(Z)` widths are finite and at most `DA+PI`'s
        at every OK step (the same ball with one more constraint), the ratio-1
        step is OK, the OK counts of `PI+IV`, `DA+PI+IV` and `DA+PI+IV(Z)` are
        RECORDED (the `(Z)` variant runs at `PI+IV`'s bound and is INFEASIBLE
        under the same floor), T1 carries the `(Z)` row under its TeX label and
        F1's outcomes are keyed exactly `HEADLINE_METHODS`; the same block with
        `DA+PI+IV` respelled `DA+PI+IV(T,Z)` keeps F1's keys and its DA band to
        1e-9, and the same block with `DA+PI+IV(Z)` removed keeps F1's DA band to
        1e-9 as well (the `(Z)` entry, listed after the default, must not reach
        the headline key); on the simulation gamma sweep both keys, finite `(Z)`
        widths, the ordering `(Z) <= DA+PI` at every OK step, the OK counts
        recorded. Catches: a recipe without the `(Z)` method, the headline lookup
        reverted (the respelled run loses its DA band), the headline key resolved
        for every mode (the `(Z)` band overwrites the default's, gap 0.9 on
        beta_pn), a `(Z)` variant wider than `DA+PI`. Misses: the full-scale
        sweeps.

    MPLBACKEND=Agg python scripts/a63_iv_variants.py [--seed 42] [--reference JSON] [--skip-digest]

`--skip-digest` drops leg (D) and exists for the break-it runs of the other legs;
the committed state is always gated with it on.
"""

import argparse
import contextlib
import os
import pickle
import shutil
import sys
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml
from loguru import logger

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import digest_leg  # noqa: E402
from munch import munchify  # noqa: E402

import src.experiments.utils.constants as constants_module  # noqa: E402
from src.data_augmentors.cigarettes import ScaleTranslation  # noqa: E402
from src.experiments.cigarettes import HEADLINE_METHODS  # noqa: E402
from src.experiments.configs import (  # noqa: E402
    ALL_METHODS,
    EPS_TOL,
    METRIC_SPECS,
    MethodRegistry,
    parse_experiment_plan,
    resolve_dataset_block,
)
from src.experiments.utils import set_seed  # noqa: E402
from src.experiments.utils.constants import (  # noqa: E402
    ALPHA_MAP,
    ARTIFACTS_DIRECTORY,
    CLAMP_YLIM,
    COLOR_MAP,
    INSTRUMENT_Z_STYLE,
    IV_MODE_METHODS,
    NORMALIZED_SWEEP_SUFFIXES,
    PLOT_CONFIGS,
    POINT_ESTIMATES,
    SUBDIR_QUERY,
    SUBDIR_SWEEP,
    TEX_MAPPER,
    parse_method,
    plot_keys_for,
    spelled_method,
    validate_plot_keys,
)
from src.experiments.utils.metrics import STATUS_CATEGORIES, rho_hat  # noqa: E402
from src.experiments.utils.model_fitting import fit_model  # noqa: E402
from src.experiments.utils.plotting import create_sweep_plot, normalize_sweep  # noqa: E402
from src.main import ORCHESTRATORS  # noqa: E402
from src.methods.regression import TwoStageLeastSquaresIV  # noqa: E402
from src.sem.cigarettes import CigaretteSEM, V, build_design  # noqa: E402

SPELLED = (
    "DA+PI+IV(Z)",
    "DA+PI+IV(T,Z)",
    "DA+PI+IV(T)",
    "PI&DA+PI+IV(Z)",
    "PI&DA+PI+IV(T,Z)",
    "PI&DA+PI+IV(T)",
    "DA+IV(Z)",
    "DA+IV(T,Z)",
    "DA+IV(T)",
)
LEGAL = {
    "DA+PI+IV(Z)": ("DA+PI+IV", "Z"),
    "DA+PI+IV( Z )": ("DA+PI+IV", "Z"),
    "DA+PI+IV(T,Z)": ("DA+PI+IV", "T,Z"),
    "DA+PI+IV(T, Z)": ("DA+PI+IV", "T,Z"),
    "DA+PI+IV(T)": ("DA+PI+IV", "T"),
    "DA+PI+IV( T )": ("DA+PI+IV", "T"),
    "PI&DA+PI+IV(Z)": ("PI&DA+PI+IV", "Z"),
    "PI&DA+PI+IV(T,Z)": ("PI&DA+PI+IV", "T,Z"),
    "DA+IV(Z)": ("DA+IV", "Z"),
    "DA+IV(T,Z)": ("DA+IV", "T,Z"),
}
REJECTED = (
    "PI+IV(Z)",
    "PI+INV+IV(T,Z)",
    "PI(Z)",
    "DA+PI(Z)",
    "DA+ERM(Z)",
    "DA+PI+IV(X)",
    "DA+PI+IV(Z,T)",
    "DA+PI+IV()",
    "DA+PI+IV(Z)(Z)",
    "DA+PI+IV(T,T)",
    "DA+PI+IV (Z)",
    "da+pi+iv(z)",
)
MODE_NAMES = (
    "PI+IV",
    "DA+PI+IV",
    "DA+PI+IV(T,Z)",
    "DA+PI+IV(Z)",
    "PI&DA+PI+IV",
    "PI&DA+PI+IV(Z)",
    "DA+IV",
    "DA+IV(T,Z)",
    "DA+IV(Z)",
)
GAMMA = 0.25
TOGGLES = dict(recalibrate=True, clipy=False, mean_match=True, n_jobs=1)
INFEASIBLE, FAILURE = STATUS_CATEGORIES.index("infeasible"), STATUS_CATEGORIES.index("solver_failure")
FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAIL.append(name)


# ------------------------------------------------------------------ helpers


@contextlib.contextmanager
def captured():
    """The loguru messages at WARNING and above emitted inside the block."""
    lines = []
    handle = logger.add(lambda message: lines.append(message.record["message"]), level="WARNING")
    try:
        yield lines
    finally:
        logger.remove(handle)


def base_block(name):
    """A block carrying its required keys and nothing else."""
    return {
        "optical_device": dict(seed=42, augmentation="all"),
        "cigarettes": dict(seed=42, augmentation="translate", target="iv", spec="t3"),
        "do_mnist": dict(seed=42, augmentation="translate", gamma=0.067, epsilon=0.1),
    }[name]


def rejection(name, **extra):
    """The ValueError message `resolve_dataset_block` raises on the block, or None."""
    try:
        resolve_dataset_block(name, {**base_block(name), **extra})
    except ValueError as error:
        return str(error)
    return None


def recipe(name):
    fname = {"cigarettes": "neighbour-price_fig12", "simulation": "iv_fig13"}[name]
    with open(os.path.join(REPO, "recipes", f"{fname}.yaml")) as handle:
        config = yaml.safe_load(handle)
    defaults = config.pop("defaults", {}) or {}
    return {**defaults, **config[name]}


def reduced_block(name, **overrides):
    block = recipe(name)
    block.pop("experiment", None)
    return resolve_dataset_block(name, {**block, "n_experiments": 1, "sweep_samples": 4, "n_jobs": 1, **overrides})


def run_reduced(name, query, sweep, **overrides):
    """The recipe through the production path at reduced scale; the artifacts folder."""
    spec = {"query": query}
    if sweep:
        spec["sweep"] = {"param": ["gamma"], "metric": recipe(name)["experiment"]["sweep"]["metric"]}
    plan = parse_experiment_plan(spec)
    block = reduced_block(name, **overrides)
    folder = os.path.join(ARTIFACTS_DIRECTORY, name)
    shutil.rmtree(folder, ignore_errors=True)
    set_seed(block["seed"])
    ORCHESTRATORS[name](**block, hyperparameters=munchify(digest_leg.HYPERPARAMETERS)).run(plan)
    return folder


def load(folder, subdir, fname):
    with open(os.path.join(folder, subdir, fname), "rb") as handle:
        return pickle.load(handle)  # noqa: S301 - our own artifact


def ok_steps(statuses, name):
    """Steps (over the one experiment) with no infeasible and no failed query."""
    counts = statuses[name][:, 0]
    return (counts[:, INFEASIBLE] == 0) & (counts[:, FAILURE] == 0)


def draw(seed):
    """The a56-vi construction: the cigarette design, the seed-`seed` DA draw, a 1-column Z."""
    design = build_design(CigaretteSEM.panel(), spec="t3", anchor="own-tax")
    X, y = design.X, design.y
    np.random.seed(seed)
    da = ScaleTranslation(V, std=float(np.std(X @ (V / np.linalg.norm(V)))))
    GX, G = da(X)
    return X, y, GX, G, da, design.Z[:, :1]


def fitted(names, X, y, GX, G, da, Z, rho):
    builders = MethodRegistry.build_methods(
        list(names),
        gamma=GAMMA,
        epsilon=EPS_TOL,
        epsilon_iv=2**-5,
        epsilon_iv_z=2**-6,
        gamma_z=2**-8,
        rho=rho,
        pad=False,
        **TOGGLES,
    )
    models = {}
    for name in names:
        model = builders[name]()
        fit_model(model=model, method_name=name, X=X, y=y, GX=GX, G=G, Z=Z, da=da)
        models[name] = model
    return models


def width(model, k, block="Z"):
    """Instrument columns of one block on a fitted IV ball: the jitter block adds
    k rows, and an absent block is 0 columns."""
    arr = model.Z_projector_R if block == "Z" else model.T_projector_R
    return 0 if arr is None else arr.shape[0] - k


def synthetic(top):
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = {
        "PI": np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]),
        "DA+PI": np.array([[0.5, 0.5], [0.9, 0.9], [1.1, 1.1], [top, top]]),
    }
    return x, y


def axis_after(x, y, **kwargs):
    """Render with `savefig=False` and return (yscale, ylim, max drawn y, messages)."""
    plt.close("all")
    with captured() as lines:
        create_sweep_plot(x, y, xlabel="x", savefig=False, **kwargs)
    ax = plt.gca()
    drawn = max((max(line.get_ydata()) for line in ax.get_lines() if len(line.get_ydata())), default=np.nan)
    return ax.get_yscale(), tuple(float(v) for v in ax.get_ylim()), drawn, lines


# ------------------------------------------------------------------ legs


def leg_d(reference):
    print("(D) the shipped configuration does not move")
    for label, ok, detail in digest_leg.compare(REPO, reference):
        check(f"(D) {label}", ok, detail)


def leg_i():
    print("(i) the grammar, the validation and the display tables")
    check("(i) every bare name parses as (base, 'T,Z')", all(parse_method(n) == (n, "T,Z") for n in ALL_METHODS))
    for name, want in LEGAL.items():
        check(f"(i) {name!r} parses as {want}", parse_method(name) == want)
    for name in REJECTED:
        try:
            parse_method(name)
            check(f"(i) {name!r} is rejected", False, "parsed")
        except ValueError as error:
            check(f"(i) {name!r} is rejected naming it", name in str(error), str(error))
    check("(i) 'DA+PI+IV( Z )' is stored as 'DA+PI+IV(Z)'", spelled_method("DA+PI+IV( Z )") == "DA+PI+IV(Z)")
    check("(i) 'DA+PI+IV(T, Z)' is stored as 'DA+PI+IV(T,Z)'", spelled_method("DA+PI+IV(T, Z)") == "DA+PI+IV(T,Z)")
    check("(i) a bare name stays bare", spelled_method("DA+PI+IV") == "DA+PI+IV")

    with_z = dict(iv=["tax_s"])
    stored = resolve_dataset_block(
        "cigarettes",
        {**base_block("cigarettes"), **with_z, "methods": ["PI", "DA+PI+IV", "DA+PI+IV( Z )", "DA+IV(T, Z)"]},
    )["methods"]
    check(
        "(i) the block stores bare, base(Z) and base(T,Z)",
        stored == ["PI", "DA+PI+IV", "DA+PI+IV(Z)", "DA+IV(T,Z)"],
        f"{stored}",
    )
    message = rejection("cigarettes", methods=["DA+PI+IV", "DA+PI+IV(T,Z)"], **with_z) or ""
    check(
        "(i) [DA+PI+IV, DA+PI+IV(T,Z)] raises naming both (one (base, mode) pair)",
        "'DA+PI+IV'" in message and "'DA+PI+IV(T,Z)'" in message,
        message or "no error",
    )
    message = rejection("cigarettes", methods=["PI", "PI"], **with_z) or ""
    check("(i) [PI, PI] raises", "'PI'" in message, message or "no error")
    check(
        "(i) [DA+PI+IV, DA+PI+IV(Z)] resolves to two methods",
        resolve_dataset_block(
            "cigarettes", {**base_block("cigarettes"), **with_z, "methods": ["DA+PI+IV", "DA+PI+IV(Z)"]}
        )["methods"]
        == ["DA+PI+IV", "DA+PI+IV(Z)"],
    )
    message = rejection("cigarettes", methods=["PI+IV(Z)"], **with_z) or ""
    check("(i) PI+IV(Z) raises naming the method", "'PI+IV(Z)'" in message, message or "no error")
    message = rejection("cigarettes", methods=["DA+IV(Z)"], iv=[]) or ""
    check(
        "(i) DA+IV(Z) under iv: [] raises naming the method and iv",
        "'DA+IV(Z)'" in message and "iv = " in message,
        message,
    )
    check("(i) DA+PI+IV(Z) under iv: [] resolves", rejection("cigarettes", methods=["DA+PI+IV(Z)"], iv=[]) is None)
    check("(i) DA+PI+IV(Z) on the optical block resolves", rejection("optical_device", methods=["DA+PI+IV(Z)"]) is None)
    for spelled in ("DA+PI+IV(Z)", "DA+PI+IV(T,Z)"):
        message = rejection("do_mnist", methods=["PI", spelled]) or ""
        check(f"(i) {spelled} on the do_mnist block raises naming it", repr(spelled) in message, message or "no error")
    flow = yaml.safe_load("[PI, DA+PI+IV(T,Z)]")
    message = rejection("cigarettes", methods=flow, **with_z) or ""
    check(
        "(i) an unquoted (T,Z) in a YAML flow list raises naming 'DA+PI+IV(T' with the quoting hint",
        flow == ["PI", "DA+PI+IV(T", "Z)"] and "'DA+PI+IV(T'" in message and "quote" in message,
        message or "no error",
    )
    quoted = yaml.safe_load('[PI, "DA+PI+IV(T,Z)"]')
    check("(i) the quoted form resolves", rejection("cigarettes", methods=quoted, **with_z) is None)

    derived = {m for m in ALL_METHODS if "DA+" in m and m.endswith("+IV")}
    check("(i) IV_MODE_METHODS is the derived set", set(IV_MODE_METHODS) == derived, f"{IV_MODE_METHODS}")
    for name in SPELLED:
        present = name in TEX_MAPPER and name in COLOR_MAP and name in ALPHA_MAP
        check(f"(i) {name} in TEX_MAPPER, COLOR_MAP, ALPHA_MAP", present)
    for base in IV_MODE_METHODS:
        tz, z = f"{base}(T,Z)", f"{base}(Z)"
        same = (
            TEX_MAPPER[tz] == TEX_MAPPER[base] and COLOR_MAP[tz] == COLOR_MAP[base] and ALPHA_MAP[tz] == ALPHA_MAP[base]
        )
        check(f"(i) {tz} copies {base}'s display entries", same)
        # a real Z on the DA family: the family's hue (DA+PI's, PI&DA+PI's, DA+IV's own)
        family = {"DA+PI+IV": "DA+PI", "PI&DA+PI+IV": "PI&DA+PI"}.get(base, base)
        check(
            f"(i) {z} shares {family}'s hue and differs from {base} in TeX",
            COLOR_MAP[z] == COLOR_MAP[family] and TEX_MAPPER[z] != TEX_MAPPER[base],
        )
    check("(i) DA+IV(Z) is a point estimate", "DA+IV(Z)" in POINT_ESTIMATES)
    order = ["PI&DA+PI+IV(Z)", "PI", "DA+PI+IV(T,Z)", "DA+PI+IV", "DA+IV(Z)", "ATE"]
    built = MethodRegistry.build_methods(
        order, gamma=GAMMA, epsilon=EPS_TOL, epsilon_iv=EPS_TOL, gamma_z=2**-8, **TOGGLES
    )
    check("(i) build_methods returns the requested spellings in order", list(built) == order, f"{list(built)}")


def leg_ii(seed):
    print("(ii) the three modes on fitted attributes")
    X, y, GX, G, da, Z = draw(seed)
    k = X.shape[1]
    rho = float(rho_hat(X, GX, y, intercept=True))
    print(f"      rho_hat of the draw {rho:.6f}")
    models = fitted(MODE_NAMES, X, y, GX, G, da, Z, rho)
    # one width per BLOCK now: the (Z) spellings carry no T block at all
    for name, z_want, t_want in (
        ("PI+IV", 1, 0),
        ("DA+PI+IV", 1, 1),
        ("DA+PI+IV(T,Z)", 1, 1),
        ("DA+PI+IV(Z)", 1, 0),
    ):
        got = width(models[name], k), width(models[name], k, block="T")
        check(f"(ii) {name} blocks Z {z_want} / T {t_want}", got == (z_want, t_want), f"{got}")
    for name, t_want in (("PI&DA+PI+IV", 1), ("PI&DA+PI+IV(Z)", 0)):
        model = models[name]
        got = width(model.augmented, k), width(model.augmented, k, block="T")
        base = width(model.baseline, k), width(model.baseline, k, block="T")
        check(
            f"(ii) {name} DA branch Z 1 / T {t_want}, baseline Z 1 / T 0",
            got == (1, t_want) and base == (1, 0),
            f"DA {got}, baseline {base}",
        )
    # every IV ball carries BOTH radii; what tells the modes apart is which blocks
    # they were FITTED with, so the (Z) spellings have no T constraint at all
    for name in ("DA+PI+IV", "DA+PI+IV(T,Z)"):
        check(f"(ii) {name} T radius is 2^-5", models[name].t_bound == 2**-5, f"{models[name].t_bound!r}")
    check("(ii) DA+PI+IV(Z) carries no T constraint", not models["DA+PI+IV(Z)"]._has_t)
    check("(ii) PI&DA+PI+IV(Z) DA branch carries no T constraint", not models["PI&DA+PI+IV(Z)"].augmented._has_t)
    check("(ii) PI&DA+PI+IV DA branch T radius is 2^-5", models["PI&DA+PI+IV"].augmented.t_bound == 2**-5)
    # a DA+ method solves on the AUGMENTED design, so its declared radius carries
    # the DA-side allowance (SS2.6) and PI+IV's does not; the two are otherwise the
    # same number, which is what this pins
    for name in ("DA+PI+IV(Z)", "DA+PI+IV"):
        model = models[name]
        declared = np.sqrt(model.sigma_sq / model.rho * model.gamma_z)
        base = models["PI+IV"].epsilon_iv_z
        want = float(np.hypot(base, declared + model._z_allowance))
        without = float(np.hypot(base, declared))
        gap = abs(model.z_bound - want)
        check(f"(ii) {name} Z radius is PI+IV's plus the DA-side allowance", gap < 1e-12, f"{gap:.2e}")
        check("(ii) and the allowance is what separates them", model.z_bound > without, f"{model.z_bound:.6f}")
    check(
        "(ii) the T constraint is what tells bare DA+PI+IV from DA+PI+IV(Z)",
        models["DA+PI+IV"]._has_t and not models["DA+PI+IV(Z)"]._has_t,
    )

    queries = np.eye(k)
    for base in ("DA+PI+IV", "PI&DA+PI+IV"):
        bare = np.asarray(models[base].predict(queries, gamma=GAMMA), dtype=float)
        if base == "DA+PI+IV":
            spelled = np.asarray(models["DA+PI+IV(T,Z)"].predict(queries, gamma=GAMMA), dtype=float)
            gap = float(np.nanmax(np.abs(spelled - bare)))
            check(f"(ii) {base}(T,Z) predicts as {base} to 1e-9", gap < 1e-9, f"{gap:.2e}")
        z_mode = np.asarray(models[f"{base}(Z)"].predict(queries, gamma=GAMMA), dtype=float)
        gap = float(np.nanmax(np.abs(z_mode - bare)))
        check(f"(ii) {base}(Z) predicts differently from {base}", gap > 1e-6, f"{gap:.2e}")
    gap = float(np.abs(models["DA+IV(T,Z)"]._W - models["DA+IV"]._W).max())
    check("(ii) DA+IV(T,Z) coefficients equal DA+IV's to 1e-12", gap < 1e-12, f"{gap:.2e}")
    by_hand = TwoStageLeastSquaresIV(fit_intercept=True).fit(X=GX, y=y, Z=Z)
    gap = float(np.abs(models["DA+IV(Z)"]._W - by_hand._W).max())
    check("(ii) DA+IV(Z) coefficients equal a fresh 2SLS on (GX, y, Z) to 1e-12", gap < 1e-12, f"{gap:.2e}")
    gap = float(np.abs(models["DA+IV(Z)"]._W - models["DA+IV"]._W).max())
    check("(ii) DA+IV(Z) differs from DA+IV", gap > 1e-6, f"{gap:.2e}")

    empty = np.zeros((len(X), 0))
    reduced = fitted(("DA+PI", "DA+PI+IV(Z)", "PI&DA+PI", "PI&DA+PI+IV(Z)"), X, y, GX, G, da, empty, rho)
    check("(ii) DA+PI+IV(Z) under an empty Z reads no instrument", not reduced["DA+PI+IV(Z)"]._has_iv)
    for base in ("DA+PI", "PI&DA+PI"):
        a = np.asarray(reduced[base].predict(queries, gamma=GAMMA), dtype=float)
        b = np.asarray(reduced[f"{base}+IV(Z)"].predict(queries, gamma=GAMMA), dtype=float)
        gap = float(np.nanmax(np.abs(a - b)))
        check(f"(ii) {base}+IV(Z) under an empty Z is {base} to exactly 0.0", gap == 0.0, f"{gap:.2e}")
    builders = MethodRegistry.build_methods(["DA+IV(Z)"], gamma=GAMMA, epsilon=EPS_TOL, epsilon_iv=EPS_TOL, **TOGGLES)
    try:
        fit_model(model=builders["DA+IV(Z)"](), method_name="DA+IV(Z)", X=X, y=y, GX=GX, G=G, Z=empty)
        check("(ii) fit_model on DA+IV(Z) with an empty Z raises", False, "fitted")
    except ValueError as error:
        check("(ii) fit_model on DA+IV(Z) with an empty Z raises", "instrument" in str(error), str(error))


def leg_iii():
    print("(iii) the rollback")
    check(
        "(iii) NORMALIZED_SWEEP_SUFFIXES is width and worst_error",
        NORMALIZED_SWEEP_SUFFIXES == ("_width", "_worst_error"),
    )
    x, y = synthetic(1.3)
    with captured() as lines:
        same, baseline = normalize_sweep(y, "gamma_approx_error")
    warned = [line for line in lines if "gamma_approx_error" in line]
    check("(iii) normalize_sweep leaves an _approx_error input untouched", same is y and baseline is None)
    check("(iii) with one warning", len(warned) == 1, f"{lines}")
    for suffix in ("_approx_error", "_coverage"):
        try:
            validate_plot_keys("x", {f"gamma{suffix}": {"normalize": True}}, plot_keys_for)
            check(f"(iii) validate_plot_keys raises on normalize under {suffix}", False, "accepted")
        except ValueError as error:
            check(f"(iii) validate_plot_keys raises on normalize under {suffix}", "normalize" in str(error))
    for suffix in ("_width", "_worst_error"):
        try:
            validate_plot_keys("x", {f"gamma{suffix}": {"normalize": True}}, plot_keys_for)
            check(f"(iii) validate_plot_keys accepts normalize under {suffix}", True)
        except ValueError as error:
            check(f"(iii) validate_plot_keys accepts normalize under {suffix}", False, str(error))
    scale, limits, _, lines = axis_after(
        x, y, ylabel=METRIC_SPECS["approx_error"].ylabel, fname="gamma_approx_error", normalize=True, yscale="asinh"
    )
    label = plt.gca().get_ylabel()
    check("(iii) an approx-error figure under the toggle keeps a label without the baseline", " / " not in label, label)
    check("(iii) and its asinh axis", scale == "asinh", scale)
    check("(iii) the plotter raised nothing", not any("Failed to plot" in line for line in lines), f"{lines}")
    plt.close("all")


def leg_iv():
    print("(iv) the clamp on rendered axes")
    x = np.array([1.0, 2.0, 3.0, 4.0])
    coverage = {
        "PI": np.array([[0.005, 0.006], [0.1, 0.1], [0.5, 0.5], [1.0, 1.0]]),
        "DA+PI": np.array([[0.01, 0.01], [0.2, 0.2], [0.6, 0.6], [1.0, 1.0]]),
    }
    saved = PLOT_CONFIGS.get("simulation")
    PLOT_CONFIGS["simulation"] = {**(saved or {}), "gamma_coverage": {"yscale": "log", "ylim": (0.5, 2.0)}}
    try:
        scale, limits, _, lines = axis_after(x, coverage, fname="gamma_coverage", ylabel="coverage rate")
    finally:
        if saved is None:
            del PLOT_CONFIGS["simulation"]
        else:
            PLOT_CONFIGS["simulation"] = saved
    check("(iv) coverage under a log config reads linear", scale == "linear", scale)
    check(f"(iv) and exactly {CLAMP_YLIM}", limits == CLAMP_YLIM, f"{limits}")
    ignored = [line for line in lines if "ignored, the axis is clamped" in line]
    check(
        "(iv) with two warnings, yscale and ylim",
        len(ignored) == 2 and any("yscale" in m for m in ignored) and any("ylim" in m for m in ignored),
        f"{lines}",
    )
    scale, limits, _, lines = axis_after(x, coverage, fname="gamma_coverage", ylabel="coverage rate")
    check(
        "(iv) a coverage series spanning 0.005 to 1 (promoted to log) reads linear on the clamp",
        scale == "linear" and limits == CLAMP_YLIM,
        f"{scale} {limits}",
    )
    check("(iv) with no warning", not lines, f"{lines}")

    x, y = synthetic(1.3)
    scale, limits, drawn, lines = axis_after(x, y, fname="gamma_width", ylabel="average interval width", normalize=True)
    check(
        "(iv) a normalised width figure reads linear on the clamp",
        scale == "linear" and limits == CLAMP_YLIM,
        f"{scale} {limits}",
    )
    check("(iv) the DA+PI line is drawn to 1.3, the frame clips it", np.isclose(drawn, 1.3), f"{drawn!r}")
    scale, limits, _, lines = axis_after(x, y, fname="gamma_width", ylabel="average interval width", normalize=False)
    check("(iv) the same figure un-normalised is not clamped", limits != CLAMP_YLIM, f"{limits}")
    scale, limits, _, lines = axis_after(
        x,
        {"DA+PI": y["DA+PI"], "PI+INV": y["PI"]},
        fname="gamma_worst_error",
        ylabel="e",
        normalize=True,
        yscale="asinh",
    )
    check(
        "(iv) a normalised worst-error figure with no baseline is not clamped",
        scale == "asinh" and limits != CLAMP_YLIM,
        f"{scale} {limits}",
    )
    check("(iv) and warns once about the baseline", sum("needs one of" in line for line in lines) == 1, f"{lines}")
    scale, limits, _, lines = axis_after(x, y, fname="gamma_approx_error", ylabel="e", normalize=True, yscale="asinh")
    check(
        "(iv) an approx-error figure under the toggle is not clamped",
        scale == "asinh" and limits != CLAMP_YLIM,
        f"{scale} {limits}",
    )
    # the (Z) sibling is told apart by its dash pattern, in the base's hue
    plt.close("all")
    with captured():
        create_sweep_plot(
            x, {"DA+PI+IV": y["PI"], "DA+PI+IV(Z)": y["DA+PI"]}, xlabel="x", fname="gamma_width", savefig=False
        )
    drawn = {line.get_label(): line for line in plt.gca().get_lines()}
    base, sibling = drawn.get(TEX_MAPPER["DA+PI+IV"]), drawn.get(TEX_MAPPER["DA+PI+IV(Z)"])
    check("(iv) a sweep draws DA+PI+IV and DA+PI+IV(Z) as two lines", base is not None and sibling is not None)
    if base is not None and sibling is not None:
        pattern = getattr(sibling, "_unscaled_dash_pattern", None)
        check(
            "(iv) the (Z) line carries INSTRUMENT_Z_STYLE",
            sibling.get_linestyle() != "-" and pattern == INSTRUMENT_Z_STYLE,
            f"{sibling.get_linestyle()} {pattern}",
        )
        check("(iv) the base line is solid", base.get_linestyle() == "-", base.get_linestyle())
        # the (Z) sibling is PI+IV on the DA'd data: DA+PI's hue, not DA+PI+IV's
        palette = sns.color_palette()
        check(
            "(iv) the (Z) line in DA+PI's hue, the base in DA+PI+IV's",
            tuple(sibling.get_color()) == tuple(palette[COLOR_MAP["DA+PI"]])
            and tuple(base.get_color()) == tuple(palette[COLOR_MAP["DA+PI+IV"]]),
            f"{base.get_color()} {sibling.get_color()}",
        )
    plt.close("all")


def leg_v():
    print("(v) both recipes resolve and run")
    for name in ("cigarettes", "simulation"):
        methods = reduced_block(name)["methods"]
        check(
            f"(v) the {name} recipe lists DA+PI+IV and DA+PI+IV(Z) with no stored duplicate",
            "DA+PI+IV" in methods and "DA+PI+IV(Z)" in methods and len(set(methods)) == len(methods),
            f"{methods}",
        )

    folder = run_reduced("cigarettes", query=True, sweep=True)
    x = load(folder, SUBDIR_SWEEP, "gamma_values.pkl")
    results = load(folder, SUBDIR_SWEEP, "gamma_results.pkl")
    statuses = load(folder, SUBDIR_SWEEP, "gamma_statuses.pkl")
    check("(v) cigarettes gamma_results.pkl carries both keys", "DA+PI+IV" in results and "DA+PI+IV(Z)" in results)
    ok = {name: ok_steps(statuses, name) for name in ("PI+IV", "DA+PI+IV", "DA+PI+IV(Z)", "DA+PI")}
    widths = {name: results[name]["interval_width"][:, 0] for name in ok}
    z_ok = ok["DA+PI+IV(Z)"]
    print(
        "      RECORDED cigarettes gamma sweep OK steps of "
        + ", ".join(f"{name} {int(ok[name].sum())}/{len(x)}" for name in ("PI+IV", "DA+PI+IV", "DA+PI+IV(Z)"))
        + f" on the ratio grid {np.round(x, 4).tolist()}"
    )
    print(
        "      RECORDED (Z) widths "
        + ", ".join(f"{w:.4f}" for w in widths["DA+PI+IV(Z)"])
        + "; DA+PI "
        + ", ".join(f"{w:.4f}" for w in widths["DA+PI"])
    )
    check("(v) the ratio-1 step is OK for DA+PI+IV(Z)", bool(z_ok[-1]), f"{statuses['DA+PI+IV(Z)'][-1, 0]}")
    check("(v) the (Z) widths are finite at every OK step", bool(np.all(np.isfinite(widths["DA+PI+IV(Z)"][z_ok]))))
    nested = np.all(widths["DA+PI+IV(Z)"][z_ok] <= widths["DA+PI"][z_ok] + 1e-6)
    check("(v) and at most DA+PI's there (the same ball, one more constraint)", bool(nested))
    with open(os.path.join(folder, SUBDIR_QUERY, "coefficients.tex")) as handle:
        table = handle.read()
    check("(v) T1 carries the (Z) row under its TeX label", f"{TEX_MAPPER['DA+PI+IV(Z)']} & " in table)
    outcomes = load(folder, SUBDIR_QUERY, "beta_pn_gamma_outcomes.pkl")
    check(
        "(v) F1's outcomes are keyed exactly HEADLINE_METHODS",
        tuple(outcomes) == HEADLINE_METHODS,
        f"{tuple(outcomes)}",
    )
    band = np.asarray(outcomes["DA+PI+IV"], dtype=float)

    methods = [m if m != "DA+PI+IV" else "DA+PI+IV(T,Z)" for m in recipe("cigarettes")["methods"]]
    folder = run_reduced("cigarettes", query=True, sweep=False, methods=methods)
    respelled = load(folder, SUBDIR_QUERY, "beta_pn_gamma_outcomes.pkl")
    check(
        "(v) with DA+PI+IV respelled (T,Z) F1's outcomes are still keyed HEADLINE_METHODS",
        tuple(respelled) == HEADLINE_METHODS,
        f"{tuple(respelled)}",
    )
    if "DA+PI+IV" in respelled:
        gap = float(np.nanmax(np.abs(np.asarray(respelled["DA+PI+IV"], dtype=float) - band)))
        check("(v) and its DA band equals the bare run's to 1e-9", gap < 1e-9, f"{gap:.2e}")
    else:
        check("(v) and its DA band equals the bare run's to 1e-9", False, "no DA band")

    # the (Z) entry sits after the default in the recipe: a headline lookup that
    # collapsed every mode onto its base would draw the (Z) band as the default
    methods = [m for m in recipe("cigarettes")["methods"] if m != "DA+PI+IV(Z)"]
    folder = run_reduced("cigarettes", query=True, sweep=False, methods=methods)
    without = load(folder, SUBDIR_QUERY, "beta_pn_gamma_outcomes.pkl")
    check(
        "(v) with DA+PI+IV(Z) removed F1's outcomes are keyed HEADLINE_METHODS less that entry",
        tuple(without) == tuple(m for m in HEADLINE_METHODS if m != "DA+PI+IV(Z)"),
        f"{tuple(without)}",
    )
    if "DA+PI+IV" in without:
        gap = float(np.nanmax(np.abs(np.asarray(without["DA+PI+IV"], dtype=float) - band)))
        check("(v) and the DA band equals the run with (Z) present to 1e-9", gap < 1e-9, f"{gap:.2e}")
    else:
        check("(v) and the DA band equals the run with (Z) present to 1e-9", False, "no DA band")

    folder = run_reduced("simulation", query=False, sweep=True)
    x = load(folder, SUBDIR_SWEEP, "gamma_values.pkl")
    results = load(folder, SUBDIR_SWEEP, "gamma_results.pkl")
    statuses = load(folder, SUBDIR_SWEEP, "gamma_statuses.pkl")
    check("(v) simulation gamma_results.pkl carries both keys", "DA+PI+IV" in results and "DA+PI+IV(Z)" in results)
    ok = {name: ok_steps(statuses, name) for name in ("DA+PI+IV", "DA+PI+IV(Z)", "DA+PI")}
    widths = {name: results[name]["interval_width"][:, 0] for name in ok}
    z_ok = ok["DA+PI+IV(Z)"]
    print(
        "      RECORDED simulation gamma sweep OK steps of "
        + ", ".join(f"{name} {int(ok[name].sum())}/{len(x)}" for name in ("DA+PI+IV", "DA+PI+IV(Z)"))
        + f" on the ratio grid {np.round(x, 4).tolist()}"
    )
    print(
        "      RECORDED (Z) widths "
        + ", ".join(f"{w:.4f}" for w in widths["DA+PI+IV(Z)"])
        + "; DA+PI "
        + ", ".join(f"{w:.4f}" for w in widths["DA+PI"])
    )
    check(
        "(v) the simulation (Z) widths are finite at every OK step",
        bool(z_ok.any()) and bool(np.all(np.isfinite(widths["DA+PI+IV(Z)"][z_ok]))),
    )
    nested = np.all(widths["DA+PI+IV(Z)"][z_ok] <= widths["DA+PI"][z_ok] + 1e-6)
    check("(v) and at most DA+PI's there", bool(nested))


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
    print(f"tree: {constants_module.__file__}")
    legs = [
        ("(i)", leg_i),
        ("(ii)", partial(leg_ii, args.seed)),
        ("(iii)", leg_iii),
        ("(iv)", leg_iv),
        ("(v)", leg_v),
    ]
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
        print("A63 PASS")
    else:
        print(f"A63 FAIL: {FAIL}")
        sys.exit(1)
