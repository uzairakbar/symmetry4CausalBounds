r"""A80: method labels, colours and line styles by attributes.

Every method is a set of attributes (solver, da, inv, iv_set, and a left and a
right side on the intersections), and everything the figures and tables show of
it is a function of those attributes and of whether the experiment has a real Z
(a non-empty `iv`). One module owns all of it, `src/experiments/utils/constants.py`.
The label table below (LABEL_ROWS) holds one row per label: the spellings that
share it and its TeX with and without a real Z. No data and no GPU; seconds,
apart from (vi). Legs:

  (i)    attributes: every `ALL_METHODS`, `DOMNIST_ONLY_METHODS` and mode spelling
         parses to a literal (solver, da, inv, iv_set) and, on an intersection, a
         literal left side, whose iv_set is {Z} in EVERY IV mode (the baseline of
         `IntersectedInstrumentalVariablePartialR2` only ever sees Z). A direct
         `Side` of inv+da, ate+inv, ate+Z or T without da raises; `DA+PI+INV`,
         `PI+IV&DA+PI` and `FOO` parse with no side and the config rejects them
         through the did-you-mean list; `PI+IV(T)` keeps today's grammar error.
         Catches: a mode read on a non-DA +IV, an intersection's baseline losing
         its Z under (T), an invalid attribute set slipping past the config.
         Misses: what is drawn of the attributes -- (ii) and (iii).
  (ii)   the label table: every row, both columns, against TeX composed here from
         the repo's blocks. The two cells without a legal null-Z method raise in
         `resolve_dataset_block` (simulation `iv: 0`, optical). Catches: a Z-only
         token printed with a null Z, a tilde on a plain Z, parentheses on a side
         without a `+`. Misses: how a figure draws the label -- (iv), (v), (ix).
  (iii)  colours, alphas, styles and legend slots over every (method, has_z) on
         literal tables: the hue (black, blue, grey, red, green, and an
         intersection's purple or pink off its DA side); the six hex literals
         equal seaborn deep at their indices and `colour` equals
         `sns.color_palette("deep")[i]` as floats; pink leans magenta; the alpha
         per hue; every point estimate dashed and every interval solid, with and
         without a real Z; with a Z `legend_order` equals a frozen literal of the
         pre-change `PAIR_ORDER`, without one every method is member 0 of its
         group. Catches: a hue rule out of order, a palette index drifting, a
         `_FAMILIES` row swap, an interval drawn with any dash. Misses: whether
         the plotting code reads these functions -- (iv), (v), (ix).
  (iv)   uniqueness: (a) within one experiment `label` is injective on the label
         rows, over every recipe block, the digest blocks, the default lists and
         exhaustively at each has_z; (b) across experiments one label is one
         (colour, alpha, dash). A mutation of each, patched into the module, proves
         it can fail. Two methods on one label in one block (PI and PI+IV at
         `iv: 0`, the (T) and bare spellings) resolve with one WARNING and no
         raise, and `create_sweep_plot` draws PI and PI+IV as one legend entry
         without a Z and two with one. Catches: two rows on one label, one label drawn two ways, a config
         that raises on a legal collision. Misses: a figure that draws a label
         without `method_style` -- the completeness greps in (viii).
  (v)    the aggregate merge: the `z_counterpart` pairs whose render differs are
         exactly the seven cross-dataset pairs (SEVEN), the two (T) -> (T,Z)
         extras render identically, a null-Z column in a merged grid draws each
         method as its Z counterpart, every multi-column recipe pairs each null-Z
         method with a method of every Z column, and validityFig9's columns fold
         to 6 legend entries. Through `sweep_grid` on synthetic trees: a Z column
         with PI+IV beside a null-Z column with PI draws one entry, pi+iv, in one
         colour and dash; validityFig9's method sets give 6 entries; all-null
         columns do not merge, nor does a Z dataset present only as a blank column;
         a missing and a truncated `labels.json` each warn once and read as null Z.
         Catches: a pair missing from `z_counterpart`, a recipe whose null-Z column
         has no Z twin, a merge that leaves two legend entries per pair, a merge
         keyed on a column that draws nothing. Misses: the perf grid (a64 (vi)).
  (vi)   the writer: one query panel per dataset through the production path
         (`resolve_dataset_block`, the orchestrator, `.run`) in a temp cwd under
         ~/scratch, simulation `iv: 2` and optical, tiny samples, `im-ci` 0:
         `artifacts/<dataset>/labels.json` holds `{has_z, methods}` with the
         orchestrator's `has_z` (true on the simulation, false on optical) beside
         `query/`. `--skip-run` keeps the property check and skips the run.
         Catches: an orchestrator without the property, a run type that writes no
         `labels.json`, a `has_z` not the orchestrator's. Misses: the
         cigarettes and do-MNIST runs (their `has_z` is the property alone).
  (vii)  do-MNIST's PI+INV reads `pi+inv` whatever the `inv_recenter`, in `label`
         and in the `domnist_table` row under off, on and inv. Catches: a label
         that follows the centring. Misses: the table's other rows (a79 (v)).
  (viii) completeness: `src/` is clean of the removed tables (TEX_MAPPER,
         COLOR_MAP, ALPHA_MAP, REAL_Z_METHODS, PAIR_ORDER, POINT_ESTIMATES,
         INSTRUMENT_Z_STYLE) and helpers (`_get_method_color`, `_line_style`),
         of `\widetilde` and `\operatorname` outside constants.py and of a
         palette indexed by hand; no dict literal in plotting.py, aggregate.py,
         panels.py or cigarettes.py has two or more method-name keys (a display
         table under another name); no displayed string (the
         xlabel/ylabel/title/label keywords, the
         set_xlabel/set_ylabel/set_title/legend arguments and the rows
         `_write_coefficients` and `domnist_table` append; `%` lines skipped)
         types a method name; every drawing call in `src/` and `scripts/` passes
         `has_z=`. Each scan is shown to fire on a probe. Catches: a site
         reading a table or typing a label, a forgotten `has_z`. Misses: a name
         built at run time from pieces.
  (ix)   the column titles: on a validityFig9-shaped tree (the simulation with a
         real Z, optical and cigarettes without) through `sweep_grid` and
         `perf_grid`, the titles read "simulation", "optical device" and
         "cigarette demand", each null-Z one followed on its line by
         `NULL_Z_TITLE_SUFFIX`, `$(Z = \varnothing)$` at 3/4 size (the
         cigarettes key stays `cigarettes` in `DATASET_ORDER`); the titles are
         inside the figure and pairwise disjoint and the legend clears them; the
         merged figure saves to PDF under usetex ([SKIP] without latex); amssymb
         is in the preamble. An all-null tree and a tree whose only Z dataset is
         a blank column carry no suffix. Catches: the suffix on a Z column or an
         unmerged grid, a suffix that collides with its neighbour or the legend,
         a preamble without `\varnothing`. Misses: the single-dataset figures,
         which carry no dataset title.

A leg, or a part of one, that needs code from a later commit prints [SKIP] and is
counted in the summary line.

Usage:
    MPLBACKEND=Agg python scripts/a80_method_labels.py [--only LEG] [--skip-run]
"""

import argparse
import contextlib
import glob
import json
import os
import pickle
import shutil
import sys
import tempfile
from itertools import pairwise

import yaml
from loguru import logger

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import digest_leg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.colors import to_hex  # noqa: E402

import src.experiments.utils.constants as constants  # noqa: E402
from src.experiments.base import METRIC_FIELDS  # noqa: E402
from src.experiments.configs import (  # noqa: E402
    ALL_METHODS,
    DOMNIST_ONLY_METHODS,
    PARAM_SPECS,
    _check_instruments,
    resolve_dataset_block,
)
from src.experiments.utils.constants import (  # noqa: E402
    DA_ERM,
    DATASET_ORDER,
    DATASET_TITLES,
    DEEP_HEX,
    DEEP_INDEX,
    ERM,
    FS_LABEL,
    INV,
    IV,
    IV_MODE_METHODS,
    IV_MODES,
    NULL_Z_TITLE_SUFFIX,
    PARTIAL_IDENTIFICATION_STYLE,
    PI,
    POINT_ESTIMATE_STYLE,
    RC_PARAMS,
    SUBDIR_PERF,
    SUBDIR_SWEEP,
    Side,
    alpha,
    colour,
    hue,
    is_method,
    label,
    label_collisions,
    legend_order,
    line_style,
    method_style,
    parse_method,
    resolve,
    spelled_method,
    z_counterpart,
)

FAIL = []
SKIPPED = []
# the synthetic trees and runs, each removed when its leg ends
SCRATCH = os.path.expanduser("~/scratch/tmp/impl_labels/a80")
TREES = []
# every spelling the config accepts: the bare names, and each IV-mode base under its three modes
SPELLINGS = [
    spelling
    for base in ALL_METHODS + DOMNIST_ONLY_METHODS
    for spelling in ([base] + ([f"{base}({mode})" for mode in IV_MODES] if base in IV_MODE_METHODS else []))
]
# the two label-table cells the config rejects: no legal (method, False) pair
REJECTED_WITHOUT_Z = ("ERM+IV", "DA+ERM+IV(Z)")
LEGAL = [(m, hz) for m in SPELLINGS for hz in (True, False) if hz or m not in REJECTED_WITHOUT_Z]


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAIL.append(name)


def skip(tag, reason):
    """A leg that cannot run here: printed and counted in the summary, never a silent PASS."""
    print(f"  [SKIP] {tag} {reason}")
    SKIPPED.append(f"{tag} {reason}")


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
    """A block carrying its required keys and nothing else (as a56)."""
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


def recipes():
    """{recipe: {dataset: (methods, has_z)}} over every dataset block of `recipes/*.yaml`,
    each resolved as src/main.py resolves it."""
    out = {}
    for path in sorted(glob.glob(os.path.join(REPO, "recipes", "*.yaml"))):
        with open(path) as handle:
            config = yaml.safe_load(handle)
        defaults = config.pop("defaults", {}) or {}
        config.pop("hyperparameters", None)
        columns = {}
        for dataset, block in config.items():
            resolved = resolve_dataset_block(dataset, {**defaults, **block, "im-ci": 0})
            columns[dataset] = (resolved["methods"], _check_instruments(dataset, resolved))
        out[os.path.basename(path)[: -len(".yaml")]] = columns
    return out


# the literal attributes: (solver, da, inv, iv_set) of the method (the DA branch of an
# intersection), and the left side's (solver, da, inv, iv_set) or None
Z, T, TZ, NONE = frozenset("Z"), frozenset("T"), frozenset("TZ"), frozenset()
MODE_SET = {"": TZ, "(T,Z)": TZ, "(Z)": Z, "(T)": T}
ATTRIBUTES = {
    "ATE": (("ate", False, False, NONE), None),
    "ERM": (("erm", False, False, NONE), None),
    "ERM+IV": (("erm", False, False, Z), None),
    "DA+ERM": (("erm", True, False, NONE), None),
    "PI+INV": (("pi", False, True, NONE), None),
    "PI": (("pi", False, False, NONE), None),
    "PI+IV": (("pi", False, False, Z), None),
    "PI+INV+IV": (("pi", False, True, Z), None),
    "DA+PI": (("pi", True, False, NONE), None),
    "PI&DA+PI": (("pi", True, False, NONE), ("pi", False, False, NONE)),
    "ERM+INV": (("erm", False, True, NONE), None),
}
for _suffix, _blocks in MODE_SET.items():
    ATTRIBUTES[f"DA+ERM+IV{_suffix}"] = (("erm", True, False, _blocks), None)
    ATTRIBUTES[f"DA+PI+IV{_suffix}"] = (("pi", True, False, _blocks), None)
    # the baseline sees Z in every mode, (T) included
    ATTRIBUTES[f"PI&DA+PI+IV{_suffix}"] = (("pi", True, False, _blocks), ("pi", False, False, Z))

# the label table, composed from the repo's TeX blocks: (spellings, has_z = true, has_z = false);
# None is "rejected by config"
ATE_TEX = r"\operatorname{ate}"
T_PI, T_IV = rf"\widetilde{{{PI}}}", rf"\widetilde{{{IV}}}"
LABEL_ROWS = [
    (["ATE"], ATE_TEX, ATE_TEX),
    (["ERM"], ERM, ERM),
    (["ERM+IV"], f"{ERM}+{IV}", None),
    (["ERM+INV"], f"{ERM}+{INV}", f"{ERM}+{INV}"),
    (["DA+ERM"], DA_ERM, DA_ERM),
    (["DA+ERM+IV(Z)"], f"{DA_ERM}+{IV}", None),
    (["DA+ERM+IV", "DA+ERM+IV(T,Z)", "DA+ERM+IV(T)"], f"{DA_ERM}+{T_IV}", f"{DA_ERM}+{T_IV}"),
    (["PI"], PI, PI),
    (["PI+IV"], f"{PI}+{IV}", PI),
    (["PI+INV"], f"{PI}+{INV}", f"{PI}+{INV}"),
    (["PI+INV+IV"], f"{PI}+{INV}+{IV}", f"{PI}+{INV}"),
    (["DA+PI"], T_PI, T_PI),
    (["DA+PI+IV(Z)"], f"{T_PI}+{IV}", T_PI),
    (["DA+PI+IV", "DA+PI+IV(T,Z)", "DA+PI+IV(T)"], f"{T_PI}+{T_IV}", f"{T_PI}+{T_IV}"),
    (["PI&DA+PI"], rf"{PI}\cap{T_PI}", rf"{PI}\cap{T_PI}"),
    (["PI&DA+PI+IV(Z)"], rf"({PI}+{IV})\cap({T_PI}+{IV})", rf"{PI}\cap{T_PI}"),
    (
        ["PI&DA+PI+IV", "PI&DA+PI+IV(T,Z)", "PI&DA+PI+IV(T)"],
        rf"({PI}+{IV})\cap({T_PI}+{T_IV})",
        rf"{PI}\cap({T_PI}+{T_IV})",
    ),
]

# the hue of each spelling on literals, which no has_z changes
HUES = {
    "ATE": "black",
    "ERM": "blue",
    "ERM+IV": "blue",
    "PI": "blue",
    "PI+IV": "blue",
    "PI+INV": "grey",
    "PI+INV+IV": "grey",
    "ERM+INV": "grey",
    "DA+ERM": "red",
    "DA+PI": "red",
    "DA+ERM+IV(Z)": "red",
    "DA+PI+IV(Z)": "red",
    "PI&DA+PI": "purple",
    "PI&DA+PI+IV(Z)": "purple",
}
for _suffix in ("", "(T,Z)", "(T)"):
    HUES[f"DA+ERM+IV{_suffix}"] = HUES[f"DA+PI+IV{_suffix}"] = "green"
    HUES[f"PI&DA+PI+IV{_suffix}"] = "pink"
# the pre-change band alpha of an interval, by hue (ALPHA_MAP); a point estimate is 1.0
ALPHAS = {"blue": 0.2, "red": 0.2, "grey": 0.8, "green": 0.4, "purple": 0.4, "pink": 0.4}
POINTS = {"ATE", "ERM", "ERM+IV", "DA+ERM", "ERM+INV"} | {f"DA+ERM+IV{s}" for s in MODE_SET}
# the pre-change PAIR_ORDER, frozen: (legend group, member) of every spelling with a real Z
LEGEND_ORDER = {
    "ATE": (0, 0),
    "PI+INV": (1, 0),
    "PI+INV+IV": (1, 1),
    "PI": (2, 0),
    "PI+IV": (2, 1),
    "ERM": (3, 0),
    "ERM+IV": (3, 1),
    "DA+PI": (4, 0),
    "DA+PI+IV(Z)": (4, 1),
    "DA+ERM": (5, 0),
    "DA+ERM+IV(Z)": (5, 1),
    "DA+PI+IV": (6, 0),
    "DA+PI+IV(T)": (6, 0),
    "DA+PI+IV(T,Z)": (6, 0),
    "DA+ERM+IV": (7, 0),
    "DA+ERM+IV(T)": (7, 0),
    "DA+ERM+IV(T,Z)": (7, 0),
    "PI&DA+PI": (8, 0),
    "PI&DA+PI+IV(Z)": (8, 1),
    "PI&DA+PI+IV": (9, 0),
    "PI&DA+PI+IV(T)": (9, 0),
    "PI&DA+PI+IV(T,Z)": (9, 0),
    "ERM+INV": (10, 0),
}
# the cross-dataset pairs of the aggregate grids, literal: (null-Z method, its Z counterpart)
SEVEN = {
    ("PI", "PI+IV"),
    ("PI+INV", "PI+INV+IV"),
    ("DA+PI", "DA+PI+IV(Z)"),
    ("PI&DA+PI", "PI&DA+PI+IV(Z)"),
    ("PI&DA+PI+IV(T)", "PI&DA+PI+IV(T,Z)"),
    ("ERM", "ERM+IV"),
    ("DA+ERM", "DA+ERM+IV(Z)"),
}
# the two further pairs `z_counterpart` gives, which render identically on both sides
EXTRAS = {("DA+PI+IV(T)", "DA+PI+IV(T,Z)"), ("DA+ERM+IV(T)", "DA+ERM+IV(T,Z)")}


def as_tuple(side):
    return None if side is None else (side.solver, side.da, side.inv, side.iv_set)


# ------------------------------------------------------------------ legs


def leg_i():
    print("(i) the attribute parse")
    check("(i) the literal table covers every spelling", set(ATTRIBUTES) == set(SPELLINGS), f"{len(SPELLINGS)}")
    for name in SPELLINGS:
        parsed = parse_method(name)
        want_side, want_left = ATTRIBUTES[name]
        got = (as_tuple(parsed.side), as_tuple(parsed.left))
        check(f"(i) {name} -> {want_side}, left {want_left}", got == (want_side, want_left), f"{got}")
        if want_left is not None:
            check(f"(i) {name}: right is the DA branch", parsed.right == parsed.side and parsed.right.da)
        else:
            check(f"(i) {name}: no right side", parsed.right is None)
    check("(i) still a (base, mode) pair", parse_method("DA+PI+IV(Z)") == ("DA+PI+IV", "Z"))
    base, mode = parse_method("PI&DA+PI+IV")
    check("(i) unpacks as before", (base, mode) == ("PI&DA+PI+IV", "T,Z"))
    invalid = {
        "inv with da": dict(solver="pi", da=True, inv=True, iv_set=NONE),
        "ate with inv": dict(solver="ate", da=False, inv=True, iv_set=NONE),
        "ate with Z": dict(solver="ate", da=False, inv=False, iv_set=Z),
        "T without da": dict(solver="pi", da=False, inv=False, iv_set=T),
        "an unknown solver": dict(solver="iv", da=False, inv=False, iv_set=NONE),
    }
    for why, kwargs in invalid.items():
        try:
            Side(**kwargs)
            raised = False
        except ValueError:
            raised = True
        check(f"(i) Side with {why} raises", raised)
    check("(i) ERM+INV (inv on erm) is legal", Side("erm", False, True, NONE).inv)
    for name in ("DA+PI+INV", "FOO", "ATE+IV", "PI+IV+INV", "DA+ERM+INV", "PI+IV&DA+PI"):
        check(f"(i) {name}: no side, not a method", parse_method(name).side is None and not is_method(name))
        message = rejection("simulation", methods=[name]) or ""
        check(f"(i) {name}: the config rejects it with the did-you-mean list", "Unknown key(s)" in message, message)
    try:
        parse_method("PI+IV(T)")
        message = ""
    except ValueError as error:
        message = str(error)
    check("(i) PI+IV(T) keeps its grammar error", "legal on" in message, message)
    try:
        resolve("FOO", True)
        raised = False
    except ValueError:
        raised = True
    check("(i) resolve raises on an unknown name", raised)
    null = resolve("PI&DA+PI+IV(T)", False)
    check("(i) resolved without Z: the baseline loses its Z", null.left.iv_set == NONE and null.side.iv_set == T)


def leg_ii():
    print("(ii) the label table, every row")
    covered = set()
    for spellings, with_z, without_z in LABEL_ROWS:
        covered |= set(spellings)
        for name in spellings:
            got = label(name, True)
            check(f"(ii) {name}, has_z true", got == f"${with_z}$", got)
            if without_z is not None:
                got = label(name, False)
                check(f"(ii) {name}, has_z false", got == f"${without_z}$", got)
                continue
            for dataset, extra in (("simulation", dict(iv=0)), ("optical_device", {})):
                message = rejection(dataset, methods=[name], **extra) or ""
                check(f"(ii) {name} without Z is rejected by the {dataset} config", repr(name) in message, message)
    check(
        "(ii) the label table covers every spelling", covered == set(SPELLINGS), f"{sorted(set(SPELLINGS) - covered)}"
    )


def leg_iii():
    print("(iii) colours, alphas and line styles")
    deep = sns.color_palette("deep")
    for name, index in DEEP_INDEX.items():
        check(f"(iii) {name} is seaborn deep {index}", DEEP_HEX[name] == to_hex(deep[index]), DEEP_HEX[name])
    pink = deep[DEEP_INDEX["pink"]]
    check("(iii) pink leans magenta", pink[0] - pink[1] > 40 / 255 and pink[2] - pink[1] > 40 / 255, f"{pink}")
    check("(iii) the hue table covers every spelling", set(HUES) == set(SPELLINGS))
    for name, has_z in LEGAL:
        want = HUES[name]
        got = hue(name, has_z)
        check(f"(iii) {name} ({has_z}) is {want}", got == want, got)
        rgb = colour(name, has_z)
        ok = rgb == "black" if want == "black" else rgb == tuple(deep[DEEP_INDEX[want]])
        check(f"(iii) {name} ({has_z}) colour is deep's float tuple", ok, f"{rgb}")
        want_alpha = 1.0 if name in POINTS else ALPHAS[want]
        check(f"(iii) {name} ({has_z}) alpha {want_alpha}", alpha(name, has_z) == want_alpha)
        style = line_style(name, has_z)
        if name in POINTS:
            check(f"(iii) {name} ({has_z}) is dashed", style == POINT_ESTIMATE_STYLE, f"{style}")
        else:
            check(f"(iii) {name} ({has_z}) is solid", style == PARTIAL_IDENTIFICATION_STYLE, f"{style}")
    check("(iii) ATE is black", colour("ATE", True) == colour("ATE", False) == "black")
    check("(iii) the legend-order table covers every spelling", set(LEGEND_ORDER) == set(SPELLINGS))
    for name, has_z in LEGAL:
        order = legend_order(name, has_z)
        if has_z:
            check(f"(iii) {name} with Z keeps its legend slot", order == LEGEND_ORDER[name], f"{order}")
        else:
            want = (LEGEND_ORDER[name][0], 0)  # a Z-only method's group is its no-IV twin's
            check(f"(iii) {name} without Z is member 0 of its group", order == want, f"{order} vs {want}")


def within(columns):
    """The label collisions (different label rows, one label) of every column."""
    return {where: label_collisions(methods, has_z) for where, (methods, has_z) in columns.items()}


def across(pairs):
    """{label: renders} where one label has more than one (colour, alpha, dash). Read
    through the module, so a patch on it is what this sees."""
    renders = {}
    for name, has_z in pairs:
        text = constants.label(name, has_z)
        render = (constants.colour(name, has_z), constants.alpha(name, has_z), constants.line_style(name, has_z))
        renders.setdefault(text, set()).add(render)
    return {text: seen for text, seen in renders.items() if len(seen) > 1}


def leg_iv():
    print("(iv) uniqueness")
    columns = {}
    for recipe, blocks in recipes().items():
        for dataset, entry in blocks.items():
            columns[f"{recipe}:{dataset}"] = entry
    for dataset, block in digest_leg.BLOCKS.items():
        resolved = resolve_dataset_block(dataset, dict(block))
        columns[f"digest_leg:{dataset}"] = (resolved["methods"], _check_instruments(dataset, resolved))
    for dataset, extra in (("simulation", dict(iv=4)), ("cigarettes", dict(iv=["tax_s"]))):
        for tag, instruments in (("on", extra), ("off", {})):
            resolved = resolve_dataset_block(dataset, {**base_block(dataset), **instruments})
            columns[f"default:{dataset}:iv {tag}"] = (resolved["methods"], _check_instruments(dataset, resolved))
    collisions = {where: pairs for where, pairs in within(columns).items() if pairs}
    check(f"(iv) (a) no label collision in {len(columns)} blocks", not collisions, f"{collisions}")
    for has_z in (True, False):
        names = [m for m, hz in LEGAL if hz == has_z]
        pairs = label_collisions(names, has_z)
        check(
            f"(iv) (a) label is injective on the label rows over every spelling, has_z {has_z}", not pairs, f"{pairs}"
        )
    clash = across(LEGAL)
    check("(iv) (b) one label is one render over every legal (method, has_z)", not clash, f"{clash}")

    # the checks can fail: a colour off on one (method, has_z), and two label rows on one label
    original_colour = constants.colour
    constants.colour = lambda m, hz: "black" if (m, bool(hz)) == ("PI+IV", False) else original_colour(m, hz)
    constants.method_style.cache_clear()
    try:
        fired = bool(across(LEGAL))
    finally:
        constants.colour = original_colour
        constants.method_style.cache_clear()
    check("(iv) (b) mutation: one colour off on PI+IV without Z fires", fired)
    original = constants.label
    constants.label = lambda m, hz: original("PI", hz) if m == "PI+INV" else original(m, hz)
    try:
        fired = bool(label_collisions(["PI", "PI+INV"], True))
    finally:
        constants.label = original
    check("(iv) (a) mutation: PI+INV labelled as PI fires", fired)
    check("(iv) (a) one row, one label: PI and PI+IV without Z", not label_collisions(["PI", "PI+IV"], False))
    for methods, iv, want in ((["PI", "PI+IV"], 0, 1), (["PI", "PI+IV"], 2, 0), (["DA+PI+IV", "DA+PI+IV(T)"], 0, 1)):
        with captured() as lines:
            resolved = rejection("simulation", methods=methods, iv=iv) is None
        shared = [line for line in lines if "one legend entry" in line]
        check(
            f"(iv) {methods} at iv = {iv} resolves with {want} shared-label WARNING",
            resolved and len(shared) == want,
            f"{lines}",
        )
    from src.experiments.utils.plotting import create_sweep_plot

    x = np.array([1.0, 2.0, 3.0])
    y = {"PI": np.full((3, 2), 1.0), "PI+IV": np.full((3, 2), 0.5)}
    for has_z, want in ((False, [label("PI", False)]), (True, [label("PI", True), label("PI+IV", True)])):
        plt.close("all")
        create_sweep_plot(x, y, xlabel="x", savefig=False, bootstrapped=False, has_z=has_z)
        legend = plt.gca().get_legend()
        texts = [t.get_text() for t in legend.get_texts()] if legend is not None else []
        check(f"(iv) create_sweep_plot of PI and PI+IV, has_z {has_z}: legend {len(want)}", texts == want, f"{texts}")
    plt.close("all")


def leg_v():
    print("(v) the aggregate merge")
    pairs = {(m, z_counterpart(m)) for m in SPELLINGS if z_counterpart(m) != m}
    check("(v) z_counterpart gives nine pairs: the seven and two extras", pairs == SEVEN | EXTRAS, f"{pairs}")
    differ = {(a, b) for a, b in pairs if method_style(a, False).signature != method_style(b, True).signature}
    check("(v) the pairs whose render differs are exactly the seven", differ == SEVEN, f"{differ}")
    for a, b in sorted(EXTRAS):
        check(f"(v) {a} -> {b} renders identically", method_style(a, False) == method_style(b, True))
    for name in SPELLINGS:
        counterpart = method_style(z_counterpart(name), True)
        check(
            f"(v) {name}, merged null Z, draws as its Z counterpart",
            method_style(name, False, merged=True) == counterpart,
        )
        check(
            f"(v) {name}, merged Z column, unchanged", method_style(name, True, merged=True) == method_style(name, True)
        )
    for name in ("ATE", "ERM+INV"):
        check(f"(v) {name} has no Z counterpart", z_counterpart(name) == name)

    multi = {recipe: blocks for recipe, blocks in recipes().items() if len(blocks) > 1}
    check("(v) validityFig9 and its siblings are multi-column recipes", "validityFig9" in multi, f"{sorted(multi)}")
    for recipe, blocks in multi.items():
        with_z = {d: [parse_method(m) for m in methods] for d, (methods, hz) in blocks.items() if hz}
        null = {d: methods for d, (methods, hz) in blocks.items() if not hz}
        if not with_z:
            continue
        found, missing = set(), []
        for dataset, methods in null.items():
            for name in methods:
                counterpart = z_counterpart(name)
                found.add((name, counterpart))
                missing += [(dataset, name, d) for d, got in with_z.items() if parse_method(counterpart) not in got]
        check(f"(v) {recipe}: each null-Z method's counterpart runs in every Z column", not missing, f"{missing}")
        paired = {(a, spelled_method(b)) for a, b in found if a != b}
        check(f"(v) {recipe}: its pairs are among the nine", paired <= SEVEN | EXTRAS, f"{paired - SEVEN - EXTRAS}")
        drawn = {(a, b) for a, b in paired if method_style(a, False).signature != method_style(b, True).signature}
        check(f"(v) {recipe}: its render-differing pairs are among the seven", drawn <= SEVEN, f"{drawn - SEVEN}")
    validity = recipes()["validityFig9"]
    merged = any(hz for _, hz in validity.values())
    entries = {method_style(m, hz, merged=merged).signature for methods, hz in validity.values() for m in methods}
    check("(v) validityFig9's three columns fold to 6 legend entries", len(entries) == 6, f"{len(entries)}")
    grid_merge()


# the production path at a toy scale: one query panel per dataset, a real Z on the simulation only
WRITER_TOGGLES = dict(recalibrate=True, pad=True, clipy=False, mean_match=True, n_jobs=-1, **{"im-ci": 0})
WRITER_BLOCKS = {
    "simulation": dict(
        seed=42,
        n_samples=256,
        kernel_dim=0,
        treatment_dim=8,
        iv=2,
        methods=["PI", "PI+IV", "DA+PI+IV(Z)"],
        augmentation="translate",
    ),
    "optical_device": dict(
        seed=42, n_samples=256, methods=["PI", "DA+PI", "DA+PI+IV(T)"], augmentation="rotation > hflip"
    ),
}


def write_column(root, dataset, methods, has_z, param="gamma", labels=True):
    """One dataset's sweep pkls on a 4-step grid (every metric, two experiments) and,
    unless `labels` is off, its `labels.json`; `labels` a str is written verbatim."""
    rng = np.random.default_rng(len(methods))
    folder = os.path.join(root, dataset, SUBDIR_SWEEP)
    os.makedirs(folder, exist_ok=True)
    grid = PARAM_SPECS[param].grid_fn(dataset, 4)
    record = {m: {key: 0.2 + 0.6 * rng.random((len(grid), 2)) for key in METRIC_FIELDS} for m in methods}
    for stem, obj in ((f"{param}_values", grid), (f"{param}_results", record)):
        with open(os.path.join(folder, f"{stem}.pkl"), "wb") as fh:
            pickle.dump(obj, fh)
    if labels is True:
        with open(os.path.join(root, dataset, "labels.json"), "w") as fh:
            json.dump({"has_z": has_z, "methods": list(methods)}, fh)
    elif isinstance(labels, str):
        with open(os.path.join(root, dataset, "labels.json"), "w") as fh:
            fh.write(labels)


def grid_legend(root, param="gamma"):
    """(legend texts, the figure, warnings) of `sweep_grid` over the tree's columns."""
    from src import aggregate

    with captured() as lines:
        fig = aggregate.sweep_grid(param, aggregate.columns(root), root)
    texts = [t.get_text() for t in fig.legends[0].get_texts()] if fig.legends else []
    return texts, fig, lines


def line_render(line):
    return tuple(matplotlib.colors.to_rgba(line.get_color())), getattr(line, "_unscaled_dash_pattern", None)


def scratch_tree(prefix):
    """A fresh directory under SCRATCH, removed when the leg ends (`__main__`)."""
    os.makedirs(SCRATCH, exist_ok=True)
    path = tempfile.mkdtemp(prefix=prefix, dir=SCRATCH)
    TREES.append(path)
    return path


def grid_merge():
    """The merge through `sweep_grid` on synthetic trees under ~/scratch."""

    def tree():
        return scratch_tree("grid_")

    root = tree()
    write_column(root, "simulation", ["PI+IV"], True)
    write_column(root, "optical_device", ["PI"], False)
    texts, fig, _ = grid_legend(root)
    check("(v) sim Z PI+IV beside optical null-Z PI: one entry pi+iv", texts == [label("PI+IV", True)], f"{texts}")
    renders = {line_render(ax.get_lines()[0]) for ax in fig.axes if ax.axison and ax.get_lines()}
    check("(v) both columns draw it in one colour and one dash", len(renders) == 1, f"{renders}")
    plt.close(fig)

    root = tree()
    for dataset, (methods, has_z) in recipes()["validityFig9"].items():
        write_column(root, dataset, methods, has_z)
    texts, fig, _ = grid_legend(root)
    check("(v) validityFig9's columns through sweep_grid: 6 entries", len(texts) == 6, f"{len(texts)}")
    plt.close(fig)

    root = tree()
    write_column(root, "simulation", ["DA+PI"], False)
    write_column(root, "optical_device", ["DA+PI"], False)
    texts, fig, _ = grid_legend(root)
    check("(v) all columns null Z: no merge", texts == [label("DA+PI", False)], f"{texts}")
    plt.close(fig)

    root = tree()
    write_column(root, "simulation", ["PI+IV"], True, param="omega")  # a Z column blank on gamma
    write_column(root, "optical_device", ["PI"], False)
    texts, fig, _ = grid_legend(root)
    check("(v) a Z dataset present only as a blank column: no merge", texts == [label("PI", False)], f"{texts}")
    plt.close(fig)

    for why, labels in (("missing", False), ("truncated", '{"has_')):
        root = tree()
        write_column(root, "simulation", ["DA+PI"], True, labels=labels)
        write_column(root, "optical_device", ["DA+PI"], False)
        texts, fig, lines = grid_legend(root)
        warned = [line for line in lines if "labels.json" in line and "null Z" in line]
        check(f"(v) a {why} labels.json warns once", len(warned) == 1, f"{lines}")
        check(f"(v) a {why} labels.json counts as null Z", texts == [label("DA+PI", False)], f"{texts}")
        plt.close(fig)


def leg_vi(skip_run):
    print("(vi) the labels.json writer")
    import shutil

    from munch import munchify

    from src.experiments.configs import parse_experiment_plan
    from src.experiments.utils import set_seed
    from src.main import ORCHESTRATORS

    missing = [name for name, cls in ORCHESTRATORS.items() if not isinstance(getattr(cls, "has_z", None), property)]
    check("(vi) every orchestrator carries a has_z property", not missing, f"{missing}")
    if skip_run:
        skip("(vi) run", "--skip-run")
        return
    root = os.path.expanduser("~/scratch/tmp/impl_labels/a80")
    os.makedirs(root, exist_ok=True)
    cwd = os.getcwd()
    scratch = tempfile.mkdtemp(prefix="writer_", dir=root)
    # data paths are relative: the run sees the repo's data/ from its own cwd
    os.symlink(os.path.join(REPO, "data"), os.path.join(scratch, "data"))
    os.chdir(scratch)
    try:
        for name, block in WRITER_BLOCKS.items():
            block = {**WRITER_TOGGLES, **block, "n_experiments": 1, "sweep_samples": 4}
            plan = parse_experiment_plan({"query": True})
            block = resolve_dataset_block(name, block)
            set_seed(block["seed"])
            orchestrator = ORCHESTRATORS[name](**block, hyperparameters=munchify({}))
            orchestrator.run(plan)
            with open(os.path.join("artifacts", name, "labels.json")) as fh:
                written = json.load(fh)
            want = {"has_z": name == "simulation", "methods": block["methods"]}
            check(f"(vi) {name}: labels.json holds has_z and the methods", written == want, f"{written}")
            check(f"(vi) {name}: has_z is the orchestrator's", written["has_z"] is orchestrator.has_z)
            check(
                f"(vi) {name}: labels.json sits beside query/, outside the hashed folders",
                os.path.isdir(os.path.join("artifacts", name, "query")),
            )
    finally:
        os.chdir(cwd)
        shutil.rmtree(scratch, ignore_errors=True)


def leg_vii():
    print("(vii) do-MNIST's PI+INV")
    want = f"${PI}+{INV}$"
    check("(vii) label('PI+INV', False) is pi+inv", label("PI+INV", False) == want, label("PI+INV", False))
    from a79_domnist_tint import _tree

    from src.aggregate import domnist_table

    with tempfile.TemporaryDirectory(dir=os.path.expanduser("~/scratch/tmp")) as scratch:
        run = _tree(scratch)
        path = os.path.join(scratch, "do_mnist", "query", "run.json")
        for recenter in ("off", "on", "inv"):
            with open(path, "w") as fh:
                json.dump({**run, "inv_recenter": recenter}, fh)
            tex = domnist_table(scratch)
            rows = tex.split("\\midrule\n")[1].split("\\bottomrule")[0].strip("\n").splitlines()
            names = [row.split(" & ")[0] for row in rows if not row.startswith(" & ")]
            check(f"(vii) the domnist_table PI+INV row under inv_recenter {recenter}", want in names, f"{names}")


# the static display tables and helpers every figure used to read, gone for the functions
REMOVED = ("TEX_MAPPER", "COLOR_MAP", "ALPHA_MAP", "REAL_Z_METHODS", "PAIR_ORDER", "POINT_ESTIMATES")
REMOVED += ("INSTRUMENT_Z_STYLE",)  # every interval is solid: no real-Z line style
REMOVED_HELPERS = ("_get_method_color", "_line_style")
# the calls that draw a method: each must say whether its experiment has a real Z
DRAWING = ("create_sweep_plot", "create_query_sweep_plot", "create_panel_plot", "create_digit_sweep_plot")
DRAWING += ("create_coverage_plot", "_draw_bands", "_draw_series", "mark_failed", "PanelBuilder")
# where a string is DISPLAYED: these keywords, these methods' arguments, and the rows
# the two table writers append
DISPLAY_KEYWORDS = ("xlabel", "ylabel", "title", "label")
DISPLAY_METHODS = ("set_xlabel", "set_ylabel", "set_title", "legend", "supxlabel", "supylabel", "suptitle")
TABLE_WRITERS = ("_write_coefficients", "domnist_table")
# the drawing modules, where a dict keyed by method names would be a display table again
DRAWING_MODULES = ("src/experiments/utils/plotting.py", "src/aggregate.py", "src/experiments/utils/panels.py")
DRAWING_MODULES += ("src/experiments/cigarettes.py",)


def method_tables(tree):
    """(line, keys) of every dict literal with two or more keys that are method names."""
    import ast

    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            keys = [k.value for k in node.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)]
            named = [k for k in keys if is_method(k)]
            if len(named) >= 2:
                out.append((node.lineno, named))
    return out


def read(path):
    with open(path) as fh:
        return fh.read()


def python_files(*roots):
    return sorted(p for root in roots for p in glob.glob(os.path.join(REPO, root, "**", "*.py"), recursive=True))


def strings_of(node):
    """The literal text of a str constant or an f-string's constant parts."""
    import ast

    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, ast.JoinedStr):
        return [part.value for part in node.values if isinstance(part, ast.Constant)]
    return [text for child in ast.iter_child_nodes(node) for text in strings_of(child)]


def displayed(tree):
    """(line, text) of every string the code displays (DISPLAY_* and TABLE_WRITERS)."""
    import ast

    out = []
    writers = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in TABLE_WRITERS]
    rows = {
        id(call)
        for writer in writers
        for call in ast.walk(writer)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == "append"
    }
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        parts = [kw.value for kw in node.keywords if kw.arg in DISPLAY_KEYWORDS]
        if isinstance(node.func, ast.Attribute) and node.func.attr in DISPLAY_METHODS or id(node) in rows:
            parts += list(node.args)
        out += [(node.lineno, text) for part in parts for text in strings_of(part)]
    return out


def leg_viii():
    print("(viii) completeness: every site reads through constants")
    import ast
    import re

    sources = {path: read(path) for path in python_files("src")}
    for name in REMOVED + REMOVED_HELPERS:
        hits = [
            f"{os.path.relpath(p, REPO)}:{i}"
            for p, text in sources.items()
            for i, line in enumerate(text.splitlines(), 1)
            if re.search(rf"\b{name}\b", line)
        ]
        check(f"(viii) {name} is gone from src/", not hits, f"{hits[:3]}")
    owner = os.path.join(REPO, "src", "experiments", "utils", "constants.py")
    tex = [
        f"{os.path.relpath(p, REPO)}:{i}"
        for p, text in sources.items()
        if p != owner
        for i, line in enumerate(text.splitlines(), 1)
        if "\\widetilde" in line or "\\operatorname" in line
    ]
    check("(viii) no \\widetilde or \\operatorname outside constants.py", not tex, f"{tex[:3]}")
    indexed = [
        f"{os.path.relpath(p, REPO)}:{i}"
        for p, text in sources.items()
        for i, line in enumerate(text.splitlines(), 1)
        if re.search(r"color_palette\([^)]*\)\s*\[", line.split("#")[0])
    ]
    check("(viii) no palette indexed by hand in src/", not indexed, f"{indexed[:3]}")

    # a method name typed into a displayed string would bypass `label`
    names = sorted({m for m in SPELLINGS} | {"DA+PI+IV(Z)"}, key=len, reverse=True)
    pattern = re.compile(r"(?<![\w\\+&])(" + "|".join(re.escape(n) for n in names) + r")(?![\w+&(])")
    plain, calls = [], []
    for path, text in sources.items():
        tree = ast.parse(text)
        for line, string in displayed(tree):
            if not string.lstrip().startswith("%") and pattern.search(string):
                plain.append(f"{os.path.relpath(path, REPO)}:{line} {string!r}")
    for path in python_files("src", "scripts"):
        tree = ast.parse(read(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)
            if func in DRAWING and not any(kw.arg == "has_z" for kw in node.keywords):
                calls.append(f"{os.path.relpath(path, REPO)}:{node.lineno} {func}")
    check("(viii) no plain-text method name in a displayed string of src/", not plain, f"{plain[:3]}")
    tables = [
        f"{module}:{line} {keys}"
        for module in DRAWING_MODULES
        for line, keys in method_tables(ast.parse(read(os.path.join(REPO, module))))
    ]
    check("(viii) no dict keyed by method names in the drawing modules", not tables, f"{tables[:3]}")
    check("(viii) every drawing call in src/ and scripts/ passes has_z", not calls, f"{calls[:5]}")
    # the scans can fail
    probe = ast.parse('ax.set_ylabel("width / PI width")\nplot(x, label=f"{n} DA+PI")\ncreate_sweep_plot(x, y)')
    found = [s for _, s in displayed(probe) if pattern.search(s)]
    check("(viii) mutation: a typed name in a y-label and a label= is found", len(found) == 2, f"{found}")
    renamed = ast.parse(
        '_HUES = {"ERM": 0, "DA+ERM": 3}\nMETHOD_NAMES = {"PI": "p", "PI+IV": "q"}\nx = {"a": 1, "PI": 2}'
    )
    found = method_tables(renamed)
    check("(viii) mutation: two renamed method tables are found, a one-name dict is not", len(found) == 2, f"{found}")
    missing = [
        n for n in ast.walk(probe) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "create_sweep_plot"
    ]
    check(
        "(viii) mutation: a drawing call without has_z is found", not any(k.arg == "has_z" for k in missing[0].keywords)
    )


def write_perf(root, dataset, methods, has_z):
    """One dataset's wall-clock perf pkls and its `labels.json`."""
    folder = os.path.join(root, dataset, SUBDIR_PERF)
    os.makedirs(folder, exist_ok=True)
    grid = PARAM_SPECS["epsilon"].grid_fn(dataset, 4)
    results = {m: (i + 1) * np.cumsum(np.ones(len(grid)))[:, None] for i, m in enumerate(methods)}
    for stem, obj in (("epsilon_values", grid), ("epsilon_wall_clock_results", results)):
        with open(os.path.join(folder, f"{stem}.pkl"), "wb") as fh:
            pickle.dump(obj, fh)
    with open(os.path.join(root, dataset, "labels.json"), "w") as fh:
        json.dump({"has_z": has_z, "methods": list(methods)}, fh)


def grids(root):
    """{kind: figure} of `sweep_grid` on gamma and `perf_grid` on the wall clock over
    the tree's columns, each where the tree carries its pkls."""
    from src import aggregate

    datasets, out = aggregate.columns(root), {}
    if any(os.path.exists(os.path.join(root, d, SUBDIR_SWEEP, "gamma_results.pkl")) for d in datasets):
        out["sweep_grid"] = aggregate.sweep_grid("gamma", datasets, root)
    if any(aggregate._has_perf(root, d, "wall_clock") for d in datasets):
        out["perf_grid"] = aggregate.perf_grid("wall_clock", ("wall_clock",), datasets, root)
    return out


def titles(fig):
    """{dataset title without the null-Z suffix: full title} of the grid's top row."""
    top = [ax.get_title() for ax in fig.axes if ax.get_title()]
    return {title.replace(NULL_Z_TITLE_SUFFIX, ""): title for title in top}


def leg_ix():
    print("(ix) the column titles")
    suffix = NULL_Z_TITLE_SUFFIX
    check(
        "(ix) the null-Z suffix is (Z = varnothing) on the title's line, at 3/4 of the title's size",
        suffix == r" {\fontsize{18}{18}\selectfont $(Z = \varnothing)$}" and FS_LABEL == 24,
        suffix,
    )
    check("(ix) amssymb is in the TeX preamble", "amssymb" in RC_PARAMS["text.latex.preamble"])
    check(
        "(ix) the cigarettes column is titled cigarette demand; its key stays",
        DATASET_TITLES["cigarettes"] == "cigarette demand" and "cigarettes" in DATASET_ORDER,
        f"{DATASET_TITLES['cigarettes']!r}",
    )
    # validityFig9's shape: its simulation column with a real Z, and optical and the
    # cigarettes as null-Z columns running optical's methods
    validity = recipes()["validityFig9"]
    shape = {
        "simulation": (validity["simulation"][0], True),
        "optical_device": (validity["optical_device"][0], False),
        "cigarettes": (validity["optical_device"][0], False),
    }
    merged = scratch_tree("titles_")
    for dataset, (methods, has_z) in shape.items():
        write_column(merged, dataset, methods, has_z)
        write_perf(merged, dataset, methods, has_z)
    literal = {
        "simulation": "simulation",
        "optical device": "optical device {\\fontsize{18}{18}\\selectfont $(Z = \\varnothing)$}",
        "cigarette demand": "cigarette demand {\\fontsize{18}{18}\\selectfont $(Z = \\varnothing)$}",
    }
    for kind, fig in grids(merged).items():
        got = titles(fig)
        check(f"(ix) {kind}, merged: the null-Z column titles carry the suffix", got == literal, f"{got}")
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        boxes = [ax.title.get_window_extent(renderer) for ax in fig.axes if ax.get_title()]
        inside = all(fig.bbox.x0 <= b.x0 and b.x1 <= fig.bbox.x1 and b.y1 <= fig.bbox.y1 for b in boxes)
        disjoint = all(a.x1 <= b.x0 for a, b in pairwise(sorted(boxes, key=lambda box: box.x0)))
        check(
            f"(ix) {kind}, merged: the titles are inside the figure and pairwise disjoint",
            inside and disjoint,
            f"{boxes}",
        )
        legend = fig.legends[0].get_window_extent(renderer)
        check(f"(ix) {kind}, merged: the legend clears every title", all(legend.y0 >= b.y1 for b in boxes), f"{legend}")
        if shutil.which("latex"):
            path = os.path.join(merged, f"{kind}.pdf")
            fig.savefig(path, format="pdf")
            check(f"(ix) {kind}, merged: saves to PDF under usetex", os.path.getsize(path) > 0)
        else:
            skip(f"(ix) {kind} pdf", "no latex on PATH")
        plt.close(fig)

    null = scratch_tree("titles_null_")
    for dataset in ("simulation", "optical_device"):
        write_column(null, dataset, ["PI", "DA+PI"], False)
        write_perf(null, dataset, ["PI", "DA+PI"], False)
    for kind, fig in grids(null).items():
        check(
            f"(ix) {kind}, all null Z: no suffix", not any(suffix in t for t in titles(fig).values()), f"{titles(fig)}"
        )
        plt.close(fig)

    blank = scratch_tree("titles_blank_")
    write_column(blank, "simulation", ["PI+IV"], True, param="omega")  # a Z column blank on gamma
    write_column(blank, "optical_device", ["PI"], False)
    fig = grids(blank)["sweep_grid"]
    check(
        "(ix) a Z dataset present only as a blank column: no suffix",
        not any(suffix in t for t in titles(fig).values()),
        f"{titles(fig)}",
    )
    plt.close(fig)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    os.environ.setdefault("MPLBACKEND", "Agg")
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=None, help="one leg: i to ix")
    parser.add_argument("--skip-run", action="store_true", help="skip (vi), the one leg that runs an experiment")
    args = parser.parse_args()
    legs = {
        "i": leg_i,
        "ii": leg_ii,
        "iii": leg_iii,
        "iv": leg_iv,
        "v": leg_v,
        "vi": lambda: leg_vi(args.skip_run),
        "vii": leg_vii,
        "viii": leg_viii,
        "ix": leg_ix,
    }
    if args.only:
        if args.only not in legs:
            parser.error(f"unknown leg {args.only!r}")
        legs = {args.only: legs[args.only]}
    for tag, leg in legs.items():
        try:
            leg()
        except Exception as error:  # a raise is a FAIL line, and the later legs still report
            check(f"({tag}) ran without raising", False, f"{type(error).__name__}: {error}")
        finally:
            for path in TREES:
                shutil.rmtree(path, ignore_errors=True)
            TREES.clear()
    skipped = f" ({len(SKIPPED)} SKIPPED: {'; '.join(SKIPPED)})" if SKIPPED else ""
    if not FAIL:
        print(f"A80 PASS{skipped}")
    else:
        print(f"A80 FAIL: {FAIL}{skipped}")
        sys.exit(1)
