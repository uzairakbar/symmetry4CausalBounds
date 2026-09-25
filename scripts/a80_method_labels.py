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
         per hue; with a Z, `line_style` and `legend_order` equal frozen literals
         of the pre-change `_line_style` and `PAIR_ORDER`; without one every
         interval is solid and every method is member 0 of its group. Catches: a
         hue rule out of order, a palette index drifting, a `_FAMILIES` row swap, a
         null-Z method drawn dash-dot. Misses: whether the plotting code reads
         these functions -- (iv), (v), (ix).
  (iv)   uniqueness: (a) within one experiment `label` is injective on the label
         rows, over every recipe block, the digest blocks, the default lists and
         exhaustively at each has_z; (b) across experiments one label is one
         (colour, alpha, dash). A mutation of each, patched into the module, proves
         it can fail. Catches: two rows on one label, one label drawn two ways.
         Misses: a figure that draws a label without `method_style` -- the
         completeness greps in (viii).
  (v)    the aggregate merge: the `z_counterpart` pairs whose render differs are
         exactly the seven cross-dataset pairs (SEVEN), the two (T) -> (T,Z)
         extras render identically, a null-Z column in a merged grid draws each
         method as its Z counterpart, every multi-column recipe pairs each null-Z
         method with a method of every Z column, and validityFig9's columns fold
         to 6 legend entries. Catches: a pair missing from `z_counterpart`, a
         recipe whose null-Z column has no Z twin, a merge that leaves two legend
         entries per pair. Misses: the aggregate's reading of `labels.json`.
  (vi)   the writer of `artifacts/<dataset>/labels.json` (a later tier).
  (vii)  do-MNIST's PI+INV reads `pi+inv` whatever the `inv_recenter`, in `label`
         and in the `domnist_table` row under off, on and inv. Catches: a label
         that follows the centring. Misses: the table's other rows (a79 (v)).
  (viii) completeness greps (a later tier).
  (ix)   the null-Z column titles and the cigarettes title (a later tier).

A leg, or a part of one, that needs code from a later commit prints [SKIP] and is
counted in the summary line.

Usage:
    MPLBACKEND=Agg python scripts/a80_method_labels.py [--only LEG] [--skip-run]
"""

import argparse
import glob
import json
import os
import sys
import tempfile

import yaml
from loguru import logger

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import digest_leg  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.colors import to_hex  # noqa: E402

import src.experiments.utils.constants as constants  # noqa: E402
from src.experiments.configs import (  # noqa: E402
    ALL_METHODS,
    DOMNIST_ONLY_METHODS,
    _check_instruments,
    resolve_dataset_block,
)
from src.experiments.utils.constants import (  # noqa: E402
    DA_ERM,
    DEEP_HEX,
    DEEP_INDEX,
    ERM,
    INSTRUMENT_Z_STYLE,
    INV,
    IV,
    IV_MODE_METHODS,
    IV_MODES,
    PARTIAL_IDENTIFICATION_STYLE,
    PI,
    POINT_ESTIMATE_STYLE,
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
# the pre-change `_line_style` of every spelling, frozen: dashed on a point estimate,
# dash-dot on REAL_Z_METHODS, solid otherwise
DASH_DOT = {"PI+IV", "PI+INV+IV", "DA+PI+IV(Z)", "PI&DA+PI+IV(Z)"}
LINE_STYLES = {
    m: POINT_ESTIMATE_STYLE if m in POINTS else INSTRUMENT_Z_STYLE if m in DASH_DOT else PARTIAL_IDENTIFICATION_STYLE
    for m in SPELLINGS
}
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
        if has_z:
            check(f"(iii) {name} with Z keeps its line style", style == LINE_STYLES[name], f"{style}")
        elif name in POINTS:
            check(f"(iii) {name} without Z is dashed", style == POINT_ESTIMATE_STYLE, f"{style}")
        else:
            check(f"(iii) {name} without Z is solid", style == PARTIAL_IDENTIFICATION_STYLE, f"{style}")
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
    skip("(iv) config warning", "the collision warning lands with the labels.json commit")
    skip("(iv) create_sweep_plot", "one legend entry for PI and PI+IV without Z lands with the plotting commit")


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
    skip("(v) sweep_grid", "the synthetic merged tree through sweep_grid lands with the aggregate commit")


def leg_vi(skip_run):
    print("(vi) the labels.json writer")
    skip("(vi)", "the writer lands with the labels.json commit")


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


def leg_viii():
    print("(viii) completeness")
    skip("(viii)", "the completeness greps land with the commit that drops the static tables")


def leg_ix():
    print("(ix) the column titles")
    skip("(ix)", "the null-Z title suffix and the cigarettes title land in their own commits")


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
    skipped = f" ({len(SKIPPED)} SKIPPED: {'; '.join(SKIPPED)})" if SKIPPED else ""
    if not FAIL:
        print(f"A80 PASS{skipped}")
    else:
        print(f"A80 FAIL: {FAIL}{skipped}")
        sys.exit(1)
