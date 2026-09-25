"""
Centralized constants for experiments.
"""

import dataclasses
import re
from functools import cache
from typing import Any, Literal, NamedTuple

# Plot formatting
FS_TICK: int = 15
FS_LABEL: int = 24
PLOT_DPI: int = 1200
PAGE_WIDTH: float = 6.75
PLOT_FORMAT: Literal["png", "pdf", "ps", "eps", "svg"] = "pdf"

# Directories
ARTIFACTS_DIRECTORY: str = "artifacts"

# One subfolder per experiment type, so artifacts/<dataset>/ stays navigable
SUBDIR_QUERY: str = "query"
SUBDIR_SWEEP: str = "sweep"
SUBDIR_PERF: str = "perf"
# the datasets as the aggregate grids order their columns, and their titles
DATASET_ORDER: tuple[str, ...] = ("simulation", "optical_device", "cigarettes", "do_mnist")
DATASET_TITLES: dict[str, str] = {
    "simulation": "simulation",
    "optical_device": "optical device",
    "cigarettes": "cigarettes",
    "do_mnist": "do-MNIST",
}

# Plotting style
RC_PARAMS: dict[str, str | int | bool] = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{bm}",
    "axes.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.linewidth": 2,
    "xtick.color": "black",
    "ytick.color": "black",
}


# LaTeX building blocks
ERM = r"\operatorname{erm}"
IV = r"\textnormal{\i}\!\operatorname{v}"
PI = r"\operatorname{p}\!\textnormal{\i}"
INV = r"\textnormal{\i}\!\operatorname{nv}"
# erm is a wide box and its accent needs the kerning; Pi does not. Shared by
# DA+ERM and by both DA-side point estimates, so their tildes sit alike
DA_ERM = rf"\mkern7mu\widetilde{{\mkern-7mu{{{ERM}}}\mkern-7mu}}\mkern7mu"

# the DA+ methods ending in +IV take an instrument mode: bare or `(T,Z)` is one
# constraint per instrument, the DA translation amount T and the configured Z,
# each at its own radius (Asm. 3, SS2.6); `(Z)` the configured Z alone, `(T)` the
# translation amount alone
IV_MODE_METHODS: tuple[str, ...] = ("DA+ERM+IV", "DA+PI+IV", "PI&DA+PI+IV")
IV_MODES: tuple[str, ...] = ("T,Z", "Z", "T")
# the non-DA methods ending in +IV: spelled bare, and always the observed Z alone
NON_DA_IV_METHODS: tuple[str, ...] = ("ERM+IV", "PI+IV", "PI+INV+IV")
# `base(mode)`: no whitespace outside the parentheses, any inside them
_METHOD_PATTERN = re.compile(r"^(?P<base>[^()\s]+)(?:\(\s*(?P<mode>T\s*,\s*Z|Z|T)\s*\))?$")


# ---- method attributes, labels, colours and line styles ----
# Every method is a set of attributes: the solver, DA or not, INV or not, and
# the instrument blocks it constrains on. Everything the figures and tables show
# of a method -- its label, hue, alpha, line style and legend slot -- is a function
# of those attributes and of whether the experiment has a real Z, never of
# the spelling. An intersection carries two sides: the PI baseline on the left and
# the DA branch on the right.


@dataclasses.dataclass(frozen=True)
class Side:
    """One estimator, or one side of an intersection. Construction rejects the
    combinations no method has (inv with da, T without da, anything on ate but
    its solver), so no attribute set outside the grammar can exist."""

    solver: str  # "ate" | "erm" | "pi"
    da: bool
    inv: bool
    iv_set: frozenset[str]  # a subset of {"Z", "T"}

    def __post_init__(self):
        if self.solver not in ("ate", "erm", "pi"):
            raise ValueError(f"unknown solver {self.solver!r}; expected ate, erm or pi.")
        if not self.iv_set <= {"Z", "T"}:
            raise ValueError(f"iv_set {sorted(self.iv_set)} is not a subset of {{Z, T}}.")
        if self.solver == "ate" and (self.da or self.inv or self.iv_set):
            raise ValueError("ate carries no attribute but its solver.")
        if self.inv and self.da:
            raise ValueError("inv together with da is not a method.")
        if "T" in self.iv_set and not self.da:
            raise ValueError("T as an instrument needs da: there is no translation amount without it.")


_SOLVERS: dict[str, str] = {"ATE": "ate", "ERM": "erm", "PI": "pi"}
_BLOCKS: dict[str, frozenset[str]] = {"T,Z": frozenset("TZ"), "Z": frozenset("Z"), "T": frozenset("T")}
# the tokens after the solver, in the one order the names spell them
_SUFFIXES: tuple[tuple[str, ...], ...] = ((), ("INV",), ("IV",), ("INV", "IV"))


def _side(text: str, mode: str, *, baseline_z: bool = False) -> Side | None:
    """The attributes of one `+`-joined name; None when the tokens are unknown or
    `Side` rejects them (the config then rejects the name through `_reject_unknown`).
    A non-DA `+IV` is the observed Z alone; a DA `+IV` takes the mode's blocks.
    `baseline_z` gives an intersection's PI baseline its Z (see `_attributes`)."""
    tokens = text.split("+")
    da = tokens[0] == "DA"
    tokens = tokens[1:] if da else tokens
    if not tokens or tokens[0] not in _SOLVERS or tuple(tokens[1:]) not in _SUFFIXES:
        return None
    inv, iv = "INV" in tokens[1:], "IV" in tokens[1:]
    instrumented = _BLOCKS[mode] if da else frozenset("Z")
    iv_set = instrumented if iv else (frozenset("Z") if baseline_z else frozenset())
    try:
        return Side(_SOLVERS[tokens[0]], da, inv, iv_set)
    except ValueError:
        return None


@cache
def _attributes(base: str, mode: str) -> tuple[Side | None, Side | None]:
    """(left, side) of a parsed name; left is None outside an intersection. The
    left side of `PI&DA+PI+IV` sees Z in EVERY mode, as
    `IntersectedInstrumentalVariablePartialR2`'s baseline does ("the baseline only
    ever sees Z"); `PI&DA+PI` has no instrument on either side."""
    if "&" not in base:
        return None, _side(base, mode)
    parts = base.split("&")
    if len(parts) != 2:
        return None, None
    right = _side(parts[1], mode)
    if right is None or not right.da or right.solver != "pi":
        return None, None
    left = _side(parts[0], mode, baseline_z=bool(right.iv_set))
    # the left side is only ever the PI baseline, and it carries Z exactly when the
    # DA branch carries an instrument (so `PI+IV&DA+PI` is not a method)
    if left is None or left.da or left.inv or left.solver != "pi":
        return None, None
    if bool(left.iv_set) != bool(right.iv_set):
        return None, None
    return left, right


class Method(NamedTuple):
    """What `parse_method` returns: still the `(base, mode)` pair (equality, hashing
    and unpacking unchanged), plus the attributes read off it."""

    base: str
    mode: str

    @property
    def left(self) -> Side | None:
        """The PI baseline of an intersection; None otherwise."""
        return _attributes(self.base, self.mode)[0]

    @property
    def right(self) -> Side | None:
        """The DA branch of an intersection; None otherwise."""
        left, side = _attributes(self.base, self.mode)
        return side if left is not None else None

    @property
    def side(self) -> Side | None:
        """The method itself (the DA branch for an intersection); None if unknown."""
        return _attributes(self.base, self.mode)[1]


def parse_method(name: str) -> Method:
    """(base, mode) of a `methods:` entry: `DA+PI+IV` -> ("DA+PI+IV", "T,Z"),
    `DA+PI+IV(Z)` -> ("DA+PI+IV", "Z"). Raises ValueError naming the entry on a
    malformed suffix or a suffix on a method outside IV_MODE_METHODS. Membership
    of the base in ALL_METHODS is the config's check, not this one's. The
    result is a `Method`: it unpacks as the pair and also carries the
    attributes (`.side`, `.left`, `.right`)."""
    match = _METHOD_PATTERN.match(name)
    if match is None:
        # a YAML flow list splits `DA+PI+IV(T,Z)` at its comma
        hint = "; quote the entry in a YAML flow list" if name.endswith("(T") else ""
        raise ValueError(f"malformed method entry {name!r}: a mode suffix is `(Z)`, `(T)` or `(T,Z)`{hint}.")
    base, mode = match.group("base"), match.group("mode")
    if mode is None:
        return Method(base, "T,Z")
    if base not in IV_MODE_METHODS:
        raise ValueError(f"method {name!r}: an instrument mode is legal on {list(IV_MODE_METHODS)} only.")
    return Method(base, re.sub(r"\s+", "", mode))


def spelled_method(name: str) -> str:
    """The stored spelling: bare for the default, `base(Z)` and `base(T)` for
    the two single-instrument modes, `base(T,Z)` when the default is spelled out."""
    base, mode = parse_method(name)
    return base if "(" not in name else f"{base}({mode})"


def iv_mode(name: str) -> str:
    return parse_method(name)[1]


# method display names
TEX_MAPPER: dict[str, str] = {
    # targets
    "Data": r"$\mathrm{data}$",
    "ATE": r"$\operatorname{ate}$",
    # estimators
    "ERM": rf"${ERM}$",
    "DA+ERM": rf"${DA_ERM}$",
    # the do-MNIST ERM trained to invariance on the DA pairs (augmented Lagrangian)
    "ERM+INV": rf"${ERM}+{INV}$",
    # instrumental variable. The tilde on the iv is the T-as-IV distinction, as on
    # the interval side: ERM+IV and DA+ERM+IV(Z) read the observed Z alone
    "ERM+IV": rf"${ERM}+{IV}$",
    "DA+ERM+IV": rf"${DA_ERM}+\widetilde{{{IV}}}$",
    "PI+IV": rf"${PI}+{IV}$",
    "PI+INV+IV": rf"${PI}+{INV}+{IV}$",
    # sensitivity models
    "PI": rf"${PI}$",
    "PI+INV": rf"${PI}+{INV}$",
    "DA+PI": rf"$\widetilde{{{PI}}}$",
    "DA+PI+IV": rf"$\widetilde{{{PI}}}+\widetilde{{{IV}}}$",
    # combinations
    "PI&DA+PI": rf"${PI}\cap\widetilde{{{PI}}}$",
    "PI&DA+PI+IV": rf"${PI}\cap(\widetilde{{{PI}}}+\widetilde{{{IV}}})$",
}

# Color mapping for methods. The hue is the FAMILY (baseline, DA, INV, T-as-IV,
# intersection); a real Z on top of a family keeps the hue and changes the line
# style (REAL_Z_METHODS below), so PI and PI+IV are one blue, DA+PI and
# DA+PI+IV(Z) one red, PI+INV and PI+INV+IV one grey. DA+PI+IV and DA+PI+IV(T)
# are the same family whether or not Z is empty, one green, one line, one label.
COLOR_MAP: dict[str, int] = {
    "ATE": 3,
    "ERM": 0,
    "DA+ERM": 3,
    "ERM+IV": 0,  # the blue family, ERM plus the observed Z
    "ERM+INV": 7,  # the INV family's grey
    "DA+ERM+IV": 2,
    "PI+INV": 7,
    "PI": 0,
    "PI+IV": 0,
    "PI+INV+IV": 7,
    "DA+PI": 3,
    "DA+PI+IV": 2,
    "PI&DA+PI": 4,
    "PI&DA+PI+IV": 6,
}

# Alpha (transparency) mapping for methods
ALPHA_MAP: dict[str, float] = {
    # Point identification methods (solid)
    "ATE": 1.0,
    "ERM": 1.0,
    "DA+ERM": 1.0,
    "ERM+IV": 1.0,
    "DA+ERM+IV": 1.0,
    "ERM+INV": 1.0,
    # Partial identification methods (transparent)
    "PI+INV": 0.8,
    "PI": 0.2,
    "PI+IV": 0.2,
    "PI+INV+IV": 0.8,
    "DA+PI": 0.2,
    "DA+PI+IV": 0.4,
    "PI&DA+PI": 0.4,
    "PI&DA+PI+IV": 0.4,
}

# the spelled modes: `(T,Z)` is the bare entry under another name, `(Z)` and
# `(T)` keep the hue and alpha and show the mode on the instrument
for _base in IV_MODE_METHODS:
    TEX_MAPPER[f"{_base}(T,Z)"] = TEX_MAPPER[_base]
    for _table in (COLOR_MAP, ALPHA_MAP):
        for _mode in IV_MODES:
            _table[f"{_base}({_mode})"] = _table[_base]
TEX_MAPPER.update(
    {
        "DA+PI+IV(Z)": rf"$\widetilde{{{PI}}}+{IV}$",
        "PI&DA+PI+IV(Z)": rf"${PI}\cap(\widetilde{{{PI}}}+{IV})$",
        "DA+ERM+IV(Z)": rf"${DA_ERM}+{IV}$",
    }
)
# the (Z) spellings are the DA family plus an observed Z: DA+PI's hue and alpha,
# told apart by the line style; the (T) spellings ARE the base whenever the
# instrument set is empty, same label, same hue, same line
for _z, _family in (("DA+PI+IV(Z)", "DA+PI"), ("PI&DA+PI+IV(Z)", "PI&DA+PI"), ("DA+ERM+IV(Z)", "DA+ERM")):
    COLOR_MAP[_z], ALPHA_MAP[_z] = COLOR_MAP[_family], ALPHA_MAP[_family]
for _base in IV_MODE_METHODS:
    TEX_MAPPER[f"{_base}(T)"] = TEX_MAPPER[_base]
# a real Z on top of a family: the family's hue, this line style. Point
# estimates stay out of it: `_line_style` reads POINT_ESTIMATES first, so an
# entry here would be inert and would break a66's conjunction
REAL_Z_METHODS: frozenset[str] = frozenset({"PI+IV", "PI+INV+IV", "DA+PI+IV(Z)", "PI&DA+PI+IV(Z)"})
# legend layout, keyed on the STORED spelling (`spelled_method`): one group is one
# legend column, member 0 on top and member 1 below, so a family and the same family
# plus an observed Z stack in one hue. The three mode spellings of a T-as-IV method
# share a (group, member) on purpose -- they render as the same pixels and `_legend`
# folds them to a single entry, which is why those groups have one member only. NOT
# derivable from REAL_Z_METHODS: the (T,Z) spellings are outside it, so keying the
# member on membership would collide groups 6 and 9.
PAIR_ORDER: dict[str, tuple[int, int]] = {
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
    # do-MNIST only; a new last group, so no existing legend moves
    "ERM+INV": (10, 0),
}
# the coefficient labels of the cigarette price elasticities, paper notation
# h_*(x) = theta_*' x on the four log treatments
COEFFICIENT_LABELS: dict[str, str] = {
    "p": r"$\theta_{\mathrm{state\,price}}$",
    "pn": r"$\theta_{\mathrm{neighbour\,price}}$",
}

# Visual style configuration
POINT_ESTIMATES: list[str] = [
    "ATE",
    "ERM",
    "DA+ERM",
    "ERM+IV",
    "DA+ERM+IV",
    "DA+ERM+IV(Z)",
    "DA+ERM+IV(T)",
    "DA+ERM+IV(T,Z)",
    "ERM+INV",
]
POINT_ESTIMATE_STYLE: str | tuple[int, tuple[int, int]] = (0, (5, 1))
# REAL_Z_METHODS on the lines and band edges: the family's hue, this dash-dot pattern
INSTRUMENT_Z_STYLE: tuple[int, tuple[int, int, int, int]] = (0, (3, 1, 1, 1))
PARTIAL_IDENTIFICATION_STYLE: str | tuple[int, tuple[int, int]] = "-"


# ---- the display of a method: every function of (method, has_z) ----
# `has_z` is whether the experiment has a real Z. Without one a Z-only
# constraint is inert and the method IS its no-IV estimator (a57 (i), a63 (ii)), so
# `resolve` drops Z from every instrument set first and nothing after it branches on
# `has_z`. The hues are seaborn "deep" at the indices the figures have always used,
# as literals so this module imports no plotting library (a80 checks them).
DEEP_HEX: dict[str, str] = {
    "blue": "#4c72b0",
    "green": "#55a868",
    "red": "#c44e52",
    "purple": "#8172b3",
    "pink": "#da8bc3",
    "grey": "#8c8c8c",
}
DEEP_INDEX: dict[str, int] = {"blue": 0, "green": 2, "red": 3, "purple": 4, "pink": 6, "grey": 7}
# the band alpha of an interval method, by hue; a point estimate is opaque
HUE_ALPHA: dict[str, float] = {"blue": 0.2, "red": 0.2, "grey": 0.8, "green": 0.4, "purple": 0.4, "pink": 0.4}


class Resolved(NamedTuple):
    """The resolved attributes: Z dropped from every side without a real Z.
    `side` is the DA branch for an intersection, `left` its PI baseline."""

    left: Side | None
    side: Side


def _drop_z(side: Side | None, has_z: bool) -> Side | None:
    if side is None or has_z:
        return side
    return dataclasses.replace(side, iv_set=side.iv_set - {"Z"})


@cache
def resolve(method: str, has_z: bool) -> Resolved:
    """The attributes `method` draws with in an experiment with (`has_z`) or
    without a real Z. Raises ValueError on a name outside the grammar."""
    parsed = parse_method(method)
    if parsed.side is None:
        raise ValueError(f"unknown method {method!r}.")
    return Resolved(_drop_z(parsed.left, bool(has_z)), _drop_z(parsed.side, bool(has_z)))


def is_method(name: str) -> bool:
    """Whether `name` spells a method of the grammar."""
    try:
        return parse_method(name).side is not None
    except ValueError:
        return False


def _point(resolved: Resolved) -> bool:
    # ATE and the ERMs: a line, not a band
    return resolved.side.solver in ("ate", "erm")


def is_point_estimate(method: str) -> bool:
    """ATE and the ERMs: a line, not a band."""
    return _point(resolve(method, True))


def is_interval(method: str) -> bool:
    """The PI methods: a band between two bounds."""
    return resolve(method, True).side.solver == "pi"


def _side_tex(side: Side) -> tuple[str, bool]:
    """(TeX, whether a `+` token was appended) of one side."""
    if side.solver == "ate":
        return r"\operatorname{ate}", False
    if side.da:
        tex = DA_ERM if side.solver == "erm" else rf"\widetilde{{{PI}}}"
    else:
        tex = ERM if side.solver == "erm" else PI
    plus = False
    if side.inv:
        tex, plus = tex + f"+{INV}", True
    # T tilde-marks the instrument with or without Z; Z alone is the plain iv;
    # nothing when the set is empty (a null-Z Z-only constraint is inert)
    if "T" in side.iv_set:
        tex, plus = tex + rf"+\widetilde{{{IV}}}", True
    elif "Z" in side.iv_set:
        tex, plus = tex + f"+{IV}", True
    return tex, plus


@cache
def label(method: str, has_z: bool) -> str:
    r"""The TeX label: an intersection is `left \cap right`, a side in
    parentheses iff it carries a `+`."""
    resolved = resolve(method, has_z)
    if resolved.left is None:
        return f"${_side_tex(resolved.side)[0]}$"

    def wrapped(tex, plus):
        return f"({tex})" if plus else tex

    return f"${wrapped(*_side_tex(resolved.left))}\\cap{wrapped(*_side_tex(resolved.side))}$"


def hue(method: str, has_z: bool) -> str:
    """The method's hue, "black" or a DEEP_HEX key; the first match wins."""
    resolved = resolve(method, has_z)
    side = resolved.side
    if side.solver == "ate":
        return "black"
    if resolved.left is not None:  # an intersection: its DA side alone decides
        return "pink" if "T" in side.iv_set else "purple"
    if not side.da:
        return "grey" if side.inv else "blue"
    return "green" if "T" in side.iv_set else "red"


def _hex_rgb(hex_colour: str) -> tuple[float, float, float]:
    # matplotlib's `to_rgb` arithmetic, so this equals sns.color_palette("deep")[i]
    return tuple(int(hex_colour[i : i + 2], 16) / 255 for i in (1, 3, 5))


def colour(method: str, has_z: bool) -> str | tuple[float, float, float]:
    """The string "black" on ATE, else the RGB float tuple of the method's deep hue."""
    name = hue(method, has_z)
    return name if name == "black" else _hex_rgb(DEEP_HEX[name])


def alpha(method: str, has_z: bool) -> float:
    """1.0 on a point estimate, the hue's band alpha on an interval."""
    if _point(resolve(method, has_z)):
        return 1.0
    return HUE_ALPHA[hue(method, has_z)]


def _z_member(resolved: Resolved) -> bool:
    # a family plus a real Z: the Z-only instrument survives resolution
    return resolved.side.iv_set == frozenset("Z")


def line_style(method: str, has_z: bool):
    """Dashed on a point estimate, dash-dot on an interval constrained by a real
    Z alone, solid on every other interval."""
    resolved = resolve(method, has_z)
    if _point(resolved):
        return POINT_ESTIMATE_STYLE
    return INSTRUMENT_Z_STYLE if _z_member(resolved) else PARTIAL_IDENTIFICATION_STYLE


# the legend's groups, one column each, by (solver, da, inv, T instrumented,
# intersection): the order the figures have always used. Member 0 sits on top and
# a real-Z family member (member 1) below it, in the same hue
_FAMILIES: tuple[tuple[str, bool, bool, bool, bool], ...] = (
    ("ate", False, False, False, False),  # ATE
    ("pi", False, True, False, False),  # PI+INV (+IV)
    ("pi", False, False, False, False),  # PI (+IV)
    ("erm", False, False, False, False),  # ERM (+IV)
    ("pi", True, False, False, False),  # DA+PI (+IV(Z))
    ("erm", True, False, False, False),  # DA+ERM (+IV(Z))
    ("pi", True, False, True, False),  # DA+PI+IV (T), (T,Z)
    ("erm", True, False, True, False),  # DA+ERM+IV (T), (T,Z)
    ("pi", True, False, False, True),  # PI&DA+PI (+IV(Z))
    ("pi", True, False, True, True),  # PI&DA+PI+IV (T), (T,Z)
    ("erm", False, True, False, False),  # ERM+INV, do-MNIST only: last, so no legend moves
)


def legend_order(method: str, has_z: bool) -> tuple[int, int]:
    """(group, member) of the method's legend entry."""
    resolved = resolve(method, has_z)
    side = resolved.side
    family = (side.solver, side.da, side.inv, "T" in side.iv_set, resolved.left is not None)
    return _FAMILIES.index(family), int(_z_member(resolved))


class MethodStyle(NamedTuple):
    """Everything a figure draws of one method in one column."""

    label: str
    colour: str | tuple[float, float, float]
    alpha: float
    linestyle: Any
    order: tuple[int, int]
    point: bool

    @property
    def signature(self) -> tuple:
        """Two entries with one signature render as the same pixels: a legend keeps one."""
        return self.label, self.colour, self.alpha, self.linestyle


def z_counterpart(method: str) -> str:
    """The method with Z added to every side that has a Z slot: PI -> PI+IV,
    PI+INV -> PI+INV+IV, ERM -> ERM+IV, DA+PI -> DA+PI+IV(Z), DA+ERM -> DA+ERM+IV(Z),
    PI&DA+PI -> PI&DA+PI+IV(Z), X(T) -> X(T,Z) on every IV-mode base; the method
    itself where that spelling is not a method (ATE, ERM+INV) or Z is already in."""
    base, mode = parse_method(method)
    if base in IV_MODE_METHODS:
        return f"{base}(T,Z)" if mode == "T" else method
    if f"{base}+IV" in IV_MODE_METHODS:
        return f"{base}+IV(Z)"
    if f"{base}+IV" in NON_DA_IV_METHODS:
        return f"{base}+IV"
    return method


@cache
def method_style(method: str, has_z: bool, *, merged: bool = False) -> MethodStyle:
    """The render record of `method` in a column with (`has_z`) or without a real
    Z. `merged` is an aggregate grid that has a Z column: there a null-Z
    column draws each method as its Z counterpart, so every cross-dataset pair
    shares one label, hue, alpha and dash, and the legend folds it to one entry."""
    if merged and not has_z:
        method, has_z = z_counterpart(method), True
    return MethodStyle(
        label=label(method, has_z),
        colour=colour(method, has_z),
        alpha=alpha(method, has_z),
        linestyle=line_style(method, has_z),
        order=legend_order(method, has_z),
        point=_point(resolve(method, has_z)),
    )


def _row_key(method: str, has_z: bool) -> tuple:
    """The label row a method falls in (a80 (ii)): its resolved attributes with each
    instrument set collapsed to the token it prints. The mode spellings of one
    row share a key, and so does a null-Z Z-only member with its no-IV twin."""

    def key(side):
        if side is None:
            return None
        token = "tilde" if "T" in side.iv_set else ("iv" if side.iv_set else "none")
        return side.solver, side.da, side.inv, token

    resolved = resolve(method, has_z)
    return key(resolved.left), key(resolved.side)


def shared_labels(methods, has_z: bool) -> list[tuple[str, str, str]]:
    """(earlier, later, label) for every later method whose label an earlier one
    already reads: one legend entry for both."""
    first, pairs = {}, []
    for method in methods:
        text = label(method, has_z)
        if text in first:
            pairs.append((first[text], method, text))
        else:
            first[text] = method
    return pairs


def label_collisions(methods, has_z: bool) -> list[tuple[str, str, str]]:
    """The `shared_labels` pairs from different label rows: the uniqueness a80 (iv) asks
    of one experiment is that this is empty."""
    return [(a, b, text) for a, b, text in shared_labels(methods, has_z) if _row_key(a, has_z) != _row_key(b, has_z)]


# Plotting defaults
DEFAULT_HILIGHT_OURS: bool = False
DEFAULT_NORMALIZE_ERROR: bool = False
# `normalize` on `create_sweep_plot` (SS10.1): divide every series of a width or
# worst-error sweep figure by the baseline's, so the baseline reads 1.0 and the
# rest as fractions of it. Off by default, so no shipped figure
# moves; the recipes turn it on through the global `normalize` toggle. NOT the
# per-query `DEFAULT_NORMALIZE_ERROR` above, which divides by the zero predictor.
DEFAULT_NORMALIZE_SWEEP: bool = False
# coverage figures and every figure drawn normalised: linear y, these limits,
# over any `PLOT_CONFIGS` scale or limit
CLAMP_YLIM: tuple[float, float] = (-0.05, 1.05)
# the figure ids it is honoured for, by suffix. A `_coverage` id is a rate and an
# `_approx_error` id a squared miss whose baseline vanishes above gamma*; neither
# is normalised (approx_error rolled back 2026-09-15 at the user's request)
NORMALIZED_SWEEP_SUFFIXES: tuple[str, ...] = ("_width", "_worst_error")
# the baseline, in this order: PI if it ran, else PI+IV, else no normalisation
NORMALIZE_BASELINES: tuple[str, ...] = ("PI", "PI+IV")

# Configuration for panel plots per experiment and per row
# Row index mapping: 0: Worst Error, 1: Width, 2: Density, 3: Predictions
PANEL_CONFIGS = {
    "simulation": {
        0: {"scale": "asinh", "ylim": (0, 10.01), "linear_width": 0.25, "linthresh": 2.0},
        1: {"scale": "asinh", "ylim": (0.1, 5), "linear_width": 1.0, "linthresh": 1.0},
        2: {"scale": "linear", "ylim": (0, 0.5), "linear_width": 1.0, "linthresh": 2.0},
        3: {"ylim": (-3, 3)},
    },
    "optical_device": {
        0: {"scale": "asinh", "ylim": (0, 100), "linear_width": 0.15, "linthresh": 2.0},
        1: {"scale": "asinh", "ylim": (0, 10.01), "linear_width": 0.25, "linthresh": 2.0},
        2: {"scale": "linear", "ylim": (0, 0.01), "linear_width": 1.0, "linthresh": 2.0},
        3: {"ylim": (-2.625, 3.375)},
    },
    # the FWL'd panel is sigma-normalised, so the rows are in units of sigma: the
    # measured PI width is 1.89 and h* has std 1. Tuned once against that.
    "cigarettes": {
        0: {"scale": "asinh", "ylim": (0, 10.01), "linear_width": 0.25, "linthresh": 2.0},
        1: {"scale": "asinh", "ylim": (0, 5), "linear_width": 0.5, "linthresh": 1.0},
        2: {"scale": "linear", "ylim": (0, 5), "linear_width": 1.0, "linthresh": 2.0},
        3: {"ylim": (-4, 4)},
    },
}


# Per-plot overrides for the sweep, perf and query-sweep figures. Keyed
# experiment -> plot id, where the id is the `fname` the orchestrator builds:
# '<param>_<metric>' (`_run_sweeps`), 'epsilon_wall_clock', 'epsilon_seed_var' and
# 'epsilon_feasibility' (`_run_perf`, the perf sweeps) and 'query' (the radial query sweep,
# `_plot_query_sweep`). No '_sweep' suffix on the id -- plotting.py appends that
# when writing the file, so 'gamma_approx_error' -> gamma_approx_error_sweep.pdf.
# '*' applies to every experiment; a named entry wins key by key, and both win
# over the plot function's own arguments (the query sweep's come from
# configs.ANNOTATE_SWEEP_PLOT).
#   xlim/ylim      (lo, hi); None on either end keeps the automatic edge
#   xscale/yscale  'linear' | 'log' | 'symlog' | 'asinh'. Two keys, not
#                  PANEL_CONFIGS' single 'scale': these plots scale both axes.
#   linear_width   asinh only; linthresh symlog only. Default: upper limit / 40.
#   normalize      width / worst_error sweeps only: divide every series by the
#                  baseline's (SS10.1); a `_coverage`, `_approx_error`, `_wall_clock`,
#                  `_seed_var` or `_feasibility` id rejects it
# Style keys, accepted by every id (and by ANNOTATE_SWEEP_PLOT):
#   legend         False hides it, True shows it, a str or (x, y) tuple is a
#                  matplotlib loc. Absent: the plot function's own default (on).
#   x_color        colour of the x-axis label and its tick labels (default 'k')
#   y_color        the same for the y-axis (default 'k')
#   title          title text (TeX allowed); absent or empty means no title
#   title_color    colour of the title (default 'k')
# Examples:
#   PLOT_CONFIGS["simulation"]["gamma_coverage"] = {"legend": False}
#   PLOT_CONFIGS["*"]["gamma_coverage"] = {"y_color": "red"}
#   PLOT_CONFIGS["optical_device"]["query"] = {"title": r"radial sweep", "title_color": "tab:blue"}
_STYLE_KEYS: set = {"legend", "x_color", "y_color", "title", "title_color"}
_PLOT_KEYS: set = {"xlim", "ylim", "xscale", "yscale", "linear_width", "linthresh", "normalize"} | _STYLE_KEYS
_PLOT_KEYS_QUERY: set = set(_STYLE_KEYS)  # limits and scale come from ANNOTATE_SWEEP_PLOT


def plot_keys_for(plot_id: str) -> set:
    """The keys PLOT_CONFIGS accepts under this id."""
    if plot_id == "query":
        return _PLOT_KEYS_QUERY
    return _PLOT_KEYS


def validate_plot_keys(name: str, table: dict[str, dict[str, Any]], allowed) -> None:
    """
    Raise at import on an unknown key, so a typo is never a silent no-op.

    `allowed` is a set applied to every id, or a callable mapping the plot id to
    its set (plot_keys_for).
    """
    for plot_id, cfg in table.items():
        keys = allowed(plot_id) if callable(allowed) else allowed
        bad = set(cfg) - keys
        if bad:
            raise ValueError(f"{name}[{plot_id!r}]: unknown key(s) {sorted(bad)}.")
        if "normalize" in cfg and plot_id.endswith(
            ("_coverage", "_approx_error", "_wall_clock", "_seed_var", "_feasibility")
        ):
            # a rate divided by a rate means nothing, a miss divided by a
            # baseline that vanishes above gamma* reads as noise, and the perf
            # figures are already normalised; loud, at import
            raise ValueError(f"{name}[{plot_id!r}]: `normalize` is meaningless on a coverage, error or perf figure.")


PLOT_CONFIGS: dict[str, dict[str, dict[str, Any]]] = {
    "*": {},
}

for _exp, _plots in PLOT_CONFIGS.items():
    validate_plot_keys(f"PLOT_CONFIGS[{_exp!r}]", _plots, plot_keys_for)
