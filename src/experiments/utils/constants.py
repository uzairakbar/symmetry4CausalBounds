"""
Centralized constants for experiments.
"""

import re
from typing import Any, Literal

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
# `base(mode)`: no whitespace outside the parentheses, any inside them
_METHOD_PATTERN = re.compile(r"^(?P<base>[^()\s]+)(?:\(\s*(?P<mode>T\s*,\s*Z|Z|T)\s*\))?$")


def parse_method(name: str) -> tuple[str, str]:
    """(base, mode) of a `methods:` entry: `DA+PI+IV` -> ("DA+PI+IV", "T,Z"),
    `DA+PI+IV(Z)` -> ("DA+PI+IV", "Z"). Raises ValueError naming the entry on a
    malformed suffix or a suffix on a method outside IV_MODE_METHODS. Membership
    of the base in ALL_METHODS is the config's check, not this one's."""
    match = _METHOD_PATTERN.match(name)
    if match is None:
        # a YAML flow list splits `DA+PI+IV(T,Z)` at its comma
        hint = "; quote the entry in a YAML flow list" if name.endswith("(T") else ""
        raise ValueError(f"malformed method entry {name!r}: a mode suffix is `(Z)`, `(T)` or `(T,Z)`{hint}.")
    base, mode = match.group("base"), match.group("mode")
    if mode is None:
        return base, "T,Z"
    if base not in IV_MODE_METHODS:
        raise ValueError(f"method {name!r}: an instrument mode is legal on {list(IV_MODE_METHODS)} only.")
    return base, re.sub(r"\s+", "", mode)


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
]
POINT_ESTIMATE_STYLE: str | tuple[int, tuple[int, int]] = (0, (5, 1))
# REAL_Z_METHODS on the lines and band edges: the family's hue, this dash-dot pattern
INSTRUMENT_Z_STYLE: tuple[int, tuple[int, int, int, int]] = (0, (3, 1, 1, 1))
PARTIAL_IDENTIFICATION_STYLE: str | tuple[int, tuple[int, int]] = "-"

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
