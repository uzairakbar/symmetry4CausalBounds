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

# the DA+ methods ending in +IV take an instrument mode: bare or `(T,Z)` is the
# joint Z-tilde = (T, Z) of Asm. 3 (the DA translation amount as an instrument
# beside the configured Z), `(Z)` the configured Z alone with no T term, `(T)`
# the translation amount alone, no Z term
IV_MODE_METHODS: tuple[str, ...] = ("DA+IV", "DA+PI+IV", "PI&DA+PI+IV")
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
    "DA+ERM": rf"$\mkern7mu\widetilde{{\mkern-7mu{{{ERM}}}\mkern-7mu}}\mkern7mu$",
    # instrumental variable
    "IV": rf"${IV}$",
    "DA+IV": rf"$\widetilde{{{IV}}}$",
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

# Color mapping for methods
COLOR_MAP: dict[str, int] = {
    "ATE": 3,
    "ERM": 0,
    "DA+ERM": 3,
    "IV": 2,  # the point-estimate pair with DA+IV
    "DA+IV": 2,
    "PI+INV": 7,
    "PI": 0,
    "PI+IV": 1,
    "PI+INV+IV": 5,
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
    "IV": 1.0,
    "DA+IV": 1.0,
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
        "DA+IV(Z)": rf"$\widetilde{{{IV}}}_{{Z}}$",
        "DA+PI+IV(T)": rf"$\widetilde{{{PI}}}+\widetilde{{{IV}}}_{{T}}$",
        "PI&DA+PI+IV(T)": rf"${PI}\cap(\widetilde{{{PI}}}+\widetilde{{{IV}}}_{{T}})$",
        "DA+IV(T)": rf"$\widetilde{{{IV}}}_{{T}}$",
    }
)

# Visual style configuration
POINT_ESTIMATES: list[str] = ["ATE", "ERM", "DA+ERM", "DA+IV", "DA+IV(Z)", "DA+IV(T)", "DA+IV(T,Z)", "IV"]
POINT_ESTIMATE_STYLE: str | tuple[int, tuple[int, int]] = (0, (5, 1))
# the (Z) siblings on the sweep lines: same hue, this dash-dot pattern
INSTRUMENT_Z_STYLE: tuple[int, tuple[int, int, int, int]] = (0, (3, 1, 1, 1))
# the (T) siblings: same hue, dotted
INSTRUMENT_T_STYLE: tuple[int, tuple[int, int]] = (0, (1, 1))
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
# '<param>_<metric>' (`_run_sweeps`), 'epsilon_wall_clock' and 'epsilon_seed_var'
# (`_run_perf`, the two perf sweeps) and 'query' (the radial query sweep,
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
#                  baseline's (SS10.1); a `_coverage`, `_approx_error`, `_wall_clock`
#                  or `_seed_var` id rejects it
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
        if "normalize" in cfg and plot_id.endswith(("_coverage", "_approx_error", "_wall_clock", "_seed_var")):
            # a rate divided by a rate means nothing, a miss divided by a
            # baseline that vanishes above gamma* reads as noise, and the perf
            # figures are already normalised; loud, at import
            raise ValueError(f"{name}[{plot_id!r}]: `normalize` is meaningless on a coverage, error or perf figure.")


PLOT_CONFIGS: dict[str, dict[str, dict[str, Any]]] = {
    "*": {},
}

for _exp, _plots in PLOT_CONFIGS.items():
    validate_plot_keys(f"PLOT_CONFIGS[{_exp!r}]", _plots, plot_keys_for)
