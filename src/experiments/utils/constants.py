"""
Centralized constants for experiments.
"""

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

# Visual style configuration
POINT_ESTIMATES: list[str] = ["ATE", "ERM", "DA+ERM", "DA+IV", "IV"]
POINT_ESTIMATE_STYLE: str | tuple[int, tuple[int, int]] = (0, (5, 1))
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
# '<param>_<metric>' (`_run_sweeps`), 'perf' (`_run_perf`) and 'query' (the
# radial query sweep, `_plot_query_sweep`). No '_sweep' suffix on the id --
# plotting.py appends that when writing the file, so
# 'gamma_approx_error' -> gamma_approx_error_sweep.pdf.
# '*' applies to every experiment; a named entry wins key by key, and both win
# over the plot function's own arguments (the query sweep's come from
# configs.ANNOTATE_SWEEP_PLOT).
#   xlim/ylim      (lo, hi); None on either end keeps the automatic edge
#   xscale/yscale  'linear' | 'log' | 'symlog' | 'asinh'. Two keys, not
#                  PANEL_CONFIGS' single 'scale': these plots scale both axes.
#   linear_width   asinh only; linthresh symlog only. Default: upper limit / 40.
#   normalize      width / worst_error sweeps only: divide every series by the
#                  baseline's (SS10.1); a `_coverage` or `_approx_error` id rejects it
#   bars           perf only; False retires the stacked reliability bars
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
_PLOT_KEYS_PERF: set = (_PLOT_KEYS - {"xlim", "xscale", "normalize"}) | {"bars"}  # categorical x
_PLOT_KEYS_QUERY: set = set(_STYLE_KEYS)  # limits and scale come from ANNOTATE_SWEEP_PLOT


def plot_keys_for(plot_id: str) -> set:
    """The keys PLOT_CONFIGS accepts under this id."""
    if plot_id == "perf":
        return _PLOT_KEYS_PERF
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
        if "normalize" in cfg and plot_id.endswith(("_coverage", "_approx_error")):
            # a rate divided by a rate means nothing, and a miss divided by a
            # baseline that vanishes above gamma* reads as noise; loud, at import
            raise ValueError(f"{name}[{plot_id!r}]: `normalize` is meaningless on a coverage or approx-error figure.")


PLOT_CONFIGS: dict[str, dict[str, dict[str, Any]]] = {
    "*": {
        "perf": {"bars": False},  # stacked reliability bars retired
    },
}

for _exp, _plots in PLOT_CONFIGS.items():
    validate_plot_keys(f"PLOT_CONFIGS[{_exp!r}]", _plots, plot_keys_for)
