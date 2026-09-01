"""
Centralized constants for experiments.
"""

from typing import Any, Dict, Literal, List

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
SUBDIR_SCATTER: str = "scatter"
SUBDIR_PERF: str = "perf"

# Plotting style
RC_PARAMS: Dict[str, str | int | bool] = {
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

from typing import Dict

# LaTeX building blocks
ERM = r"\operatorname{erm}"
IV = r"\textnormal{\i}\!\operatorname{v}"
PI = r"\operatorname{p}\!\textnormal{\i}"
INV = r"\textnormal{\i}\!\operatorname{nv}"

# method display names
TEX_MAPPER: Dict[str, str] = {
    # targets
    "Data": r"$\mathrm{data}$",
    "ATE": r"$\operatorname{ate}$",
    # estimators
    "ERM": rf"${ERM}$",
    "DA+ERM": rf"$\mkern7mu\widetilde{{\mkern-7mu{{{ERM}}}\mkern-7mu}}\mkern7mu$",
    # instrumental variable
    "DA+IV": rf"$\widetilde{{{IV}}}$",
    "PI_IV": rf"${PI}+{IV}$",
    # sensitivity models
    "PI": rf"${PI}$",
    "PI_INV": rf"${PI}+{INV}$",
    "DA+PI": rf"$\widetilde{{{PI}}}$",
    "DA+PI_IV": rf"$\widetilde{{{PI}}}+\widetilde{{{IV}}}$",
    # combinations
    "PI&DA+PI": rf"${PI}\cap\widetilde{{{PI}}}$",
    "PI&DA+PI_IV": rf"${PI}\cap(\widetilde{{{PI}}}+\widetilde{{{IV}}})$",
}

# Color mapping for methods
COLOR_MAP: Dict[str, int] = {
    "ATE": 3,
    "ERM": 0,
    "DA+ERM": 3,
    "DA+IV": 2,
    "PI_INV": 7,
    "PI": 0,
    "PI_IV": 1,
    "DA+PI": 3,
    "DA+PI_IV": 2,
    "PI&DA+PI": 4,
    "PI&DA+PI_IV": 6,
}

# Alpha (transparency) mapping for methods
ALPHA_MAP: Dict[str, float] = {
    # Point identification methods (solid)
    "ATE": 1.0,
    "ERM": 1.0,
    "DA+ERM": 1.0,
    "DA+IV": 1.0,
    # Partial identification methods (transparent)
    "PI_INV": 0.8,
    "PI": 0.2,
    "PI_IV": 0.2,
    "DA+PI": 0.2,
    "DA+PI_IV": 0.4,
    "PI&DA+PI": 0.4,
    "PI&DA+PI_IV": 0.4,
}

# Visual style configuration
POINT_ESTIMATES: List[str] = ["ATE", "ERM", "DA+ERM", "DA+IV"]
POINT_ESTIMATE_STYLE: str | tuple[int, tuple[int, int]] = (0, (5, 1))
PARTIAL_IDENTIFICATION_STYLE: str | tuple[int, tuple[int, int]] = "-"

# Plotting defaults
DEFAULT_HILIGHT_OURS: bool = False
DEFAULT_NORMALIZE_ERROR: bool = False

# Panel Plot Specific Configurations
# Requirement 2: specification of y-axis clip values for Row 4 (index 3: h^T x)
PANEL_PREDICTION_LIMITS: Dict[str, tuple[float, float]] = {
    "simulation": (-3, 3),
    "optical_device": (-2.625, 3.375),
}

# Requirement 3: specification of scale (log vs linear) for rows 1 and 2 (indices 0 and 1)
PANEL_Y_SCALES: Dict[str, Literal["linear", "log"]] = {
    "worst_error": "asinh",  # Row 1 (index 0)
    "width": "asinh",  # Row 2 (index 1)
}


# src/experiments/utils/constants.py

# Configuration for panel plots per experiment and per row
# Row index mapping: 0: Worst Error, 1: Width, 2: Density, 3: Predictions
PANEL_CONFIGS = {
    "simulation": {
        0: {"scale": "asinh", "ylim": (0, 10.01), "linear_width": 0.25, "linthresh": 2.0},
        1: {"scale": "asinh", "ylim": (0, 10.01), "linear_width": 0.25, "linthresh": 2.0},
        2: {"scale": "linear", "ylim": (0, 0.5), "linear_width": 1.0, "linthresh": 2.0},
        3: {"ylim": (-3, 3)},
    },
    "optical_device": {
        0: {"scale": "asinh", "ylim": (0, 100), "linear_width": 0.15, "linthresh": 2.0},
        1: {"scale": "asinh", "ylim": (0, 10.01), "linear_width": 0.25, "linthresh": 2.0},
        2: {"scale": "linear", "ylim": (0, 0.01), "linear_width": 1.0, "linthresh": 2.0},
        3: {"ylim": (-2.625, 3.375)},
    },
}


# Per-plot rescale/clip overrides for the sweep/scatter/perf figures. Keyed
# experiment -> plot id, where the id is the `fname` base.py already builds:
# '<param>_<metric>' (base.py:533), 'scatter_<param>_<mx>_vs_<my>' (base.py:627),
# 'perf' (base.py:601). No '_sweep' suffix on the id -- plotting.py appends that when
# writing the file, so 'gamma_approx_error' -> gamma_approx_error_sweep.pdf.
# '*' applies to every experiment; a named entry wins key by key.
#   xlim/ylim      (lo, hi); None on either end keeps the automatic edge
#   xscale/yscale  'linear' | 'log' | 'symlog' | 'asinh'. Two keys, not
#                  PANEL_CONFIGS' single 'scale': these plots scale both axes.
#   linear_width   asinh only; linthresh symlog only. Default: upper limit / 40.
#   legend         False hides it, True is automatic, a str is a matplotlib loc
#   bars           perf only; False retires the stacked reliability bars
_PLOT_KEYS: set = {"xlim", "ylim", "xscale", "yscale", "linear_width", "linthresh", "legend"}
_PLOT_KEYS_PERF: set = (_PLOT_KEYS - {"xlim", "xscale"}) | {"bars"}  # categorical x

PLOT_CONFIGS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "*": {
        "perf": {"bars": False},  # stacked reliability bars retired
    },
}

for _exp, _plots in PLOT_CONFIGS.items():  # a typo must not be a silent no-op
    for _id, _cfg in _plots.items():
        _bad = set(_cfg) - (_PLOT_KEYS_PERF if _id == "perf" else _PLOT_KEYS)
        if _bad:
            raise ValueError(f"PLOT_CONFIGS[{_exp!r}][{_id!r}]: unknown key(s) {sorted(_bad)}.")
