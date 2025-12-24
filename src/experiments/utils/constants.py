"""
Centralized constants for experiments.
"""
from typing import Dict, Literal

# Plot formatting
FS_TICK: int = 15
FS_LABEL: int = 24
PLOT_DPI: int = 1200
PAGE_WIDTH: float = 6.75
PLOT_FORMAT: Literal['png', 'pdf', 'ps', 'eps', 'svg'] = 'pdf'

# Directories
ARTIFACTS_DIRECTORY: str = 'artifacts'

# Plotting style
RC_PARAMS: Dict[str, str | int | bool] = {
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern'],
    'text.latex.preamble': r'\usepackage{amsmath}\usepackage{bm}',
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.linewidth': 2,
    'xtick.color': 'black',
    'ytick.color': 'black',
}

# Method display names
TEX_MAPPER: Dict[str, str] = {
    'Data': r'Data',
    'ATE': r'$\operatorname{ate}$',
    'PI': r'$\operatorname{pi}$',
    'DA+PI': r'$\operatorname{da}+\operatorname{pi}$',
    'INV+PI': r'$\operatorname{inv}+\operatorname{pi}$',
    'ERM': r'$\operatorname{erm}$',
    'DA+ERM': r'$\operatorname{da}+\operatorname{erm}$',
}

# Color mapping for methods
COLOR_MAP: Dict[str, int] = {
    'ATE': 3,
    'ERM': 0,
    'DA+ERM': 3,
    'PI': 0,
    'DA+PI': 3,
    'INV+PI': 2,
}

# Plotting defaults
DEFAULT_HILIGHT_OURS: bool = False
DEFAULT_NORMALIZE_ERROR: bool = False