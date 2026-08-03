"""
Centralized constants for experiments.
"""
from typing import Dict, Literal, List

# Plot formatting
FS_TICK: int = 15
FS_LABEL: int = 24
PLOT_DPI: int = 1200
PAGE_WIDTH: float = 6.75
PLOT_FORMAT: Literal['png', 'pdf', 'ps', 'eps', 'svg'] = 'pdf'

# Directories
ARTIFACTS_DIRECTORY: str = 'artifacts'

# One subfolder per experiment type, so artifacts/<dataset>/ stays navigable
SUBDIR_QUERY: str = 'query'
SUBDIR_SWEEP: str = 'sweep'
SUBDIR_SCATTER: str = 'scatter'
SUBDIR_PERF: str = 'perf'

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

from typing import Dict

# LaTeX building blocks
PI  = r'\operatorname{p}\!\textnormal{\i}'
IV  = r'\textnormal{\i}\!\operatorname{v}'
INV = r'\textnormal{\i}\!\operatorname{nv}'

# method display names
TEX_MAPPER: Dict[str, str] = {
    # targets
    'Data':     r'$\mathrm{data}$',
    'ATE':      r'$\operatorname{ate}$',
    # estimators
    'ERM':      r'$\operatorname{erm}$',
    'DA+ERM':   r'$\tilde{\operatorname{erm}}$',
    # instrumental variable
    'DA+IV':    fr'$\tilde{{{IV}}}$',
    'PI_IV':    fr'${PI}+{IV}$',
    # sensitivity models
    'PI':       fr'${PI}$',
    'PI_INV':   fr'${PI}+{INV}$',
    'DA+PI':    fr'$\tilde{{{PI}}}$',
    'DA+PI_IV': fr'$\tilde{{{PI}}}+\tilde{{{IV}}}$',
    # combinations
    'PI&DA+PI':     fr'${PI}\cap\tilde{{{PI}}}$',
    'PI&DA+PI_IV':  fr'${PI}\cap(\tilde{{{PI}}}+\tilde{{{IV}}})$',
}

# Color mapping for methods
COLOR_MAP: Dict[str, int] = {
    'ATE': 3,
    'ERM': 0,
    'DA+ERM': 3,
    'DA+IV': 2,
    'PI_INV': 7,
    'PI': 0,
    'PI_IV': 1,
    'DA+PI': 3,
    'DA+PI_IV': 2,
    'PI&DA+PI': 4,
    'PI&DA+PI_IV': 6,
}

# Alpha (transparency) mapping for methods
ALPHA_MAP: Dict[str, float] = {
    # Point identification methods (solid)
    'ATE': 1.0,
    'ERM': 1.0,
    'DA+ERM': 1.0,
    'DA+IV': 1.0,
    # Partial identification methods (transparent)
    'PI_INV': 0.8,
    'PI': 0.2,
    'PI_IV': 0.2,
    'DA+PI': 0.2,
    'DA+PI_IV': 0.4,
    'PI&DA+PI': 0.4,
    'PI&DA+PI_IV': 0.4,
}

# Visual style configuration
POINT_ESTIMATES: List[str] = ['ATE', 'ERM', 'DA+ERM', 'DA+IV']
POINT_ESTIMATE_STYLE: str | tuple[int, tuple[int, int]] = (0, (5, 1))
PARTIAL_IDENTIFICATION_STYLE: str | tuple[int, tuple[int, int]] = '-'

# Plotting defaults
DEFAULT_HILIGHT_OURS: bool = False
DEFAULT_NORMALIZE_ERROR: bool = False

# Panel Plot Specific Configurations
# Requirement 2: specification of y-axis clip values for Row 4 (index 3: h^T x)
PANEL_PREDICTION_LIMITS: Dict[str, tuple[float, float]] = {
    'simulation': (-3, 3),
    'optical_device': (-2.625, 3.375),
}

# Requirement 3: specification of scale (log vs linear) for rows 1 and 2 (indices 0 and 1)
PANEL_Y_SCALES: Dict[str, Literal['linear', 'log']] = {
    'worst_error': 'asinh',  # Row 1 (index 0)
    'width': 'asinh'         # Row 2 (index 1)
}



# src/experiments/utils/constants.py

# Configuration for panel plots per experiment and per row
# Row index mapping: 0: Worst Error, 1: Width, 2: Density, 3: Predictions
PANEL_CONFIGS = {
    'simulation': {
        0: {'scale': 'asinh', 'ylim': (0, 10.01), 'linear_width': 0.25, 'linthresh': 2.0},
        1: {'scale': 'asinh', 'ylim': (0, 10.01), 'linear_width': 0.25, 'linthresh': 2.0},
        2: {'scale': 'linear', 'ylim': (0, 0.5), 'linear_width': 1.0, 'linthresh': 2.0},
        3: {'ylim': (-3, 3)}
    },
    'optical_device': {
        0: {'scale': 'asinh', 'ylim': (0, 100), 'linear_width': 0.15, 'linthresh': 2.0},
        1: {'scale': 'asinh', 'ylim': (0, 10.01), 'linear_width': 0.25, 'linthresh': 2.0},
        2: {'scale': 'linear', 'ylim': (0, 0.01), 'linear_width': 1.0, 'linthresh': 2.0},
        3: {'ylim': (-2.625, 3.375)}
    }
}