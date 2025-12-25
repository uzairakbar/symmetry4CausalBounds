"""
Experiments utilities package.
Provides metrics, plotting, data operations, model fitting, and panels.
"""

from .constants import *
from .metrics import *
from .plotting import *
from .data_operations import *
from .model_fitting import *
from .panels import *

__all__ = [
    # Constants
    'RC_PARAMS', 'TEX_MAPPER', 'COLOR_MAP', 'ALPHA_MAP',
    'FS_TICK', 'FS_LABEL', 'PLOT_DPI', 'PLOT_FORMAT',
    'ARTIFACTS_DIRECTORY',
    
    # Metrics
    'estimation_error', 'approximation_error', 
    'worst_error', 'interval_width',
    
    # Plotting
    'create_param_sweep_plot', 'create_query_sweep_plot',
    'create_panel_plot',
    
    # Data operations
    'set_seed', 'bootstrap', 'save', 'load',
    'radial_sweep_pcs', 'sweep_along_pc', 'project_onto_pc',
    
    # Model fitting
    'fit_model',
    
    # Panels
    'PanelBuilder',
]