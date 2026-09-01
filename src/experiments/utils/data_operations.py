"""
Data manipulation and I/O utilities.
"""
import os
import json
import pickle
import random
import typing
import warnings
import torch
import numpy as np
from loguru import logger
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from typing import Any, Dict, Literal, Tuple, Optional

from .constants import ARTIFACTS_DIRECTORY, TEX_MAPPER

PlotFormat = Literal['png', 'pdf', 'ps', 'eps', 'svg']
ExperimentType = Literal['simulation', 'optical_device', 'colored_mnist', 'rotated_mnist']


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility across numpy, random, and torch.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    logger.info(f'Random seed set as {seed}.')


# Bootstrap draws its own stream: off the global one the band moved run to run, and
# the sweep limits/linear_width are derived FROM it.
BOOTSTRAP_SEED: int = 0


def bootstrap(
    data: Dict[str, NDArray] | Dict[str, Dict[str, NDArray]],
    n_samples: int = 1000,
    seed: int = BOOTSTRAP_SEED,
) -> Dict:
    """
    Generate bootstrap samples from data.

    Args:
        data: Dictionary of arrays or nested dictionary
        n_samples: Number of bootstrap samples to generate
        seed: resample stream; fixed so the CI band is reproducible

    Returns:
        Bootstrapped data with same structure as input
    """
    rng = np.random.default_rng(seed)

    def _bootstrap_single_dict(data_dict: Dict[str, NDArray], n_bootstrap: int = n_samples):
        """Bootstrap a single level dictionary."""
        # Ensure 2D arrays
        data_dict = {
            key: value.copy().reshape(1, -1) if len(value.shape) == 1 else value.copy()
            for key, value in data_dict.items()
        }

        def _bootstrap_sample(array: NDArray, n_boot: Optional[int] = None) -> NDArray:
            """Generate one bootstrap sample."""
            if len(array.shape) == 1:
                array = array.reshape(1, -1)
            if n_boot is None:
                n_boot = array.shape[-1]

            n_rows, n_cols = array.shape
            indices = rng.integers(0, n_cols, (n_rows, n_boot))
            return np.take_along_axis(array, indices, axis=1)

        # Generate bootstrap samples
        bootstrapped = {
            model: np.zeros((data_dict[model].shape[0], n_samples))
            for model in data_dict
        }

        # nanmean, not mean: one NaN replicate in a resample used to NaN the whole
        # bootstrap draw, so a method with a finite mean at a step was drawn as a gap.
        # An all-NaN step still comes back NaN -- that one is honest.
        for model in data_dict:
            for i in range(n_samples):
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    bootstrapped[model][:, i] = np.nanmean(
                        _bootstrap_sample(data_dict[model]), axis=1)

        return bootstrapped
    
    # Check if top-level keys are method names (single row) or experiment names (nested)
    is_single_level = set(data.keys()) <= set(TEX_MAPPER.keys())
    
    if is_single_level:
        return _bootstrap_single_dict(data)
    else:
        return {key: _bootstrap_single_dict(data[key]) for key in data}


def _json_default(obj: Any):
    """JSON serialization helper for numpy types."""
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    raise TypeError(f'Unknown type: {type(obj)}.')


def save(
    obj: Any,
    fname: str,
    experiment: ExperimentType,
    format: PlotFormat | Literal['pkl', 'json', 'tex'],
    subdir: Optional[str] = None,
    **kwargs
):
    """
    Save object to file with appropriate serialization.

    Args:
        obj: Object to save (figure, dict, array, etc.)
        fname: Filename without extension
        experiment: Experiment name for organizing artifacts
        format: File format ('pkl', 'json', 'tex', or plot format)
        subdir: experiment-type folder ('query'/'sweep'/'scatter'/'perf'),
            so one run's output does not pile into a single directory
        **kwargs: Additional arguments for format-specific saving
    """
    save_path = f'{ARTIFACTS_DIRECTORY}/{experiment}'
    if subdir:
        save_path = f'{save_path}/{subdir}'

    os.makedirs(save_path, exist_ok=True)

    full_path = f'{save_path}/{fname}.{format}'
    
    try:
        if format == 'pkl':
            with open(full_path, 'wb+') as file:
                pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)
        
        elif format == 'json':
            with open(full_path, 'w+') as file:
                json.dump(
                    obj, file,
                    separators=(',', ':'),
                    sort_keys=True,
                    indent=4,
                    default=_json_default
                )
        
        elif format == 'tex':
            with open(full_path, 'w+') as file:
                file.write(obj)
        
        elif format in typing.get_args(PlotFormat):
            obj.savefig(full_path, format=format, **kwargs)
        
        else:
            raise NotImplementedError(f'Save not implemented for {format} file.')
        
        logger.info(f'Saved file {fname}.{format} at path {save_path}.')
    
    except Exception as e:
        logger.error(f'Could not save file {fname}.{format} at path {save_path}.')
        raise e


def load(path: str):
    """
    Load data from pickle or JSON file.
    
    Args:
        path: Full path to file
        
    Returns:
        Loaded data
    """
    if not os.path.exists(path):
        raise ValueError(f'Path {path} does not exist.')
    
    file_format = path.split('.')[-1]
    assert file_format in ['pkl', 'json'], \
        f'Incorrect format {file_format}, can only accept pkl or json.'
    
    try:
        if file_format == 'pkl':
            with open(path, 'rb') as file:
                data = pickle.load(file)  # noqa: S301 - our own artifacts, no untrusted input
        else:  # json
            with open(path, 'r') as file:
                data = json.load(file)
        
        logger.info(f'Loaded data from file {path}.')
        return data
    
    except Exception as e:
        logger.error(f'Could not load data from file {path}.')
        raise e


def radial_sweep_pcs(X: NDArray, n_points: int = 100) -> NDArray:
    """
    Generate sweep points on a circle along first 2 principal components.
    
    Creates points on a 1-standard-deviation circle in PC space,
    then maps them back to original feature space.
    
    Args:
        X: Input data, shape (n_samples, n_features)
        n_points: Number of points around the circle
        
    Returns:
        Sweep points in original space, shape (n_points, n_features)
    """
    # Center and perform PCA
    X_centered = X - X.mean(axis=0)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_centered)
    
    # Get standard deviations along first 2 PCs
    std_pc1, std_pc2 = X_pca.std(axis=0)
    
    # Create circle in PC space
    angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    circle_pc_space = np.column_stack([
        std_pc1 * np.cos(angles),
        std_pc2 * np.sin(angles)
    ])
    
    # Map back to original space
    sweep_points = circle_pc_space @ pca.components_ + X.mean(axis=0)
    
    return sweep_points


def sweep_along_pc(
    X: NDArray,
    pc_index: int = 0,
    n_steps: int = 21,
    std_range: float = 1.0
) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Generate sweep along a specified principal component.
    
    Args:
        X: Input data, shape (n_samples, n_features)
        pc_index: Which principal component to use (0-based)
        n_steps: Number of points along the sweep
        std_range: How many standard deviations in each direction
        
    Returns:
        Tuple of:
        - sweep_points: Points along PC, shape (n_steps, n_features)
        - t_values: Scaling factors along PC, shape (n_steps,)
        - mean: Data mean, shape (n_features,)
        - pc_vector: Unit vector of chosen PC, shape (n_features,)
    """
    # Center data and compute PCA via SVD
    mean = np.mean(X, axis=0)
    X_centered = X - mean
    
    _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
    principal_components = Vt  # Rows are PCs
    pc_vector = principal_components[pc_index]
    
    # Compute standard deviation along this component
    projections = X_centered @ pc_vector
    std_dev = np.std(projections)
    
    # Generate sweep points
    t_values = np.linspace(-std_range * std_dev, std_range * std_dev, n_steps)
    sweep_points = np.outer(t_values, pc_vector)
    
    return sweep_points, t_values, mean, pc_vector


def project_onto_pc(
    X: NDArray,
    pc_vector: NDArray,
    mean: Optional[NDArray] = None
) -> Tuple[NDArray, NDArray]:
    """
    Project all points onto a specified principal component.
    
    Args:
        X: Input data, shape (n_samples, n_features)
        pc_vector: Unit vector of PC, shape (n_features,)
        mean: Optional data mean (computed if not provided)
        
    Returns:
        Tuple of:
        - projections: Projected points, shape (n_samples, n_features)
        - t_values: Scaling along pc_vector, shape (n_samples,)
    """
    if mean is None:
        mean = np.mean(X, axis=0)
    
    X_centered = X - mean
    t_values = X_centered @ pc_vector
    projections = mean + np.outer(t_values, pc_vector)
    
    return projections, t_values