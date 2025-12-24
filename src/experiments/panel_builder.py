"""
Panel plot builder for 4x3 visualization panels.
"""
import numpy as np
from typing import Dict, Tuple

from src.experiments.utils import radial_sweep_pcs, sweep_along_pc
from src.experiments.utils.plotting import create_panel_plot


class PanelBuilder:
    """Builds 4x3 panel plots for experiments."""
    
    def __init__(self, runner, experiment_name: str, use_augmented_geometry: bool = True):
        """
        Initialize panel builder.
        
        Args:
            runner: QuerySweepRunner instance with data already loaded
            experiment_name: Name for saving ('simulation', 'optical_device', etc.)
            use_augmented_geometry: Use GX_raw (True) or X_raw (False) for PC calculations
        """
        self.runner = runner
        self.experiment_name = experiment_name
        self.use_augmented = use_augmented_geometry
    
    def build(self, sweep_samples: int):
        """
        Generate complete 4x3 panel plot.
        
        Args:
            sweep_samples: Number of points in each sweep
        """
        # Choose geometry for PC calculations
        geometry_for_pcs = self.runner.GX_raw if self.use_augmented else self.runner.X_raw
        
        # Calculate sweep points along principal components
        pc1_points, _, mean, pc1_vector = sweep_along_pc(geometry_for_pcs, 0, sweep_samples, 3.0)
        pc2_points, _, _, pc2_vector = sweep_along_pc(geometry_for_pcs, 1, sweep_samples, 3.0)
        radial_points = radial_sweep_pcs(geometry_for_pcs, sweep_samples)
        
        # Run predictions on these geometries
        results_pc1, ground_truth_pc1 = self._run_sweep(pc1_points)
        results_pc2, ground_truth_pc2 = self._run_sweep(pc2_points)
        results_radial, ground_truth_radial = self._run_sweep(radial_points)
        
        # Build histogram data (ALWAYS compare X vs GX)
        histogram_data = self._build_histogram_data(
            self.runner.X_raw, self.runner.GX_raw, mean, pc1_vector, pc2_vector
        )
        
        # Define plot grids using same geometry as PC calculations
        std_pc1 = np.std((geometry_for_pcs - mean) @ pc1_vector)
        std_pc2 = np.std((geometry_for_pcs - mean) @ pc2_vector)
        
        t_values_pc1 = np.linspace(-3 * std_pc1, 3 * std_pc1, sweep_samples)
        t_values_pc2 = np.linspace(-3 * std_pc2, 3 * std_pc2, sweep_samples)
        theta_values = np.linspace(0, 2*np.pi, sweep_samples)
        
        # Organize column data: (results, ground_truth, x_axis)
        column_data = [
            (results_pc1, ground_truth_pc1, t_values_pc1),
            (results_radial, ground_truth_radial, theta_values),
            (results_pc2, ground_truth_pc2, t_values_pc2),
        ]
        
        # Generate panel plot
        create_panel_plot(self.experiment_name, column_data, histogram_data)
    
    def _run_sweep(self, raw_points: np.ndarray) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        Run models on specific geometry.
        
        Args:
            raw_points: Query points in raw feature space
            
        Returns:
            Tuple of (predictions_dict, ground_truth)
        """
        # Apply polynomial transformation if needed
        if hasattr(self.runner, 'poly'):
            transformed_points = self.runner.poly.fit_transform(raw_points)
        else:
            transformed_points = raw_points
        
        # Temporarily override sweep values
        original_get_sweep = self.runner.get_sweep_values
        self.runner.get_sweep_values = lambda: transformed_points
        
        _, results = self.runner.run(desc="Panel Sweep")
        
        # Restore original method
        self.runner.get_sweep_values = original_get_sweep
        
        # Compute ground truth
        ground_truth = transformed_points @ self.runner.sem.solution
        
        return results, ground_truth
    
    def _build_histogram_data(
        self,
        X_original: np.ndarray,
        GX_augmented: np.ndarray,
        mean: np.ndarray,
        pc1_vector: np.ndarray,
        pc2_vector: np.ndarray
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Build histogram projection data for original vs augmented data.
        
        Args:
            X_original: Original data
            GX_augmented: Augmented data
            mean: Data mean
            pc1_vector: First principal component
            pc2_vector: Second principal component
            
        Returns:
            Dictionary with 'pc1' and 'pc2' projections
        """
        # Project onto PC1
        proj_X_pc1 = (X_original - mean) @ pc1_vector
        proj_GX_pc1 = (GX_augmented - mean) @ pc1_vector
        
        # Project onto PC2
        proj_X_pc2 = (X_original - mean) @ pc2_vector
        proj_GX_pc2 = (GX_augmented - mean) @ pc2_vector
        
        return {
            'pc1': (proj_X_pc1, proj_GX_pc1),
            'pc2': (proj_X_pc2, proj_GX_pc2),
        }