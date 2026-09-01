"""
Panel plot utilities for 4x3 visualization panels.
"""


import numpy as np

from src.experiments.utils.data_operations import radial_sweep_pcs, sweep_along_pc
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
        self.fitted_models = {}  # Cache fitted models
        self.radial_results = None  # Cache radial sweep results

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

        # Fit models ONCE using radial sweep (standard query sweep)
        self._fit_all_models()

        # Run predictions on different geometries (reusing fitted models)
        results_pc1, ground_truth_pc1 = self._predict_sweep(pc1_points)
        results_pc2, ground_truth_pc2 = self._predict_sweep(pc2_points)
        results_radial, ground_truth_radial = self._predict_sweep(radial_points)

        # Cache radial results for later use
        self.radial_results = results_radial

        # Build histogram data (ALWAYS compare X vs GX)
        histogram_data = self._build_histogram_data(self.runner.X_raw, self.runner.GX_raw, mean, pc1_vector, pc2_vector)

        # Define plot grids using same geometry as PC calculations
        std_pc1 = np.std((geometry_for_pcs - mean) @ pc1_vector)
        std_pc2 = np.std((geometry_for_pcs - mean) @ pc2_vector)

        t_values_pc1 = np.linspace(-3 * std_pc1, 3 * std_pc1, sweep_samples)
        t_values_pc2 = np.linspace(-3 * std_pc2, 3 * std_pc2, sweep_samples)
        theta_values = np.linspace(0, 2 * np.pi, sweep_samples, endpoint=False)

        # Organize column data: (results, ground_truth, x_axis)
        column_data = [
            (results_pc1, ground_truth_pc1, t_values_pc1),
            (results_radial, ground_truth_radial, theta_values),
            (results_pc2, ground_truth_pc2, t_values_pc2),
        ]

        # Generate panel plot
        create_panel_plot(self.experiment_name, column_data, histogram_data)

    def get_radial_results(self):
        """Return cached radial sweep results."""
        return self.radial_results

    def _fit_all_models(self):
        """Fit all models once and cache them."""
        from src.experiments.utils import fit_model

        context = self.runner.setup_data()

        for name, builder in self.runner.methods.items():
            if name == "ATE":
                self.fitted_models[name] = None  # ATE has no model
            else:
                model = builder()
                fit_model(
                    model=model,
                    method_name=name,
                    X=context.X,
                    y=context.y,
                    GX=context.GX,
                    G=context.G,
                    hyperparameters=self.runner.hyperparameters,
                    da=context.da,
                )
                self.fitted_models[name] = model

    def _predict_sweep(self, raw_points: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """
        Run predictions on specific geometry using cached models.

        Args:
            raw_points: Query points in raw feature space

        Returns:
            Tuple of (predictions_dict, ground_truth)
        """
        # Apply polynomial transformation if needed
        if hasattr(self.runner, "poly") and self.runner.poly is not None:
            transformed_points = self.runner.poly.fit_transform(raw_points)
        else:
            transformed_points = raw_points

        results = {}
        for name in self.runner.methods.keys():
            if name == "ATE":
                predictions = self.runner.sem.f(transformed_points)
            else:
                model = self.fitted_models[name]
                predictions = model.predict(transformed_points)

            # Reshape for consistent output format
            if "PI" in name:
                results[name] = predictions[:, np.newaxis, :]
            else:
                results[name] = predictions.reshape(len(transformed_points), 1)

        # Compute ground truth
        ground_truth = self.runner.sem.f(transformed_points)

        return results, ground_truth

    def _build_histogram_data(
        self,
        X_original: np.ndarray,
        GX_augmented: np.ndarray,
        mean: np.ndarray,
        pc1_vector: np.ndarray,
        pc2_vector: np.ndarray,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
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
            "pc1": (proj_X_pc1, proj_GX_pc1),
            "pc2": (proj_X_pc2, proj_GX_pc2),
        }
