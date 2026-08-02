"""
Optical device experiment using generic runners.
Dramatically reduced code duplication.
"""
from typing import Type, List, Tuple, Dict
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

from src.data_augmentors.optical_device import OpticalDeviceDA as DA
from src.sem.optical_device import OpticalDeviceSEM as SEM
from src.experiments.base import ExperimentOrchestrator
from src.experiments.generic_runner import (
    GenericQuerySweep,
    GammaSweep,
    AugmentationFoldSweep,
    GenericParamSweep
)
from src.experiments.configs import (
    MethodRegistry,
    OPTICAL_CONFIG,
    SWEEP_CONFIGS
)
from src.experiments.utils.metrics import trace_S_over_k
from src.oracle import gamma_star

EXPERIMENT_NAME = 'optical_device'


# =============================================================================
# ORCHESTRATOR
# =============================================================================

class OpticalOrchestrator(ExperimentOrchestrator):
    """Orchestrator for optical device experiments."""
    
    def __init__(self, augmentation: str, **kwargs):
        """
        Initialize optical orchestrator.
        """
        self.augmentation = augmentation
        toggles = dict(
            calibrate=kwargs.get('calibrate', False),
            pad=kwargs.get('pad', False),
            clipy=kwargs.get('clipy', True),
        )

        # Create registry with optical-specific parameters
        class OpticalRegistry(MethodRegistry):
            @staticmethod
            def build_methods(names):
                return MethodRegistry.build_methods(
                    names,
                    gamma=OPTICAL_CONFIG.gamma,
                    epsilon=OPTICAL_CONFIG.epsilon,
                    **toggles
                )
        
        super().__init__(EXPERIMENT_NAME, OpticalRegistry(), **kwargs)
    
    def _sem_factory(self):
        """Factory for creating SEM instances."""
        return SEM(
            experiment=OPTICAL_CONFIG.dataset_index,
            ground_truth=OPTICAL_CONFIG.ground_truth_model
        )
    
    def _da_factory(self, sem=None):
        """Factory for creating DA instances."""
        return DA(self.augmentation)
    
    def _poly_factory(self):
        """Factory for creating polynomial transformer."""
        # Get degree from a sample SEM
        sem = self._sem_factory()
        return PolynomialFeatures(sem.poly_degree, include_bias=False)
    
    def get_query_runner_cls(self) -> Type[GenericQuerySweep]:
        """Return query sweep runner."""
        class OpticalQuerySweep(GenericQuerySweep):
            def __init__(inner_self, **kwargs):
                super().__init__(
                    sem_factory=self._sem_factory,
                    da_factory=self._da_factory,
                    poly_transform=self._poly_factory(),
                    epsilon_true=OPTICAL_CONFIG.epsilon_true,
                    **kwargs
                )

        return OpticalQuerySweep
    
    def get_param_sweeps(self) -> List[Tuple[Type, str]]:
        """Return parameter sweeps to run."""
        
        # 1. Standard Gamma Sweep
        class OpticalGammaSweep(GammaSweep):
            def __init__(inner_self, **kwargs):
                super().__init__(
                    sem_factory=self._sem_factory,
                    da_factory=self._da_factory,
                    poly_transform=self._poly_factory(),
                    test_fraction=OPTICAL_CONFIG.test_fraction,
                    sweep_config=SWEEP_CONFIGS['optical_device']['gamma'],
                    epsilon_true=OPTICAL_CONFIG.epsilon_true,
                    use_train_test_split=True,
                    **kwargs
                )

        # 2. Augmentation Folds Sweep
        class OpticalFoldSweep(AugmentationFoldSweep):
            def __init__(inner_self, **kwargs):
                super().__init__(
                    sem_factory=self._sem_factory,
                    da_factory=self._da_factory,
                    poly_transform=self._poly_factory(),
                    test_fraction=OPTICAL_CONFIG.test_fraction,
                    sweep_config=SWEEP_CONFIGS['optical_device']['folds'],
                    epsilon_true=OPTICAL_CONFIG.epsilon_true,
                    **kwargs
                )

        # 3. gamma* (Dataset) Sweep
        class OpticalGammaStarSweep(GenericParamSweep):
            def __init__(inner_self, **kwargs):
                # Manually extract config
                config = SWEEP_CONFIGS['optical_device']['gamma_star']

                super().__init__(
                    sem_factory=self._sem_factory,
                    da_factory=self._da_factory,
                    poly_transform=self._poly_factory(),
                    test_fraction=OPTICAL_CONFIG.test_fraction,
                    epsilon_true=OPTICAL_CONFIG.epsilon_true,
                    sweep_config=config,
                    **kwargs
                )

                # Attach to THIS runner instance
                inner_self.sweep_config = config
                inner_self.gamma_stars = []

            @property
            def data_depends_on_param(inner_self) -> bool:
                return True 

            def get_param_range(inner_self) -> np.ndarray:
                return inner_self.sweep_config.range_fn(12) 

            def generate_data(inner_self, experiment_index: int, dataset_idx):
                dataset_idx = int(dataset_idx)
                
                # FRESH SEM for this specific dataset
                sem = SEM(experiment=dataset_idx, ground_truth=OPTICAL_CONFIG.ground_truth_model)
                
                # Oracle confounding budget of this dataset
                gamma_star_est = gamma_star(sem, calibrate=inner_self.calibrate)

                # Store gamma* (once per dataset index)
                if experiment_index == 0:
                    inner_self.gamma_stars.append((dataset_idx, gamma_star_est))

                # Load Data (DA is dataset-independent, reuse the prepared one)
                da = inner_self.get_da(experiment_index)
                X_raw, y = sem(N=inner_self.n_samples) 
                GX_raw, G = da(X_raw)
                
                # Transform
                # FIX: Use the specific degree from the current SEM, not the global default
                poly = PolynomialFeatures(sem.poly_degree, include_bias=False)
                X = poly.fit_transform(X_raw)
                GX = poly.fit_transform(GX_raw)
                
                X_train, X_test, y_train, _, GX_train, _, G_train, _ = train_test_split(
                    X, y, GX, G,
                    test_size=inner_self.test_fraction,
                    random_state=inner_self.seed + experiment_index
                )
                
                estimand = X_test @ sem.solution
                return X_train, y_train, GX_train, G_train, X_test, estimand

            def run(inner_self, desc: str):
                """Sort results by gamma* after running."""
                # Run standard sweep
                indices, raw_results = super().run(desc)

                # Sort by calculated gamma* (value is at index 1)
                inner_self.gamma_stars.sort(key=lambda x: x[1])

                # Extract sorted values and dataset indices
                sorted_k_values = np.array([g for _, g in inner_self.gamma_stars])
                sorted_dataset_indices = [idx for idx, _ in inner_self.gamma_stars]
                
                sorted_results = {}
                for method, matrix in raw_results.items():
                    # We map row i to sorted position
                    new_matrix = np.zeros_like(matrix)
                    
                    # The param range was 0..11
                    for new_row_idx, original_dataset_idx in enumerate(sorted_dataset_indices):
                         new_matrix[new_row_idx] = matrix[original_dataset_idx]
                    
                    sorted_results[method] = new_matrix
                
                return sorted_k_values, sorted_results

        # 4. tr(S)/k Sweep (Augmentation Strength)
        class OpticalTraceSSweep(GenericParamSweep):
            def __init__(inner_self, **kwargs):
                # Manually extract config
                config = SWEEP_CONFIGS['optical_device']['trS']
                inner_self.sweep_config = config

                super().__init__(
                    sem_factory=self._sem_factory,
                    da_factory=self._da_factory,
                    poly_transform=self._poly_factory(),
                    test_fraction=OPTICAL_CONFIG.test_fraction,
                    **kwargs
                )
                inner_self.strength_values = []

            @property
            def data_depends_on_param(inner_self) -> bool:
                return True

            def get_param_range(inner_self) -> np.ndarray:
                return inner_self.sweep_config.range_fn(inner_self.sweep_samples)

            def run(inner_self, desc: str = "tr(S)/k Sweep") -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
                """Run sweep and return measured strength values as x-axis."""
                inner_self.strength_values = []
                _, results = super().run(desc)
                
                # Average strengths across experiments
                strength_array = np.array(inner_self.strength_values).reshape(inner_self.n_experiments, inner_self.sweep_samples)
                mean_strength = strength_array.mean(axis=0)
                
                return mean_strength, results

            def generate_data(inner_self, experiment_index: int, strength: float):
                # Map strength [0, 1] to specific parameters
                
                # 1. Flip/Rotation Probability: 0.0 to 0.5
                p = 0.5 * strength
                
                # 2. Gaussian Noise Coeff: 0.0 to sqrt(0.1)
                # Matches target variance of 0.0 to 0.1 * Var(X)
                noise_coeff = np.sqrt(0.1 * strength)

                # FIX: use inner_self to access GenericParamSweep's stored list of SEMs/DAs
                sem = inner_self.sems[experiment_index]
                da = inner_self.das[experiment_index]
                
                # Load Raw Data
                X_raw, y = sem(N=inner_self.n_samples) 
                
                # Augment with dynamic strength parameters
                GX_raw, G = da(X_raw, p=p, noise_coeff=noise_coeff)

                # Transform using the runner's poly transformer
                X = inner_self.apply_transform(X_raw)
                GX = inner_self.apply_transform(GX_raw)

                # tr(S)/k in the feature space the methods use
                inner_self.strength_values.append(trace_S_over_k(X, GX))
                
                X_train, X_test, y_train, _, GX_train, _, G_train, _ = train_test_split(
                    X, y, GX, G,
                    test_size=inner_self.test_fraction,
                    random_state=inner_self.seed + experiment_index
                )
                
                estimand = X_test @ sem.solution
                return X_train, y_train, GX_train, G_train, X_test, estimand

        return [
            (OpticalGammaSweep, 'gamma'),
            (OpticalFoldSweep, 'folds'),
            (OpticalGammaStarSweep, 'gamma_star'),
            (OpticalTraceSSweep, 'trS')
        ]