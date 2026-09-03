import os

import fsspec
import numpy as np
from loguru import logger
from numpy.typing import NDArray

from src.sem.abstract import StructuralEquationModel as SEM
from src.sem.utils import (
    fit_ground_truth_f,
    select_best_degree,
)

# change to 5 if verification needed -- best fit is only either 1 or 2.
# kept at 2 for faster runtimes in generating experiment results/plots.
MAX_PLOYNOMIAL_DEGREE: int = 2


class OpticalDeviceSEM(SEM):
    @staticmethod
    def load_dataset(directory: str = "data/optical_device") -> dict[int, NDArray]:

        def download_dataset(directory: str):
            fs = fsspec.filesystem("github", org="janzing", repo="janzing.github.io")
            fs.get(fs.ls("code/data_from_optical_device"), directory)

        def dataset_exists(directory: str) -> bool:
            if not os.path.isdir(directory):
                return False
            files = [f for f in os.listdir(directory) if "confounder" in f and "random" not in f]
            return len(files) > 0

        if not dataset_exists(directory):
            download_dataset(directory)

        file_list = sorted([f for f in os.listdir(directory) if "confounder" in f and "random" not in f])

        dataset = {}
        for experiment, file_name in enumerate(file_list):
            dataset[experiment] = np.genfromtxt(f"{directory}/{file_name}", delimiter=" ")
        return dataset

    _DATASET: dict[int, NDArray] | None = None  # filled on first use, not on import

    @classmethod
    def dataset(cls) -> dict[int, NDArray]:
        """The cached pool, loaded on FIRST USE.

        `load_dataset` downloads from a remote when the directory is missing, so
        binding it at class-definition time made merely importing this module --
        or anything that transitively imports it -- reach for the network. It also
        made the module unimportable in a fresh checkout with no network, which is
        how a worktree of an earlier commit fails.
        """
        if cls._DATASET is None:
            cls._DATASET = cls.load_dataset()
        return cls._DATASET

    def __init__(self, experiment: int = 0, center: bool = True, ground_truth: str = "polynomial"):
        # a COPY: `get_experiment_data` hands back the shared class cache, and the
        # centring below is in place. Without this, constructing a SEM mutates the
        # pool every other SEM will be built from -- repeated construction drifts
        # the fit, and a `center=False` SEM built after a `center=True` one gets
        # silently centred data.
        experiment_data = np.array(self.get_experiment_data(experiment), dtype=float, copy=True)

        if center:
            experiment_data -= experiment_data.mean(axis=0)

        y = experiment_data[:, -1:]  # outcome
        XC = experiment_data[:, :-1]  # treatment and confounder
        X = XC[:, :-1]  # treatment
        C = XC[:, -1:]  # confounder

        best_degree = 1
        if ground_truth == "linear":
            W_XY, b_XY, features, epsilon = fit_ground_truth_f(X, y, C, 1)
        elif ground_truth == "polynomial":
            best_degree, _ = select_best_degree(X, y, C, max_degree=MAX_PLOYNOMIAL_DEGREE)
            logger.info(f"Experiment {experiment} polynomial degree: {best_degree}")
            W_XY, b_XY, features, epsilon = fit_ground_truth_f(X, y, C, best_degree)
        else:
            raise ValueError(f"Ground truth {ground_truth} model not supported/implemented.")

        # 1. Calculate original noise statistics
        # epsilon is the coefficient of C. Noise xi = epsilon * C.
        original_varXi = np.var(epsilon * C)

        # 2. Calculate the scaling factor (sigma_xi)
        noise_scale = np.sqrt(original_varXi)

        # 3. Scale Y and the Ground Truth weights
        # Y_new = Y_old / sigma
        # f_new(X) = f_old(X) / sigma
        self.y = y / noise_scale
        self.W_XY = W_XY / noise_scale
        self.b_XY = b_XY / noise_scale
        self.X = X
        self.C = C
        self.poly_degree = best_degree

        # 4. Normalized variances. Var(xi_new) = Var(xi_old) / sigma^2 = 1.0
        # Bias lives in the feature space PI is fit on, i.e. phi(X), not X -- and
        # in the class Asm. 1 actually describes, which is closed under constant
        # shifts, so the projection is onto span(phi, 1) and NOT span(phi). Same
        # class the solver searches under `mean_match`; measuring gamma* over one
        # class while solving over another is what let h_* fall out of the set.
        Phi = features.fit_transform(self.X)
        design = np.column_stack([Phi, np.ones(len(Phi))])
        xi_hat = design @ np.linalg.pinv(design) @ (epsilon * self.C)
        self._bias_sq = float(np.var(xi_hat) / (noise_scale**2))

        # sigma^2 = E[Var(Y|X)] carries BOTH parts of the conditional spread: what
        # the confounder leaves unexplained, E[Var(U|X)] = Var(U) - Var(E[U|X]) =
        # 1 - bias_sq after the normalisation above, AND the exogenous noise -- the
        # residual of the joint (phi, C) fit, which is the paper's xi. Dropping the
        # second understates sigma^2 and so OVERSTATES gamma* = bias_sq/sigma^2,
        # making the oracle budget loose rather than tight (it is documented as
        # "tightest gamma keeping h_* inside"). Small on the shipped device (1.8%,
        # gamma* 0.6738 -> 0.6619) and enormous where the device is noisy
        # (experiment 6: 801x, gamma* 0.0510 -> 0.0001).
        self._noise_sq = float(np.var(self.y.ravel() - self.f(Phi).ravel() - (epsilon / noise_scale) * self.C.ravel()))

    def f(self, X) -> NDArray:
        """Ground truth on the FEATURE space: f(phi) = phi W + b.

        `X` is already phi(x) at every call site. The intercept is part of the
        estimand (see `fit_ground_truth_f`); without it h_* is off Lem. 2's slice.
        """
        return X @ self.W_XY + self.b_XY

    @property
    def bias_sq(self) -> float:
        return self._bias_sq

    @property
    def sigma_sq(self) -> float:
        # E[Var(U|X)] = 1 - bias_sq after the normalisation, PLUS the exogenous
        # noise; see `_noise_sq`.
        return 1.0 - self._bias_sq + self._noise_sq

    @property
    def pool(self) -> tuple[NDArray, NDArray]:
        """The recorded rows themselves -- see `SEM.pool`."""
        return self.X, self.y

    def sample(self, N: int = 1, **kwargs) -> tuple[NDArray, NDArray]:
        N_max, M = self.X.shape
        indices = np.arange(N_max)
        replace = N_max < N
        if replace:
            # not an error -- the sweeps legitimately ask for more rows than the
            # device recorded -- but the draw is then a BOOTSTRAP, so anything
            # read off it carries resampling noise on top of the device's own
            logger.debug(f"OpticalDeviceSEM: {N} rows requested from a pool of {N_max}; resampling with replacement.")
        sampled = np.random.choice(indices, N, replace)
        return self.X[sampled], self.y[sampled]

    @classmethod
    def get_experiment_data(cls, n: int) -> NDArray:
        return cls.dataset()[n]

    @classmethod
    def num_experiments(cls) -> int:
        return len(cls.dataset())
