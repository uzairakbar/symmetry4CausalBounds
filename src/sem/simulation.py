########################################
#
#  Linear Gaussian SEM
#
########################################


import numpy as np
from numpy.typing import NDArray

from src.sem.abstract import StructuralEquationModel as SEM

OUTCOME_DIMENSION: int = 1
TREATMENT_DIMENSION: int = 32  # the root yaml's `treatment_dim` fallback (DatasetDefaults)
OUTCOME_NOISE_STD: float = 0.1  # s: unconfoundable outcome noise
MAX_GAMMA: float = 1.0 / OUTCOME_NOISE_STD**2
# the instrument's loading, A = alpha U. At 1 the instrument owns range(A)
# outright and X U = Z exactly; below it V still carries a share of those
# directions. Fixed: the run configures the WIDTH m, not the strength.
IV_ALPHA: float = 1.0


class LinearSimulationSEM(SEM):
    """
    SEM:
        X ~ N(0, I)
        xi = kappa * e^T X + sqrt(1 - kappa^2) * eps
        Y = f^T X + xi + s * nu

    Confounding is set by the true budget `gamma` (paper's calibrated gamma,
    i.e. bias^2/sigma^2 = kappa^2 / (1 - kappa^2 + s^2)), inverted as
    kappa^2 = gamma (1 + s^2) / (1 + gamma) so that gamma* == gamma exactly.
    `gamma=None` means kappa=1 (fully confounded, sigma^2 = s^2, gamma = 1/s^2).

    With `iv_dim` = m > 0 a standard-Gaussian instrument reaches X through a
    rank-m map (SS5.1):
        U    orthonormal d x m, drawn at construction, with U' e = 0
        A    = alpha U
        Z    ~ N(0, I_m)                         per draw
        X    = Z A' + V,  V ~ N(0, I - A A')     so Var(X) = I, unchanged
    and the draw comes back as [X | Z], the last m columns being the
    instrument (`iv_width`). `U' e = 0` makes Cov(Z, xi) = 0 exactly, a valid
    instrument; rank m << d makes it incomplete: it identifies m of d
    directions and nothing else. The confounding, `f` and the budgets are
    untouched, so gamma* == gamma still. At m = 0 nothing is drawn and every
    number is the one shipped before the instrument existed.
    """

    def __init__(
        self,
        outcome_dimension: int = OUTCOME_DIMENSION,
        treatment_dimension: int = TREATMENT_DIMENSION,
        gamma: float | None = None,
        iv_dim: int = 0,
        alpha: float = IV_ALPHA,
    ):
        self.outcome_dimension = outcome_dimension
        self.treatment_dimension = treatment_dimension

        # Confounding direction W_XXi ~ N(0, I), normalized
        W_XXi = np.random.randn(treatment_dimension)
        self.W_XXi = W_XXi / np.linalg.norm(W_XXi)

        # Structural coefficients f ~ N(0, I)
        self.W_XY = np.random.randn(treatment_dimension, outcome_dimension)

        self.gamma = gamma

        # the instrument map, drawn AFTER W_XXi and W_XY and only when asked for:
        # the orchestrator builds n_experiments SEMs off one stream, so an
        # unconditional draw here would shift every later SEM's f
        self.iv_dim = int(iv_dim)
        self.alpha = float(alpha)
        if self.iv_dim < 0 or self.iv_dim > treatment_dimension:
            raise ValueError(f"`iv_dim` must be in [0, {treatment_dimension}]; got {iv_dim!r}.")
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError(f"`alpha` must be in (0, 1]; got {alpha!r}.")
        if self.iv_dim > 0:
            basis = np.random.randn(treatment_dimension, self.iv_dim)
            # kill the confounding direction, then orthonormalise: U' W_XXi = 0
            basis -= np.outer(self.W_XXi, self.W_XXi @ basis)
            self.U = np.linalg.qr(basis)[0]
            self.A = self.alpha * self.U
            # the exact symmetric root of I - A A': (I - (1 - sqrt(1 - alpha^2)) U U')
            self.root = np.eye(treatment_dimension) - (1.0 - np.sqrt(1.0 - self.alpha**2)) * (self.U @ self.U.T)
        super().__init__()

    # ---------------------------------------------------------- instruments

    @property
    def iv_width(self) -> int:
        return self.iv_dim

    # ---------------------------------------------------------- confounding

    @property
    def gamma(self) -> float | None:
        """True (calibrated) confounding budget; None = inf."""
        return self._gamma

    @gamma.setter
    def gamma(self, gamma: float | None):
        if gamma is not None:
            if gamma < 0.0:
                raise ValueError("`gamma` must be non-negative.")
            if gamma > MAX_GAMMA:
                raise ValueError(f"`gamma` must be <= 1/s^2 = {MAX_GAMMA}.")
        self._gamma = gamma

    @property
    def kappa(self) -> float:
        """kappa^2 = gamma (1 + s^2) / (1 + gamma), in [0, 1]."""
        if self._gamma is None:
            return 1.0
        s_sq = OUTCOME_NOISE_STD**2
        kappa_sq = self._gamma * (1.0 + s_sq) / (1.0 + self._gamma)
        return float(np.sqrt(min(kappa_sq, 1.0)))  # guard gamma == MAX_GAMMA

    @property
    def bias_sq(self) -> float:
        return self.kappa**2

    @property
    def sigma_sq(self) -> float:
        return 1.0 - self.kappa**2 + OUTCOME_NOISE_STD**2

    # -------------------------------------------------------------- sampling

    def _draw_treatment(self, N: int) -> tuple[NDArray, NDArray | None]:
        """(X, Z): X ~ N(0, I) directly, or through the instrument when there is one."""
        if self.iv_dim == 0:
            return np.random.randn(N, self.treatment_dimension), None
        Z = np.random.randn(N, self.iv_dim)
        return Z @ self.A.T + np.random.randn(N, self.treatment_dimension) @ self.root, Z

    def sample(
        self, N: int = 1, gamma: float | None = None, intervention: bool = False, **kwargs
    ) -> tuple[NDArray, NDArray]:
        if gamma is not None:
            self.gamma = gamma
        kappa = self.kappa

        # 1. Sample the underlying treatment X (and the instrument behind it)
        X, Z = self._draw_treatment(N)

        # 2. Sample the independent error eps
        eps = np.random.randn(N, self.outcome_dimension)

        if intervention:
            # do(X): the "true" xi must be independent of the intervened X.
            # Sample a phantom X_confound for the xi that 'would have' happened.
            X_confound, _ = self._draw_treatment(N)
            xi = kappa * (X_confound @ self.W_XXi).reshape(-1, 1) + np.sqrt(1 - kappa**2) * eps
        else:
            # Observational: xi is explicitly tied to X
            xi = kappa * (X @ self.W_XXi).reshape(-1, 1) + np.sqrt(1 - kappa**2) * eps

        # 3. Y is generated by the invariant f + the confounding xi + noise
        Y = X @ self.W_XY + xi + OUTCOME_NOISE_STD * np.random.randn(N, self.outcome_dimension)

        # the instrument rides as the trailing columns, on the interventional draw
        # too, so train and test have one width and the runner splits both alike
        if Z is not None:
            X = np.column_stack([X, Z])
        return X, Y
