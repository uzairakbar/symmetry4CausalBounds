"""US state cigarette demand: the panel loader, the FWL design and its IV diagnostics.

log q = b_p log p + b_y log y + b_n log p_n + b_P log CPI + state FE + trend(t),
every regressor NOMINAL. Homogeneity of degree zero is v'b = 0 with v = (1,1,1,1):
deflating by the CPI would impose it by construction, so the CPI enters as a
regressor and never as a deflator. Year FE are inadmissible for the same reason --
log CPI_t is collinear with them and b_P is absorbed.

numpy and the stdlib `csv` module only. pandas is not a dependency of this project
(it reaches the venv as a transitive of seaborn), and the work here is a long-to-wide
pivot, a melt, two joins and a groupby-min.
"""

import csv
import os
import subprocess
import sys
from dataclasses import dataclass

import numpy as np
from loguru import logger
from numpy.typing import NDArray

# raw sources and the built panel live under $SYM4CB_DATA_DIR (large files stay
# off the repo and off $HOME); `data/cigarettes` is a symlink at that directory
DATA_ROOT: str = "~/scratch/data"
DATASET: str = "cigarettes"
PANEL_DIR: str = "data/cigarettes"
PANEL_FILE: str = "panel.csv"
PANEL_COLUMNS: tuple[str, ...] = (
    "st",
    "year",
    "p",
    "q",
    "tax_s",
    "tax_fs",
    "tax_f",
    "y",
    "pop",
    "cpi",
    "pn",
    "pn_mean",
    "tax_sn",
    "tax_sn_mean",
)

# X = (log p, log y, log p_n, log CPI); the symmetry is x -> x + c v
TREATMENTS: tuple[str, ...] = ("p", "y", "pn", "cpi")
V: NDArray = np.ones(len(TREATMENTS))
OUTCOME: str = "q"

# state FE plus a polynomial trend of this order, t = (year - 1994) / 25
SPECS: dict[str, int] = {"s": 0, "t1": 1, "t2": 2, "t3": 3, "t4": 4}
TREND_CENTRE: float = 1994.0
TREND_SCALE: float = 25.0

# anchor -> (instrument columns, indices of X they instrument)
ANCHORS: dict[str, tuple[tuple[str, ...], tuple[int, ...]]] = {
    "own-tax": (("tax_s",), (0,)),
    "own-and-neighbour-tax": (("tax_s", "tax_sn"), (0, 2)),
}
# neighbour aggregator: `min` is Baltagi's bootlegging term, `mean` the robustness row
NEIGHBOURS: dict[str, tuple[str, str]] = {"min": ("pn", "tax_sn"), "mean": ("pn_mean", "tax_sn_mean")}


def data_directory() -> str:
    """Where the raw sources and the built panel go."""
    return os.path.join(os.path.expanduser(os.environ.get("SYM4CB_DATA_DIR") or DATA_ROOT), DATASET)


def null_basis(direction: NDArray = V) -> NDArray:
    """Orthonormal k x (k-1) basis N of null(v): the homogeneous subspace."""
    return np.linalg.svd(np.asarray(direction, dtype=float)[None, :])[2][1:].T


# =============================================================================
# LOADER
# =============================================================================


def load_panel(directory: str = PANEL_DIR) -> dict[str, NDArray]:
    """The balanced state-year panel as {column: array}, built on a miss.

    Same shape as `OpticalDeviceSEM.load_dataset`: the recorded data is not in the
    repo, so a missing directory means fetch it rather than fail.
    """
    path = os.path.join(os.path.expanduser(directory), PANEL_FILE)
    if not os.path.isfile(path):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        script = os.path.join(repo, "scripts", "fetch_cigarettes.py")
        logger.info(f"{path} missing; building the panel with {script}")
        subprocess.run([sys.executable, script], check=True, cwd=repo)  # noqa: S603 - our own interpreter on a repo script

    with open(path, newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = list(reader)
    if tuple(header) != PANEL_COLUMNS:
        raise ValueError(f"{path} has columns {header}; expected {list(PANEL_COLUMNS)}.")

    panel = {"st": np.array([row[0] for row in rows]), "year": np.array([int(row[1]) for row in rows])}
    for index, name in enumerate(PANEL_COLUMNS[2:], start=2):
        panel[name] = np.array([float(row[index]) for row in rows])
    return panel


def controls(panel: dict[str, NDArray], spec: str) -> NDArray:
    """State dummies plus t, t^2, ... up to the spec's trend order.

    The dummies span the intercept, which is why the FWL residuals below are
    exactly mean zero and `mean_match`'s mu is zero on every query.
    """
    if spec not in SPECS:
        raise ValueError(f"spec {spec!r} is not one of {sorted(SPECS)}.")
    states = np.unique(panel["st"])
    dummies = (panel["st"][:, None] == states[None, :]).astype(float)
    t = (panel["year"].astype(float) - TREND_CENTRE) / TREND_SCALE
    trend = [t**order for order in range(1, SPECS[spec] + 1)]
    return np.column_stack([dummies, *trend]) if trend else dummies


def residualise(A: NDArray, C: NDArray) -> NDArray:
    """FWL: what the controls cannot explain."""
    return A - C @ np.linalg.lstsq(C, A, rcond=None)[0]


@dataclass(frozen=True)
class PanelDesign:
    """One (spec, anchor) FWL design, in the paper's units.

    `y` and `b_ols` are divided by the OLS residual sd, so sigma^2 = 1 and
    gamma* = bias^2 -- what `OpticalDeviceSEM.__init__` does, and what makes
    `EPS_TOL` (a module constant in outcome units) mean the same thing here.
    `X` and `Z` are left in log units; a coefficient is read back raw by
    multiplying by `sigma`.
    """

    X: NDArray
    y: NDArray
    Z: NDArray
    b_ols: NDArray
    sigma: float
    C: NDArray
    state: NDArray
    year: NDArray
    spec: str
    anchor: str

    @property
    def n(self) -> int:
        return len(self.y)

    @property
    def k(self) -> int:
        return self.X.shape[1]

    @property
    def K(self) -> int:
        """Parameters of the pre-FWL regression, for the finite-sample factor."""
        return self.k + self.C.shape[1]

    @property
    def Sigma(self) -> NDArray:
        return self.X.T @ self.X / self.n

    @property
    def exogenous(self) -> list[int]:
        """Columns of X the anchor leaves as their own instruments."""
        endogenous = ANCHORS[self.anchor][1]
        return [j for j in range(self.k) if j not in endogenous]

    @property
    def instruments(self) -> NDArray:
        """The full instrument set: included exogenous regressors, then the excise."""
        return np.column_stack([self.X[:, self.exogenous], self.Z])

    @property
    def Q_Z(self) -> NDArray:
        """Orthonormal basis of span(instruments)."""
        return np.linalg.qr(self.instruments)[0]


def build_design(
    panel: dict[str, NDArray],
    spec: str = "t3",
    anchor: str = "own-tax",
    neighbour: str = "min",
) -> PanelDesign:
    """FWL the spec's controls out of (log q, log X, log Z) and normalise by sigma."""
    if anchor not in ANCHORS:
        raise ValueError(f"anchor {anchor!r} is not one of {sorted(ANCHORS)}.")
    if neighbour not in NEIGHBOURS:
        raise ValueError(f"neighbour aggregator {neighbour!r} is not one of {sorted(NEIGHBOURS)}.")
    price_column, tax_column = NEIGHBOURS[neighbour]
    columns = {"pn": price_column, "tax_sn": tax_column}

    C = controls(panel, spec)
    treatments = np.column_stack([panel[columns.get(name, name)] for name in TREATMENTS])
    instruments = np.column_stack([panel[columns.get(name, name)] for name in ANCHORS[anchor][0]])

    # NOT re-centred afterwards: the state dummies span the intercept, so the
    # residuals come out mean zero on their own (4e-15). Forcing it would hide a
    # control matrix that had lost the dummies, which is the one way this design
    # can silently stop being the one the plan measured.
    y = residualise(np.log(panel[OUTCOME]), C)
    X = residualise(np.log(treatments), C)
    Z = residualise(np.log(instruments), C)

    b_ols = np.linalg.lstsq(X, y, rcond=None)[0]
    sigma = float(np.sqrt(np.mean((y - X @ b_ols) ** 2)))
    return PanelDesign(
        X=X,
        y=y / sigma,
        Z=Z,
        b_ols=b_ols / sigma,
        sigma=sigma,
        C=C,
        state=panel["st"],
        year=panel["year"],
        spec=spec,
        anchor=anchor,
    )


# =============================================================================
# TARGETS
# =============================================================================


def restricted_fit(design: PanelDesign) -> tuple[NDArray, float]:
    """(b_r, r0): 2SLS restricted to the homogeneous subspace, and its IV misfit.

    b_r = N argmin_a || Q_Z' (y - X N a) ||^2, so it satisfies the tax moment
    condition as closely as any homogeneous function can and satisfies v'b = 0
    exactly. Four moments and three free parameters: over-identified by one, and
    r0 = || Q_Z'(y - X b_r) || / sqrt(n) is what is left over.
    """
    N = null_basis()
    Q = design.Q_Z
    A = Q.T @ (design.X @ N)
    d = Q.T @ design.y
    a = np.linalg.lstsq(A, d, rcond=None)[0]
    return N @ a, float(np.linalg.norm(A @ a - d) / np.sqrt(design.n))


def two_stage_fit(design: PanelDesign) -> tuple[NDArray, NDArray]:
    """(b_u, X_hat): the unrestricted 2SLS fit. A DIAGNOSTIC, never the target --
    its Wald on v'b is the compatibility test conditional on IV validity."""
    Q = design.Q_Z
    X_hat = Q @ (Q.T @ design.X)
    return np.linalg.solve(X_hat.T @ design.X, X_hat.T @ design.y), X_hat


def restricted_ols_fit(design: PanelDesign) -> NDArray:
    """OLS restricted to null(v); `rho_max` is its SSR over the unrestricted one."""
    N = null_basis()
    return N @ np.linalg.lstsq(design.X @ N, design.y, rcond=None)[0]


def rho_max(design: PanelDesign) -> float:
    """SSR(restricted OLS) / SSR(OLS): how much the restriction costs the fit."""
    residual = design.y - design.X @ restricted_ols_fit(design)
    free = design.y - design.X @ design.b_ols
    return float(np.sum(residual**2) / np.sum(free**2))


# =============================================================================
# DIAGNOSTICS
# =============================================================================


def clustered_vcov(D: NDArray, u: NDArray, bread: NDArray, K: int, cluster: NDArray | None) -> NDArray:
    """Sandwich variance; `cluster=None` is the iid one.

    log CPI is national and annual, so the effective sample for b_CPI is about 50
    and the year clustering is not decoration. Serial correlation within a state
    over 50 years is strong, which is why the state clustering is the default read.
    """
    if cluster is None:
        meat = (D * u[:, None]).T @ (D * u[:, None])
        return bread @ meat @ bread
    groups = np.unique(cluster)
    meat = np.zeros((D.shape[1], D.shape[1]))
    for group in groups:
        rows = cluster == group
        score = D[rows].T @ u[rows]
        meat += np.outer(score, score)
    n = len(u)
    adjustment = len(groups) / (len(groups) - 1) * (n - 1) / (n - K)
    return adjustment * bread @ meat @ bread


def clusters(design: PanelDesign, by: str) -> NDArray | None:
    if by == "iid":
        return None
    if by == "state":
        return design.state
    if by == "year":
        return design.year
    raise ValueError(f"clustering {by!r} is not one of ['iid', 'state', 'year'].")


def homogeneity_wald(design: PanelDesign, by: str = "state", estimator: str = "iv") -> float:
    """Wald on v'b = 0, df 1. `estimator` picks the OLS ladder or the 2SLS anchor."""
    if estimator == "iv":
        beta, D = two_stage_fit(design)
        bread = np.linalg.inv(D.T @ D)
    else:
        beta, D = design.b_ols, design.X
        bread = np.linalg.inv(D.T @ D)
    u = design.y - design.X @ beta
    vcov = clustered_vcov(D, u, bread, design.K, clusters(design, by))
    return float(beta.sum() ** 2 / vcov.sum())


def homogeneity_f(design: PanelDesign) -> float:
    """Classical iid statistic for v'b = 0 on the OLS ladder: the SSR the
    restriction costs, over the unrestricted MSE.

    Reported beside the clustered Walds precisely because it ignores serial
    correlation: over 50 years within a state that is not the right variance, and
    it is the column that rejects where the others do not.
    """
    free = design.y - design.X @ design.b_ols
    restricted = design.y - design.X @ restricted_ols_fit(design)
    return float((np.sum(restricted**2) - np.sum(free**2)) / (np.sum(free**2) / (design.n - design.K)))


def first_stage_f(design: PanelDesign, by: str = "state") -> list[float]:
    """Sanderson-Windmeijer F per endogenous regressor: with one excluded
    instrument each this is the usual first-stage F."""
    Z = design.instruments
    bread = np.linalg.inv(Z.T @ Z)
    n_excluded = design.Z.shape[1]
    out = []
    for j in ANCHORS[design.anchor][1]:
        coefficients = bread @ Z.T @ design.X[:, j]
        u = design.X[:, j] - Z @ coefficients
        vcov = clustered_vcov(Z, u, bread, Z.shape[1] + design.C.shape[1], clusters(design, by))
        block = vcov[-n_excluded:, -n_excluded:]
        excluded = coefficients[-n_excluded:]
        out.append(float(excluded @ np.linalg.solve(block, excluded) / n_excluded))
    return out
