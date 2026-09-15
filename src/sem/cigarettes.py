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
from typing import Literal

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from src.sem.abstract import StructuralEquationModel as SEM

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


# plasmode defaults; the config dataclass overrides them (SS6)
GAMMA_TRUE: float = 0.25
OUTCOME_NOISE_STD: float = 0.1
CONFOUND_DIRECTIONS: tuple[str, ...] = ("v", "own_price", "worst_case")
# the leaky-IV guard (SS5): FIXED, never swept
GAMMA_Z: float = 2**-8
# one state history; the cluster bootstrap deals in these, not in rows
YEARS_PER_STATE: int = 50


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


def instrument_set(design: PanelDesign, names) -> NDArray:
    """The CONFIGURED instrument matrix, (n, m), from panel column names in the
    order given (the yaml's `iv:`). A treatment name is that FWL'd regressor
    instrumenting itself (included exogenous), an excise name the FWL'd excise
    column the anchor carries: the same residualised columns the design uses,
    never the raw ones (SS2.1). Empty names give (n, 0), no instrument. The
    anchor decides which excise columns exist; a name it does not carry raises.
    """
    excise = ANCHORS[design.anchor][0]
    columns = []
    for name in names:
        if name in TREATMENTS:
            columns.append(design.X[:, TREATMENTS.index(name)])
        elif name in excise:
            columns.append(design.Z[:, excise.index(name)])
        else:
            raise ValueError(
                f"instrument {name!r} is neither a treatment {TREATMENTS} nor an excise column of anchor "
                f"{design.anchor!r} {excise}."
            )
    return np.column_stack(columns) if columns else np.zeros((design.n, 0))


# =============================================================================
# TARGETS
# =============================================================================


def restricted_fit(design: PanelDesign, instruments: NDArray | None = None) -> tuple[NDArray, float]:
    """(b_r, r0): 2SLS restricted to the homogeneous subspace, and its IV misfit.

    b_r = N argmin_a || Q_Z' (y - X N a) ||^2, so it satisfies the tax moment
    condition as closely as any homogeneous function can and satisfies v'b = 0
    exactly. On the anchor set (the default) that is four moments and three free
    parameters: over-identified by one, and r0 = || Q_Z'(y - X b_r) || / sqrt(n)
    is what is left over. `instruments` fits against a configured set instead
    (decision 1): with three moments the point is exactly identified and r0 is 0.
    """
    N = null_basis()
    Q = design.Q_Z if instruments is None else np.linalg.qr(instruments)[0]
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
    meat = score_meat(D, u, cluster)
    if cluster is None:
        return bread @ meat @ bread
    groups = len(np.unique(cluster))
    n = len(u)
    adjustment = groups / (groups - 1) * (n - 1) / (n - K)
    return adjustment * bread @ meat @ bread


def score_meat(D: NDArray, u: NDArray, cluster: NDArray | None) -> NDArray:
    """Sum of outer products of the cluster scores; no finite-sample factor.

    The GMM weight is the inverse of THIS, so it must not carry the adjustment the
    Wald's sandwich does, or J and the Wald stop being the same statistic.
    """
    if cluster is None:
        return (D * u[:, None]).T @ (D * u[:, None])
    meat = np.zeros((D.shape[1], D.shape[1]))
    for group in np.unique(cluster):
        rows = cluster == group
        score = D[rows].T @ u[rows]
        meat += np.outer(score, score)
    return meat


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


def restricted_gmm(design: PanelDesign, by: str = "state") -> tuple[float, float, NDArray]:
    """(J1w, J2, b): the restricted-GMM overid statistic of HD0 given IV validity.

    Four moments, three free parameters after v'b = 0, so df 1 and J is a real test.

    `J1w` evaluates the clustered-weight objective at the ONE-STEP (Z'Z)^-1
    restricted estimate, which does not minimise it: an upper bound that rejects too
    often, and the whole of the apparent disagreement with the Wald. `J2` is the
    efficient two-step statistic -- re-estimate AT the weight, then evaluate -- and
    is the one of the program actually being solved, so it is what gets reported.
    """
    N = null_basis()
    Z = design.instruments
    A = design.X @ N
    Q = design.Q_Z
    one_step = N @ np.linalg.lstsq(Q.T @ A, Q.T @ design.y, rcond=None)[0]

    meat = score_meat(Z, design.y - design.X @ one_step, clusters(design, by))
    moments = Z.T @ (design.y - design.X @ one_step)
    J1w = float(moments @ np.linalg.solve(meat, moments))

    weight = np.linalg.inv(meat)
    jacobian = Z.T @ A
    two_step = N @ np.linalg.solve(jacobian.T @ weight @ jacobian, jacobian.T @ weight @ (Z.T @ design.y))
    moments = Z.T @ (design.y - design.X @ two_step)
    return J1w, float(moments @ np.linalg.solve(meat, moments)), two_step


# =============================================================================
# SEM
# =============================================================================


class CigaretteSEM(SEM):
    """The panel as a recorded SEM, under one of two targets.

    `iv`       h_* is the restricted-IV point of SS0.2: the tax-moment program
               solved subject to v'b = 0, so it satisfies the instrument and
               homogeneity at once and eps* is 0 by construction. With
               `iv_columns` empty the moments are the anchor set's and the tax
               column is read once, here, and never reaches a solver. With a
               configured set (the yaml's `iv:`) the moments ARE that set, the
               target is the restricted fit against it (decision 1) and the same
               columns ride beside every draw as the trailing `iv_width` columns
               (`iv_pool` beside `pool`), so the solvers see them too. Coverage
               of this target is validity CONDITIONAL on those two assumptions;
               what probes the assumptions is the compatibility statistic above
               and the sliver below.
    `plasmode` the real FWL'd design with a synthetic exactly homogeneous h_* and
               synthetic confounding of known strength, in the simulation SEM's
               convention, so gamma* == `gamma_true` exactly. Validity against a
               truth nobody has to believe in.

    Units: y and every coefficient vector are divided by the OLS residual sd, so
    sigma^2 = 1 on the `iv` path and gamma* = bias^2. `EPS_TOL` is a module constant
    in outcome units, which on the raw panel (sigma 0.1455) would be 21% of sigma.
    """

    load_panel = staticmethod(load_panel)

    _PANEL: dict[str, NDArray] | None = None  # filled on first use, not on import

    @classmethod
    def panel(cls) -> dict[str, NDArray]:
        """The cached panel. Built from the raw sources on first use, as
        `OpticalDeviceSEM.dataset` downloads on first use: importing this module
        must not reach for the network or the disk."""
        if cls._PANEL is None:
            cls._PANEL = cls.load_panel()
        return cls._PANEL

    def __init__(
        self,
        spec: str = "t3",
        target: Literal["iv", "plasmode"] = "iv",
        anchor: str = "own-tax",
        neighbour: str = "min",
        bootstrap: bool = False,
        sliver: bool = False,
        gamma_z: float = GAMMA_Z,
        gamma_true: float = GAMMA_TRUE,
        confound_direction: str = "v",
        outcome_noise_std: float = OUTCOME_NOISE_STD,
        iv_columns=(),
    ):
        if target not in ("iv", "plasmode"):
            raise ValueError(f"target {target!r} is not one of ['iv', 'plasmode'].")
        if sliver and target != "iv":
            raise ValueError("the leaky-IV sliver is a set around the restricted point; it needs target 'iv'.")
        self.iv_columns = tuple(iv_columns)
        if sliver and self.iv_columns:
            # the sliver is the set around the ANCHOR set's restricted point at a
            # fixed gamma_z; under a configured set the target moves and the
            # ellipsoid would need that set's geometry. Not defined here.
            raise ValueError("the leaky-IV sliver is defined on the anchor set; it does not combine with `iv`.")

        self.design = build_design(self.panel(), spec=spec, anchor=anchor, neighbour=neighbour)
        self.target = target
        self.bootstrap = bool(bootstrap)
        self._sliver = bool(sliver)
        self._gamma_z = float(gamma_z)
        self.X = self.design.X
        # the anchor's tax column, read ONCE, here, for the ladder diagnostics and
        # the anchor-set fallback; no solver sees it
        self._Z = self.design.Z
        # the CONFIGURED instrument set, (n, m): what the solvers see, row-aligned
        # with `X`, and (n, 0) when there is none
        self._Z_iv = instrument_set(self.design, self.iv_columns)

        # the target: the restricted fit against the configured set when there is
        # one, else the anchor set (decision 15), exactly today's b_r
        b_r, self._misfit = restricted_fit(self.design, self._Z_iv if self.iv_columns else None)
        if target == "iv":
            self.W_XY = b_r.reshape(-1, 1)
            self.y = self.design.y.reshape(-1, 1)
            gap = self.design.b_ols - b_r
            self._bias_sq = float(gap @ self.design.Sigma @ gap)
            self._sigma_sq = 1.0
        else:
            self.W_XY = self._plasmode_target().reshape(-1, 1)
            self._noise_std = float(outcome_noise_std)
            self._kappa_sq = self._calibrate(float(gamma_true))
            self.y = self._draw_outcome(confound_direction)
            self._bias_sq = self._kappa_sq
            self._sigma_sq = 1.0 - self._kappa_sq + self._noise_std**2

    # ------------------------------------------------------------- plasmode

    def _plasmode_target(self) -> NDArray:
        """P_null(v) b_u: the unrestricted 2SLS fit projected onto the homogeneous
        subspace. Exactly homogeneous, and near enough the data's own elasticities
        that Var(h_*(X)) and the confounding are the same order."""
        N = null_basis()
        b_u, _ = two_stage_fit(self.design)
        return N @ (N.T @ b_u)

    def _calibrate(self, gamma_true: float) -> float:
        """kappa^2 = gamma (1 + s^2) / (1 + gamma), the simulation's inversion, so
        gamma* = bias^2/sigma^2 comes out at `gamma_true` exactly."""
        if gamma_true < 0.0:
            raise ValueError("`gamma_true` must be non-negative.")
        s_sq = self._noise_std**2
        return float(min(gamma_true * (1.0 + s_sq) / (1.0 + gamma_true), 1.0))

    def _draw_outcome(self, direction: str) -> NDArray:
        """Y = h_*(X) + kappa u_d + sqrt(1 - kappa^2) e + s nu, with u_d the design
        projected on `direction` and STANDARDISED: without the unit variance the
        confounding no longer has the magnitude gamma_true was inverted for."""
        if direction not in CONFOUND_DIRECTIONS:
            raise ValueError(f"confound_direction {direction!r} is not one of {list(CONFOUND_DIRECTIONS)}.")
        if direction == "v":
            # the spec-S story made explicit: an unobserved taste drift co-moving
            # with the price level, which is the direction the symmetry is about
            d = V / np.linalg.norm(V)
        elif direction == "own_price":
            d = np.eye(self.design.k)[0]
        else:
            d = np.linalg.solve(self.design.Sigma, self.W_XY.ravel())
        confounder = self.X @ d
        confounder = confounder / confounder.std()
        kappa = np.sqrt(self._kappa_sq)
        n = len(self.X)
        noise = np.sqrt(1.0 - self._kappa_sq) * np.random.randn(n) + self._noise_std * np.random.randn(n)
        return (self.f(self.X).ravel() + kappa * confounder + noise).reshape(-1, 1)

    # -------------------------------------------------------------- the target

    @property
    def bias_sq(self) -> float:
        return self._bias_sq

    @property
    def sigma_sq(self) -> float:
        return self._sigma_sq

    @property
    def pool(self) -> tuple[NDArray, NDArray]:
        """The panel itself -- see `SEM.pool`. The oracle reads THIS, never a
        `sample`, so h_*, gamma* and eps* do not move with the replicate draw."""
        return self.X, self.y

    # ------------------------------------------------------------ instruments

    @property
    def iv_width(self) -> int:
        return self._Z_iv.shape[1]

    @property
    def iv_pool(self) -> NDArray | None:
        """The configured instrument beside `pool`, row for row; None without one."""
        return self._Z_iv if self.iv_width > 0 else None

    def _rows(self, index) -> tuple[NDArray, NDArray]:
        """(X | Z, y) at `index`: the instrument rides as the trailing columns of a
        draw when there is one, so a row resample keeps Z aligned with X."""
        X = self.X[index]
        if self.iv_width > 0:
            X = np.column_stack([X, self._Z_iv[index]])
        return X, self.y[index]

    def extent(self, X) -> NDArray:
        """Half-width of the target SET at each query; zero means a point target.

        With `sliver` on, the target is everything consistent with the tax moment at
        a FIXED gamma_z and with v'b = 0 (SS5),

            S = { b : v'b = 0,  ||Q_Z'(y - X b)|| / sqrt(n) <= sigma sqrt(gamma_z) },

        an ellipsoid in the three free parameters centred at b_r, so the per-query
        extent is closed form: one 3x3 inverse, no solver. S is non-empty iff the
        minimised misfit r0 is under the radius, which is itself a test of
        "leaky-by-gamma_z IV and HD0" -- empty is information, not a failure.
        """
        X = np.asarray(X, dtype=float)
        if not self._sliver:
            return np.zeros(len(X))
        slack = self._sigma_sq * self._gamma_z - self._misfit**2
        if slack <= 0.0:
            logger.info(
                f"leaky-IV sliver EMPTY at {self.design.spec}/{self.design.anchor}: misfit {self._misfit:.4f} "
                f"exceeds the radius {np.sqrt(self._sigma_sq * self._gamma_z):.4f}; falling back to the point target."
            )
            return np.zeros(len(X))
        N = null_basis()
        A = self.design.Q_Z.T @ (self.X @ N)
        precision = np.linalg.inv(A.T @ A)
        projected = X @ N
        return np.sqrt(slack * len(self.X)) * np.sqrt(np.einsum("ij,jk,ik->i", projected, precision, projected))

    # ------------------------------------------------------------- replicates

    def sample(self, N: int = 1, **kwargs) -> tuple[NDArray, NDArray]:
        """The replicate mechanism (SS6). `pool` is untouched either way.

        `bootstrap=False`: the first rows of the panel, so at N = 2450 the draw IS
        the panel, deterministically -- what the query figures and the coefficient
        table are fit on.
        `bootstrap=True`: whole state histories drawn WITH replacement, the sweep
        replicate. Clusters and not rows, because within-state serial correlation
        over 50 years is strong and an iid row bootstrap would understate the spread
        and inflate coverage.
        """
        n_total = len(self.X)
        if not self.bootstrap:
            if n_total >= N:
                return self._rows(slice(None, N))
            # not an error -- a sweep may legitimately ask for more rows than the
            # panel has -- but the draw is then a BOOTSTRAP, so anything read off it
            # carries resampling noise the panel itself does not have
            logger.warning(f"CigaretteSEM: {N} rows requested from a panel of {n_total}; padding by cluster resample.")
        states = np.unique(self.design.state)
        rows = {state: np.flatnonzero(self.design.state == state) for state in states}
        drawn = np.random.choice(states, int(np.ceil(N / YEARS_PER_STATE)), replace=True)
        index = np.concatenate([rows[state] for state in drawn])
        return self._rows(index)
