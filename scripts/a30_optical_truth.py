"""A30: the optical estimand is the one the solver searches.

The optical ground truth is FITTED, not declared, so it is only the paper's h_*
if the fit is done in the paper's hypothesis class. Asm. 1's base clause closes
that class under constant shifts and Lem. 2 puts the identified set on the slice
H_X = {h : E[h(X)] = E[Y]}; a ground truth fitted without a free intercept sits
off that slice and is excluded from the set by construction -- a VALIDITY loss
that no coverage average will show, because it hides in the one query where the
interval pinches to a point.

  (i)   h_* lands on the empirical slice, and phi(x) = mean(phi) -- where Cor. 3
        pinches the interval to {ybar} -- is exactly the query that proves it;
  (ii)  bias_sq/sigma_sq/gamma* are measured over span(phi, 1), the SAME class
        the solver searches, and differ from the span(phi) numbers;
  (iii) the PI+INV budget is at least the MEASURED eps*, so h_* is admitted;
  (iv)  that budget is reproducible -- it goes into published intervals;
  (v)   h_* is inside the PI and PI+INV intervals at gamma* on real queries;
  (vi)  importing the SEM does not reach for the network.

    python scripts/a30_optical_truth.py
"""

import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.preprocessing import PolynomialFeatures  # noqa: E402

from src.data_augmentors.optical_device import OpticalDeviceDA as DA  # noqa: E402
from src.experiments.configs import OPTICAL_CONFIG  # noqa: E402
from src.experiments.optical_device import OpticalOrchestrator  # noqa: E402
from src.methods.sensitivity_models import InvarianceConstrainedPartialR2, PartialR2  # noqa: E402
from src.sem.optical_device import OpticalDeviceSEM as SEM  # noqa: E402

# 1 SE of the level is std(y)/sqrt(n) ~ 0.039 here; the slice identity is exact
# up to the least-squares solve, so hold it to solver noise, not to sampling noise
SLICE_TOL = 1e-9
FAIL = []


def check(name, ok, detail="", hard=True):
    print(f"[{'PASS' if ok else ('FAIL' if hard else 'WARN')}] {name} {detail}")
    if not ok and hard:
        FAIL.append(name)


def fixture():
    sem = SEM(experiment=OPTICAL_CONFIG.dataset_index, ground_truth=OPTICAL_CONFIG.ground_truth_model)
    poly = PolynomialFeatures(sem.poly_degree, include_bias=False).fit(sem.X)
    return sem, poly, poly.transform(sem.X), sem.y.ravel()


def a30_slice(sem, poly, Phi, y):
    """(i) h_* is ON Lem. 2's slice, and the pinch query is the witness."""
    h = sem.f(Phi).ravel()
    defect = float(np.mean(h) - np.mean(y))
    standard_error = float(np.std(y) / np.sqrt(len(y)))
    check(
        "A30 (i) h_* is on the empirical slice",
        abs(defect) <= SLICE_TOL,
        f"|mean h_* - ybar| {abs(defect):.3e} ({abs(defect) / standard_error:.3f} SE)",
    )

    # Cor. 3 at the design mean: the centred feature vector vanishes, so the
    # interval is the single point {ybar} whatever gamma is. h_* is covered there
    # IF AND ONLY IF it is on the slice -- which is why this query, and not the
    # coverage average over the pool, is the one that catches a dropped intercept.
    mean_query = Phi.mean(axis=0, keepdims=True)
    h_at_mean = float(sem.f(mean_query).ravel()[0])
    check(
        "A30 (i) h_* at the design mean == ybar (the pinch query)",
        abs(h_at_mean - float(np.mean(y))) <= SLICE_TOL,
        f"h_*(mean phi) {h_at_mean:+.3e} vs ybar {float(np.mean(y)):+.3e}",
    )


def a30_projection(sem, Phi):
    """(ii) gamma* is measured over span(phi, 1), and that is not span(phi)."""

    def bias_over(design):
        """Var of the projection of the (already noise-scaled) confounding term."""
        xi = sem.C.ravel() * epsilon_coefficient(sem, Phi)
        return float(np.var(design @ np.linalg.pinv(design) @ xi))

    with_intercept = np.column_stack([Phi, np.ones(len(Phi))])
    check(
        "A30 (ii) bias_sq is the span(phi, 1) projection",
        abs(bias_over(with_intercept) - sem.bias_sq) <= 1e-12,
        f"{sem.bias_sq:.6f}",
    )
    # and the intercept-free projection is a DIFFERENT number: without this the
    # check above would pass on either class and so gate nothing
    intercept_free = bias_over(Phi)
    check(
        "A30 (ii) the span(phi) projection differs",
        abs(intercept_free - sem.bias_sq) > 1e-6,
        f"span(phi) {intercept_free:.6f} vs span(phi, 1) {sem.bias_sq:.6f}",
    )
    check("A30 (ii) sigma_sq == 1 - bias_sq", abs(sem.sigma_sq - (1.0 - sem.bias_sq)) <= 1e-12, f"{sem.sigma_sq:.6f}")


def epsilon_coefficient(sem, Phi):
    """The fitted confounding coefficient, recovered from the stored pieces:
    y = phi W + b + epsilon C exactly on the fit, so epsilon C = y - f(phi)."""
    residual = sem.y.ravel() - sem.f(Phi).ravel()
    return float(np.dot(residual, sem.C.ravel()) / np.dot(sem.C.ravel(), sem.C.ravel()))


def a30_budget():
    """(iii)+(iv) the PI+INV budget admits h_*, and does not move between runs."""
    values = []
    for seed in (1, 12345):
        np.random.seed(seed)
        orchestrator = OpticalOrchestrator("all", methods=["PI"], n_jobs=1)
        values.append((orchestrator.measured_epsilon_star(), orchestrator._epsilon_budget(None)))

    (eps_star, budget), (eps_star_again, budget_again) = values
    check(
        "A30 (iii) the PI+INV budget admits h_* (budget >= eps*)",
        budget >= eps_star,
        f"budget {budget:.6f} vs eps* {eps_star:.6f}",
    )
    # to a tolerance, not exactly: the two calls refit the SEM and redraw the DA,
    # so BLAS accumulation order moves the last couple of digits. A real dependence
    # on the incoming stream is a percent-level effect (measured: 0.268 vs 0.255
    # across two unseeded draws), nowhere near this.
    check(
        "A30 (iv) the budget does not depend on the incoming RNG state",
        abs(eps_star - eps_star_again) <= 1e-9 * eps_star and abs(budget - budget_again) <= 1e-9 * budget,
        f"{eps_star!r} vs {eps_star_again!r}",
    )
    # a configured float must still win, or the published number could not be pinned
    orchestrator = OpticalOrchestrator("all", methods=["PI"], n_jobs=1)
    check("A30 (iv) a configured budget is passed through", orchestrator._epsilon_budget(0.125) == 0.125)
    return eps_star, budget


def a30_membership(sem, poly, Phi, y, budget):
    """(v) h_* is inside the intervals at gamma*, on the pool's own queries."""
    gamma_star = sem.bias_sq / sem.sigma_sq
    np.random.seed(0)
    GX_raw, _ = DA("all")(sem.X)
    GX = poly.transform(GX_raw)

    queries = np.vstack([Phi[:200], Phi.mean(axis=0, keepdims=True)])
    truth = sem.f(queries).ravel()

    for name, model in (
        ("PI", PartialR2(gamma=gamma_star, calibrate=True, clipy=False, n_jobs=1)),
        (
            "PI+INV",
            InvarianceConstrainedPartialR2(gamma=gamma_star, epsilon=budget, calibrate=True, clipy=False, n_jobs=1),
        ),
    ):
        fitted = model.fit(Phi, y, GX=GX) if name == "PI+INV" else model.fit(Phi, y)
        bounds = fitted.predict(queries)
        inside = (bounds[:, 0] - 1e-9 <= truth) & (truth <= bounds[:, 1] + 1e-9)
        check(f"A30 (v) {name}: h_* inside on every query", bool(inside.all()), f"{inside.mean():.1%}")
        check(
            f"A30 (v) {name}: h_* inside at the pinch query",
            bool(inside[-1]),
            f"[{bounds[-1, 0]:+.3e}, {bounds[-1, 1]:+.3e}] vs h_* {truth[-1]:+.3e}",
        )
        # negative control: the SAME query with the intercept dropped from the
        # ground truth -- the pre-fix estimand -- must fall OUTSIDE. Without this
        # the check above is satisfied by any interval that happens to contain 0.
        intercept_free = truth[-1] - float(np.ravel(sem.b_XY)[0])
        check(
            f"A30 (v) {name}: the intercept-free h_* is EXCLUDED at the pinch query",
            not (bounds[-1, 0] - 1e-9 <= intercept_free <= bounds[-1, 1] + 1e-9),
            f"[{bounds[-1, 0]:+.3e}, {bounds[-1, 1]:+.3e}] vs {intercept_free:+.4f}",
        )


def a30_lazy_load():
    """(vi) importing the SEM must not reach for the network."""
    code = "import src.sem.optical_device as m; print(m.OpticalDeviceSEM._DATASET is None)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=os.getcwd())
    check("A30 (vi) importing the SEM loads no data", result.stdout.strip().endswith("True"), result.stdout.strip())


if __name__ == "__main__":
    sem, poly, Phi, y = fixture()
    a30_slice(sem, poly, Phi, y)
    a30_projection(sem, Phi)
    _, budget = a30_budget()
    a30_membership(sem, poly, Phi, y, budget)
    a30_lazy_load()
    print(f"\n{'A30 ALL PASS' if not FAIL else 'A30 FAILURES: ' + ', '.join(FAIL)}")
    sys.exit(bool(FAIL))
