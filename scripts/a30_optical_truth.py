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

from src.data_augmentors.optical_device import ALL_AUGMENTATIONS  # noqa: E402
from src.data_augmentors.optical_device import OpticalDeviceDA as DA  # noqa: E402
from src.experiments.configs import OPTICAL_CONFIG  # noqa: E402
from src.experiments.optical_device import (  # noqa: E402
    EPSILON_STAR_DRAWS,
    EPSILON_STAR_SEED,
    OpticalOrchestrator,
)
from src.methods.sensitivity_models import InvarianceConstrainedPartialR2, PartialR2  # noqa: E402
from src.oracle import PAD_QUANTILE, epsilon_star, preserve_rng  # noqa: E402
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
    # `sigma_sq` returning `1 - bias_sq` is a definition, so asserting it gates
    # nothing. What the definition RESTS on is the noise normalisation and the
    # orthogonality of the projection -- check those.
    xi = sem.C.ravel() * epsilon_coefficient(sem, Phi)
    check(
        "A30 (ii) the confounding noise is normalised to unit variance",
        abs(np.var(xi) - 1.0) <= 1e-12,
        f"{np.var(xi):.12f}",
    )
    residual = xi - with_intercept @ np.linalg.pinv(with_intercept) @ xi
    fitted = with_intercept @ np.linalg.pinv(with_intercept) @ xi
    check(
        "A30 (ii) bias_sq + the orthogonal remainder == Var(xi)",
        abs(np.mean(xi**2) - (np.mean(fitted**2) + np.mean(residual**2))) <= 1e-12,
        f"{np.mean(xi**2):.12f}",
    )


def epsilon_coefficient(sem, Phi):
    """The fitted confounding coefficient, recovered from the stored pieces:
    y = phi W + b + epsilon C exactly on the fit, so epsilon C = y - f(phi)."""
    residual = sem.y.ravel() - sem.f(Phi).ravel()
    return float(np.dot(residual, sem.C.ravel()) / np.dot(sem.C.ravel(), sem.C.ravel()))


def shipped_augmentations():
    """Every augmentation this repo can actually run optical under: whatever
    config.yaml names right now, plus the alternatives sitting commented beside it
    and the registry default. Gating only the ACTIVE one would let a budget defect
    hide behind an uncommented line -- and gating a hard-coded one (this gate's own
    first mistake) checks a configuration nothing runs."""
    candidates = ["all", "rotation > gaussian-noise", "rotation > hflip > vflip > random-permutation"]
    try:
        with open("config.yaml") as handle:
            for line in handle:
                stripped = line.strip().lstrip("#").strip()
                if stripped.startswith("augmentation:"):
                    named = stripped.split(":", 1)[1].split("#")[0].strip()
                    # config.yaml names a simulation augmentation too, and this
                    # registry does not know it -- keep only what optical can run
                    parts = [part.strip() for part in named.split(">")] if named else []
                    known = parts and all(part in ALL_AUGMENTATIONS for part in parts)
                    if (named == "all" or known) and named not in candidates:
                        candidates.append(named)
    except OSError:
        pass
    return candidates


def a30_budget():
    """(iii)+(iv) the budgets admit h_*, in the right norms, for every augmentation
    the repo can run -- and the CONFIGURED value is the one gated, not a default."""
    budgets = {}
    for augmentation in shipped_augmentations():
        orchestrator = OpticalOrchestrator(augmentation, methods=["PI"], n_jobs=1)
        eps_star = orchestrator.measured_epsilon_star()
        eps_pad = orchestrator.measured_epsilon_pad()

        # THE check: whatever OpticalDeviceConfig actually carries -- None or a
        # pinned float -- must clear the measured defect in its own norm
        configured = orchestrator._epsilon_budget(OPTICAL_CONFIG.epsilon)
        check(
            f"A30 (iii) [{augmentation}] the configured PI+INV budget admits h_*",
            configured >= eps_star,
            f"budget {configured:.4f} vs eps* {eps_star:.4f}",
        )
        configured_pad = orchestrator._pad_budget(OPTICAL_CONFIG.pad_epsilon)
        check(
            f"A30 (iii) [{augmentation}] the configured pad covers the pointwise defect",
            configured_pad >= eps_pad,
            f"pad {configured_pad:.4f} vs q{PAD_QUANTILE:g}|W| {eps_pad:.4f}",
        )
        # the two norms must not be conflated: padding by the L2 budget is the
        # bug this separation exists to prevent, so assert they really differ
        check(
            f"A30 (iii) [{augmentation}] the pad budget is the POINTWISE one",
            configured_pad > 1.5 * eps_star,
            f"pad {configured_pad:.4f} vs L2 eps* {eps_star:.4f}",
        )
        budgets[augmentation] = (configured, configured_pad)

    # (iv) reproducible across incoming RNG states -- it goes into published intervals
    values = []
    for seed in (1, 12345):
        np.random.seed(seed)
        orchestrator = OpticalOrchestrator("all", methods=["PI"], n_jobs=1)
        values.append((orchestrator.measured_epsilon_star(), orchestrator.measured_epsilon_pad()))
    # to a tolerance, not exactly: OpticalDeviceSEM centres its cached array in
    # place, so repeated construction drifts the fit in the last few digits. A real
    # dependence on the incoming stream is percent-level (measured: 0.2553 vs
    # 0.2684 across two unseeded draws), nowhere near this.
    check(
        "A30 (iv) the budgets do not depend on the incoming RNG state",
        all(abs(a - b) <= 1e-9 * max(a, 1e-30) for a, b in zip(values[0], values[1], strict=True)),
        f"{values[0]!r} vs {values[1]!r}",
    )
    orchestrator = OpticalOrchestrator("all", methods=["PI"], n_jobs=1)
    check("A30 (iv) a configured budget is passed through", orchestrator._epsilon_budget(0.125) == 0.125)

    # the pooling must actually pool: seeding once and looping would evaluate one
    # realisation N times (it did), so require the draws to be distinct
    sem, da, features = orchestrator._oracle_pieces()
    with preserve_rng():
        draws = set()
        for draw in range(EPSILON_STAR_DRAWS):
            np.random.seed(EPSILON_STAR_SEED + draw)
            draws.add(epsilon_star(sem, da, X=sem.X, features=features))
    check(
        "A30 (iv) the pooled eps* draws are distinct",
        len(draws) == EPSILON_STAR_DRAWS,
        f"{len(draws)}/{EPSILON_STAR_DRAWS} distinct",
    )
    return budgets


def a30_membership(sem, poly, Phi, y, budgets):
    """(v) h_* is inside the intervals at gamma*, on the pool's own queries, under
    the SHIPPED toggles -- `pad` and `clipy` included, since the pad is where Thm.
    3.A's epsilon does its work and an unpadded run never reads it at all."""
    gamma_star = sem.bias_sq / sem.sigma_sq
    augmentation = shipped_augmentations()[1]
    budget, pad_budget = budgets[augmentation]
    np.random.seed(0)
    GX_raw, _ = DA(augmentation)(sem.X)
    GX = poly.transform(GX_raw)

    queries = np.vstack([Phi[:200], Phi.mean(axis=0, keepdims=True)])
    truth = sem.f(queries).ravel()

    common = dict(calibrate=True, clipy=True, n_jobs=1)
    for name, model in (
        ("PI", PartialR2(gamma=gamma_star, pad=False, **common)),
        ("PI+INV", InvarianceConstrainedPartialR2(gamma=gamma_star, epsilon=budget, pad=False, **common)),
        # the DA branch padded exactly as config.yaml ships it -- the only leg in
        # which `pad_epsilon` is read
        ("DA+PI", PartialR2(gamma=gamma_star, pad=True, pad_epsilon=pad_budget, **common)),
    ):
        design = GX if name == "DA+PI" else Phi
        fitted = model.fit(Phi, y, GX=GX) if name == "PI+INV" else model.fit(design, y)
        bounds = fitted.predict(queries)
        inside = (bounds[:, 0] - 1e-9 <= truth) & (truth <= bounds[:, 1] + 1e-9)
        check(f"A30 (v) {name}: h_* inside on every query", bool(inside.all()), f"{inside.mean():.1%}")
        check(
            f"A30 (v) {name}: h_* inside at the pinch query",
            bool(inside[-1]),
            f"[{bounds[-1, 0]:+.3e}, {bounds[-1, 1]:+.3e}] vs h_* {truth[-1]:+.3e}",
        )
        # Negative control: the SAME query with the intercept dropped from the
        # ground truth -- the pre-fix estimand -- must fall OUTSIDE. Without it the
        # check above is satisfied by any interval that happens to contain 0.
        # UNPADDED methods only: Thm. 3.A's pad is +-0.69 here, four times the
        # 0.333 defect, so a padded interval cannot discriminate and asserting it
        # would only be asserting that the pad is small. The pinch witness is a
        # statement about Cor. 3's geometry, and the pad sits outside that.
        if not fitted.pad:
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
    budgets = a30_budget()
    a30_membership(sem, poly, Phi, y, budgets)
    a30_lazy_load()
    print(f"\n{'A30 ALL PASS' if not FAIL else 'A30 FAILURES: ' + ', '.join(FAIL)}")
    sys.exit(bool(FAIL))
