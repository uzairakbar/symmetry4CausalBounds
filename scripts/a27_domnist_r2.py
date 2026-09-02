"""A27: the partial_r2_net backend's own gates.

Four properties, none of which SOURCE parity can pin for a new model:
  1. nesting -- a constrained method never comes out WIDER than the unconstrained
     model on the same ball (adding a constraint can only shrink the feasible set);
  2. membership -- the ANALYTIC h_* (exact for this SEM) sits inside the PI bounds
     at gamma = gamma* for >= 95% of held-out queries. The thing CopSens could
     never certify: h_* was not in its candidate class.
  3. JAX == numpy-mirror finite differences on every term of the NLP (A8 style);
  4. l=2 exercised at reduced scale: correctness + nesting on the AL path.

PI+INV is REPORTED but exempt from hard-fail: realizability under a shallow refit
is an open empirical question (all-INFEASIBLE there is honest behaviour, not a
bug -- read the floor/budget pair it logs).

    python scripts/a27_domnist_r2.py
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_augmentors.do_mnist import DoMNISTDA  # noqa: E402
from src.experiments.configs import DOMNIST_CONFIG, MethodRegistry  # noqa: E402
from src.experiments.do_mnist import NESTED_IN, Flatten  # noqa: E402
from src.experiments.utils import set_seed  # noqa: E402
from src.methods.partial_r2_net import PartialR2Net, head_index  # noqa: E402
from src.methods.regression import GradientDescentERM  # noqa: E402
from src.methods.sensitivity_models import SolveStatus  # noqa: E402
from src.sem.do_mnist import DoMNISTSEM  # noqa: E402

N_SAMPLES, N_PI, N_QUERIES = 60_000, 6_000, 32
L2_SAMPLES, L2_PI, L2_QUERIES = 20_000, 2_000, 4
GAMMA_STAR = 0.010000000000000002 / 0.15  # bias_sq / sigma_sq, sem/do_mnist.py
EPSILON, EPSILON_IV = 0.1, 0.1
# Numeric slack for the nesting check. Child subset-of parent holds EXACTLY at the
# set level whenever the two share a ball (gated structurally below); the interval
# endpoints are INDEPENDENT multi-start inner approximations though, and the
# lagging side is not always the child's -- measured asymmetry ~9e-4 with an inert
# IV constraint. A real nesting break (wrong ball, wrong constraint sign) shows up
# at width scale, orders above this.
NEST_TOL = 2e-3
N_JOBS = 16
EXEMPT = {"PI+INV"}  # reported, never hard-failed (user decision, PLAN stage 4)
FAIL = []


def check(name, ok, detail="", hard=True):
    tag = "PASS" if ok else ("FAIL" if hard else "WARN")
    print(f"[{tag}] {name} {detail}")
    if hard and not ok:
        FAIL.append(name)


def fixture(n_samples, n_pi, n_queries):
    set_seed(42)
    sem = DoMNISTSEM(
        seed=42,
        train=True,
        target_samples=1,
        alpha=DOMNIST_CONFIG.alpha,
        beta=DOMNIST_CONFIG.beta,
        eta=DOMNIST_CONFIG.eta,
    )
    flat = Flatten()
    X_img, y, _ = sem.sample_paired(n_samples, seed=42)
    GX_img, G = DoMNISTDA()(X_img)
    X, GX = flat.fit_transform(X_img), flat.fit_transform(GX_img)
    nets = {
        "X": GradientDescentERM().fit(X, y, init_seed=42, epochs=1),
        "GX": GradientDescentERM().fit(GX, y, init_seed=42, epochs=1),
    }
    keep = np.random.default_rng(42).choice(len(X), n_pi, replace=False)

    # held-out interventional queries, with the ANALYTIC h_* off the digit label
    sem_test = DoMNISTSEM(
        seed=42,
        train=False,
        target_samples=1,
        alpha=DOMNIST_CONFIG.alpha,
        beta=DOMNIST_CONFIG.beta,
        eta=DOMNIST_CONFIG.eta,
    )
    Q_img, _ = sem_test(N=n_queries, intervention=True, seed=9)
    h_star = sem_test.h_star(sem_test.last_["f"])
    return nets, X[keep], GX[keep], y[keep], G[keep], flat.fit_transform(Q_img), h_star


def fit_and_predict(nets, X, GX, y, G, Q, unfrozen_layers):
    """{method: (bounds, status, model)} for the six PI variants, wall-clock logged."""
    builders = MethodRegistry.build_methods(
        ["PI", "DA+PI", "PI+INV", "DA+PI+IV", "PI&DA+PI", "PI&DA+PI+IV"],
        gamma=GAMMA_STAR,
        epsilon=EPSILON,
        epsilon_iv=EPSILON_IV,
        calibrate=True,
        clipy=True,
        n_jobs=N_JOBS,
        backend="partial_r2_net",
        outcome_models=nets,
        unfrozen_layers=unfrozen_layers,
    )
    fit_data = {
        "PI": dict(X=X, y=y),
        "DA+PI": dict(X=GX, y=y),
        "PI+INV": dict(X=X, y=y, GX=GX),
        "DA+PI+IV": dict(X=GX, y=y, Z=G),
        "PI&DA+PI": dict(X=X, y=y, GX=GX, G=G),
        "PI&DA+PI+IV": dict(X=X, y=y, GX=GX, G=G),
    }
    out = {}
    for name, builder in builders.items():
        start = time.perf_counter()
        model = builder().fit(**fit_data[name])
        bounds = model.predict(Q)
        elapsed = time.perf_counter() - start
        out[name] = (bounds, model.query_status.copy(), model)
        counts = np.bincount(model.query_status, minlength=3).tolist()
        width = np.nanmean(bounds[:, 1] - bounds[:, 0])
        print(f"  {name:12s} {elapsed:6.1f}s ({elapsed / len(Q):.2f}s/q)  ok/inf/fail {counts}  width {width:.4f}")
    return out


def gate_statuses(results, label):
    """No FAILURE anywhere: the multi-start must always produce a feasible point
    (the anchor at worst). INFEASIBLE is a legitimate outcome only for PI+INV."""
    for name, (_, status, _) in results.items():
        hard = name not in EXEMPT
        check(f"A27 {label} {name}: no FAILURE status", not (status == SolveStatus.FAILURE).any(), hard=hard)
        if name in EXEMPT:
            infeasible = int((status == SolveStatus.INFEASIBLE).sum())
            check(f"A27 {label} {name}: feasibility report", True, f"{infeasible}/{len(status)} INFEASIBLE", hard=False)


def gate_nesting(results, label):
    """Constrained subset-of parent: structurally (same ball => the SETS nest
    exactly) and numerically on the intervals, up to NEST_TOL of solver noise."""
    for name, parent in NESTED_IN.items():
        if not {name, parent} <= set(results):
            continue
        child, child_status, child_model = results[name]
        base, _, base_model = results[parent]
        # same centre and radius is what MAKES the child's set a subset -- gate it
        # exactly, so the numeric check below only ever measures solver noise.
        # For intersections the DA BRANCH is the one sharing the parent's ball.
        branch = getattr(child_model, "augmented", child_model)
        same_ball = branch.gamma == base_model.gamma and branch.sigma2_ == base_model.sigma2_
        check(f"A27 {label} {name} in {parent}: same ball", same_ball, hard=name not in EXEMPT)
        both = np.isfinite(child).all(axis=1) & np.isfinite(base).all(axis=1)
        if not both.any():
            check(f"A27 {label} {name} in {parent}: nesting", True, "no mutually-finite queries", hard=False)
            continue
        low_gap = float(np.max(base[both, 0] - child[both, 0]))  # parent lower above child's
        high_gap = float(np.max(child[both, 1] - base[both, 1]))
        violation = max(low_gap, high_gap, 0.0)
        check(
            f"A27 {label} {name} in {parent}: nesting",
            violation <= NEST_TOL,
            f"max violation {violation:.2e} on {int(both.sum())} queries",
            hard=name not in EXEMPT,
        )


def gate_membership(results, h_star, label):
    """Lemma 2 at work: the analytic h_* must sit inside the PI ball's bounds at
    gamma = gamma* on (almost) every query."""
    for name, hard in (("PI", True), ("DA+PI", False)):
        bounds, _, _ = results[name]
        finite = np.isfinite(bounds).all(axis=1)
        inside = (bounds[:, 0] - 1e-9 <= h_star) & (h_star <= bounds[:, 1] + 1e-9) & finite
        rate = float(inside.sum()) / len(h_star)
        check(f"A27 {label} {name}: h_* membership >= 95%", rate >= 0.95, f"{rate:.1%}", hard=hard)


def _fd_check(label, jax_vg, numpy_value, theta, coords, step=1e-6):
    """value: JAX == numpy mirror exactly; gradient: JAX == central FD of the mirror."""
    value, gradient = jax_vg(theta)
    value, gradient = float(value), np.asarray(gradient, dtype=float)
    delta = abs(value - numpy_value(theta))
    check(f"A27 grad: JAX {label} == numpy mirror", delta <= 1e-12, f"|d| {delta:.2e}")

    worst = 0.0
    for j in coords:
        up, down = theta.copy(), theta.copy()
        up[j] += step
        down[j] -= step
        fd = (numpy_value(up) - numpy_value(down)) / (2 * step)
        worst = max(worst, abs(fd - gradient[j]) / (1e-4 + abs(gradient[j])))
    check(f"A27 grad: JAX {label} grad == finite differences", worst <= 1e-4, f"max rel|d| {worst:.2e}")


def gate_gradients(nets, X, GX, y, G):
    """A8 discipline for the new module: every JAX term against the numpy mirror,
    values exactly, gradients by central finite differences."""
    from src.methods.partial_r2_net import IVConstrainedPartialR2Net, RecentredInvPartialR2Net

    pi = PartialR2Net(gamma=GAMMA_STAR, calibrate=True, outcome_model=nets["X"], n_jobs=1).fit(X, y)
    inv = RecentredInvPartialR2Net(
        gamma=GAMMA_STAR, epsilon=EPSILON, calibrate=True, outcome_model=nets["GX"], n_jobs=1
    ).fit(X, y, GX=GX)
    iv = IVConstrainedPartialR2Net(
        gamma=GAMMA_STAR, epsilon_iv=EPSILON_IV, calibrate=True, outcome_model=nets["GX"], n_jobs=1
    ).fit(GX, y, Z=G)

    rng = np.random.default_rng(0)
    theta = pi.theta_c_ + 0.05 * np.sqrt(np.mean(pi.theta_c_**2)) * rng.standard_normal(pi.theta_c_.size)
    coords = rng.choice(theta.size, 24, replace=False)
    phi_q = pi.phi_[0]

    def index_np(t):
        return head_index(t, phi_q[None, :], pi.head_shapes_)[0]

    objective, r2_vg, _, _ = pi._get_terms()
    _fd_check("E_r2", r2_vg, pi.r2_value, theta, coords)
    _fd_check("index", lambda t: objective(t, phi_q), index_np, theta, coords)
    _fd_check("E_inv", inv._get_terms()[2], inv._extra_value, theta, coords)
    _fd_check("R_iv", iv._get_terms()[2], iv._extra_value, theta, coords)


def main():
    print(f"== l=1 at n={N_SAMPLES:,} n_pi={N_PI:,} q={N_QUERIES} gamma*={GAMMA_STAR:.4f} ==")
    nets, X, GX, y, G, Q, h_star = fixture(N_SAMPLES, N_PI, N_QUERIES)
    results = fit_and_predict(nets, X, GX, y, G, Q, unfrozen_layers=1)
    gate_statuses(results, "l1")
    gate_nesting(results, "l1")
    gate_membership(results, h_star, "l1")
    gate_gradients(nets, X, GX, y, G)

    print(f"== l=2 at n={L2_SAMPLES:,} n_pi={L2_PI:,} q={L2_QUERIES} (correctness only; AL path) ==")
    nets, X, GX, y, G, Q, h_star = fixture(L2_SAMPLES, L2_PI, L2_QUERIES)
    results = fit_and_predict(nets, X, GX, y, G, Q, unfrozen_layers=2)
    gate_statuses(results, "l2")
    gate_nesting(results, "l2")
    for name, (bounds, _, _) in results.items():
        finite = np.isfinite(bounds).all(axis=1)
        ordered = bool((bounds[finite, 0] <= bounds[finite, 1] + 1e-12).all())
        in_range = bool((bounds[finite] >= -1e-9).all() and (bounds[finite] <= 1 + 1e-9).all())
        check(f"A27 l2 {name}: bounds ordered and in [0, 1]", ordered and in_range, hard=name not in EXEMPT)

    print("\n" + ("ALL PASS" if not FAIL else f"FAILURES: {FAIL}"))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
