"""A28: the linear PI programs live on Lem. 2's mean-matched slice.

Lem. 2 puts the identified set on H_X = {h : E[h(X)] = E[Y]}, a COVARIANCE ball
around the with-intercept ERM. Production reaches that set by ELIMINATING the
intercept (centred design and outcome, bounds shifted back by ybar); this gate
restates it EXPLICITLY -- a free intercept coordinate plus an equality
constraint -- and checks the two agree, query by query.

  (i)   classes == the explicit cvxpy reference, both calibrations, and n_jobs
        1 == 4 bit-identically;
  (ii)  the closed-form shortcut == the SOCP == Cor. 3 by hand;
  (iii) the returned optimum really sits on the slice;
  (iv)  `constraint_floor` == the explicit floor (the budget guard must measure
        the ball the solver actually uses);
  (v)   coverage of h_* at gamma* does not degrade against the uncentred run.

    python scripts/a28_mean_match.py
"""

import os
import sys

import cvxpy as cp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.methods.sensitivity_models as sm  # noqa: E402
from src.experiments.base import SweepData  # noqa: E402
from src.experiments.optical_device import OpticalOrchestrator  # noqa: E402
from src.experiments.simulation import SimulationOrchestrator  # noqa: E402
from src.experiments.utils import set_seed  # noqa: E402
from src.methods.sensitivity_models import (  # noqa: E402
    constraint_floor,
    inv_constraint_terms,
    iv_constraint_terms,
)

GAMMA, EPSILON, EPSILON_IV = 0.5, 0.3, 0.2
BOUND_TOL = 1e-5  # bounds are O(5) on these fixtures
FAIL = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAIL.append(name)


# ------------------------------------------------------------------- fixtures


def a10_fixture(shift=0.0):
    """a10's own `direct_solves` draw (scripts/a10_partial_r2_regression.py:37-49),
    optionally shifted so the design mean is far from zero -- centring is a no-op
    on a mean-zero design, which would make leg (i) vacuous."""
    set_seed(3)
    X = np.random.randn(200, 6)
    y = X @ np.random.randn(6, 1) + 0.3 * np.random.randn(200, 1)
    GX = X + 0.2 * np.random.randn(*X.shape)
    Q = np.random.randn(60, 6)
    # X and GX move TOGETHER: the invariance constraint is about GX - X, and
    # shifting only one would silently turn it into a constant-offset constraint
    return X + shift, GX + shift, y, Q + shift


def optical_fixture():
    set_seed(69)
    orch = OpticalOrchestrator(
        seed=69,
        n_samples=1000,
        n_experiments=1,
        sweep_samples=8,
        methods=["PI", "DA+PI"],
        hyperparameters={},
        n_jobs=1,
        calibrate=True,
        pad=False,
        clipy=True,
        augmentation="rotation > gaussian-noise",
    )
    runner = orch.get_sweep_runner_cls("gamma")(
        methods=orch.methods,
        method_factory=orch.build_methods,
        **{k: v for k, v in orch.kwargs.items() if k != "methods"},
    )
    data = SweepData.coerce(runner.generate_data(0, runner.get_param_range()[0]))
    gamma_star = runner.fit_gamma(0)
    return data, gamma_star


# --------------------------------------------------- the explicit L1 reference


def _ball(design, y, gamma, calibrate):
    """(R, h1_erm, delta, D, ybar) for the explicit parameterisation on [X, 1]."""
    design, y = np.asarray(design), np.asarray(y).flatten()
    N = len(design)
    D = np.hstack([design, np.ones((N, 1))])
    h1_erm = np.linalg.lstsq(D, y, rcond=None)[0]
    residual = y - D @ h1_erm
    scale = float(np.sqrt(np.mean(residual**2))) if calibrate else 1.0
    _, R = np.linalg.qr(D)
    return R, h1_erm, np.sqrt(N) * scale * np.sqrt(max(gamma, 0.0)), D, float(np.mean(y))


def l1_bounds(design, y, queries, gamma, *, calibrate, kind=None, GX=None, Z=None, epsilon=None, want_h=False):
    """min/max h(x) over Lem. 2's set, stated explicitly: an intercept coordinate
    and the equality mean_n([X, 1]) h = ybar, solved by cvxpy.

    The extra constraint's TERMS are production's (they did not change in this
    branch); the GEOMETRY -- ball, slice, parameterisation -- is restated here.
    """
    design = np.asarray(design)
    N, M = design.shape
    R, h1_erm, delta, D, ybar = _ball(design, y, gamma, calibrate)
    mu = design.mean(axis=0)
    centred = design - mu

    h1 = cp.Variable(M + 1)
    constraints = [
        cp.norm(cp.Constant(R) @ (h1 - cp.Constant(h1_erm)), 2) <= delta,
        cp.Constant(D.mean(axis=0)) @ h1 == ybar,
    ]
    if kind == "inv":
        A, b = inv_constraint_terms(centred, np.asarray(GX) - mu)
        constraints.append(cp.norm(cp.Constant(A) @ h1[:M] - cp.Constant(b), 2) <= np.sqrt(N) * epsilon)
    elif kind == "iv":
        A, b = iv_constraint_terms(centred, np.asarray(y).flatten() - ybar, Z)
        constraints.append(cp.norm(cp.Constant(A) @ h1[:M] - cp.Constant(b), 2) <= np.sqrt(N) * epsilon)

    out, attained = [], []
    for x in np.asarray(queries):
        row = cp.Constant(np.append(x, 1.0))
        pair, solutions = [], []
        for sense in (cp.Minimize, cp.Maximize):
            problem = cp.Problem(sense(row @ h1), constraints)
            value = np.nan
            for solver in (cp.CLARABEL, cp.ECOS):
                try:
                    problem.solve(solver=solver, warm_start=False, verbose=False)
                except Exception:  # noqa: S112 - solver fallback chain
                    continue
                if problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                    value = problem.value
                    solutions.append(np.asarray(h1.value, dtype=float))
                    break
                if problem.status in (cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE):
                    break
            pair.append(value)
        out.append(pair)
        attained.append(solutions)
    return (np.array(out), attained, D) if want_h else np.array(out)


# ------------------------------------------- (i) classes == explicit reference


def leg_i():
    for shift, label in ((1.5, "shifted design"), (0.0, "a10's own draw")):
        X, GX, y, Q = a10_fixture(shift)
        for calibrate in (True, False):
            tag = f"{label}, calibrate={calibrate}"
            # clipy=False: this leg compares the BALL, and `_finalize`'s clip to
            # the observed y range would otherwise mask (or fake) an agreement.
            # The clip itself is unchanged code, exercised by a10 and the runs.
            common = dict(gamma=GAMMA, epsilon=EPSILON, calibrate=calibrate, mean_match=True, clipy=False)
            # every fixture array is bound as a default: ruff B023, and the
            # builders are called later in the loop body
            cases = {
                "PI": (
                    lambda nj, c=common, X=X, y=y: sm.PartialR2(n_jobs=nj, **c).fit(X, y),
                    dict(design=X, y=y),
                ),
                "PI+INV": (
                    lambda nj, c=common, X=X, y=y, GX=GX: sm.InvarianceConstrainedPartialR2(n_jobs=nj, **c).fit(
                        X, y, GX=GX
                    ),
                    dict(design=X, y=y, kind="inv", GX=GX, epsilon=EPSILON),
                ),
                "PI+IV(null Z)": (
                    lambda nj, c=common, X=X, y=y: sm.InstrumentalVariablePartialR2(
                        epsilon_iv=EPSILON_IV, n_jobs=nj, **c
                    ).fit(X, y, Z=None),
                    dict(design=X, y=y),
                ),
                "DA+PI+IV": (
                    lambda nj, c=common, y=y, GX=GX: sm.InstrumentalVariablePartialR2(
                        epsilon_iv=EPSILON_IV, n_jobs=nj, **c
                    ).fit(GX, y, Z=GX),
                    dict(design=GX, y=y, kind="iv", Z=GX, epsilon=EPSILON_IV),
                ),
            }
            for name, (build, reference_kw) in cases.items():
                model = build(1)
                got = model.predict(Q)
                want = l1_bounds(queries=Q, gamma=GAMMA, calibrate=calibrate, **reference_kw)
                finite = np.isfinite(got).all() and np.isfinite(want).all()
                check(f"A28 (i) {name} [{tag}]: both sides solved", finite)
                if finite:
                    delta = float(np.abs(got - want).max())
                    check(f"A28 (i) {name} [{tag}]: == explicit reference", delta <= BOUND_TOL, f"max |d| {delta:.2e}")
                parallel = build(4).predict(Q)
                check(f"A28 (i) {name} [{tag}]: n_jobs 1 == 4", np.array_equal(got, parallel))

            # the intersections: Cor. 1 at the interval level, on both branches
            for name, model, branches in (
                (
                    "PI&DA+PI",
                    sm.IntersectedPartialR2(n_jobs=1, **common).fit(X, y, GX=GX, G=GX),
                    (dict(design=X, y=y), dict(design=GX, y=y)),
                ),
                (
                    "PI&DA+PI+IV",
                    sm.IntersectedInstrumentalVariablePartialR2(epsilon_iv=EPSILON_IV, n_jobs=1, **common).fit(
                        X, y, GX=GX, G=GX
                    ),
                    (dict(design=X, y=y), dict(design=GX, y=y, kind="iv", Z=GX, epsilon=EPSILON_IV)),
                ),
            ):
                got = model.predict(Q)
                base = l1_bounds(queries=Q, gamma=GAMMA, calibrate=calibrate, **branches[0])
                augmented = l1_bounds(queries=Q, gamma=GAMMA, calibrate=calibrate, **branches[1])
                want = np.column_stack(
                    [np.maximum(base[:, 0], augmented[:, 0]), np.minimum(base[:, 1], augmented[:, 1])]
                )
                delta = float(np.abs(got - want).max())
                check(f"A28 (i) {name} [{tag}]: == explicit reference", delta <= BOUND_TOL, f"max |d| {delta:.2e}")


# --------------------------------------- (ii) closed form == SOCP == Cor. 3


def leg_ii():
    X, GX, y, Q = a10_fixture(1.5)
    for calibrate in (True, False):
        kw = dict(gamma=GAMMA, epsilon=EPSILON, calibrate=calibrate, mean_match=True, clipy=False, n_jobs=1)
        socp = sm.PartialR2(**kw).fit(X, y)
        socp_bounds = socp.predict(Q)

        sm.CLOSED_FORM_SOLUTION = True
        try:
            closed = sm.PartialR2(**kw).fit(X, y)
            closed_bounds = closed.predict(Q)
        finally:
            sm.CLOSED_FORM_SOLUTION = False

        delta = float(np.abs(socp_bounds - closed_bounds).max())
        check(f"A28 (ii) closed form == SOCP, calibrate={calibrate}", delta <= 1e-6, f"max |d| {delta:.2e}")

        # Cor. 3 by hand: h_erm(x) +- s sqrt(gamma) ||g_x||, g_x the representer
        mu, ybar = X.mean(axis=0), float(np.mean(y))
        Xc, yc = X - mu, np.asarray(y).flatten() - ybar
        h_erm = np.linalg.lstsq(Xc, yc, rcond=None)[0]
        scale = float(np.sqrt(np.mean((yc - Xc @ h_erm) ** 2))) if calibrate else 1.0
        cov_inv = np.linalg.pinv(Xc.T @ Xc / len(Xc))
        Qc = Q - mu
        margin = scale * np.sqrt(GAMMA) * np.sqrt(np.maximum(0.0, np.sum((Qc @ cov_inv) * Qc, axis=1)))
        centre = Qc @ h_erm + ybar
        want = np.column_stack([centre - margin, centre + margin])
        delta = float(np.abs(socp_bounds - want).max())
        check(f"A28 (ii) SOCP == Cor. 3 closed form, calibrate={calibrate}", delta <= 1e-6, f"max |d| {delta:.2e}")


# ------------------------------------------------- (iii) on the slice, really


def leg_iii():
    X, GX, y, Q = a10_fixture(1.5)
    ybar = float(np.mean(y))
    _, attained, D = l1_bounds(X, y, Q[:12], GAMMA, calibrate=True, want_h=True, kind="inv", GX=GX, epsilon=EPSILON)
    worst = 0.0
    for solutions in attained:
        for h1 in solutions:
            worst = max(worst, abs(float(D.mean(axis=0) @ h1) - ybar))
    check("A28 (iii) explicit optimum satisfies E_n[h(X)] = ybar", worst <= 1e-8, f"worst |d| {worst:.2e}")

    # elimination side: the slice is structural, h_erm and the Cor. 3 extremals
    # all satisfy it by construction -- check the identity the code relies on
    model = sm.PartialR2(gamma=GAMMA, epsilon=EPSILON, calibrate=True, mean_match=True, n_jobs=1).fit(X, y)
    residual = abs(float(np.mean((X - model.mu_) @ model.h_erm)))
    check("A28 (iii) eliminated form: mean_n((X - mu) h_erm) == 0", residual <= 1e-12, f"{residual:.2e}")
    fitted = float(np.mean((X - model.mu_) @ model.h_erm + model.y_offset_))
    check(
        "A28 (iii) eliminated form: mean_n(h_erm(X)) == ybar", abs(fitted - ybar) <= 1e-12, f"{abs(fitted - ybar):.2e}"
    )


# ------------------------------------------------------- (iv) the floor guard


def leg_iv():
    from scripts.a25_floor_guard import cvxpy_floor  # the explicit reference

    data, gamma_star = optical_fixture()
    for gamma in (0.05, gamma_star, 4 * gamma_star):
        for calibrate in (True, False):
            for kind, kw in (("inv", dict(GX=data.GX)), ("iv", dict(Z=data.G))):
                design = data.X if kind == "inv" else data.GX
                got = constraint_floor(design, data.y, gamma, kind=kind, calibrate=calibrate, mean_match=True, **kw)
                want = cvxpy_floor(design, data.y, gamma, kind, calibrate=calibrate, mean_match=True, **kw)
                if abs(got - want) < 1e-11:  # both zero to solver precision
                    check(f"A28 (iv) optical {kind} floor, gamma={gamma:.4g}, calibrate={calibrate}", True, "both ~0")
                    continue
                relative = abs(got - want) / max(abs(want), 1e-300)
                check(
                    f"A28 (iv) optical {kind} floor, gamma={gamma:.4g}, calibrate={calibrate}",
                    relative < 1e-6,
                    f"rel {relative:.2e}",
                )


# ------------------------------------------------------------ (v) coverage


def _gamma_step(orchestrator_factory, mean_match):
    """Coverage/width at gamma = gamma* on a 1-point gamma sweep."""
    orch = orchestrator_factory(mean_match)
    runner = orch.get_sweep_runner_cls("gamma")(
        methods=orch.methods,
        method_factory=orch.build_methods,
        param_grid_override=[1.0],
        **{k: v for k, v in orch.kwargs.items() if k != "methods"},
    )
    _, results, _ = runner.run("a28 gamma*")
    return {
        name: (float(np.nanmean(r["coverage"])), float(np.nanmean(r["interval_width"]))) for name, r in results.items()
    }


def leg_v():
    methods = ["PI", "DA+PI", "PI&DA+PI"]

    def simulation(mean_match):
        set_seed(42)
        return SimulationOrchestrator(
            seed=42,
            n_samples=2048,
            n_experiments=2,
            sweep_samples=8,
            kernel_dim=0,
            methods=methods,
            hyperparameters={},
            n_jobs=-1,
            calibrate=True,
            pad=True,
            clipy=True,
            mean_match=mean_match,
        )

    def optical(mean_match):
        set_seed(42)
        return OpticalOrchestrator(
            seed=42,
            n_samples=1000,
            n_experiments=2,
            sweep_samples=8,
            methods=methods,
            hyperparameters={},
            n_jobs=-1,
            calibrate=True,
            pad=True,
            clipy=True,
            mean_match=mean_match,
            augmentation="rotation > gaussian-noise",
        )

    for label, factory in (("sim", simulation), ("optical", optical)):
        centred = _gamma_step(factory, True)
        uncentred = _gamma_step(factory, False)
        for name in methods:
            coverage, width = centred[name]
            base_coverage, base_width = uncentred[name]
            check(
                f"A28 (v) {label} {name}: coverage at gamma* does not degrade",
                coverage >= base_coverage - 0.02,
                f"{coverage:.3f} vs {base_coverage:.3f} uncentred | width {width:.4f} vs {base_width:.4f}",
            )


if __name__ == "__main__":
    leg_i()
    leg_ii()
    leg_iii()
    leg_iv()
    leg_v()
    print(f"\n{'A28 ALL PASS' if not FAIL else 'A28 FAILURES: ' + chr(10) + chr(10).join('  ' + f for f in FAIL)}")
    sys.exit(bool(FAIL))
