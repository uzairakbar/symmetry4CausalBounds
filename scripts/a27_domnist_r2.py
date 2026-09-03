"""A27: the partial_r2_net backend's own gates.

Four properties, none of which SOURCE parity can pin for a new model:
  1. nesting -- a constrained method never comes out WIDER than the unconstrained
     model on the same ball (adding a constraint can only shrink the feasible set);
  2. membership -- the ANALYTIC h_* (exact for this SEM) sits inside the PI bounds
     at gamma = gamma* for >= 95% of held-out queries -- something the retired
     latent-factor model could never certify: h_* was not in its candidate class.
  3. JAX == numpy-mirror finite differences on every term of the NLP (A8 style);
  4. l=2 exercised at reduced scale: correctness + nesting on the AL path.

PI+INV is REPORTED but exempt from hard-fail: realizability under a shallow refit
is an open empirical question (all-INFEASIBLE there is honest behaviour, not a
bug -- read the floor/budget pair it logs).

    python scripts/a27_domnist_r2.py
    python scripts/a27_domnist_r2.py --micro                 # 4k/512/4, minutes
    python scripts/a27_domnist_r2.py --micro --band-se 0.146 # forces the slab anchor

0.146 is the largest MEAN_BAND_SE that puts tau BELOW the micro fixture's own
|m(theta_c)|, so `--band-se 0.146` is the leg that drives theta_c out of the band
and exercises `_band_anchor`'s constrained branch; production runs never take it.
    python scripts/a27_domnist_r2.py --polish-compare        # single vs full polish widths
    python scripts/a27_domnist_r2.py --compare-off          # what the band costs and moves
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_augmentors.do_mnist import DoMNISTDA  # noqa: E402
from src.experiments.configs import DOMNIST_CONFIG, MethodRegistry  # noqa: E402
from src.experiments.do_mnist import NESTED_IN, Flatten  # noqa: E402
from src.experiments.utils import set_seed  # noqa: E402
from src.methods.partial_r2_net import (  # noqa: E402
    MEAN_BAND_SE,
    PartialR2Net,
    head_index,
)
from src.methods.regression import GradientDescentERM  # noqa: E402
from src.methods.sensitivity_models import SolveStatus  # noqa: E402
from src.sem.do_mnist import DoMNISTSEM  # noqa: E402

N_SAMPLES, N_PI, N_QUERIES = 60_000, 6_000, 32
L2_SAMPLES, L2_PI, L2_QUERIES = 20_000, 2_000, 4
# --micro: the only do-MNIST scale this refactor is allowed to run (PLAN v2 C4).
# Small enough to finish in minutes, large enough to exercise every path.
MICRO = (4_000, 512, 4)
MICRO_L2 = (2_000, 256, 1)
# micro runs SERIAL: with 4 queries the loky workers spend more time importing
# torch and JIT-compiling the jitted terms than they save on the solves
MICRO_JOBS = 1
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
# reported, never hard-failed: realizability under a shallow refit is an open
# empirical question (see RecentredInvPartialR2Net); PLAN decision 2 additionally
# bars hard-failing its INFEASIBILITY at l=1
EXEMPT = {"PI+INV"}
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


def fit_and_predict(nets, X, GX, y, G, Q, unfrozen_layers, n_jobs=N_JOBS, mean_match=True):
    """{method: (bounds, status, model)} for the six PI variants, wall-clock logged."""
    builders = MethodRegistry.build_methods(
        ["PI", "DA+PI", "PI+INV", "DA+PI+IV", "PI&DA+PI", "PI&DA+PI+IV"],
        gamma=GAMMA_STAR,
        epsilon=EPSILON,
        epsilon_iv=EPSILON_IV,
        calibrate=True,
        clipy=True,
        n_jobs=n_jobs,
        mean_match=mean_match,
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
        # the SLAB is part of the set too: same ball + same band => the child's
        # feasible set really is a subset, whatever the extra constraint does
        same_ball = (
            branch.gamma == base_model.gamma and branch.sigma2_ == base_model.sigma2_ and branch._tau == base_model._tau
        )
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

    objective, r2_vg, _, _, band_vg = pi._get_terms()
    _fd_check("E_r2", r2_vg, pi.r2_value, theta, coords)
    _fd_check("index", lambda t: objective(t, phi_q), index_np, theta, coords)
    _fd_check("E_inv", inv._get_terms()[2], inv._extra_value, theta, coords)
    _fd_check("R_iv", iv._get_terms()[2], iv._extra_value, theta, coords)
    if band_vg is not None:  # Lem. 2's band; both sides are the SIGNED defect m
        _fd_check("E_mean", band_vg, pi.mean_value, theta, coords)


def gate_band(results, label, n_pi):
    """Lem. 2's slab: the right width, enforced on acceptance, and actually
    travelled in (a slab too thin to move in shows up as every start backtracking
    onto the anchor -- the silent-narrowing failure mode).

    Written so that DELETING the band clause from `PartialR2Net._feasible` makes
    this fail. That is a real hazard: the obvious "shift the level far out and
    check it is rejected" probe is answered by the R^2 BALL long before the band
    is consulted, and passes a model with no band at all.
    """
    for name, (_, _, model) in results.items():
        branch = getattr(model, "augmented", model)
        if branch._tau is None:
            check(f"A27 {label} {name}: band is on", False)
            continue

        # from the sensitivity model's own bound on Var(U + xi) (see MEAN_BAND_SE),
        # written out rather than read back from `_band_tau` -- and via the BUDGET,
        # so the `calibrate=False` units are covered too
        budget = branch.scale**2 * branch.gamma
        want = MEAN_BAND_SE * np.sqrt((branch.sigma2_ + budget) / n_pi)
        check(
            f"A27 {label} {name}: tau == {MEAN_BAND_SE} sqrt((sigma^2 + b_r2)/n)",
            abs(branch._tau - want) < 1e-12,
            f"{branch._tau:.5g}",
        )
        # tau must MOVE with the budget: a tau that ignored gamma would pass the
        # identity above at the single gamma this fixture runs at
        check(
            f"A27 {label} {name}: tau grows with the budget",
            branch._band_tau(4.0 * branch.gamma + 1.0) > branch._tau,
            f"{branch._tau:.5g} -> {branch._band_tau(4.0 * branch.gamma + 1.0):.5g}",
        )

        if branch._ctx is None:  # _prepare bailed: every query INFEASIBLE, no anchor
            check(f"A27 {label} {name}: an anchor was prepared", False)
            continue

        anchor = branch._ctx["anchor"]
        check(
            f"A27 {label} {name}: the anchor is band-feasible",
            abs(branch.mean_value(anchor)) <= branch._tau * (1 + 1e-6) + 1e-9,
            f"|m| {abs(branch.mean_value(anchor)):.4g} vs tau {branch._tau:.4g}",
        )

        # Step along grad m so the LEVEL moves and little else: m(anchor + t g/|g|^2)
        # ~ m(anchor) + t, so aim just past the slab. Deliberately NOT a big shift --
        # the point is that the ball still accepts this point, so the only thing that
        # can reject it is the band.
        gradient = np.asarray(branch._get_terms()[4](anchor)[1], dtype=float)
        scale = float(gradient @ gradient)
        target = np.sign(branch.mean_value(anchor) or 1.0) * 1.5 * branch._tau
        outside = anchor + ((target - branch.mean_value(anchor)) / max(scale, 1e-30)) * gradient
        check(
            f"A27 {label} {name}: the shifted point clears the ball on its own",
            branch.r2_value(outside) <= branch._ctx["b_r2"] * (1 + 1e-6) + 1e-9,
            f"E_r2 {branch.r2_value(outside):.4g} vs budget {branch._ctx['b_r2']:.4g}",
        )
        check(
            f"A27 {label} {name}: a level shift past tau is rejected",
            not branch._feasible(outside, branch._ctx["b_r2"], None),
            f"|m| {abs(branch.mean_value(outside)):.4g} vs tau {branch._tau:.4g}",
        )

        diagnostics = getattr(model, "query_diagnostics", None)
        check(f"A27 {label} {name}: backtrack diagnostics reported", diagnostics is not None and len(diagnostics) > 0)
        if diagnostics is not None and len(diagnostics):
            fraction = float(np.mean(diagnostics[:, 0]))
            check(f"A27 {label} {name}: starts backtracked to the anchor < 50%", fraction < 0.5, f"{fraction:.1%}")


def polish_compare(nets, X, GX, y, G, Q):
    """Single-polish vs full multi-start widths at the CURRENT tau, for the two
    programs whose only extra constraint is the band. `<= 1e-3` everywhere is what
    would license SINGLE_POLISH_WITH_BAND; measure it at production scale, never
    on a micro fixture where the band is inert."""
    import src.methods.partial_r2_net as pr2

    for name, design, net in (("PI", X, "X"), ("DA+PI", GX, "GX")):
        model = PartialR2Net(
            gamma=GAMMA_STAR, epsilon=EPSILON, calibrate=True, outcome_model=nets[net], n_jobs=N_JOBS
        ).fit(design, y)
        full = model.predict(Q)
        try:  # same feasible set, fewer polish solves -- the only thing that moves
            pr2.SINGLE_POLISH_WITH_BAND = True
            single = model.predict(Q)
        finally:
            pr2.SINGLE_POLISH_WITH_BAND = False
        delta = float(np.nanmax(np.abs((full[:, 1] - full[:, 0]) - (single[:, 1] - single[:, 0]))))
        print(f"  polish-compare {name}: max |width(full) - width(single)| = {delta:.3e} (tau {model._tau:.4g})")


# The band-on and band-off solves differ in more than the feasible set: the anchor,
# the floor point, the directed starts (tangent-projected only under the band) and,
# for PI/DA+PI, the polish policy all move with it. So the widths are two HEURISTIC
# optima of nested sets, not two exact ones, and band-on coming out slightly wider
# means the band-OFF solve was the looser of the two -- informative about the old
# path, not a violated monotonicity. Only a gap past this is worth a look.
BAND_COMPARE_TOL = 0.02


def compare_off(nets, X, GX, y, G, Q, jobs, results_on):
    """What the band actually costs and moves, measured rather than asserted: the
    same fixture solved with `mean_match=False`. PLAN v2 SS4f asked for this next to
    the per-solve timings."""
    print("== the same l=1 fixture with mean matching OFF ==")
    results_off = fit_and_predict(nets, X, GX, y, G, Q, unfrozen_layers=1, n_jobs=jobs, mean_match=False)
    for name, (bounds_off, _, _) in results_off.items():
        bounds_on = results_on[name][0]
        width_on = float(np.nanmean(bounds_on[:, 1] - bounds_on[:, 0]))
        width_off = float(np.nanmean(bounds_off[:, 1] - bounds_off[:, 0]))
        print(f"  band cost {name:12s} width {width_off:.4f} (off) -> {width_on:.4f} (on)")
        check(
            f"A27 l1 {name}: band-on and band-off widths agree to {BAND_COMPARE_TOL}",
            width_on <= width_off + BAND_COMPARE_TOL,
            f"{width_off:.4f} -> {width_on:.4f}",
            hard=False,
        )


def main(micro=False, polish=False, compare=False):
    sizes = MICRO if micro else (N_SAMPLES, N_PI, N_QUERIES)
    sizes_l2 = MICRO_L2 if micro else (L2_SAMPLES, L2_PI, L2_QUERIES)
    print(
        f"== l=1 at n={sizes[0]:,} n_pi={sizes[1]:,} q={sizes[2]} gamma*={GAMMA_STAR:.4f} "
        f"MEAN_BAND_SE={MEAN_BAND_SE} =="
    )
    nets, X, GX, y, G, Q, h_star = fixture(*sizes)
    jobs = MICRO_JOBS if micro else N_JOBS
    results = fit_and_predict(nets, X, GX, y, G, Q, unfrozen_layers=1, n_jobs=jobs)
    gate_statuses(results, "l1")
    gate_nesting(results, "l1")
    gate_membership(results, h_star, "l1")
    gate_band(results, "l1", sizes[1])
    gate_gradients(nets, X, GX, y, G)
    if polish:
        polish_compare(nets, X, GX, y, G, Q)
    if compare:
        compare_off(nets, X, GX, y, G, Q, jobs, results)

    print(f"== l=2 at n={sizes_l2[0]:,} n_pi={sizes_l2[1]:,} q={sizes_l2[2]} (correctness only; AL path) ==")
    nets, X, GX, y, G, Q, h_star = fixture(*sizes_l2)
    results = fit_and_predict(nets, X, GX, y, G, Q, unfrozen_layers=2, n_jobs=jobs)
    gate_statuses(results, "l2")
    gate_nesting(results, "l2")
    gate_band(results, "l2", sizes_l2[1])  # the AL path has its own band plumbing
    for name, (bounds, _, _) in results.items():
        finite = np.isfinite(bounds).all(axis=1)
        ordered = bool((bounds[finite, 0] <= bounds[finite, 1] + 1e-12).all())
        in_range = bool((bounds[finite] >= -1e-9).all() and (bounds[finite] <= 1 + 1e-9).all())
        check(f"A27 l2 {name}: bounds ordered and in [0, 1]", ordered and in_range, hard=name not in EXEMPT)

    print("\n" + ("ALL PASS" if not FAIL else f"FAILURES: {FAIL}"))
    return 1 if FAIL else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--micro", action="store_true", help="4k/512/4 fixture (PLAN v2 C4)")
    parser.add_argument("--band-se", type=float, default=None, help="override MEAN_BAND_SE BEFORE any fit")
    parser.add_argument("--polish-compare", action="store_true", help="single vs full polish widths")
    parser.add_argument("--compare-off", action="store_true", help="also solve l=1 with mean_match=False")
    args = parser.parse_args()
    if args.band_se is not None:
        # BEFORE the fits: tau, the anchor and the floor cache are all built from
        # it, so a post-fit override would leave a band-infeasible anchor behind
        import src.methods.partial_r2_net as _pr2

        _pr2.MEAN_BAND_SE = MEAN_BAND_SE = args.band_se
        print(f"MEAN_BAND_SE overridden to {args.band_se}")
    sys.exit(main(micro=args.micro, polish=args.polish_compare, compare=args.compare_off))
