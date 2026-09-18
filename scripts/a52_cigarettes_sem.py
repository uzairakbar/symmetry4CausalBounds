"""A52: `CigaretteSEM` carries the right target under both flags, replicates the way
SS6 says, and never lets the real instrument near a solver.

  (i)    `iv`: `solution` is the restricted-IV point, recomputed here from the
         normal equations of the projected program rather than by calling the
         loader's QR form; v'solution is machine zero; gamma* is 0.1922 and
         sigma^2 is 1. Catches: the unrestricted 2SLS passed off as the target
         (v'b is 0.271/sigma there), a target that forgot the normalisation.
  (ii)   `sample` under both flags. `pool` is byte-identical across constructions
         and across draws either way -- the oracle reads `pool`, so what `sample`
         does must never move h_*. `bootstrap=False` returns the panel itself at
         N = 2450; `bootstrap=True` returns WHOLE state histories drawn with
         replacement, and over 20 seeds the MEAN distinct-state count sits at the
         49(1 - (48/49)^49) = 31.2 the scheme implies. The mean and not a point
         value: single draws run 28 to 36. Catches: an iid row bootstrap, a
         re-fitted target inside `sample`.
  (iii)  `plasmode`: bias^2 and sigma^2 are the POPULATION values the calibration
         was inverted for, exactly, so gamma* == gamma_true; and the sample
         gamma-hat* of a construction draw tracks it. Catches: a bias read off the
         draw. The sample band is wide because it has to be -- see BAND below.
  (iv)   all three confounding directions calibrate, in the declared value AND in
         the draw. The draw half is what makes this falsifiable: the declared
         gamma* is analytic in kappa alone, so it cannot see a confounder that was
         never standardised, and the sample ratio can.
  (v)    the real instrument never reaches a solver. `fit_model` hands DA+PI+IV the
         DA's translation amounts as Z, so (a) what the method received is not the
         tax column and is uncorrelated with it, and (b) the IV constraint it built
         is exactly `iv_constraint_terms` of those amounts. Catches: a SEM or DA
         that smuggles the tax column in as G, which no shape assertion would --
         the tax column is (n, 1) too.
  (vi)   the compatibility statistics: J2 under three clusterings, its agreement
         with the unrestricted Wald, and the gap to the one-step J1w that caused
         the apparent disagreement. Catches: the GMM weight carrying the Wald's
         finite-sample factor, J evaluated at the one-step estimate.
  (vii)  `extent` degenerates to the point target under the shipped flag, is 6.3%
         of the PI width with the guard on at t3, and is EMPTY at t2. Catches: a
         slack that forgot the minimised misfit r0 (t2 would then report a set).
  (viii) the sliver radius is sigma sqrt(gamma_z), not gamma_z: doubling the guard
         multiplies the extent by exactly sqrt((2g - r0^2)/(g - r0^2)).

    python scripts/a52_cigarettes_sem.py
"""

import os
import sys

import numpy as np
from loguru import logger

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from src.data_augmentors.cigarettes import ScaleTranslation  # noqa: E402
from src.experiments.configs import EPS_TOL, MethodRegistry  # noqa: E402
from src.experiments.utils.metrics import sigma_sq_hat  # noqa: E402
from src.experiments.utils.model_fitting import fit_model  # noqa: E402
from src.methods.sensitivity_models import iv_constraint_terms  # noqa: E402
from src.oracle import gamma_star  # noqa: E402
from src.sem.cigarettes import (  # noqa: E402
    YEARS_PER_STATE,
    CigaretteSEM,
    V,
    homogeneity_wald,
    null_basis,
    restricted_gmm,
)

SEED = 42
SPEC = "t3"
GAMMA_STAR = 0.192215  # MEASURED; the plan quotes it to 4 decimals as 0.1922
GAMMA_TRUE = 0.25
NOISE_STD = 0.1
# gamma-hat* of ONE plasmode draw carries a real spread: b_ols picks up the cross
# term between the confounder and the outcome noise, whose sd is
# 2 kappa sqrt(sigma^2 / n) / sd(X d) = 8% of bias^2 at n = 2450. So the mean over
# draws is pinned tightly and the individual draws loosely; the plan's [0.85, 1.15]
# on 20 draws is a 1.9-sigma band and rejects a correct SEM most of the time.
PLASMODE_DRAWS = 40
BAND_MEAN = (0.96, 1.07)
BAND_DRAW = (0.6, 1.5)
BOOTSTRAP_SEEDS = 20
DISTINCT_MEAN = (29.0, 33.0)
DISTINCT_ANY = (22, 40)
PADDED = 2500
# J1w, J2, Wald at t3; MEASURED, plan SS0.3
COMPATIBILITY = {"iid": (6.68, 6.62, 6.73), "state": (8.93, 2.74, 2.79), "year": (2.18, 1.75, 1.90)}
WALD_GAP = 0.2
ONE_STEP_GAP = 3.0
SLIVER_RATIO = 0.063
SLIVER_RATIO_TOL = 0.005
SLIVER_CPI = 0.533
SLIVER_CPI_TOL = 0.005
DOUBLING = 2.635
QUERIES = 512
FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAIL.append(name)


def restricted_by_projector(design):
    """The restricted-IV point, recomputed INDEPENDENTLY of the loader: the normal
    equations of the projected program, where the loader solves the QR least
    squares. Same estimator, different arithmetic."""
    N = null_basis()
    Z = design.instruments
    P = Z @ np.linalg.solve(Z.T @ Z, Z.T)
    A = design.X @ N
    return N @ np.linalg.solve(A.T @ P @ A, A.T @ P @ design.y)


def sliver_by_projector(design, gamma_z, queries):
    """The sliver half-width, recomputed INDEPENDENTLY of the SEM: the projector
    Z(Z'Z)^-1 Z' from the normal equations where the SEM carries an orthonormal
    basis of span(Z), and the restricted point from `restricted_by_projector`.

    Same set, different arithmetic. A wrong metric -- the plain design second moment
    N'X'XN in place of the IV one (Q_Z'XN)'(Q_Z'XN) -- moves every coefficient
    half-width, and pinning all four against this is what sees it.
    """
    N = null_basis()
    Z = design.instruments
    P = Z @ np.linalg.solve(Z.T @ Z, Z.T)
    A = design.X @ N
    metric = A.T @ P @ A
    residual = design.y - design.X @ restricted_by_projector(design)
    misfit_sq = float(residual @ P @ residual) / len(design.y)
    slack = gamma_z - misfit_sq
    if slack <= 0.0:
        return np.zeros(len(queries))
    projected = np.asarray(queries, dtype=float) @ N
    spread = np.einsum("ij,jk,ik->i", projected, np.linalg.inv(metric), projected)
    return np.sqrt(slack * len(design.y)) * np.sqrt(spread)


def leg_i(sem):
    print("(i) the iv target is the restricted-IV point")
    independent = restricted_by_projector(sem.design)
    gap = float(np.abs(sem.solution.ravel() - independent).max())
    check("(i) solution == an independent restricted 2SLS", gap < 1e-10, f"max |diff| {gap:.2e}")
    homogeneity = abs(float(V @ sem.solution.ravel()))
    check("(i) v' solution == 0", homogeneity < 1e-12, f"{homogeneity:.2e}")
    gamma = gamma_star(sem)
    check("(i) gamma*", abs(gamma - GAMMA_STAR) < 1e-6, f"{gamma:.8f} vs {GAMMA_STAR}")
    check("(i) sigma^2 == 1", abs(sem.sigma_sq - 1.0) < 1e-12, f"{sem.sigma_sq:.15f}")


def index_of(pool_X):
    return {row.tobytes(): position for position, row in enumerate(pool_X)}


def whole_histories(drawn_X, lookup, states, years):
    """True when the draw is made of COMPLETE state histories: every state in it
    contributes a whole multiple of 50 rows and covers all 50 years. An iid row
    bootstrap fails this on its first draw."""
    positions = np.array([lookup[row.tobytes()] for row in drawn_X])
    drawn_states = states[positions]
    for state in np.unique(drawn_states):
        rows = drawn_states == state
        if int(rows.sum()) % YEARS_PER_STATE != 0:
            return False
        if len(np.unique(years[positions][rows])) != YEARS_PER_STATE:
            return False
    return True


def leg_ii():
    print("(ii) the replicate mechanism, under both flags")
    reference = CigaretteSEM()
    pool_X, pool_y = reference.pool
    for flag in (False, True):
        np.random.seed(SEED)
        sems = [CigaretteSEM(bootstrap=flag) for _ in range(8)]
        same = all(np.array_equal(sem.pool[0], pool_X) and np.array_equal(sem.pool[1], pool_y) for sem in sems)
        check(f"(ii) bootstrap={flag}: pool identical across 8 constructions", same)
        solutions = all(np.array_equal(sem.solution, reference.solution) for sem in sems)
        check(f"(ii) bootstrap={flag}: solution identical across 8 constructions", solutions)
        sem = sems[0]
        sem.sample(len(pool_X))
        sem.sample(len(pool_X))
        check(
            f"(ii) bootstrap={flag}: pool unmoved by sampling",
            np.array_equal(sem.pool[0], pool_X) and np.array_equal(sem.pool[1], pool_y),
        )
        check(
            f"(ii) bootstrap={flag}: solution unmoved by sampling",
            np.array_equal(sem.solution, reference.solution),
        )

    plain = CigaretteSEM(bootstrap=False)
    lookup = index_of(pool_X)
    states, years = plain.design.state, plain.design.year
    first, second = plain.sample(len(pool_X)), plain.sample(len(pool_X))
    check("(ii) bootstrap=False: sample(2450) IS the panel", np.array_equal(first[0], pool_X))
    check("(ii) bootstrap=False: two draws agree", np.array_equal(first[0], second[0]))

    # above the panel the draw is a BOOTSTRAP and says so out loud, whatever the flag
    messages = []
    handle = logger.add(messages.append, level="WARNING")
    padded_X, padded_y = plain.sample(PADDED)
    logger.remove(handle)
    expected = YEARS_PER_STATE * int(np.ceil(PADDED / YEARS_PER_STATE))
    check("(ii) bootstrap=False: a request above the panel is padded", len(padded_X) == expected, f"{len(padded_X)}")
    check("(ii) bootstrap=False: outcomes padded alike", len(padded_y) == expected, f"{len(padded_y)}")
    check(
        "(ii) bootstrap=False: the padding warns",
        any("padding by cluster resample" in str(message) for message in messages),
        f"{len(messages)} warnings",
    )
    check(
        "(ii) bootstrap=False: padding leaves the pool alone",
        np.array_equal(plain.pool[0], pool_X) and np.array_equal(plain.pool[1], pool_y),
    )
    check(
        "(ii) bootstrap=False: the padding is by CLUSTER",
        whole_histories(padded_X, lookup, states, years),
        "every state in the padded draw contributes complete 50-year histories",
    )

    booted = CigaretteSEM(bootstrap=True)
    np.random.seed(SEED)
    draw_X, draw_y = booted.sample(len(pool_X))
    again_X, _ = booted.sample(len(pool_X))
    positions = np.array([lookup[row.tobytes()] for row in draw_X])
    distinct_states = len(np.unique(states[positions]))
    check("(ii) bootstrap=True: 2450 rows", len(draw_X) == len(pool_X) and len(draw_y) == len(pool_X))
    check(
        "(ii) bootstrap=True: whole state histories",
        whole_histories(draw_X, lookup, states, years),
        f"{distinct_states} distinct states",
    )
    check("(ii) bootstrap=True: draws differ", not np.array_equal(draw_X, again_X))
    distinct = []
    for seed in range(BOOTSTRAP_SEEDS):
        np.random.seed(SEED + seed)
        sample_X, _ = booted.sample(len(pool_X))
        distinct.append(len(np.unique(states[[lookup[row.tobytes()] for row in sample_X]])))
    mean = float(np.mean(distinct))
    print(
        f"      distinct states over {BOOTSTRAP_SEEDS} seeds: min {min(distinct)} mean {mean:.1f} max {max(distinct)}"
    )
    check("(ii) mean distinct-state count", DISTINCT_MEAN[0] <= mean <= DISTINCT_MEAN[1], f"{mean:.1f}, expected 31.2")
    check(
        "(ii) every draw's distinct-state count",
        DISTINCT_ANY[0] <= min(distinct) and max(distinct) <= DISTINCT_ANY[1],
        f"[{min(distinct)}, {max(distinct)}]",
    )


def sample_ratio(sem):
    """gamma-hat* of the construction draw, over gamma_true."""
    X, y = sem.pool
    b = np.linalg.lstsq(X, y.ravel(), rcond=None)[0]
    gap = b - sem.solution.ravel()
    return float(gap @ sem.design.Sigma @ gap) / sigma_sq_hat(X, y, intercept=True) / GAMMA_TRUE


def plasmode_band(direction, tag):
    np.random.seed(SEED)
    ratios = np.array(
        [sample_ratio(CigaretteSEM(target="plasmode", confound_direction=direction)) for _ in range(PLASMODE_DRAWS)]
    )
    print(
        f"      {tag}: gamma-hat*/gamma_true over {PLASMODE_DRAWS} draws "
        f"min {ratios.min():.3f} mean {ratios.mean():.3f} max {ratios.max():.3f}"
    )
    check(f"{tag} mean sample ratio", BAND_MEAN[0] <= ratios.mean() <= BAND_MEAN[1], f"{ratios.mean():.4f}")
    check(
        f"{tag} every draw in band",
        bool(np.all((ratios > BAND_DRAW[0]) & (ratios < BAND_DRAW[1]))),
        f"[{ratios.min():.3f}, {ratios.max():.3f}]",
    )


def leg_iii():
    print("(iii) the plasmode calibrates exactly, and the draw follows")
    kappa_sq = GAMMA_TRUE * (1.0 + NOISE_STD**2) / (1.0 + GAMMA_TRUE)
    np.random.seed(SEED)
    sem = CigaretteSEM(target="plasmode")
    check("(iii) bias^2 == kappa^2", sem.bias_sq == kappa_sq, f"{sem.bias_sq!r} vs {kappa_sq!r}")
    check(
        "(iii) sigma^2 == 1 - kappa^2 + s^2",
        sem.sigma_sq == 1.0 - kappa_sq + NOISE_STD**2,
        f"{sem.sigma_sq!r}",
    )
    check("(iii) v' b_* == 0", abs(float(V @ sem.solution.ravel())) < 1e-12, f"{float(V @ sem.solution.ravel()):.2e}")
    plasmode_band("v", "(iii)")


def leg_iv():
    print("(iv) every confounding direction calibrates")
    for direction in ("v", "own_price", "worst_case"):
        np.random.seed(SEED)
        sem = CigaretteSEM(target="plasmode", confound_direction=direction)
        gamma = gamma_star(sem)
        check(f"(iv) {direction}: gamma* == gamma_true", abs(gamma - GAMMA_TRUE) < 1e-12, f"{gamma:.15f}")
        plasmode_band(direction, f"(iv) {direction}")


def leg_v(sem):
    print("(v) the real instrument never reaches a solver")
    amplitude = float(np.std(sem.X @ (V / np.linalg.norm(V))))
    np.random.seed(SEED)
    da = ScaleTranslation(V, std=amplitude)
    X, y = sem.pool
    GX, G = da(X)
    tax = sem.design.Z[:, 0]
    gap = float(np.abs(G.ravel() - tax).max())
    correlation = abs(float(np.corrcoef(G.ravel(), tax)[0, 1]))
    check("(v) G is not the tax column", gap > 0.01, f"max |G - log tax_s| {gap:.4f}")
    check("(v) G is uncorrelated with it", correlation < 0.2, f"|corr| {correlation:.4f}")
    builders = MethodRegistry.build_methods(
        ["DA+PI+IV", "PI&DA+PI+IV"],
        gamma=gamma_star(sem),
        epsilon=EPS_TOL,
        epsilon_iv=EPS_TOL,
        rho=1.0,
        recalibrate=True,
        pad=False,
        clipy=False,
        n_jobs=1,
        mean_match=True,
    )
    # the solvers centre on the mean-matched slice before precomputing (Lem. 2),
    # so the constraint to compare against is built on the centred augmented design
    want = iv_constraint_terms(GX - GX.mean(axis=0), y - y.mean(), G)[0]
    for name in ("DA+PI+IV", "PI&DA+PI+IV"):
        model = builders[name]()
        fit_model(model=model, method_name=name, X=X, y=y, GX=GX, G=G)
        # DA+PI+IV IS the IV method; the intersection holds it on `.augmented`
        built = model.augmented.T_projector_R if name.startswith("PI&") else model.T_projector_R
        difference = float(np.abs(np.asarray(built) - want).max())
        check(f"(v) {name}: the IV constraint is built from G", difference < 1e-12, f"max |diff| {difference:.2e}")


def leg_vi(design):
    print("(vi) the compatibility statistics agree once they are weighted alike")
    print(f"      {'cluster':>8s} {'J1w':>8s} {'J2':>8s} {'Wald':>8s}")
    for by, (want_one, want_two, want_wald) in COMPATIBILITY.items():
        one_step, two_step, _ = restricted_gmm(design, by)
        wald = homogeneity_wald(design, by, "iv")
        print(f"      {by:>8s} {one_step:8.2f} {two_step:8.2f} {wald:8.2f}")
        check(f"(vi) {by} J1w", abs(one_step - want_one) < 0.005, f"{one_step:.2f} vs {want_one}")
        check(f"(vi) {by} J2", abs(two_step - want_two) < 0.005, f"{two_step:.2f} vs {want_two}")
        check(f"(vi) {by} Wald", abs(wald - want_wald) < 0.005, f"{wald:.2f} vs {want_wald}")
        check(f"(vi) {by} J2 agrees with the Wald", abs(two_step - wald) < WALD_GAP, f"gap {abs(two_step - wald):.2f}")
    one_step, two_step, _ = restricted_gmm(design, "state")
    check("(vi) the one-step J is the outlier", one_step - two_step > ONE_STEP_GAP, f"gap {one_step - two_step:.2f}")


def leg_vii(sem):
    print("(vii) the target set degenerates to the point under the shipped flag")
    queries = sem.pool[0][np.random.default_rng(1).choice(len(sem.X), QUERIES, replace=False)]
    coefficients = np.eye(sem.design.k)
    point = sem.extent(queries)
    zero = bool(np.array_equal(point, np.zeros(QUERIES)))
    check("(vii) sliver false: extent is exactly zero", zero, f"max {point.max()}")
    check(
        "(vii) sliver false: zero on the coefficient queries",
        bool(np.array_equal(sem.extent(coefficients), np.zeros(sem.design.k))),
    )

    guarded = CigaretteSEM(spec=SPEC, sliver=True)
    precision = np.linalg.inv(guarded.design.Sigma)
    width = 2.0 * np.sqrt(gamma_star(guarded)) * np.sqrt(np.einsum("ij,jk,ik->i", queries, precision, queries))
    ratio = float(np.mean(2.0 * guarded.extent(queries) / width))
    half_widths = guarded.extent(coefficients)
    cpi = float(half_widths[3])
    print(f"      t3 misfit r0 {guarded._misfit:.4f}, extent/PI width {ratio:.4f}, e_cpi half-width {cpi:.4f}")
    print("      coefficient half-widths " + " ".join(f"{value:.6f}" for value in half_widths))
    check("(vii) sliver true: extent / PI width", abs(ratio - SLIVER_RATIO) < SLIVER_RATIO_TOL, f"{ratio:.4f}")
    check("(vii) sliver true: e_cpi half-width", abs(cpi - SLIVER_CPI) < SLIVER_CPI_TOL, f"{cpi:.4f}")
    independent = sliver_by_projector(guarded.design, 2**-8, coefficients)
    for index, name in enumerate(("log p", "log y", "log p_n", "log CPI")):
        gap = abs(float(half_widths[index] - independent[index]))
        check(f"(vii) {name} half-width == the independent closed form", gap < 1e-10, f"gap {gap:.2e}")
    observed = float(np.abs(guarded.extent(queries) - sliver_by_projector(guarded.design, 2**-8, queries)).max())
    check("(vii) the observed queries agree too", observed < 1e-10, f"max gap {observed:.2e}")

    messages = []
    handle = logger.add(messages.append, level="INFO")
    empty = CigaretteSEM(spec="t2", sliver=True)
    values = empty.extent(queries)
    logger.remove(handle)
    check("(vii) t2: the set is empty", bool(np.array_equal(values, np.zeros(QUERIES))), f"max {values.max()}")
    logged = any("EMPTY" in str(message) for message in messages)
    check("(vii) t2: emptiness is logged", logged, f"{len(messages)} lines captured")


def leg_viii():
    print("(viii) the guard is a radius, not a budget")
    tight = CigaretteSEM(spec=SPEC, sliver=True, gamma_z=2**-8)
    loose = CigaretteSEM(spec=SPEC, sliver=True, gamma_z=2**-7)
    coefficients = np.eye(tight.design.k)
    ratio = float(np.max(loose.extent(coefficients) / tight.extent(coefficients)))
    misfit, guard = tight._misfit, 2**-8
    law = float(np.sqrt((2 * guard - misfit**2) / (guard - misfit**2)))
    check("(viii) doubling gamma_z", abs(ratio - law) < 1e-9, f"{ratio:.6f} vs {law:.6f}")
    check("(viii) matches the plan's value", abs(ratio - DOUBLING) < 5e-4, f"{ratio:.4f} vs {DOUBLING}")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    os.chdir(REPO)
    np.random.seed(SEED)
    sem = CigaretteSEM(spec=SPEC)
    leg_i(sem)
    leg_ii()
    leg_iii()
    leg_iv()
    leg_v(sem)
    leg_vi(sem.design)
    leg_vii(sem)
    leg_viii()
    if FAIL:
        print(f"A52 FAIL: {FAIL}")
        sys.exit(1)
    print("A52 PASS")
