"""A54: the cigarette orchestrator is wired -- config in, budgets and replicates out.

refactor5 adds one orchestrator, one config dataclass, three registry entries, the
per-spec `QUERY_GAMMA` table, the validation branch and the root-yaml block. Nothing
it adds is a number of its own: the budgets come from `QUERY_GAMMA` and the oracle,
the replicate scheme from the `bootstrap` flag the two runners are built with. Legs:

  (i)   validation. The shipped block resolves; an unknown `target`, `spec` or
        `anchor`, a non-bool `sliver` and a non-positive `da_amplitude` each raise.
        Catches: the branch deleted (an unknown spec then runs silently at the
        loader's default, which is a DIFFERENT experiment). Misses: a value that is
        in the set but wrong for the run.
  (ii)  budget plumbing. The half-width at three fixed queries equals the closed
        form sigma sqrt(gamma) sqrt(x' Sigma^-1 x) to 1e-6, at the spec's declared
        budget and at a quarter of it. The panel is sigma-normalised, so this is the
        one leg that says the DECLARED gamma reached the solver as gamma and not as
        something monotone in it. The ball it reads is whichever of PI_BALLS the
        block lists, and it reports rather than raises when the block lists none.
        Catches: a squared, halved or rescaled budget, a `QUERY_GAMMA` lookup on the
        wrong spec, a block with no plain-PI ball. Misses: which epsilon was used --
        the PI dispatch carries no invariance constraint.
  (iii) the gamma sweep moves. Four ratio steps, one experiment: the baseline's
        coverage at r = 1 is far above its coverage at r = 2^-4, the curve is
        monotone, and the INV method's width stays inside it at the pair's pinned
        ratio. The (INV, baseline) pair comes from WIDTH_PAIRS against the block's
        own list, so the leg follows the config instead of indexing it blind. A
        block that lists no pair at all SKIPS the leg, named in the summary line.
        Catches: a budget that does not reach `predict`, an inverted ratio axis, a
        block with an INV method but not its baseline.
  (iv)  the omega axis is the recalibrated one. Four knobs, one experiment: the
        PLOTTED x is monotone decreasing in the knob, never above 1 (Prop. 2), and
        the runner's label is `OMEGA_XLABEL[True]` under `recalibrate: true`.
        Catches: the label and the factor disagreeing, an amplitude that does not
        reach the DA.
  (v)   `target` routes, and the two runners get DIFFERENT replicate schemes. Run
        on the block as given AND on its other declaration (`iv: [tax_s, y, cpi]`
        with `gamma_z: 0.0177` added, or both keys dropped), so both schemes are
        exercised whichever the block declares. gamma* at t3 is
        `GAMMA_STAR_T3[declared]` under `iv` and gamma_true under `plasmode`;
        `pool` and `solution` are identical across the sweep SEMs under both, so
        the oracle does not move with the draw; under `plasmode` the outcome
        differs between experiments. The sweep SEM bootstraps state clusters only
        under `iv` with no instrument declared (`cigarettes.py`); otherwise its draw
        IS the panel on the treatment columns, with `iv_pool` as the trailing
        columns when an instrument is declared. Under `iv` the SWEEP fit sets are
        pairwise different either way: bootstrapped they carry duplicated state
        histories, row-split no state has more than its 50 rows. The QUERY design
        is the full panel, 2450 distinct rows, elementwise `pool[0]`. Catches: both
        runners built from one factory (the whole point of SS6), a target that does
        not route, an oracle that reads the resample.

    MPLBACKEND=Agg python scripts/a54_cigarettes_runner.py [--seed 42] [--config PATH]

Reads the cigarettes block of `--config` (default config.yaml) the way main.py
does, and never assumes which methods it lists: every leg derives its witness from
the block and reports when one is missing. `--config` points the gate at a block
listing the IV pair (a recipe, or a temp copy of one). Never touches do-MNIST.
"""

import argparse
import os
import sys

import numpy as np
import yaml
from loguru import logger

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from src.experiments.cigarettes import CigaretteOrchestrator  # noqa: E402
from src.experiments.configs import (  # noqa: E402
    OMEGA_XLABEL,
    QUERY_GAMMA,
    resolve_dataset_block,
)
from src.experiments.utils import fit_model, set_seed  # noqa: E402
from src.sem.cigarettes import V, null_basis  # noqa: E402

# small enough to stay under a minute, large enough for every leg to move
SWEEP_STEPS = 4
N_JOBS = 4
GAMMA_GRID = (2.0**-4, 2.0**-2, 2.0**-1, 1.0)
OMEGA_GRID = (0.125, 0.5, 2.0, 8.0)
# leg (ii): three queries, none of them degenerate
QUERIES = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.4, -0.2, 0.3, 0.1]])
HALF_WIDTH_TOL = 1e-6
# leg (iii), plan SS0.7: PI+INV is 0.869 of PI at t3. The COVERAGE pins are relative
# to PI's own curve, not to 1.000: the sweeps fit state-cluster bootstrap replicates
# against an oracle read off the whole pool (SS6), so nothing covers 1.000 here and a
# pin at 1.0 would fail a correct implementation (measured PI at r = 1: 0.963).
# leg (ii) reads the budget off whichever plain-PI ball the block lists, in this
# order. `fit_model(method_name="PI", ...)` solves the plain ball whatever the
# builder configured, so the INV cone and the instrument are not applied on that
# path: measured gap to the closed form 1.2e-09 (PI+INV+IV) and 2.8e-09 (PI+IV)
# against 7.3e-03 for a DA ball, which carries the translation and is NOT a witness
PI_BALLS = ("PI", "PI+IV", "PI+INV", "PI+INV+IV")
# leg (iii): (the INV method, its baseline) in preference order, and the pinned
# (ratio, tolerance) of their widths at r = 1. The pair must differ by the INV cone
# alone, so the +IV spellings pair with each other. `PI+INV / PI` is plan SS0.7's
# 0.869 at t3 (no-iv seed span 0.8497 to 0.8792, inside its tolerance). The +IV pair
# moves with the seed AND with the block's observed instrument, so it is sized on
# the union of both spans, measured over seeds 42, 7, 2024, 101, 0, 1, 2, 3:
#   with `iv: [tax_s, y, cpi]`: 0.8932 0.8433 0.8498 0.8490 0.8828 0.8566 0.8612 0.8545
#   without it:                0.8609 0.8709 0.8782 0.8691 0.8792 0.8683 0.8497 0.8574
# union 0.8433 to 0.8932: pinned at its midpoint, each end cleared by 0.0097 or more,
# and a dropped INV cone (1.0000) still fails
WIDTH_PAIRS = (("PI+INV", "PI"), ("PI+INV+IV", "PI+IV"))
WIDTH_RATIO = {("PI+INV", "PI"): (0.869, 0.03), ("PI+INV+IV", "PI+IV"): (0.868, 0.035)}
COVERAGE_DROP = 0.3
# leg (v): gamma* at t3 under `target: iv`, keyed on "the block declares an
# instrument": the declared set moves the restricted target. The True value holds
# under DECLARED_IV's set [tax_s, y, cpi]; another set reads another gamma*
GAMMA_STAR_T3 = {False: 0.19221455, True: 0.37052167}
GAMMA_STAR_TOL = 1e-6
# the other declaration leg (v) runs beside the block's own: the query recipe's set
DECLARED_IV = {"iv": ["tax_s", "y", "cpi"], "gamma_z": 0.0177}
PANEL_ROWS = 2450
STATE_YEARS = 50  # one state history

FAIL = []
SKIPPED = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAIL.append(name)


def skip(name, reason):
    """A leg that cannot run on this block: printed and counted in the summary, never a silent PASS."""
    print(f"  [SKIP] {name} {reason}")
    SKIPPED.append(f"{name} {reason}")


def shipped_block(path):
    """The cigarettes block of the yaml at `path`, merged with the defaults as
    main.py merges it, minus the `experiment` plan."""
    with open(path) as handle:
        config = yaml.safe_load(handle) or {}
    block = {**(config.get("defaults") or {}), **(config.get("cigarettes") or {})}
    block.pop("experiment", None)
    return block


def build(block, seed, **overrides):
    """An orchestrator on the shipped block, shrunk for the gate."""
    block = resolve_dataset_block("cigarettes", dict(block))
    block.update(seed=seed, n_jobs=N_JOBS, **overrides)
    set_seed(seed)
    return CigaretteOrchestrator(**block, hyperparameters={})


def sweep_runner(orch, param, grid, **overrides):
    kwargs = {k: v for k, v in orch.kwargs.items() if k != "methods"}
    kwargs.update(overrides)
    return orch.get_sweep_runner_cls(param)(
        methods=orch.methods,
        method_factory=orch.build_methods,
        param_grid_override=list(grid),
        **kwargs,
    )


def query_runner(orch):
    kwargs = {k: v for k, v in orch.kwargs.items() if k != "methods"}
    kwargs["n_experiments"] = 1
    return orch.get_query_runner_cls()(methods=orch.methods, **kwargs)


# =============================================================================
# LEG (i): VALIDATION
# =============================================================================


def leg_i(block):
    print("(i) the cigarettes validation branch")
    try:
        resolved = resolve_dataset_block("cigarettes", dict(block))
        ok = resolved["target"] in ("iv", "plasmode") and resolved["n_samples"] == PANEL_ROWS
    except Exception as error:  # noqa: BLE001
        resolved, ok = None, False
        print(f"      shipped block raised: {error}")
    check("(i) the shipped block resolves", ok, f"target {None if resolved is None else resolved['target']}")

    bad = {
        "target": "ols",
        "spec": "t9",
        "anchor": "federal-tax",
        "sliver": "yes",
        "da_amplitude": -1.0,
    }
    for key, value in bad.items():
        candidate = {**block, key: value}
        try:
            resolve_dataset_block("cigarettes", candidate)
            raised = False
        except ValueError as error:
            raised = key in str(error)
        check(f"(i) {key}={value!r} is rejected", raised)

    # a bool is an int subclass; `da_amplitude: true` must not mean 1.0
    try:
        resolve_dataset_block("cigarettes", {**block, "da_amplitude": True})
        raised = False
    except ValueError:
        raised = True
    check("(i) da_amplitude=True is rejected", raised)


# =============================================================================
# LEG (ii): BUDGET PLUMBING
# =============================================================================


def _witness(available, preferred):
    """The first of `preferred` the config-derived `available` actually holds, or
    None. The block's method list is the owner's, so nothing may index it blind."""
    return next((name for name in preferred if name in available), None)


def _pi_half_width(runner, model):
    """Half-width of a model fitted through the PI dispatch, at the three fixed
    queries. The dispatch is what makes the ball the plain one; see PI_BALLS."""
    fit_model(
        model=model,
        method_name="PI",
        X=runner.X,
        y=runner.y,
        GX=runner.GX,
        G=runner.G,
        hyperparameters={},
        da=runner.da,
    )
    bounds = model.predict(QUERIES)
    return (bounds[:, 1] - bounds[:, 0]) / 2.0


def leg_ii(orch, runner):
    print("(ii) the declared budget reaches the solver as gamma")
    precision = np.linalg.inv(runner.X.T @ runner.X / len(runner.X))
    # the closed form is anchored on the TABLE, never on what the runner happens to
    # be holding: a budget mangled on the way in must show up as a width, not cancel
    declared = QUERY_GAMMA[orch.spec]
    check("(ii) the runner's gamma is QUERY_GAMMA[spec]", abs(runner.default_gamma - declared) < 1e-15, f"{declared}")

    name = _witness(runner.methods, PI_BALLS)
    check("(ii) the block lists a plain-PI ball to read the budget off", name is not None, f"{list(runner.methods)}")
    if name is None:
        return
    cases = [
        (declared, f"the runner's own {name} model", runner.methods[name]()),
        (
            declared / 4.0,
            f"build_methods at a quarter budget ({name})",
            orch.build_methods(gamma=declared / 4.0, epsilon=runner.default_epsilon)[name](),
        ),
    ]
    for gamma, label, model in cases:
        measured = _pi_half_width(runner, model)
        closed = np.sqrt(gamma) * np.sqrt(np.einsum("ij,jk,ik->i", QUERIES, precision, QUERIES))
        gap = float(np.max(np.abs(measured - closed)))
        check(
            f"(ii) {label}: half-width = sigma sqrt({gamma:g}) sqrt(x' Sigma^-1 x)",
            gap < HALF_WIDTH_TOL,
            f"max gap {gap:.2e}",
        )


# =============================================================================
# LEG (iii): THE GAMMA SWEEP
# =============================================================================


def leg_iii(orch):
    print("(iii) the gamma sweep")
    runner = sweep_runner(orch, "gamma", GAMMA_GRID, n_experiments=1)
    x, results, _ = runner.run("gamma")
    coverage = {name: np.nanmean(record["coverage"], axis=1) for name, record in results.items()}
    width = {name: np.nanmean(record["interval_width"], axis=1) for name, record in results.items()}

    pair = next((p for p in WIDTH_PAIRS if p[0] in width and p[1] in width), None)
    if pair is None:
        # an INV method without its baseline is a broken block; no pair at all is a
        # block this leg has nothing to say about
        orphans = [inv for inv, _ in WIDTH_PAIRS if inv in width]
        if orphans:
            check("(iii) the block lists an INV method beside its baseline", False, f"{orphans}: {list(coverage)}")
        else:
            missing = " nor ".join(f"({inv}, {base})" for inv, base in WIDTH_PAIRS)
            skip("(iii)", f"the block lists neither {missing}: {list(coverage)}")
        return x, None
    inv, base = pair

    pi = coverage[base]
    check(f"(iii) {base} coverage rises with the budget", bool(np.all(np.diff(pi) > -1e-12)), f"{np.round(pi, 3)}")
    check(
        f"(iii) {base} coverage at r = 1 clears r = 2^-4 by 0.3",
        float(pi[-1] - pi[0]) > COVERAGE_DROP,
        f"{pi[0]:.3f} -> {pi[-1]:.3f}",
    )
    check(f"(iii) {base} coverage at r = 2^-4 is well under 1", float(pi[0]) < 0.6, f"{pi[0]:.3f}")

    ratio = float(width[inv][-1] / width[base][-1])
    want, tol = WIDTH_RATIO[pair]
    check(f"(iii) {inv} / {base} at r = 1", abs(ratio - want) < tol, f"{ratio:.4f} vs {want} +- {tol}")
    return x, pi


# =============================================================================
# LEG (iv): THE OMEGA AXIS
# =============================================================================


def leg_iv(orch):
    print("(iv) the omega axis is the recalibrated one")
    runner = sweep_runner(orch, "omega", OMEGA_GRID, n_experiments=1)
    x, _, _ = runner.run("omega")
    x = np.asarray(x, dtype=float)
    check("(iv) the plotted x falls with the knob", bool(np.all(np.diff(x) < 0.0)), f"{np.round(x, 4)}")
    check("(iv) every x is at or under 1 (Prop. 2)", bool(np.all(x <= 1.0 + 1e-9)), f"max {x.max():.5f}")
    # against the CONFIGURED toggle, not the runner's own: comparing the runner to
    # itself would pass whatever ball it went on to solve
    configured = bool(orch.toggles["recalibrate"])
    check("(iv) the runner solves the configured ball", runner.recalibrate == configured, f"{configured}")
    check("(iv) the label follows the factor", runner.xlabel == OMEGA_XLABEL[configured], f"recalibrate={configured}")


# =============================================================================
# LEG (v): TARGET ROUTING AND THE REPLICATE SCHEME
# =============================================================================


def _rows_per_state(design_rows, pool_rows, states):
    """How many rows of `design_rows` each state contributes, matched back through
    the pool. A state drawn twice by the cluster bootstrap contributes about twice
    its 50-year history, minus what the test split took."""
    lookup = {row.tobytes(): index for index, row in enumerate(pool_rows)}
    drawn = [states[lookup[row.tobytes()]] for row in design_rows if row.tobytes() in lookup]
    return np.unique(np.asarray(drawn), return_counts=True)


def declarations(block):
    """The block as given, then its other declaration: `DECLARED_IV` added when it
    declares no instrument, both keys dropped when it does."""
    declared = bool(block.get("iv"))
    other = {k: v for k, v in block.items() if k not in DECLARED_IV} if declared else {**block, **DECLARED_IV}
    return ((f"as given (iv {block.get('iv')!r})", block), (f"the other declaration (iv {other.get('iv')!r})", other))


def leg_v(block, seed):
    print("(v) target routing and the replicate scheme, on both declarations")
    N = null_basis()
    for label, variant in declarations(block):
        for target in ("iv", "plasmode"):
            _leg_v_case(variant, seed, target, N, label)


def _leg_v_case(block, seed, target, N, label):
    orch = build(block, seed, target=target, n_experiments=2, sweep_samples=8)
    declared = bool(orch.iv_columns)
    # `cigarettes.py`: the sweep SEM bootstraps only under `iv` with no set declared
    bootstrap = target == "iv" and not declared
    tag = f"(v) {target}, {'declared' if declared else 'no'} instrument"
    print(f"    {label}: {tag[4:]}")
    runner = sweep_runner(orch, "gamma", GAMMA_GRID[:1], n_experiments=2)
    gamma_star = float(runner.get_oracle(0).gamma_star)
    want = GAMMA_STAR_T3[declared] if target == "iv" else orch.kwargs.get("gamma_true", 0.25)
    check(
        f"{tag}: gamma*",
        abs(gamma_star - want) < (GAMMA_STAR_TOL if target == "iv" else 1e-12),
        f"{gamma_star:.8f} vs {want}",
    )

    sems = runner.sems
    # BEFORE anything draws: `sample` is the replicate mechanism and the oracle
    # reads `pool`, so a draw that moved the pool would move h_* with it
    snapshot = tuple(np.array(part, copy=True) for part in sems[0].pool)
    for _ in range(2):
        sems[0].sample(PANEL_ROWS)
    check(
        f"{tag}: `sample` leaves the pool alone",
        all(np.array_equal(before, after) for before, after in zip(snapshot, sems[0].pool, strict=True)),
    )

    pools = [sem.pool[0] for sem in sems]
    solutions = [sem.solution for sem in sems]
    check(
        f"{tag}: pool identical across sweep SEMs",
        all(np.array_equal(pools[0], other) for other in pools[1:]),
    )
    check(
        f"{tag}: solution identical across sweep SEMs",
        all(np.array_equal(solutions[0], other) for other in solutions[1:]),
    )
    check(
        f"{tag}: h_* is homogeneous",
        float(abs(V @ solutions[0].ravel())) < 1e-12,
        f"|v'b| {float(abs(V @ solutions[0].ravel())):.2e}",
    )
    check(
        f"{tag}: h_* lies in null(v)",
        float(np.max(np.abs(solutions[0].ravel() - N @ (N.T @ solutions[0].ravel())))) < 1e-12,
    )

    outcomes = [sem.pool[1] for sem in sems]
    differs = not np.array_equal(outcomes[0], outcomes[1])
    check(
        f"{tag}: the outcome {'differs' if target == 'plasmode' else 'is fixed'} between experiments",
        differs == (target == "plasmode"),
    )

    # `sample` is the sweep replicate mechanism: a cluster bootstrap, or the panel
    # itself, where the replicate is the row split (and the outcome draw under
    # `plasmode`). A declared instrument rides as the trailing columns of the draw
    # (`_rows`) while `pool` is the treatment alone, so the panel is compared on the
    # treatment columns and the rest against `iv_pool`
    draws = [sems[0].sample(PANEL_ROWS)[0] for _ in range(2)]
    resamples = not np.array_equal(draws[0], draws[1])
    check(f"{tag}: the sweep SEM {'resamples' if bootstrap else 'returns the panel'}", resamples == bootstrap)
    k = pools[0].shape[1]
    if not bootstrap:
        check(f"{tag}: the sweep draw IS the panel on the treatment columns", np.array_equal(draws[0][:, :k], pools[0]))
    if declared:
        check(
            f"{tag}: the trailing columns are the declared instrument (`iv_pool`)",
            sems[0].iv_pool is not None and np.array_equal(draws[0][:, k:], sems[0].iv_pool),
            f"draw {draws[0].shape}, pool {pools[0].shape}",
        )
    else:
        check(f"{tag}: the draw carries the treatment alone", draws[0].shape[1] == k, f"{draws[0].shape}")

    # the SWEEP fit sets under `iv`: pairwise different under both schemes (the
    # row split is seeded per experiment), whole state histories repeated only
    # under the bootstrap
    if target == "iv":
        fits = [runner._base_data(j)[0] for j in range(2)]
        check(f"{tag}: sweep fit sets are pairwise different", not np.array_equal(fits[0], fits[1]))
        states, counts = _rows_per_state(fits[0], pools[0], sems[0].design.state)
        detail = f"{len(states)} distinct states, up to {int(counts.max())} rows each"
        if bootstrap:
            check(
                f"{tag}: the sweep fit set carries duplicated state histories", int(counts.max()) > STATE_YEARS, detail
            )
        else:
            check(f"{tag}: no state has more than its {STATE_YEARS} rows", int(counts.max()) <= STATE_YEARS, detail)

    # the QUERY runner: the panel itself, every row once
    runner_q = query_runner(orch)
    design = runner_q.X
    check(f"{tag}: the query design IS the pool", np.array_equal(design, pools[0]), f"{design.shape}")
    check(
        f"{tag}: the query design has {PANEL_ROWS} distinct rows",
        len(np.unique(design, axis=0)) == PANEL_ROWS,
        f"{len(np.unique(design, axis=0))}",
    )


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", default=os.path.join(REPO, "config.yaml"), help="yaml with a cigarettes block")
    args = parser.parse_args()
    seed, config = args.seed, os.path.abspath(args.config)  # before the chdir, as a32's --artifacts
    os.chdir(REPO)
    print(f"config {config}; seed {seed}")

    block = shipped_block(config)
    leg_i(block)
    orch = build(block, seed, n_experiments=1, sweep_samples=8)
    leg_ii(orch, query_runner(orch))
    leg_iii(orch)
    leg_iv(orch)
    leg_v(block, seed)

    skipped = f" ({len(SKIPPED)} SKIPPED: {'; '.join(SKIPPED)})" if SKIPPED else ""
    if not FAIL:
        print(f"A54 PASS{skipped}")
    else:
        print(f"A54 FAIL: {FAIL}{skipped}")
        sys.exit(1)
