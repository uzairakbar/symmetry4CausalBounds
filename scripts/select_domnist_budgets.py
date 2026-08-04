"""Select do-MNIST budgets by POPULATION coverage. Run once; hand-copy into config.yaml.

Not part of the pipeline -- nothing in src/ imports this. It answers "what gamma,
epsilon and epsilon_iv actually hold coverage at this (pad, calibrate, n_pi, net)",
which the oracle cannot: do-MNIST's h_* is a frozen CNN, `sem.solution` raises, so no
oracle quantity certifies membership on the copsens backend.

Three sequential legs at ONE target coverage X, each fixing its knob for good:

    1. lowest gamma      with coverage >= X on PI          -> FIXED, never re-selected
    2. lowest epsilon_iv with coverage >= X on DA+PI_IV    (at that gamma)
    3. lowest epsilon    with coverage >= X on PI_INV      (at that gamma)

gamma is fixed by leg 1 because the pipeline consumes ONE gamma for every method; a
per-method gamma would report a tuple no experiment can use. DA+PI_IV and PI_INV are
subsets of DA+PI, so if DA+PI misses X at that gamma no budget can reach it -- the
report says so and marks the leg `target_reachable: false` rather than pretending.

    python scripts/select_domnist_budgets.py            # population: 1.2M draws, 10k eval
    python scripts/select_domnist_budgets.py --smoke    # minutes, for gates

Writes artifacts/domnist-budget_report.json. Budgets are conditional on every setting
recorded in there -- change one and they are stale.
"""
import os
import sys
import json
import time
import argparse
import subprocess
from datetime import datetime, timezone

import numpy as np
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sem.do_mnist import DoMNISTSEM                          # noqa: E402
from src.data_augmentors.do_mnist import DoMNISTDA               # noqa: E402
from src.experiments.do_mnist import Flatten, train_pair, pi_subset, TEST_SEED_OFFSET  # noqa: E402
from src.experiments.configs import MethodRegistry, DOMNIST_CONFIG  # noqa: E402
from src.experiments.utils import set_seed                       # noqa: E402
from src.experiments.utils.metrics import (evaluate_queries,     # noqa: E402
                                           STATUS_CATEGORIES)
from src.oracle import compute_oracle_parameters                 # noqa: E402

OUT_DEFAULT = 'artifacts/domnist-budget_report.json'
START = time.time()

# bisection brackets. gamma is a latent-space ball radius^2; the budgets are bracketed
# below by their own constraint floor, which is where INFEASIBLE starts.
GAMMA_BRACKET = (1e-3, 4.0)
BUDGET_R_HI = 4096.0            # budget^2 / floor; above this the constraint is inert
FLOOR_MARGIN = 1.02             # first probe just above the floor, not on it


# =============================================================================
# COVERAGE PROBES
# =============================================================================

def probe(model, X_eval, estimand, **predict_kw):
    """One (coverage, width, statuses) at whatever budget the model currently holds."""
    started = time.time()
    record = evaluate_queries(estimand, model.predict(X_eval, **predict_kw),
                              getattr(model, 'query_status', None),
                              time.time() - started)
    return record


def min_knob_for_coverage(evaluate, lo, hi, target, max_iter=12, tol=0.01):
    """Smallest knob in [lo, hi] whose coverage clears `target`.

    Coverage is monotone in every knob here -- a larger ball or a larger budget is a
    superset feasible set, hence a weakly wider interval. Bisect on log(knob);
    `curve` records every probe so the assumption is auditable rather than assumed.
    """
    curve = []

    def at(value):
        record = evaluate(value)
        curve.append({'knob': float(value), 'coverage': record.coverage,
                      'mean_width': record.interval_width,
                      'status_counts': [int(c) for c in record.status_counts]})
        logger.info(f'  probe {value:.6g}: coverage {record.coverage:.4f} '
                    f'width {record.interval_width:.5f}')
        return record

    top = at(hi)
    if top.coverage < target:               # unreachable: report the ceiling honestly
        return dict(value=float(hi), record=top, curve=curve, n_iter=1,
                    bracket=[float(hi), float(hi)], reached=False)

    bottom = at(lo)
    if bottom.coverage >= target:
        return dict(value=float(lo), record=bottom, curve=curve, n_iter=2,
                    bracket=[float(lo), float(lo)], reached=True)

    best, log_lo, log_hi = top, np.log(lo), np.log(hi)
    for iteration in range(max_iter):
        if (log_hi - log_lo) < np.log(1.0 + tol):
            break
        mid = float(np.exp(0.5 * (log_lo + log_hi)))
        record = at(mid)
        if record.coverage >= target:
            log_hi, best = np.log(mid), record
        else:
            log_lo = np.log(mid)

    return dict(value=float(np.exp(log_hi)), record=best, curve=curve,
                n_iter=len(curve), reached=True,
                bracket=[float(np.exp(log_lo)), float(np.exp(log_hi))])


def leg_report(selected, floor=None, reference_width=None, ceiling=None,
               target_reachable=True):
    record = selected['record']
    out = {
        'value': selected['value'],
        'achieved_coverage': record.coverage,
        'mean_width': record.interval_width,
        'bracket': selected['bracket'],
        'n_iter': selected['n_iter'],
        'target_reachable': bool(target_reachable and selected['reached']),
        'status_counts': dict(zip(STATUS_CATEGORIES,
                                  (int(c) for c in record.status_counts))),
        'curve': selected['curve'],
    }
    if floor is not None:
        out.update(floor=float(floor), floor_units='squared',
                   sqrt_floor=float(np.sqrt(max(floor, 0.0))),
                   r=float(selected['value'] ** 2 / floor) if floor > 0 else None)
    if reference_width:
        out['delta_width_vs_DA_PI'] = record.interval_width / reference_width - 1.0
    if ceiling is not None:
        out['ceiling_coverage'] = float(ceiling)
    return out


# =============================================================================
# DRIVER
# =============================================================================

def load_block():
    """config.yaml's do_mnist block over defaults. The budgets are conditional on it."""
    import yaml
    with open('config.yaml') as handle:
        config = yaml.safe_load(handle) or {}
    defaults = config.get('defaults') or {}
    block = {**defaults, **(config.get('do_mnist') or {})}
    return block, (config.get('hyperparameters') or {})


def main(args):
    block, hyperparameters = load_block()
    target = args.target_coverage or block.get('target_coverage', 0.95)

    n_samples = args.n_samples or block.get('n_samples', 1_200_000)
    n_pi = args.n_pi or block.get('n_pi', 60_000)
    seed = block.get('seed', 42)
    toggles = dict(calibrate=block.get('calibrate', False), pad=block.get('pad', False),
                   clipy=block.get('clipy', True), n_jobs=block.get('n_jobs', 1))

    logger.info(f'select: target coverage {target}, n_samples {n_samples:,}, '
                f'n_pi {n_pi:,}, n_eval {args.n_eval:,}, toggles {toggles}')
    set_seed(seed)

    sem_kw = dict(seed=seed, net=block.get('net', 'domnist-fast'),
                  alpha=DOMNIST_CONFIG.alpha, beta=DOMNIST_CONFIG.beta,
                  eta=DOMNIST_CONFIG.eta, subsample=DOMNIST_CONFIG.subsample,
                  target_samples=n_samples, target_kw=dict(hyperparameters))
    sem = DoMNISTSEM(train=True, **sem_kw)
    sem_test = DoMNISTSEM(train=False, **sem_kw)
    da = DoMNISTDA(block['augmentation'])
    flat = Flatten()

    # ---------------------------------------------------------------- the draw
    X_raw, y, _ = sem.sample_paired(n_samples, seed=sem.seed)
    GX_raw, G = da(X_raw)
    nets = train_pair(flat.fit_transform(X_raw), flat.fit_transform(GX_raw), y,
                      init_seed=sem.seed, net=sem.net, **hyperparameters)

    keep = pi_subset(len(X_raw), n_pi, seed)
    X, GX, y_pi, G_pi = X_raw[keep], GX_raw[keep], y[keep], G[keep]
    del X_raw, GX_raw, y, G

    # ------------------------------------------------------------ the eval set
    # POPULATION, not the pipeline's 512 queries: a balanced interventional resample
    # over the HELD-OUT split (sem_test is train=False), scored against the same
    # target net the experiments score against.
    X_eval_raw, _ = sem_test(N=args.n_eval, intervention=True,
                             seed=seed + TEST_SEED_OFFSET)
    digits_eval = np.asarray(sem_test.last_['digits'])      # read before any redraw
    X_eval = flat.fit_transform(X_eval_raw)
    estimand = sem.f(X_eval)
    estimand_analytic = sem.ate_of(digits_eval)             # diagnostic, not the target

    def build(names, gamma, epsilon, epsilon_iv):
        return MethodRegistry.build_methods(
            names, gamma=gamma, epsilon=epsilon, epsilon_iv=epsilon_iv,
            backend='copsens', outcome_models=nets,
            n_components=block.get('n_components', 32), **toggles)

    def fitted(name, gamma, epsilon=1.0, epsilon_iv=1.0):
        model = build([name], gamma, epsilon, epsilon_iv)[name]()   # registry: builders
        fit_kw = {'PI': dict(X=X), 'DA+PI': dict(X=GX),
                  'PI_INV': dict(X=X, GX=GX, G=G_pi),
                  'DA+PI_IV': dict(X=GX, Z=G_pi),
                  'PI&DA+PI': dict(X=X, GX=GX, G=G_pi),
                  'PI&DA+PI_IV': dict(X=X, GX=GX, G=G_pi)}[name]
        return model.fit(y=y_pi, **fit_kw)

    # ------------------------------------------------- leg 1: gamma on baseline PI
    # The latent fit does not depend on gamma, so fit once and sweep at predict.
    logger.info('leg 1/3: gamma on baseline PI')
    pi = fitted('PI', gamma=GAMMA_BRACKET[1])
    gamma_sel = min_knob_for_coverage(
        lambda g: probe(pi, X_eval, estimand, gamma=g),
        *GAMMA_BRACKET, target=target, max_iter=args.max_iter, tol=args.tol)
    gamma = gamma_sel['value']
    logger.info(f'leg 1: gamma = {gamma:.6g} at coverage '
                f'{gamma_sel["record"].coverage:.4f}')

    # ------------------------------------ DA+PI at that SAME gamma: the shared ceiling
    # Reference only. gamma is FIXED by leg 1 and never re-selected: the pipeline
    # consumes one gamma for every method, so a per-method gamma would report a tuple
    # no experiment can use.
    da_pi = fitted('DA+PI', gamma=gamma)
    da_pi_record = probe(da_pi, X_eval, estimand)
    ceiling = da_pi_record.coverage
    warnings = []
    if ceiling < target:
        # DA+PI_IV and PI_INV are subsets of DA+PI, so neither can out-cover it. The
        # target is then unattainable at this gamma -- a finding about the fixture,
        # reported as such. Legs 2/3 fall back to the ceiling so the report still
        # carries a usable number, flagged `target_reachable: false`.
        logger.warning(f'DA+PI covers {ceiling:.4f} < target {target} at gamma '
                       f'{gamma:.6g}: no epsilon_iv or epsilon can reach the target, '
                       'since both constraints only shrink DA+PI.')
        warnings.append(
            f'target {target} is UNATTAINABLE for DA+PI_IV and PI_INV at gamma '
            f'{gamma:.6g}: DA+PI itself covers {ceiling:.4f} and both methods are '
            'subsets of it. Legs 2/3 report the lowest budget reaching that ceiling '
            f'({ceiling:.4f}) instead, marked target_reachable=false. Raise the '
            'target-coverage input or revisit the fixture; do NOT re-select gamma '
            'per method.')
    aim = min(target, ceiling)

    def budget_leg(name, attr, label):
        model = fitted(name, gamma=gamma)
        floor = model.constraint_floor(model._radius(gamma))
        lo = float(np.sqrt(max(floor, 0.0)) * FLOOR_MARGIN)
        hi = float(np.sqrt(max(floor, 1e-12) * BUDGET_R_HI))
        logger.info(f'{label}: floor {floor:.6g} (squared) -> bracket '
                    f'[{lo:.6g}, {hi:.6g}], aim {aim:.4f}')

        def evaluate(value):
            setattr(model, attr, float(value))
            return probe(model, X_eval, estimand)

        selected = min_knob_for_coverage(evaluate, lo, hi, target=aim,
                                         max_iter=args.max_iter, tol=args.tol)
        selected['reached'] = selected['reached'] and ceiling >= target
        setattr(model, attr, selected['value'])
        return selected, floor

    logger.info('leg 2/3: epsilon_iv on DA+PI_IV')
    iv_sel, iv_floor = budget_leg('DA+PI_IV', 'epsilon_iv', 'leg 2')

    logger.info('leg 3/3: epsilon on PI_INV')
    inv_sel, inv_floor = budget_leg('PI_INV', 'epsilon', 'leg 3')

    epsilon, epsilon_iv = inv_sel['value'], iv_sel['value']

    # ------------------------------------- intersections: NOT bounded by the legs above
    # PI & DA+PI_IV can be empty when neither branch is, so marginal selection on
    # DA+PI_IV says nothing about it. Measure, do not infer.
    # a budget that buys no width is a duplicate column, whatever its coverage
    def flag_inert(label, width, reference, where=''):
        # relative, not absolute: buying 0.005% of width is inert at any scale
        if width >= reference * (1 - 1e-3):
            warnings.append(
                f'{label} is INERT{where}: width {width:.5f} vs DA+PI '
                f'{reference:.5f}. The constraint buys nothing at this budget -- do '
                'not plot the column as a distinct method.')

    for label, selected in (('epsilon_iv', iv_sel), ('epsilon', inv_sel)):
        flag_inert(f'{label} = {selected["value"]:.6g}',
                   selected['record'].interval_width, da_pi_record.interval_width)


    intersections = {}
    for name in ('PI&DA+PI', 'PI&DA+PI_IV'):
        if name not in block.get('methods', []):
            continue
        model = fitted(name, gamma=gamma, epsilon=epsilon, epsilon_iv=epsilon_iv)
        record = probe(model, X_eval, estimand)
        parent = da_pi_record.coverage if name == 'PI&DA+PI' else iv_sel['record'].coverage
        empty = float(record.status_counts[STATUS_CATEGORIES.index('infeasible')]
                      / max(args.n_eval, 1))
        intersections[name] = {'coverage': record.coverage,
                               'mean_width': record.interval_width,
                               'empty_rate': empty}
        se = float(np.sqrt(max(parent * (1 - parent), 0.0) / max(args.n_eval, 1)))
        if record.coverage < parent - 2 * se:
            warnings.append(f'{name} covers {record.coverage:.4f} vs its parent '
                            f'{parent:.4f}: the intersection loses more than 2 SE.')

    # ------------------------------------------------------------------- report
    try:                                    # diagnostic only; never gates the report
        oracle = compute_oracle_parameters(sem, da, features=flat.fit_transform)
    except Exception as error:
        logger.warning(f'oracle unavailable, recording nulls: {error}')
        oracle = None
    report = {
        'schema_version': 1,
        'generated_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'wall_clock_s': round(time.time() - START, 1),
        'provenance': {
            'git_sha': _git('rev-parse', '--short', 'HEAD'),
            'git_dirty': bool(_git('status', '--porcelain')),
            'python': sys.version.split()[0], 'numpy': np.__version__,
        },
        'settings': {
            'seed': seed, 'n_samples': n_samples, 'n_pi': n_pi,
            'n_eval': args.n_eval, 'target_coverage': target,
            'eval_draw': 'balanced interventional resample over the held-out split',
            **toggles,
            'n_components': block.get('n_components', 32), 'net': block.get('net'),
            'augmentation': block['augmentation'],
            'link': DOMNIST_CONFIG.link, 'n_anchors': DOMNIST_CONFIG.n_anchors,
            'n_anchors_c': DOMNIST_CONFIG.n_anchors_c,
            'n_constraint_inv': DOMNIST_CONFIG.n_constraint_inv,
            'n_constraint_iv': DOMNIST_CONFIG.n_constraint_iv,
            'mu_clip': DOMNIST_CONFIG.mu_clip, 'jax_grad': DOMNIST_CONFIG.jax_grad,
            'exemplar_seed': DOMNIST_CONFIG.exemplar_seed,
            'sem': {'alpha': DOMNIST_CONFIG.alpha, 'beta': DOMNIST_CONFIG.beta,
                    'eta': DOMNIST_CONFIG.eta, 'subsample': DOMNIST_CONFIG.subsample},
            'hyperparameters': hyperparameters,
        },
        # recorded, never consumed: the contrast with what coverage actually wants
        # is the reason this script exists.
        'oracle': {
            'gamma_star': _f(getattr(oracle, 'gamma_star', None)),
            'epsilon_star': _f(getattr(oracle, 'epsilon_star', None)),
            'eps_iv_star': _f(getattr(oracle, 'eps_iv_star', None)),
            'eps_rms': _f(getattr(oracle, 'eps_rms', None)),
            'eta': _f(getattr(oracle, 'eta', None)),
            'epsilon_star_analytic': 0.0,   # provable: DoMNISTDA.exact_invariance
            'sigma_sq': _f(sem.sigma_sq), 'bias_sq': _f(sem.bias_sq),
            'note': 'sem.solution raises on do-MNIST; no oracle quantity certifies '
                    'membership on the copsens backend. Selected by coverage instead.',
        },
        'reference': {
            'PI': {'coverage': gamma_sel['record'].coverage,
                   'mean_width': gamma_sel['record'].interval_width},
            # at the SAME fixed gamma; the ceiling legs 2/3 inherit
            'DA+PI': {'coverage': da_pi_record.coverage,
                      'mean_width': da_pi_record.interval_width},
        },
        'selected': {
            'gamma': leg_report(gamma_sel),
            'epsilon_iv': leg_report(iv_sel, floor=iv_floor,
                                     reference_width=da_pi_record.interval_width,
                                     ceiling=ceiling, target_reachable=ceiling >= target),
            'epsilon': leg_report(inv_sel, floor=inv_floor,
                                  reference_width=da_pi_record.interval_width,
                                  ceiling=ceiling, target_reachable=ceiling >= target),
        },
        'intersections_at_selected': intersections,
        # THE deliverable: one gamma, fixed by leg 1, and the two budgets selected
        # against it. Paste verbatim into config.yaml::do_mnist.
        'config_patch': {'gamma': gamma, 'epsilon': epsilon, 'epsilon_iv': epsilon_iv},
        'warnings': warnings,
    }

    # coverage against the ANALYTIC h_* as well: sem.f is a fitted net, so a gap
    # between the two is target-net error, not a budget problem.
    report['reference']['PI_vs_analytic_coverage'] = float(np.mean(
        _covered(pi.predict(X_eval, gamma=gamma), estimand_analytic)))

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as handle:
        json.dump(report, handle, indent=2)

    logger.info(f'\nwrote {args.out}\n\n  gamma: {gamma:.6g}\n  epsilon: '
                f'{epsilon:.6g}\n  epsilon_iv: {epsilon_iv:.6g}\n')
    for entry in warnings:
        logger.warning(entry)
    return report


def _covered(interval, estimand):
    interval = np.asarray(interval, dtype=float)
    truth = np.asarray(estimand).squeeze()
    inside = (truth >= interval[:, 0]) & (truth <= interval[:, 1])
    return np.where(np.isnan(interval).any(axis=1), False, inside)


def _f(value):
    return None if value is None or not np.isfinite(value) else float(value)


def _git(*command):
    try:
        return subprocess.run(('git',) + command, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return None


if __name__ == '__main__':
    START = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target-coverage', type=float, default=None)
    parser.add_argument('--n-samples', type=int, default=None)
    parser.add_argument('--n-pi', type=int, default=None)
    parser.add_argument('--n-eval', type=int, default=10_000)
    parser.add_argument('--max-iter', type=int, default=12)
    parser.add_argument('--tol', type=float, default=0.01)
    parser.add_argument('--out', default=OUT_DEFAULT)
    parser.add_argument('--smoke', action='store_true',
                        help='60k draws / 6k n_pi / 2k eval: minutes, for gates')
    parsed = parser.parse_args()
    if parsed.smoke:
        parsed.n_samples = parsed.n_samples or 60_000
        parsed.n_pi = parsed.n_pi or 6_000
        parsed.n_eval = 2_000 if parsed.n_eval == 10_000 else parsed.n_eval
        parsed.out = (OUT_DEFAULT.replace('.json', '_smoke.json')
                      if parsed.out == OUT_DEFAULT else parsed.out)
    main(parsed)
