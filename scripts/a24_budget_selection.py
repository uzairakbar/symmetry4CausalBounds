"""A24: the do-MNIST budget-selection utility.

Gates the parts that are cheap to check exactly -- the bisection contract, the
floor cache, and the report schema. It does NOT re-run a selection: that costs a
CNN fit. Run `select_domnist_budgets.py --smoke` for the end-to-end path.

    python scripts/a24_budget_selection.py
"""
import os
import sys
import json

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sem.do_mnist import DoMNISTSEM                             # noqa: E402
from src.data_augmentors.do_mnist import DoMNISTDA                  # noqa: E402
from src.methods.regression import GradientDescentERM               # noqa: E402
from src.methods.copsens import RecentredInvCopSens                 # noqa: E402
from src.experiments.configs import DOMNIST_CONFIG                  # noqa: E402
from src.experiments.do_mnist import Flatten                        # noqa: E402
from src.experiments.utils import set_seed                          # noqa: E402
from src.experiments.utils.metrics import QueryEval                 # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from select_domnist_budgets import min_knob_for_coverage, leg_report  # noqa: E402

N_SAMPLES, N_PI = 20_000, 1_500
FAIL = []


def check(name, ok, detail=''):
    print(f'[{"PASS" if ok else "FAIL"}] {name} {detail}')
    if not ok:
        FAIL.append(name)


def _eval(coverage, width=1.0):
    return QueryEval(approximation_error=0.0, worst_error=0.0, interval_width=width,
                     coverage=coverage, wall_clock=0.0,
                     status_counts=np.zeros(4, dtype=int))


# ------------------------------------------------------------------ bisection

def a24_bisection():
    """Coverage is monotone in the knob, so the search must return the MINIMUM
    knob clearing the target -- not merely some knob that clears it."""
    # step function: coverage 0.99 at/above 0.30, else 0.10. True answer: 0.30.
    calls = []

    def evaluate(value):
        calls.append(value)
        return _eval(0.99 if value >= 0.30 else 0.10)

    got = min_knob_for_coverage(evaluate, 0.01, 10.0, target=0.95,
                                max_iter=40, tol=1e-4)
    check('A24 bisection finds the true threshold', abs(got['value'] - 0.30) < 3e-4,
          f'got {got["value"]:.6f}')
    check('A24 bisection brackets the threshold',
          got['bracket'][0] <= 0.30 <= got['bracket'][1] * (1 + 1e-9))
    check('A24 bisection records every probe', len(got['curve']) == len(calls))
    check('A24 bisection reports reached', got['reached'] is True)

    # unreachable target: report the ceiling, never a silent pass
    got = min_knob_for_coverage(lambda v: _eval(0.40), 0.01, 10.0, target=0.95)
    check('A24 unreachable target is flagged', got['reached'] is False)
    check('A24 unreachable returns the top of the bracket', got['value'] == 10.0)

    # already covered at the bottom: return the bottom, do not bisect upward
    got = min_knob_for_coverage(lambda v: _eval(0.99), 0.01, 10.0, target=0.95)
    check('A24 already-covered returns the floor of the bracket', got['value'] == 0.01)

    # r is reported in SQUARED units, matching CopSensPI._budget()
    report = leg_report(min_knob_for_coverage(lambda v: _eval(0.99, 0.5), 0.2, 10.0,
                                              target=0.95), floor=0.01)
    check('A24 r is budget^2 / floor',
          abs(report['r'] - 0.2 ** 2 / 0.01) < 1e-9, f'got {report["r"]}')
    check('A24 floor units are declared', report['floor_units'] == 'squared')


# ----------------------------------------------------------------- floor cache

def a24_floor_cache():
    """The cache must be exact and must not survive a refit -- `_prepare` asks for
    the floor on every predict, and a stale floor silently gates the NaN return."""
    set_seed(42)
    sem = DoMNISTSEM(seed=42, train=True, target_samples=N_SAMPLES,
                     target_kw=dict(epochs=1), alpha=DOMNIST_CONFIG.alpha,
                     beta=DOMNIST_CONFIG.beta, eta=DOMNIST_CONFIG.eta)
    flat = Flatten()
    X_img, y, _ = sem.sample_paired(N_SAMPLES, seed=sem.seed)
    GX_img, G = DoMNISTDA()(X_img)
    X, GX = flat.fit_transform(X_img), flat.fit_transform(GX_img)
    net = GradientDescentERM().fit(GX, y, init_seed=42, epochs=1)

    keep = np.random.default_rng(42).choice(len(X), N_PI, replace=False)
    X, GX, y, G = X[keep], GX[keep], y[keep], G[keep]

    model = RecentredInvCopSens(gamma=0.1, epsilon=0.1, outcome_model=net,
                                n_components=32, link=DOMNIST_CONFIG.link,
                                n_anchors=DOMNIST_CONFIG.n_anchors,
                                n_anchors_c=DOMNIST_CONFIG.n_anchors_c,
                                n_constraint=DOMNIST_CONFIG.n_constraint_inv,
                                jax_grad=DOMNIST_CONFIG.jax_grad, calibrate=True)
    model.fit(X=X, y=y, GX=GX, G=G)

    radius = model._radius(0.1)
    first = model.constraint_floor(radius)
    second = model.constraint_floor(radius)
    check('A24 floor cache is bit-exact', first == second, f'{first!r} vs {second!r}')
    check('A24 floor cache is populated', len(model._floor_cache) == 1)

    other = model.constraint_floor(model._radius(0.05))
    check('A24 a different radius is a different entry', len(model._floor_cache) == 2)
    check('A24 a smaller ball has a higher floor', other >= first,
          f'{other:.6g} vs {first:.6g}')

    model.fit(X=X, y=y, GX=GX, G=G)
    check('A24 a refit clears the cache', model._floor_cache == {})

    # bypassing the cache reproduces it: the guard is memoisation, not a shortcut
    model.constraint_floor(radius)
    model._floor_cache = {}
    check('A24 cached == uncached', model.constraint_floor(radius) == first)


# ---------------------------------------------------------------- report schema

def a24_report_schema(path):
    if not os.path.exists(path):
        print(f'[SKIP] A24 report schema -- {path} absent; '
              'run select_domnist_budgets.py --smoke')
        return
    report = json.load(open(path))

    for key in ('schema_version', 'provenance', 'settings', 'oracle', 'reference',
                'selected', 'intersections_at_selected', 'config_patch', 'warnings'):
        check(f'A24 report has `{key}`', key in report)

    for leg in ('gamma', 'epsilon', 'epsilon_iv'):
        entry = report['selected'].get(leg, {})
        for key in ('value', 'achieved_coverage', 'bracket', 'curve',
                    'target_reachable', 'status_counts'):
            check(f'A24 {leg} reports `{key}`', key in entry)

    check('A24 config_patch is exactly the three budgets',
          set(report['config_patch']) == {'gamma', 'epsilon', 'epsilon_iv'})
    check('A24 config_patch matches the selected values',
          all(report['config_patch'][k] == report['selected'][k]['value']
              for k in ('gamma', 'epsilon', 'epsilon_iv')))

    # every setting a budget is conditional on must be recorded, or the constants
    # cannot be traced back to the run that produced them
    for key in ('pad', 'calibrate', 'n_pi', 'n_eval', 'n_components', 'net',
                'augmentation', 'target_coverage', 'seed'):
        check(f'A24 settings record `{key}`', key in report['settings'])

    for leg in ('epsilon', 'epsilon_iv'):
        entry = report['selected'][leg]
        if entry.get('floor'):
            check(f'A24 {leg} clears its own floor',
                  entry['value'] ** 2 > entry['floor'],
                  f'{entry["value"]**2:.4g} vs {entry["floor"]:.4g}')


if __name__ == '__main__':
    a24_bisection()
    a24_floor_cache()
    a24_report_schema(sys.argv[1] if len(sys.argv) > 1
                      else 'artifacts/domnist-budget_report_smoke.json')
    print(f'\n{"A24 ALL PASS" if not FAIL else "A24 FAILURES: " + ", ".join(FAIL)}')
    sys.exit(bool(FAIL))
