"""A1 (SEM correctness) + A2 (SEM parity with SOURCE).

    <source-venv>/bin/python scripts/a1_a2_sem.py --source /tmp/a2.npz
    <s4cb>/bin/python       scripts/a1_a2_sem.py --check  /tmp/a2.npz
"""
import os
import sys
import argparse

import numpy as np

SOURCE = os.path.expanduser(
    '~/neurips26/ZULFI/doMNIST/symmetry4CausalBoundsDoMNIST')
SEM_KW = dict(alpha=0.0, beta=0.4, eta=0.25, subsample=2)
SEED, N_PARITY, N_LAW = 11, 4096, 200_000

FAIL = []


def _path(path):
    """np.savez_compressed appends .npz, np.load does not. Normalise both ends."""
    return path if path.endswith('.npz') else path + '.npz'


def check(name, ok, detail=''):
    print(f'[{"PASS" if ok else "FAIL"}] {name} {detail}')
    if not ok:
        FAIL.append(name)


def dump(path):
    sys.path.insert(0, SOURCE)
    os.chdir(SOURCE)
    from src.sem.domnist import DoMNISTSEM                  # noqa: PLC0415

    sem = DoMNISTSEM(train=False, **SEM_KW)
    X, y = sem.sample(N=N_PARITY, mode='obs', seed=SEED)
    Xd, yd = sem.sample(N=N_PARITY, mode='do', seed=SEED)
    Xp, y_obs, y_do = sem.sample_paired(N_PARITY, seed=SEED)
    ex, digits = sem.exemplars(420, colors='alternating')
    np.savez_compressed(_path(path), X=X, y=y, Xd=Xd, yd=yd, Xp=Xp, y_obs=y_obs,
                        y_do=y_do, ex=ex, digits=digits)
    print(f'wrote {_path(path)}')


def a1(sem):
    """The structural laws, measured. Cells and closed forms must agree."""
    X, y = sem.sample(N=N_LAW, seed=3)
    last = sem.last_
    for f in (0.0, 1.0):
        for c in (0.0, 1.0):
            mask = (last['f'] == f) & (last['C'] == c)
            got, want = float(y[mask].mean()), float(sem.h_erm(f, c))
            check(f'obs cell f={f:.0f} C={c:.0f} -> {want}',
                  abs(got - want) < 0.013, f'got {got:.4f}')

    _, y_do = sem.sample(N=N_LAW, intervention=True, seed=3)
    last = sem.last_
    for f in (0.0, 1.0):
        got, want = float(y_do[last['f'] == f].mean()), float(sem.h_star(f))
        check(f'do cell f={f:.0f} -> {want}', abs(got - want) < 0.013,
              f'got {got:.4f}')

    check('bias_sq closed form', abs(sem.bias_sq - 0.01) < 1e-12, f'{sem.bias_sq}')
    check('sigma_sq closed form', abs(sem.sigma_sq - 0.15) < 1e-12, f'{sem.sigma_sq}')
    check('attainable closed form',
          np.allclose(sem.attainable, (0.1, 0.9), atol=1e-12), f'{sem.attainable}')

    # h_* is colour-free: the do draw must not depend on C at fixed f
    spread = max(abs(float(y_do[(last['f'] == f) & (last['C'] == c)].mean())
                     - float(sem.h_star(f)))
                 for f in (0.0, 1.0) for c in (0.0, 1.0))
    check('h_* is colour-free', spread < 0.013, f'max cell dev {spread:.4f}')

    try:
        sem.solution
        check('solution raises', False, 'returned a value')
    except NotImplementedError:
        check('solution raises', True)


def a2(sem, path):
    """Bit-identical draws against SOURCE at the same seed."""
    ref = np.load(_path(path))
    X, y = sem.sample(N=N_PARITY, seed=SEED)
    Xd, yd = sem.sample(N=N_PARITY, intervention=True, seed=SEED)
    Xp, y_obs, y_do = sem.sample_paired(N_PARITY, seed=SEED)
    ex, digits = sem.exemplars(420, colors='alternating')

    for name, got, want in (('obs X', X, ref['X']), ('obs y', y, ref['y']),
                            ('do X', Xd, ref['Xd']), ('do y', yd, ref['yd']),
                            ('paired X', Xp, ref['Xp']),
                            ('paired y_obs', y_obs, ref['y_obs']),
                            ('paired y_do', y_do, ref['y_do']),
                            ('exemplars', ex, ref['ex']),
                            ('exemplar digits', digits, ref['digits'])):
        check(f'A2 {name} bit-identical to SOURCE', np.array_equal(got, want))


def run(path):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.sem.do_mnist import DoMNISTSEM                 # noqa: PLC0415

    sem = DoMNISTSEM(seed=SEED, train=False, **SEM_KW)
    a1(sem)
    a2(sem, path)
    print('\n' + ('ALL PASS' if not FAIL else f'FAILURES: {FAIL}'))
    return 1 if FAIL else 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source')
    parser.add_argument('--check')
    args = parser.parse_args()
    if args.source:
        dump(args.source)
    else:
        sys.exit(run(args.check))
