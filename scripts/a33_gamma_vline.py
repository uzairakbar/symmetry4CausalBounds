"""A33: the gamma sweep draws its spec vlines only, no measured line, no label.

`thm1_gamma_min` stays in `src/oracle.py` (a29 legs 1-3 test the formula); its
use as a plotted annotation is gone. Four legs:

  (i)   no strategy overrides `ParamSweepRunner.vlines`: `GammaRatioStrategy` has
        no `vlines` of its own and every `STRATEGIES` class resolves to the base
        property (the spec constant), so a re-added measured line is caught;
  (ii)  `create_sweep_plot` on a synthetic gamma sweep with two requested vlines:
        zero text artists (the rotated Thm. 1 label is gone), exactly one vertical
        line per requested vline inside the resolved xlim (computed from the
        resolved limits, not hard-coded), no swallowed plotting error;
  (iii) `generic_runner` no longer imports `thm1_gamma_min`;
  (iv)  production path: a 1-experiment `SimulationOrchestrator.sweep_record("gamma")`
        stores `PARAM_SPECS["gamma"].vlines` and nothing else in `_sweep_vlines`.

    MPLBACKEND=Agg python scripts/a33_gamma_vline.py
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from loguru import logger  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from src.experiments import generic_runner  # noqa: E402
from src.experiments.base import ParamSweepRunner  # noqa: E402
from src.experiments.configs import _RATIO_GRID, PARAM_SPECS  # noqa: E402
from src.experiments.generic_runner import STRATEGIES, GammaRatioStrategy  # noqa: E402
from src.experiments.simulation import SimulationOrchestrator  # noqa: E402
from src.experiments.utils import set_seed  # noqa: E402
from src.experiments.utils.plotting import create_sweep_plot  # noqa: E402

TMPROOT = os.path.expanduser("~/scratch/tmp/a33")
FAIL = []
_errors = []
logger.add(lambda m: _errors.append(m), level="ERROR")


def check(tag, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {tag} {detail}")
    if not ok:
        FAIL.append(tag)


def vertical_lines(ax):
    """x of every Line2D that is a vertical line (two equal x, spanning the axes)."""
    out = []
    for ln in ax.get_lines():
        x = np.asarray(ln.get_xdata(), dtype=float)
        if x.size == 2 and x[0] == x[1] and len(ln.get_ydata()) == 2:
            out.append(float(x[0]))
    return out


def leg_i():
    print("(i) no strategy overrides the spec vlines")
    check("(i) GammaRatioStrategy has no vlines of its own", "vlines" not in vars(GammaRatioStrategy))
    for name, cls in STRATEGIES.items():
        check(f"(i) STRATEGIES[{name!r}].vlines is the base property", cls.vlines is ParamSweepRunner.vlines)


def leg_ii():
    print("(ii) synthetic gamma sweep render")
    rng = np.random.default_rng(0)
    x = _RATIO_GRID("simulation", 8)
    y = {"PI": 1.0 + 0.1 * rng.random((8, 3))}
    requested = (0.45, 0.2)
    before = len(_errors)
    plt.close("all")
    create_sweep_plot(
        x,
        y,
        xlabel=PARAM_SPECS["gamma"].xlabel,
        ylabel="width",
        xscale=PARAM_SPECS["gamma"].xscale,
        experiment="simulation",
        fname="gamma_width",
        savefig=False,
        vlines=requested,
    )
    ax = plt.gca()
    x_lo, x_hi = ax.get_xlim()
    in_view = [v for v in requested if x_lo <= v <= x_hi]
    drawn = vertical_lines(ax)
    print(f"      xlim ({x_lo:.4f}, {x_hi:.4f}); requested in view {in_view}; drawn {drawn}")
    check("(ii) no text artist on the sweep axes", len(ax.texts) == 0, f"{len(ax.texts)} texts")
    check(
        "(ii) one vertical line per in-view requested vline",
        len(drawn) == len(in_view) and np.allclose(sorted(drawn), sorted(in_view)),
        f"{len(drawn)} drawn vs {len(in_view)} in view",
    )
    check("(ii) no plotting error swallowed", len(_errors) == before)
    plt.close("all")


def leg_iii():
    print("(iii) the runner module does not import the threshold")
    check("(iii) thm1_gamma_min not in generic_runner", "thm1_gamma_min" not in vars(generic_runner))


def leg_iv():
    print("(iv) production sweep_record stores the spec vlines only")
    set_seed(42)
    orch = SimulationOrchestrator(
        seed=42,
        n_samples=512,
        n_experiments=1,
        sweep_samples=3,
        kernel_dim=0,
        treatment_dim=32,
        methods=["PI"],
        hyperparameters={},
        n_jobs=1,
        calibrate=True,
        pad=False,
        clipy=True,
    )
    orch.sweep_record("gamma")
    got = tuple(orch._sweep_vlines["gamma"])
    want = tuple(PARAM_SPECS["gamma"].vlines)
    check("(iv) _sweep_vlines['gamma'] == PARAM_SPECS['gamma'].vlines", got == want, f"{got} vs {want}")


if __name__ == "__main__":
    os.makedirs(TMPROOT, exist_ok=True)
    os.chdir(TMPROOT)
    leg_i()
    leg_ii()
    leg_iii()
    leg_iv()
    print(f"\n{'A33 ALL PASS' if not FAIL else 'A33 FAILURES: ' + ', '.join(FAIL)}")
    sys.exit(bool(FAIL))
