"""A35: the simulation treatment dimension is configured, not hard-coded.

`config.yaml`'s simulation block carries `treatment_dim` (optional; omitted =
`DatasetDefaults`, which reads the SEM's own `TREATMENT_DIMENSION` = 32), and
`SimulationOrchestrator` takes it as a required argument that reaches the SEM
draw. Four legs:

  (i)   the constants agree: `TREATMENT_DIMENSION == 32`, the SEM's default is
        that constant, the `DatasetDefaults` fallback is that constant, and the
        orchestrator has NO default for `treatment_dim` (a script must state it);
  (ii)  the yaml layer: `config.yaml` carries an int `treatment_dim` (yaml load,
        not regex); `resolve_dataset_block` accepts it, fills 32 when the key is
        omitted (every recipe with a simulation block resolves to 32 that way),
        and raises `ValueError` on `true`, `0`, `-3` and `"32"`;
  (iii) the configured value reaches the draw: at the config.yaml value the SEM
        has that dimension and `sem(N=8)[0]` has that many columns; at 16 it has
        16, and `NullSpaceTranslation(sem.W_XY, kernel_dim=0)` maps to 16 columns,
        so nothing downstream keeps the old constant;
  (iv)  every script under `scripts/` that constructs `SimulationOrchestrator(`
        also spells `treatment_dim=` (text scan), so no script regresses to the
        implicit constant.

    python scripts/a35_sim_dim.py
"""

import glob
import inspect
import os
import sys

import numpy as np
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import src.sem.simulation as sem_module  # noqa: E402
from src.data_augmentors.simulation import NullSpaceTranslation  # noqa: E402
from src.experiments.configs import DATASET_DEFAULTS, REQUIRED_KEYS, resolve_dataset_block  # noqa: E402
from src.experiments.simulation import SimulationOrchestrator  # noqa: E402
from src.experiments.utils import set_seed  # noqa: E402
from src.sem.simulation import TREATMENT_DIMENSION, LinearSimulationSEM  # noqa: E402

EXPECT = 32
FAIL = []


def check(tag, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {tag} {detail}")
    if not ok:
        FAIL.append(tag)


def raises(fn):
    try:
        fn()
    except ValueError:
        return True
    return False


def load_block(path):
    """The simulation block as main.py merges it (defaults under the block), or None."""
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    if "simulation" not in cfg:
        return None
    return {**cfg.get("defaults", {}), **cfg["simulation"]}


def orchestrator(treatment_dim):
    set_seed(42)
    return SimulationOrchestrator(
        seed=42,
        n_samples=64,
        n_experiments=1,
        sweep_samples=2,
        kernel_dim=0,
        treatment_dim=treatment_dim,
        methods=["PI"],
        hyperparameters={},
        n_jobs=1,
        calibrate=True,
        pad=False,
        clipy=True,
    )


def leg_i():
    print("(i) constants and signatures")
    check("(i) TREATMENT_DIMENSION == 32", sem_module.TREATMENT_DIMENSION == EXPECT, f"{TREATMENT_DIMENSION}")
    sem_default = inspect.signature(LinearSimulationSEM).parameters["treatment_dimension"].default
    check("(i) SEM default treatment_dimension is the constant", sem_default == TREATMENT_DIMENSION, f"{sem_default}")
    fallback = DATASET_DEFAULTS["simulation"].treatment_dim
    check("(i) DatasetDefaults fallback is the constant", fallback == TREATMENT_DIMENSION, f"{fallback}")
    orch_param = inspect.signature(SimulationOrchestrator).parameters.get("treatment_dim")
    check(
        "(i) SimulationOrchestrator.treatment_dim is required (no default)",
        orch_param is not None and orch_param.default is inspect.Parameter.empty,
    )
    check("(i) treatment_dim is optional in the yaml", "treatment_dim" not in REQUIRED_KEYS["simulation"])


def leg_ii():
    print("(ii) yaml layer")
    root = load_block(f"{REPO}/config.yaml")
    dim = None if root is None else root.get("treatment_dim")
    check(
        "(ii) config.yaml simulation block carries an int treatment_dim",
        isinstance(dim, int) and not isinstance(dim, bool),
        f"{dim!r}",
    )
    resolved = resolve_dataset_block("simulation", root)
    check("(ii) resolve_dataset_block keeps the configured value", resolved["treatment_dim"] == dim)

    without = {k: v for k, v in root.items() if k != "treatment_dim"}
    check(
        "(ii) omitted treatment_dim falls back to 32",
        resolve_dataset_block("simulation", without)["treatment_dim"] == EXPECT,
    )
    for bad in (True, 0, -3, "32"):
        check(
            f"(ii) treatment_dim: {bad!r} is a ValueError",
            raises(lambda bad=bad: resolve_dataset_block("simulation", {**root, "treatment_dim": bad})),
        )

    recipes = sorted(glob.glob(f"{REPO}/recipes/*.yaml"))
    n_sim = 0
    for path in recipes:
        block = load_block(path)
        if block is None:
            continue
        n_sim += 1
        got = resolve_dataset_block("simulation", block)["treatment_dim"]
        check(
            f"(ii) {os.path.relpath(path, REPO)} resolves treatment_dim",
            got == block.get("treatment_dim", EXPECT),
            f"{got}",
        )
    check("(ii) at least one recipe has a simulation block", n_sim >= 1, f"{n_sim} of {len(recipes)}")
    return dim


def leg_iii(configured):
    print("(iii) the configured value reaches the draw")
    for dim in (configured, 16):
        sem = orchestrator(dim)._sem_factory()
        X, _ = sem(N=8)
        check(
            f"(iii) treatment_dim={dim}: sem.treatment_dimension",
            sem.treatment_dimension == dim,
            f"{sem.treatment_dimension}",
        )
        check(f"(iii) treatment_dim={dim}: sem(N=8)[0].shape == (8, {dim})", X.shape == (8, dim), f"{X.shape}")
        if dim == 16:
            da = NullSpaceTranslation(sem.W_XY, kernel_dim=0)
            GX, _ = da(np.asarray(X))
            check(
                "(iii) treatment_dim=16: NullSpaceTranslation output has 16 columns", GX.shape[1] == 16, f"{GX.shape}"
            )


def leg_iv():
    print("(iv) every script that builds the orchestrator states treatment_dim")
    for path in sorted(glob.glob(f"{REPO}/scripts/*.py")):
        with open(path) as fh:
            text = fh.read()
        if "SimulationOrchestrator(" not in text:
            continue
        check(f"(iv) {os.path.basename(path)} spells treatment_dim=", "treatment_dim=" in text)


if __name__ == "__main__":
    leg_i()
    configured = leg_ii()
    leg_iii(configured)
    leg_iv()
    print(f"\n{'A35 ALL PASS' if not FAIL else 'A35 FAILURES: ' + ', '.join(FAIL)}")
    sys.exit(bool(FAIL))
