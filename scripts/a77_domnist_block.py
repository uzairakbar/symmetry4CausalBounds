"""A77: the do-MNIST block, the recipe and the registry wiring, STATIC (no data, no
net, no run; seconds on a CPU).

  (i)   `resolve_dataset_block("do_mnist", ...)`: the five required keys, every new key
        optional; `inv_recenter` accepts off/on/inv and YAML's bare booleans and
        rejects `fallback`; `erm_inv_tau` must be a positive number; `ERM+INV` is
        accepted on do-MNIST and rejected on every other block; `mix_in` outside
        [0, 1), a `split` without exactly A/B/C, a
        non-positive split size, `pop_seed == seed + 1`, an unknown augmentation amount,
        an instrument-mode spelling and the old `backend`/`unfrozen_layers`/`link`/
        `solver` keys all raise; `im-ci` is forced to 0.
  (ii)  the shipped block of config.yaml (commented or not) and the recipe resolve, carry
        the seven methods with PI+INV last and `ERM+INV` commented out, and agree on
        every key.
  (iii) the registry: `backend="copsens"` builds exactly the ten `COPSENS_METHODS`, an
        unknown backend raises, `partial_r2` still builds `ALL_METHODS`, and the
        do-MNIST orchestrator's own registry builds the block's method list lazily
        (no net needed until a builder is CALLED).
  (iv)  the orchestrator: only the `gamma` sweep is wired, perf is skipped with a
        warning, the query runner class carries the block's knobs, and
        `DoMNISTSEM.f` raises (the estimand is analytic).

Run: uv run python scripts/a77_domnist_block.py
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml  # noqa: E402

from src.experiments.configs import (  # noqa: E402
    ALL_METHODS,
    COPSENS_METHODS,
    DOMNIST_ONLY_METHODS,
    MethodRegistry,
    resolve_dataset_block,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAILURES: list[str] = []
PASSES = 0


def check(name: str, condition: bool, detail: str = ""):
    global PASSES
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f": {detail}" if detail and not condition else ""))
    if condition:
        PASSES += 1
    else:
        FAILURES.append(name)


MINIMAL = dict(seed=42, augmentation="translate", gamma=0.085, epsilon=0.04, methods=["PI"])


def rejection(**overrides) -> str | None:
    try:
        resolve_dataset_block("do_mnist", {**MINIMAL, **overrides})
    except ValueError as error:
        return str(error)
    return None


def shipped_block() -> dict:
    """The `do_mnist:` block of config.yaml, uncommenting it if it is commented."""
    with open(os.path.join(REPO, "config.yaml")) as fh:
        text = fh.read()
    if re.search(r"^do_mnist:", text, flags=re.M):
        return yaml.safe_load(text)["do_mnist"]
    lines = text.split("\n")
    start = next(i for i, line in enumerate(lines) if line.startswith("# do_mnist:"))
    block = []
    for line in lines[start:]:
        if line.startswith("# ") or line == "#":
            block.append(line[2:])
        elif line.startswith("#"):
            block.append(line[1:])
        else:
            break
    return yaml.safe_load("\n".join(block))["do_mnist"]


def leg_i():
    print("(i) block validation")
    for key in ("seed", "augmentation", "gamma", "epsilon", "methods"):
        block = {k: v for k, v in MINIMAL.items() if k != key}
        try:
            resolve_dataset_block("do_mnist", block)
            missing = False
        except ValueError as error:
            missing = key in str(error)
        check(f"(i) {key} is required", missing)
    check("(i) the minimal block resolves", rejection() is None)
    resolved = resolve_dataset_block("do_mnist", {**MINIMAL, "im-ci": 95})
    check("(i) im-ci is forced to 0", resolved["im_ci"] == 0.0)
    check("(i) n_samples defaults to 1.2M", resolved["n_samples"] == 1_200_000)
    for value in ("off", "on", True, False, "ON", " off "):
        block = resolve_dataset_block("do_mnist", {**MINIMAL, "inv_recenter": value})
        check(f"(i) inv_recenter {value!r} resolves to off/on", block["inv_recenter"] in ("off", "on"))
    for value in ("inv", "INV", " inv "):
        block = resolve_dataset_block("do_mnist", {**MINIMAL, "inv_recenter": value})
        check(f"(i) inv_recenter {value!r} resolves to inv", block["inv_recenter"] == "inv")
    check("(i) inv_recenter fallback rejected", rejection(inv_recenter="fallback") is not None)
    check("(i) inv_recenter erm-inv rejected", rejection(inv_recenter="erm-inv") is not None)
    check("(i) erm_inv_tau 4e-4 resolves", rejection(erm_inv_tau=4e-4) is None)
    check("(i) erm_inv_tau 1 resolves (an int)", rejection(erm_inv_tau=1) is None)
    for value in (0, 0.0, -1e-4, True, "4e-4", None, float("inf")):
        check(f"(i) erm_inv_tau {value!r} rejected", rejection(erm_inv_tau=value) is not None)
    check("(i) erm_inv_tau is optional", "erm_inv_tau" not in resolve_dataset_block("do_mnist", dict(MINIMAL)))
    check("(i) ERM+INV resolves on do-MNIST", rejection(methods=["ERM", "ERM+INV", "PI+INV"]) is None)
    others = {
        "simulation": dict(seed=42, kernel_dim=0),
        "optical_device": dict(seed=42, augmentation="rotation"),
        "cigarettes": dict(seed=42, augmentation="rotation", target="plasmode", spec="s"),
    }
    for name, block in others.items():
        try:
            resolve_dataset_block(name, {**block, "methods": ["PI"], "im-ci": 0})
            baseline = True
        except ValueError:
            baseline = False
        try:
            resolve_dataset_block(name, {**block, "methods": ["PI", "ERM+INV"], "im-ci": 0})
            rejected = False
        except ValueError as error:
            rejected = "ERM+INV" in str(error)
        check(f"(i) ERM+INV rejected on {name} (its PI-only block resolves)", baseline and rejected)
    check("(i) mix_in 1.0 rejected", rejection(mix_in=1.0) is not None)
    check("(i) mix_in -0.1 rejected", rejection(mix_in=-0.1) is not None)
    check("(i) mix_in 0.05 resolves", rejection(mix_in=0.05) is None)
    check("(i) mix_in 0 resolves", rejection(mix_in=0) is None)
    check("(i) split without C rejected", rejection(split={"A": 1, "B": 2}) is not None)
    check("(i) split with a zero size rejected", rejection(split={"A": 1, "B": 2, "C": 0}) is not None)
    check("(i) split A/B/C resolves", rejection(split={"A": 40_000, "B": 10_000, "C": 10_000}) is None)
    check("(i) pop_seed == seed + 1 rejected", rejection(pop_seed=43) is not None)
    check("(i) pop_seed 44 resolves", rejection(pop_seed=44) is None)
    check("(i) unknown augmentation amount rejected", rejection(augmentation_amounts={"recolor": 0.1}) is not None)
    check("(i) augmentation_amounts {hue: 0.4} resolves", rejection(augmentation_amounts={"hue": 0.4}) is None)
    check("(i) augmentation_amounts null resolves", rejection(augmentation_amounts=None) is None)
    check("(i) target_coverage 1.5 rejected", rejection(target_coverage=1.5) is not None)
    check("(i) gamma_z_star -1 rejected", rejection(gamma_z_star=-1) is not None)
    check("(i) n_components 0 rejected", rejection(n_components=0) is not None)
    check("(i) calibrate_sigma 'yes' rejected", rejection(calibrate_sigma="yes") is not None)
    for key in ("backend", "unfrozen_layers", "link", "solver", "spread", "n_seeds", "model"):
        check(f"(i) old key {key} rejected", rejection(**{key: 1}) is not None)
    check("(i) an instrument-mode spelling rejected", rejection(methods=["DA+PI+IV(T)"]) is not None)
    check("(i) ERM+IV rejected (no instrument on do-MNIST)", rejection(methods=["ERM+IV"]) is not None)


def leg_ii():
    print("(ii) the shipped block and the recipe")
    shipped = shipped_block()
    with open(os.path.join(REPO, "recipes", "doMnistFigF1.yaml")) as fh:
        recipe = yaml.safe_load(fh)
    block_r = recipe["do_mnist"]
    for name, block in (("config.yaml", shipped), ("recipe", block_r)):
        resolved = resolve_dataset_block("do_mnist", {**block, "im-ci": 0})
        check(f"(ii) {name} block resolves", resolved is not None)
        methods = resolved["methods"]
        seven = len(methods) == 7 and methods[-1] == "PI+INV"
        check(f"(ii) {name} lists seven methods, PI+INV last", seven, str(methods))
        check(f"(ii) {name} does not list ERM+INV", "ERM+INV" not in methods)
        check(f"(ii) {name} inv_recenter off", resolved["inv_recenter"] == "off")
        check(f"(ii) {name} gamma 0.08505", abs(resolved["gamma"] - 0.08505258154439962) < 1e-12)
        check(f"(ii) {name} epsilon 0.04", resolved["epsilon"] == 0.04)
        check(f"(ii) {name} mix_in 0.05", resolved["mix_in"] == 0.05)
        check(f"(ii) {name} exemplar_seed 420", resolved["exemplar_seed"] == 420)
        check(f"(ii) {name} split 40k/10k/10k", resolved["split"] == {"A": 40_000, "B": 10_000, "C": 10_000})
    for name in ("config.yaml", os.path.join("recipes", "doMnistFigF1.yaml")):
        with open(os.path.join(REPO, name)) as fh:
            listed = re.search(r"^\s*# - ERM\+INV\b", fh.read(), flags=re.M) is not None
        check(f"(ii) {name} carries ERM+INV commented out", listed)
    shared = set(shipped) & set(block_r) - {"experiment"}
    same = [k for k in shared if shipped[k] == block_r[k]]
    differ = sorted(set(shared) - set(same))
    check("(ii) the two blocks agree on every shared key", not differ, str(differ))
    check("(ii) the recipe carries the query experiment only", block_r["experiment"] == {"query": True})
    toggles = recipe["defaults"]
    check("(ii) the recipe pins the toggles", toggles["recalibrate"] is False and toggles["pad"] is False)


def leg_iii():
    print("(iii) the registry")
    built = MethodRegistry.build_methods(list(COPSENS_METHODS), gamma=0.085, epsilon=0.04, backend="copsens")
    check("(iii) copsens builds exactly its ten", tuple(built) == COPSENS_METHODS, str(tuple(built)))
    check("(iii) ERM+INV is a copsens method", "ERM+INV" in COPSENS_METHODS)
    check("(iii) ERM+INV is not in ALL_METHODS", "ERM+INV" not in ALL_METHODS and "ERM+INV" in DOMNIST_ONLY_METHODS)
    try:
        MethodRegistry.build_methods(
            ["ERM+INV"], gamma=0.085, epsilon=0.04, backend="copsens", outcome_models={"X": 1, "GX": 2}
        )["ERM+INV"]()
        raised = False
    except ValueError as error:
        raised = "INV" in str(error)
    check("(iii) ERM+INV without an INV net raises naming it", raised)
    try:
        MethodRegistry.build_methods(["PI"], gamma=0.085, epsilon=0.04, backend="partial_r2_net")
        raised = False
    except ValueError:
        raised = True
    check("(iii) partial_r2_net is no longer a backend", raised)
    try:
        MethodRegistry.build_methods(["ERM+IV"], gamma=0.085, epsilon=0.04, backend="copsens")
        raised = False
    except ValueError:
        raised = True
    check("(iii) copsens refuses ERM+IV", raised)
    linear = MethodRegistry.build_methods(list(ALL_METHODS), gamma=1.0, epsilon=0.1)
    check("(iii) partial_r2 still builds ALL_METHODS", tuple(linear) == ALL_METHODS)
    try:
        built["PI"]()
        raised = False
    except ValueError as error:
        raised = "prefit" in str(error)
    check("(iii) a copsens builder without nets raises when CALLED, not when built", raised)


def leg_iv():
    print("(iv) the orchestrator, static")
    from munch import munchify

    from src.experiments.do_mnist import DoMNISTOrchestrator, DoMNISTQuerySweep
    from src.sem.do_mnist import DoMNISTSEM

    block = resolve_dataset_block("do_mnist", {**shipped_block(), "im-ci": 0})
    block.pop("experiment", None)
    orchestrator = DoMNISTOrchestrator(**block, hyperparameters=munchify({"epochs": 1}))
    check("(iv) the registry names the block's methods", tuple(orchestrator.methods) == tuple(block["methods"]))
    runner_cls = orchestrator.get_query_runner_cls()
    check("(iv) the query runner is a DoMNISTQuerySweep", issubclass(runner_cls, DoMNISTQuerySweep))
    sweep_cls = orchestrator.get_sweep_runner_cls("gamma")
    check("(iv) the gamma sweep is wired", sweep_cls is not None)
    for param in ("epsilon", "omega", "n", "m", "recalibrate"):
        try:
            orchestrator.get_sweep_runner_cls(param)
            raised = False
        except NotImplementedError:
            raised = True
        check(f"(iv) the {param} sweep raises NotImplementedError", raised)
    check("(iv) inv_recenter is read as off", orchestrator.inv_recenter == "off")
    check("(iv) the split is the block's", orchestrator.split == {"A": 40_000, "B": 10_000, "C": 10_000})
    check("(iv) DoMNISTSEM has no target net", not hasattr(DoMNISTSEM, "target"))
    with open(os.path.join(REPO, "src", "sem", "do_mnist.py")) as fh:
        source = fh.read()
    check("(iv) DoMNISTSEM.f raises", "def f(self, X)" in source and "raise NotImplementedError" in source)
    for path in ("src/methods/partial_r2_net.py", "src/methods/partial_r2_net_jax.py"):
        check(f"(iv) {path} is gone", not os.path.exists(os.path.join(REPO, path)))


def main():
    leg_i()
    leg_ii()
    leg_iii()
    leg_iv()
    print(f"\nA77 {'PASS' if not FAILURES else 'FAIL'} ({PASSES} passed, {len(FAILURES)} failed)")
    for name in FAILURES:
        print(f"  FAILED: {name}")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
