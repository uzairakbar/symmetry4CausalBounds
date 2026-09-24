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
        the seven methods with PI+INV last and `ERM+INV` commented out, carry the
        pinned values (`inv_recenter: inv`, PI's selected gamma, `target_coverage`
        0.995, `erm_inv_tau` 4e-4), agree on every key, and the smoke script's
        BLOCK mirrors them; the tint recipe F2 is F1's block plus the ten-digit
        tint spec.
  (iii) the registry: `backend="copsens"` builds exactly the ten `COPSENS_METHODS`, an
        unknown backend raises, `partial_r2` still builds `ALL_METHODS`, and the
        do-MNIST orchestrator's own registry builds the block's method list lazily
        (no net needed until a builder is CALLED).
  (iv)  the orchestrator: only the `gamma` sweep is wired, perf is skipped with a
        warning, the query runner class carries the block's knobs,
        `DoMNISTSEM.f` raises (the estimand is analytic), and the selection script's
        shared gamma is PI's alone (a source check: no max over methods).
  (v)   the query tint spec: `experiment.query.tint` parses (an int or a list of
        digits, default 8 samples on [0, 1]) and implies `query`; bad digits, a
        repeat, too few samples, a bad range and unknown keys raise; `query: true`
        parses to no tint; the base orchestrator refuses a tint spec, and the
        do-MNIST one keeps it and forwards a plan without it.

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


#: the shipped gamma: PI's selection at target_coverage 0.995 on split C (5,000 rows)
GAMMA = 0.059352292722969865
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
        check(f"(ii) {name} inv_recenter inv", resolved["inv_recenter"] == "inv")
        check(f"(ii) {name} gamma is PI's selected {GAMMA}", resolved["gamma"] == GAMMA)
        check(f"(ii) {name} target_coverage 0.995", resolved["target_coverage"] == 0.995)
        check(f"(ii) {name} erm_inv_tau 4e-4", resolved["erm_inv_tau"] == 4e-4)
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
    with open(os.path.join(REPO, "recipes", "doMnistTintFigF2.yaml")) as fh:
        tint_recipe = yaml.safe_load(fh)
    block_t = tint_recipe["do_mnist"]
    same_t = {k for k in set(block_t) - {"experiment"} if block_t[k] == block_r.get(k)}
    differ_t = sorted(set(block_t) - {"experiment"} - same_t)
    check("(ii) the tint recipe F2 is F1's block key for key", not differ_t, str(differ_t))
    check(
        "(ii) F2 and F1 share their defaults and hyperparameters",
        tint_recipe["defaults"] == recipe["defaults"] and tint_recipe["hyperparameters"] == recipe["hyperparameters"],
    )
    from src.experiments.configs import parse_experiment_plan

    tint = parse_experiment_plan(block_t["experiment"]).tint
    check(
        "(ii) F2 sweeps all ten digits at 8 tints on [0, 1]",
        tint is not None and tint.digits == tuple(range(10)) and tint.sweep_samples == 8 and tint.range == (0.0, 1.0),
    )
    from smoke_do_mnist import BLOCK

    for key in ("gamma", "target_coverage", "erm_inv_tau", "inv_recenter", "epsilon"):
        check(
            f"(ii) the smoke BLOCK mirrors the shipped {key}", BLOCK.get(key) == shipped.get(key), str(BLOCK.get(key))
        )


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
    check("(iv) inv_recenter is read as inv", orchestrator.inv_recenter == "inv")
    check("(iv) the ERM+INV net trains under the shipped block", orchestrator.train_inv is True)
    check("(iv) the split is the block's", orchestrator.split == {"A": 40_000, "B": 10_000, "C": 10_000})
    check("(iv) DoMNISTSEM has no target net", not hasattr(DoMNISTSEM, "target"))
    with open(os.path.join(REPO, "src", "sem", "do_mnist.py")) as fh:
        source = fh.read()
    check("(iv) DoMNISTSEM.f raises", "def f(self, X)" in source and "raise NotImplementedError" in source)
    with open(os.path.join(REPO, "scripts", "select_domnist_gamma.py")) as fh:
        selection = fh.read()
    check(
        "(iv) the selection calibrates on PI", re.search(r'^CALIBRATED_ON = "PI"$', selection, flags=re.M) is not None
    )
    check(
        "(iv) the shared gamma is PI's alone",
        'shared = float(record[CALIBRATED_ON]["gamma"])' in selection and "shared_gamma=shared" in selection,
    )
    no_max = re.search(r"\bmax\([^)]*\[.gamma.\]", selection) is None and "put the max" not in selection
    check("(iv) the selection takes no max over the methods' gammas", no_max)
    for path in ("src/methods/partial_r2_net.py", "src/methods/partial_r2_net_jax.py"):
        check(f"(iv) {path} is gone", not os.path.exists(os.path.join(REPO, path)))


def leg_v():
    print("(v) the query tint spec")
    from src.experiments.base import ExperimentOrchestrator
    from src.experiments.configs import TintSpec, parse_experiment_plan

    plan = parse_experiment_plan({"query": {"tint": {"digit": 7}}})
    check("(v) a bare digit parses, query implied", plan.query is True and plan.tint == TintSpec(digits=(7,)))
    check("(v) the defaults are 8 samples on [0, 1]", plan.tint.sweep_samples == 8 and plan.tint.range == (0.0, 1.0))
    plan = parse_experiment_plan({"query": {"tint": {"digit": [9, 0, 3], "sweep_samples": 5, "range": [0.1, 0.9]}}})
    check("(v) a list, samples and range parse", plan.tint == TintSpec((9, 0, 3), 5, (0.1, 0.9)))
    check("(v) query: true has no tint", parse_experiment_plan({"query": True}).tint is None)
    check("(v) query: {} is a query without tint", parse_experiment_plan({"query": {}}).query is True)
    bad = (
        ({"digit": 10}, "digit 10"),
        ({"digit": -1}, "digit -1"),
        ({"digit": [1, 1]}, "a repeated digit"),
        ({"digit": []}, "no digit"),
        ({"digit": True}, "a bool digit"),
        ({"digit": "7"}, "a string digit"),
        ({}, "a missing digit"),
        ({"digit": 7, "sweep_samples": 1}, "one sample"),
        ({"digit": 7, "sweep_samples": 2.0}, "a float sample count"),
        ({"digit": 7, "range": [0.5, 0.5]}, "an empty range"),
        ({"digit": 7, "range": [0.9, 0.1]}, "a reversed range"),
        ({"digit": 7, "range": [-0.1, 1.0]}, "a range below 0"),
        ({"digit": 7, "range": [0.0, 1.1]}, "a range above 1"),
        ({"digit": 7, "range": [0.0]}, "a one-sided range"),
        ({"digit": 7, "colour": "red"}, "an unknown key"),
    )
    for tint, why in bad:
        try:
            parse_experiment_plan({"query": {"tint": tint}})
            raised = False
        except ValueError:
            raised = True
        check(f"(v) {why} raises", raised)
    try:
        parse_experiment_plan({"query": {"tint": {"digit": 7}, "exemplars": True}})
        raised = False
    except ValueError:
        raised = True
    check("(v) an unknown query key raises", raised)

    tinted = parse_experiment_plan({"query": {"tint": {"digit": 7}}})
    try:
        ExperimentOrchestrator.run(type("Stub", (), {"name": "optical_device"})(), tinted)
        raised = False
    except ValueError as error:
        raised = "tint" in str(error)
    check("(v) the base orchestrator refuses a tint spec", raised)

    from munch import munchify

    from src.experiments.do_mnist import DoMNISTOrchestrator

    block = resolve_dataset_block("do_mnist", {**shipped_block(), "im-ci": 0})
    block.pop("experiment", None)
    orchestrator = DoMNISTOrchestrator(**block, hyperparameters=munchify({"epochs": 1}))
    forwarded = []
    orchestrator._run_query_sweep = lambda: forwarded.append("query")
    orchestrator.run(tinted)
    check("(v) the do-MNIST orchestrator keeps the spec", orchestrator.tint_ == tinted.tint)
    check("(v) and still runs the query", forwarded == ["query"])


def main():
    leg_i()
    leg_ii()
    leg_iii()
    leg_iv()
    leg_v()
    print(f"\nA77 {'PASS' if not FAILURES else 'FAIL'} ({PASSES} passed, {len(FAILURES)} failed)")
    for name in FAILURES:
        print(f"  FAILED: {name}")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
