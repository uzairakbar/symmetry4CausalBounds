"""do-MNIST pipeline gates: A6, A7, A8, A9, A11, A12, A13, A14.

A1/A2 (SEM), A3 (DA), A4 (CopSens parity) and A5 (n_jobs exactness) live in their
own scripts -- they each need SOURCE's environment or a long solve.

    python scripts/a6_a14_pipeline.py
"""

import os
import resource
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_augmentors.do_mnist import DoMNISTDA  # noqa: E402
from src.experiments.configs import DOMNIST_CONFIG, resolve_dataset_block  # noqa: E402
from src.experiments.do_mnist import DoMNISTOrchestrator, Flatten  # noqa: E402
from src.experiments.utils import set_seed  # noqa: E402
from src.experiments.utils.metrics import evaluate_queries  # noqa: E402
from src.methods.copsens import CopSensPI, IVConstrainedCopSens  # noqa: E402
from src.methods.regression import GradientDescentERM  # noqa: E402
from src.methods.sensitivity_models import SolveStatus  # noqa: E402
from src.sem.do_mnist import TINT_HI, TINT_LO, DoMNISTSEM, grey_of, tint  # noqa: E402

N_SAMPLES, N_PI, N_QUERIES = 60_000, 6_000, 48
FAIL = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAIL.append(name)


# --------------------------------------------------------------------- fixture


def fixture():
    set_seed(42)
    sem = DoMNISTSEM(
        seed=42,
        train=True,
        target_samples=N_SAMPLES,
        target_kw=dict(epochs=1),
        alpha=DOMNIST_CONFIG.alpha,
        beta=DOMNIST_CONFIG.beta,
        eta=DOMNIST_CONFIG.eta,
    )
    flat = Flatten()
    X_img, y, _ = sem.sample_paired(N_SAMPLES, seed=sem.seed)
    GX_img, G = DoMNISTDA()(X_img)
    X, GX = flat.fit_transform(X_img), flat.fit_transform(GX_img)

    nets = {
        "X": GradientDescentERM().fit(X, y, init_seed=42, epochs=1),
        "GX": GradientDescentERM().fit(GX, y, init_seed=42, epochs=1),
    }
    keep = np.random.default_rng(42).choice(len(X), N_PI, replace=False)
    return sem, nets, X[keep], GX[keep], y[keep], G[keep], X[:N_QUERIES]


# ------------------------------------------------------------------------ A13


def a13_cost_profile(sem, nets, X, GX, y, G, Q):
    """Time each stage separately. A reporting gate: record, do not assert."""
    common = dict(
        n_components=32,
        calibrate=True,
        clipy=True,
        mu_clip=DOMNIST_CONFIG.attainable,
        n_anchors=DOMNIST_CONFIG.n_anchors,
    )
    profile = {}

    start = time.perf_counter()
    net = GradientDescentERM().fit(X, y, init_seed=42, epochs=1)
    profile["net_fit"] = time.perf_counter() - start

    model = CopSensPI(gamma=0.1, outcome_model=nets["X"], **common)
    start = time.perf_counter()
    model.fit(X, y)
    profile["fit (FactorAnalysis + anchors)"] = time.perf_counter() - start

    start = time.perf_counter()
    payloads = model._prepare(Q, 0.1)
    profile["_prepare"] = time.perf_counter() - start

    model._begin_chunk()
    start = time.perf_counter()
    model._solve_single(payloads[0])
    profile["_solve_single (warm)"] = time.perf_counter() - start

    print("  cost profile (seconds):")
    for stage, seconds in profile.items():
        print(f"    {stage:34s} {seconds:8.3f}")
    check("A13 cost profile recorded", True)
    return net


def a14_memory(sem):
    """Flatten must stay a VIEW; peak RSS must not blow up."""
    X_img, _ = sem.sample(N=20_000, seed=5)
    flat = Flatten().fit_transform(X_img)
    check("A14 Flatten returns a view, not a copy", np.shares_memory(flat, X_img))

    try:
        Flatten().fit_transform(X_img[:, :, ::2])
        check("A14 Flatten rejects non-contiguous input", False, "accepted silently")
    except ValueError:
        check("A14 Flatten rejects non-contiguous input", True)

    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    check("A14 peak RSS < 8 GB", peak_gb < 8.0, f"{peak_gb:.2f} GB")


# ---------------------------------------------------------------- A7, A8, A12


def a7_query_status(sem, nets, X, GX, y, G, Q):
    common = dict(
        n_components=32,
        calibrate=True,
        clipy=True,
        mu_clip=DOMNIST_CONFIG.attainable,
        n_anchors=DOMNIST_CONFIG.n_anchors,
    )

    model = CopSensPI(gamma=0.1, outcome_model=nets["X"], **common).fit(X, y)
    bounds = model.predict(Q)
    check("A7 status length == n_queries", len(model.query_status) == len(Q))
    ok_rows = model.query_status == SolveStatus.OK
    check("A7 no NaN bound carries OK", not np.isnan(bounds[ok_rows]).any())

    record = evaluate_queries(sem.f(Q), bounds, model.query_status, 1.0)
    check("A7 status_counts sums to n_queries", int(record.status_counts.sum()) == len(Q), f"{record.status_counts}")

    # a budget below the floor must gate the WHOLE batch, via _prepare -> None
    starved = IVConstrainedCopSens(gamma=0.1, epsilon_iv=1e-9, outcome_model=nets["GX"], **common).fit(GX, y, Z=G)
    bounds = starved.predict(Q)
    check(
        "A7 sub-floor budget => all INFEASIBLE",
        bool((starved.query_status == SolveStatus.INFEASIBLE).all()) and bool(np.isnan(bounds).all()),
    )


def a21_intersection_wiring(sem, nets, X, GX, y, G, Q):
    """A21: each branch must get the right NET and be fit on the right BALL.

    Both halves are needed. Giving PI_INV the X net instead of the GX net moves an
    inactive constraint's bounds ~7.7%; and a mis-routed `fit(X, ...)` that lets
    `GX=GX` fall into **kwargs silently fits the DA branch on the baseline ball.
    Neither raises; both quietly change every number.
    """
    from src.methods.copsens import IntersectedCopSens, IntersectedIVCopSens

    common = dict(
        gamma=0.1,
        n_components=32,
        calibrate=True,
        clipy=True,
        mu_clip=DOMNIST_CONFIG.attainable,
        n_anchors=DOMNIST_CONFIG.n_anchors,
    )
    models = {
        "PI&DA+PI": IntersectedCopSens(epsilon=0.1, pad=False, outcome_models=nets, **common).fit(X, y, GX=GX, G=G),
        "PI&DA+PI_IV": IntersectedIVCopSens(epsilon=0.1, epsilon_iv=0.1, pad=False, outcome_models=nets, **common).fit(
            X, y, GX=GX, G=G
        ),
    }

    for name, model in models.items():
        check(f"A21 {name}: baseline gets the X net", model.baseline.outcome_ is nets["X"])
        check(f"A21 {name}: DA branch gets the GX net", model.augmented.outcome_ is nets["GX"])
        # the ball each branch was fit on, read off the latent model's own mean
        check(
            f"A21 {name}: baseline fit on the X ball",
            np.allclose(model.baseline.latent_.mean_, X.mean(axis=0), atol=1e-5),
        )
        check(
            f"A21 {name}: DA branch fit on the GX ball",
            np.allclose(model.augmented.latent_.mean_, GX.mean(axis=0), atol=1e-5),
        )
        # padding reaches the DA branch only (Cor. 1)
        check(f"A21 {name}: pad is DA-only", not model.baseline.pad)

    # Cor. 1: the intersection can never be wider than either branch
    for name, model in models.items():
        bounds = model.predict(Q)
        widths = {}
        for label, branch in (("baseline", model.baseline), ("DA", model.augmented)):
            b = branch.predict(Q)
            widths[label] = np.nanmean(b[:, 1] - b[:, 0])
        intersection = np.nanmean(bounds[:, 1] - bounds[:, 0])
        check(
            f"A21 {name}: width <= min(branches)",
            intersection <= min(widths.values()) + 1e-9,
            f"{intersection:.5f} vs {min(widths.values()):.5f}",
        )

    # an infeasible branch must take the whole intersection down, with its status
    starved = IntersectedIVCopSens(epsilon=0.1, epsilon_iv=1e-9, pad=False, outcome_models=nets, **common).fit(
        X, y, GX=GX, G=G
    )
    bounds = starved.predict(Q)
    check("A21 starved IV branch => intersection all-NaN", bool(np.isnan(bounds).all()))
    check("A21 starved IV branch => all INFEASIBLE", bool((starved.query_status == SolveStatus.INFEASIBLE).all()))
    check("A21 the other branch still solved", bool(np.isfinite(starved.baseline.predict(Q)).all()))

    # parallelism must not perturb an intersection either
    serial = IntersectedCopSens(epsilon=0.1, pad=False, outcome_models=nets, **{**common, "n_jobs": 1}).fit(
        X, y, GX=GX, G=G
    )
    parallel = IntersectedCopSens(epsilon=0.1, pad=False, outcome_models=nets, **{**common, "n_jobs": 8}).fit(
        X, y, GX=GX, G=G
    )
    check("A21 intersection n_jobs 1 == 8", np.array_equal(serial.predict(Q), parallel.predict(Q), equal_nan=True))

    # pad=False is H_pi n H_pi~; pad=True is H_pi n (H_pi~ +- eps), so it can only widen
    padded = IntersectedCopSens(epsilon=0.1, pad=True, outcome_models=nets, **common).fit(X, y, GX=GX, G=G)
    width_padded = np.nanmean(np.diff(padded.predict(Q), axis=1))
    width_plain = np.nanmean(np.diff(serial.predict(Q), axis=1))
    check(
        "A21 pad widens the intersection",
        width_padded >= width_plain - 1e-12,
        f"{width_plain:.5f} -> {width_padded:.5f}",
    )
    check("A21 pad=True leaves the baseline branch unpadded", not padded.baseline.pad and padded.augmented.pad)

    # IVConstrainedCopSens must refuse Z=None rather than instrument on X
    from src.methods.copsens import IVConstrainedCopSens

    try:
        IVConstrainedCopSens(epsilon_iv=0.1, outcome_model=nets["GX"], **common).fit(GX, y, Z=None)
        check("A21 IVConstrainedCopSens refuses Z=None", False, "instrumented on X")
    except ValueError:
        check("A21 IVConstrainedCopSens refuses Z=None", True)


def a8_jax_equals_fd(nets, X, GX, y, G, Q):
    common = dict(
        n_components=32,
        calibrate=True,
        clipy=True,
        n_jobs=1,
        mu_clip=DOMNIST_CONFIG.attainable,
        n_anchors=DOMNIST_CONFIG.n_anchors,
    )
    for name, build in (
        ("PI", lambda jg: CopSensPI(gamma=0.1, outcome_model=nets["X"], jax_grad=jg, **common).fit(X, y)),
        (
            "DA+PI_IV",
            lambda jg: IVConstrainedCopSens(
                gamma=0.1, epsilon_iv=0.12, outcome_model=nets["GX"], jax_grad=jg, **common
            ).fit(GX, y, Z=G),
        ),
    ):
        with_jax, without = build(True).predict(Q[:16]), build(False).predict(Q[:16])
        delta = float(np.nanmax(np.abs(with_jax - without)))
        check(f"A8 {name}: JAX == finite differences", delta <= 1e-6, f"max|d| {delta:.3g}")


def a12_training_recipe():
    """The ported _fit must have no early-stopping path at all -- not a default
    that yaml has to cancel."""
    import inspect

    signature = inspect.signature(GradientDescentERM._fit)
    for dead in ("val_frac", "patience", "temperature", "lr_gamma"):
        check(f"A12 `{dead}` is gone from _fit", dead not in signature.parameters)
    check("A12 weight_decay defaults to 0", signature.parameters["weight_decay"].default == 0.0)
    check("A12 epochs defaults to 1", signature.parameters["epochs"].default == 1)

    source = inspect.getsource(GradientDescentERM._fit)
    check("A12 no best_state restore path", "best_state" not in source)


# ------------------------------------------------------------------------- A9


def a9_target_net(sem):
    """Is the estimated target the CAUSAL one?

    The load-bearing checks are the two CONDITIONAL MEANS and the colour
    sensitivity: together they say the net recovers h_* in {0.2, 0.8} and ignores
    the confounding channel. RMSE against the analytic h_* is a per-sample number
    that also carries the net's own approximation error -- which is R2's whole
    point, not a defect -- so it is bounded loosely and reported.

    Measured at the production 1.2M draws, 1 epoch: RMSE 0.0660, means
    0.2084 / 0.7874, colour sensitivity 0.0164. SOURCE reports 0.208 / 0.791.
    At 60k draws the net memorises and every number here degrades.
    """
    sem_test = DoMNISTSEM(
        seed=42,
        train=False,
        target_samples=N_SAMPLES,
        target_kw=dict(epochs=1),
        alpha=DOMNIST_CONFIG.alpha,
        beta=DOMNIST_CONFIG.beta,
        eta=DOMNIST_CONFIG.eta,
    )
    flat = Flatten()
    X_img, _ = sem_test(N=10_000, intervention=True, seed=9)
    last = dict(sem_test.last_)
    X = flat.fit_transform(X_img)

    # the TRAIN SEM's target net, evaluated out of sample -- what the runner uses
    predicted = sem.f(X).ravel()
    h_star = sem_test.h_star(last["f"])
    rmse = float(np.sqrt(np.mean((predicted - h_star) ** 2)))
    check("A9 RMSE(target, h*) < 0.08", rmse < 0.08, f"{rmse:.4f}")
    for value, want in ((predicted[last["f"] < 0.5].mean(), 0.2), (predicted[last["f"] > 0.5].mean(), 0.8)):
        check(f"A9 mean at h*={want} within 0.02", abs(float(value) - want) < 0.02, f"{float(value):.4f}")

    # Y_do is independent of colour, so the target must ignore it
    grey = grey_of(X_img)
    low = sem.f(flat.fit_transform(tint(grey, np.full(len(grey), TINT_LO)))).ravel()
    high = sem.f(flat.fit_transform(tint(grey, np.full(len(grey), TINT_HI)))).ravel()
    sensitivity = float(np.abs(high - low).mean())
    check("A9 colour sensitivity < 0.02", sensitivity < 0.02, f"{sensitivity:.4f}")


# ------------------------------------------------------------------ A11 config


def a11_config_strictness():
    good = dict(seed=42, augmentation="translate", gamma=0.1, epsilon=0.05, methods=["PI"])
    resolve_dataset_block("do_mnist", dict(good))
    check("A11 a valid block resolves", True)

    for missing in ("gamma", "epsilon", "methods", "seed", "augmentation"):
        block = {k: v for k, v in good.items() if k != missing}
        try:
            resolve_dataset_block("do_mnist", block)
            check(f"A11 missing `{missing}` raises", False, "accepted")
        except ValueError:
            check(f"A11 missing `{missing}` raises", True)

    for bad_key in ("n_pii", "link", "solver"):
        try:
            resolve_dataset_block("do_mnist", {**good, bad_key: 1})
            check(f"A11 unknown key `{bad_key}` raises", False, "accepted")
        except ValueError:
            check(f"A11 unknown key `{bad_key}` raises", True)

    try:
        resolve_dataset_block("do_mnist", {**good, "n_jobs": True})
        check("A11 n_jobs: true raises", False, "accepted")
    except ValueError:
        check("A11 n_jobs: true raises", True)

    # the copsens backend must reject a method it does not define, not drop it
    from src.experiments.configs import MethodRegistry

    # `DA+IV` is 2SLS -- one of the two names copsens genuinely does not define.
    # (`PI&DA+PI` used to be the probe here; the backend defines it now.)
    try:
        MethodRegistry.build_methods(
            ["PI", "DA+IV"], gamma=0.1, epsilon=0.05, backend="copsens", outcome_models={"X": None}
        )
        check("A11 copsens rejects an undefined method", False, "filtered silently")
    except ValueError:
        check("A11 copsens rejects an undefined method", True)

    for name in ("PI&DA+PI", "PI&DA+PI_IV"):
        built = MethodRegistry.build_methods(
            [name], gamma=0.1, epsilon=0.05, epsilon_iv=0.05, backend="copsens", outcome_models={"X": None, "GX": None}
        )
        check(f"A11 copsens now defines {name}", name in built)

    try:
        MethodRegistry.build_methods(["PI"], gamma=0.1, epsilon=0.05, backend="copsens")["PI"]()
        check("A11 missing outcome_models raises clearly", False)
    except ValueError as exc:
        check("A11 missing outcome_models raises clearly", "prefit outcome nets" in str(exc))


# ------------------------------------------------------------------------- A6


def a6_perf_fairness():
    """Behavioural, not a source grep: assert the models perf builds are serial,
    and that the PI/ERM cost ratio is the one a serial run produces."""
    orchestrator = DoMNISTOrchestrator(
        seed=42,
        n_samples=1024,
        n_experiments=1,
        sweep_samples=10,
        methods=["ERM", "PI"],
        hyperparameters=dict(epochs=1),
        n_jobs=16,
        augmentation="translate > rotation",
        gamma=0.1,
        epsilon=0.05,
        n_pi=512,
        n_queries=8,
        calibrate=True,
        pad=False,
        clipy=True,
    )
    nets = {"X": None, "GX": None}
    from functools import partial

    sweep = orchestrator.build_methods(gamma=0.1, epsilon=0.05, outcome_models=nets)
    perf = partial(orchestrator.build_methods, n_jobs=1)(gamma=0.1, epsilon=0.05, outcome_models=nets)

    check("A6 sweep models keep the configured n_jobs", sweep["PI"]().n_jobs == 16, f"{sweep['PI']().n_jobs}")
    check("A6 perf models are actually serial", perf["PI"]().n_jobs == 1, f"{perf['PI']().n_jobs}")

    import inspect

    import src.experiments.base as base

    source = inspect.getsource(base.ExperimentOrchestrator._run_perf)
    check("A6 perf binds n_jobs=1 on the factory too", "partial(self.build_methods, n_jobs=1)" in source)


def a26_budget_wiring():
    """The config's gamma/epsilon must reach the BUILT models.

    A runner that assigns its budgets before `super().__init__()` has them
    silently overwritten by the base class's own defaults. That is invisible in
    every unit check -- the models build, fit and solve fine, they just do it on
    the wrong ball: measured, gamma 0.1 -> 1.0 gives flat vacuous bounds on
    every method, baseline PI included.
    """
    from munch import munchify

    from src.experiments.configs import resolve_dataset_block

    gamma, epsilon = 0.1, 0.051
    block = resolve_dataset_block(
        "do_mnist",
        dict(
            seed=42,
            n_experiments=1,
            sweep_samples=10,
            n_components=32,
            net="domnist-fast",
            gamma=gamma,
            epsilon=epsilon,
            calibrate=True,
            n_jobs=8,
            augmentation="translate > rotation > contrast > saturation > hue",
            methods=["PI", "DA+PI", "PI_INV"],
            n_samples=20_000,
            n_pi=2_000,
            n_queries=16,
            experiment={"query": True},
        ),
    )

    set_seed(42)
    orchestrator = DoMNISTOrchestrator(
        **block,
        hyperparameters=munchify(
            dict(lr=0.01, batch=256, epochs=1, optimizer="adam", betas=(0.7, 0.9), onecycle=True, loss="mse")
        ),
    )
    runner = orchestrator.get_query_runner_cls()(
        methods=orchestrator.methods,
        **{k: v for k, v in orchestrator.kwargs.items() if k not in ("methods", "n_queries")},
    )

    check("A26 runner keeps the config gamma", runner.default_gamma == gamma, f"{runner.default_gamma}")
    check("A26 runner keeps the config epsilon", runner.default_epsilon == epsilon, f"{runner.default_epsilon}")
    for name in ("PI", "DA+PI", "PI_INV"):
        built = runner.methods[name]()
        check(f"A26 {name} is built on the config gamma", built.gamma == gamma, f"{built.gamma}")

    # and the bounds are not vacuous: a too-large ball pins every query to the
    # clip range, so the spread across queries collapses
    bounds = runner.methods["PI"]().fit(X=runner.X, y=runner.y).predict(runner.get_sweep_values())
    widths = bounds[:, 1] - bounds[:, 0]
    check("A26 PI bounds are not vacuous", float(np.max(widths)) < 0.99, f"max width {np.max(widths):.4f}")
    check("A26 PI bounds vary across queries", float(np.ptp(widths)) > 1e-6, f"ptp {np.ptp(widths):.6f}")


if __name__ == "__main__":
    # A9 gates the ESTIMAND, so it only means anything at the production draw
    # count -- at the small fixture the net memorises and it fails for that reason
    # alone. Skipped by default; run it before trusting any coverage number.
    full = "--full" in sys.argv

    a11_config_strictness()
    a12_training_recipe()
    a6_perf_fairness()
    a26_budget_wiring()

    sem, nets, X, GX, y, G, Q = fixture()
    a14_memory(sem)
    a13_cost_profile(sem, nets, X, GX, y, G, Q)
    a7_query_status(sem, nets, X, GX, y, G, Q)
    a21_intersection_wiring(sem, nets, X, GX, y, G, Q)
    a8_jax_equals_fd(nets, X, GX, y, G, Q)

    if full:
        N_SAMPLES = 1_200_000
        a9_target_net(
            DoMNISTSEM(
                seed=42,
                train=True,
                target_samples=N_SAMPLES,
                target_kw=dict(epochs=1),
                alpha=DOMNIST_CONFIG.alpha,
                beta=DOMNIST_CONFIG.beta,
                eta=DOMNIST_CONFIG.eta,
            )
        )
    else:
        print("[SKIP] A9 target-net gate (needs --full: 1.2M draws)")

    print("\n" + ("ALL PASS" if not FAIL else f"FAILURES: {FAIL}"))
    sys.exit(1 if FAIL else 0)
