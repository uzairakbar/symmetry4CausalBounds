"""A38: the recalibrated post-DA budget (SS4.2).

Every ball has the radius sigma-hat sqrt(gamma~) with gamma~ = gamma ((1 - t) + t / rho),
t = `recalibrate` in [0, 1] and rho the information-loss factor of the data the ball
was fit on (1 on a baseline). t = 1 solves the DA+ methods at gamma/rho, t = 0 at the
inherited gamma.

  (i)    the old toggle name (the one without the `re`) occurs nowhere in src, scripts,
         config.yaml, README.md, scripts/README.md (whole word; recipes are the user's
         and are reported, not gated);
  (ii)   closed form on a synthetic fixture: PartialR2(gamma, rho, recalibrate=t) equals
         Cor. 3 at gamma~ to 1e-6 for t in (0, 0.5, 1) and rho in (1.3, 2.0); rho = 1
         makes t = 0 and t = 1 bit-identical and rho = 0.5 is read as 1 (DPI), so the
         budget never exceeds the inherited gamma; the registry hands rho to the standalone
         DA+ balls only and the intersections read theirs off their branches;
  (iii)  predict-time knob: `predict(Q, recalibrate=1.0)` on a t = 0 model equals the
         t = 1 model, and the intersection follows its branches;
  (iv)   identity leg, one sim experiment, pad off, clipy off: PI bit-identical across
         the toggle, width_DA+PI(False) / width_DA+PI(True) == sqrt(rho_hat) per query
         (rtol 1e-6, rho_hat recomputed here), the runner's rho == the intersection's;
  (iv')  configured leg (toggles read from config.yaml): PI bit-identical, DA+PI and
         DA+PI+IV narrower under True at every query and strictly in the mean, the pad
         still applied;
  (v)    both intersections == max/min of their standalone branches under True, both legs;
  (vi)   `constraint_floor(gamma, rho=r, recalibrate=t)` == `constraint_floor(gamma~)`;
  (vii)  oracle units: gamma* == bias^2/sigma^2, `thm1_gamma_min` is its closed form,
         and no oracle function still takes the old flag;
  (viii) config: `TOGGLE_KEYS`, the old key rejected by `resolve_dataset_block`,
         config.yaml loads with a bool `defaults.recalibrate`.

Runs from the repo root, writes nothing. Nothing here touches do-MNIST.

    python scripts/a38_recalibrate.py
"""

import glob
import inspect
import os
import subprocess
import sys

import numpy as np
import yaml
from loguru import logger

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import src.oracle as oracle  # noqa: E402
from src.data_augmentors.simulation import NullSpaceTranslation  # noqa: E402
from src.experiments.base import SweepData  # noqa: E402
from src.experiments.configs import TOGGLE_KEYS, MethodRegistry, resolve_dataset_block  # noqa: E402
from src.experiments.simulation import SimulationOrchestrator  # noqa: E402
from src.experiments.utils import set_seed  # noqa: E402
from src.experiments.utils.metrics import rho_hat  # noqa: E402
from src.methods.sensitivity_models import (  # noqa: E402
    IntersectedPartialR2,
    PartialR2,
    constraint_floor,
    recalibrated_gamma,
)
from src.sem.simulation import LinearSimulationSEM  # noqa: E402

# built at runtime so this file passes its own grep
OLD = "".join(("cali", "brate"))
GAMMA, EPSILON, EPSILON_IV = 0.5, 0.3, 0.2
METHODS = ["PI", "DA+PI", "DA+PI+IV", "PI&DA+PI", "PI&DA+PI+IV"]
N_QUERIES = 32
N_JOBS = 4
FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAIL.append(name)


def width(bounds):
    return bounds[:, 1] - bounds[:, 0]


# ------------------------------------------------------------------ fixtures


def synthetic():
    """a28's draw: a 6-column design, a noisy linear outcome, a jittered GX."""
    set_seed(3)
    X = np.random.randn(200, 6)
    y = X @ np.random.randn(6, 1) + 0.3 * np.random.randn(200, 1)
    GX = X + 0.2 * np.random.randn(*X.shape)
    Q = np.random.randn(60, 6)
    return X + 1.5, GX + 1.5, y, Q + 1.5


def cor3(design, y, Q, gamma):
    """Cor. 3 on Lem. 2's slice: h_erm(q) +- sigma-hat sqrt(gamma) ||g_q||."""
    mu, ybar = design.mean(axis=0), float(np.mean(y))
    Dc, yc = design - mu, np.asarray(y).flatten() - ybar
    h_erm = np.linalg.lstsq(Dc, yc, rcond=None)[0]
    sigma = float(np.sqrt(np.mean((yc - Dc @ h_erm) ** 2)))
    cov_inv = np.linalg.pinv(Dc.T @ Dc / len(Dc))
    Qc = Q - mu
    margin = sigma * np.sqrt(gamma) * np.sqrt(np.maximum(0.0, np.sum((Qc @ cov_inv) * Qc, axis=1)))
    centre = Qc @ h_erm + ybar
    return np.column_stack([centre - margin, centre + margin])


def sim_models(recalibrate, pad, clipy, mean_match):
    """One sim experiment through the production gamma runner at gamma*: the data,
    the runner, the fitted models and their bounds on the first N_QUERIES test rows."""
    set_seed(42)
    orch = SimulationOrchestrator(
        seed=42,
        n_samples=2048,
        n_experiments=1,
        sweep_samples=1,
        kernel_dim=0,
        treatment_dim=32,
        methods=METHODS,
        hyperparameters={},
        n_jobs=N_JOBS,
        recalibrate=recalibrate,
        pad=pad,
        clipy=clipy,
        mean_match=mean_match,
    )
    runner = orch.get_sweep_runner_cls("gamma")(
        methods=orch.methods, method_factory=orch.build_methods, param_grid_override=[1.0], **orch._get_clean_kwargs()
    )
    data = SweepData.coerce(runner.generate_data(0, 1.0))
    models = runner.build_models(0, 0, data)
    Xq = data.X_test[:N_QUERIES]
    bounds = {name: model.predict(Xq, **runner.get_predict_kwargs(1.0, 0)) for name, model in models.items()}
    return data, runner, models, bounds


# ------------------------------------------------------------------ (i) grep


def leg_i():
    print("(i) the old token is gone")
    targets = ["src", "scripts", "config.yaml", "README.md"]
    result = subprocess.run(["grep", "-rnw", OLD, *targets], cwd=REPO, capture_output=True, text=True)
    hits = [line for line in result.stdout.splitlines() if line]
    check(f"(i) grep -rnw {OLD} over {targets} is empty", not hits, "; ".join(hits[:3]))
    recipes = []
    for path in sorted(glob.glob(os.path.join(REPO, "recipes", "*.yaml"))):
        with open(path) as handle:
            if any(line.strip().startswith(f"{OLD}:") for line in handle):
                recipes.append(os.path.basename(path))
    print(f"      report: recipes still carrying `{OLD}:` (not edited here, the user's): {recipes}")


# ------------------------------------------------------ (ii) closed form, registry


def leg_ii():
    print("(ii) closed form and registry")
    X, GX, y, Q = synthetic()
    common = dict(gamma=GAMMA, epsilon=EPSILON, mean_match=True, clipy=False, n_jobs=1)
    for rho in (1.3, 2.0):
        for t in (0.0, 0.5, 1.0):
            got = PartialR2(rho=rho, recalibrate=t, **common).fit(GX, y).predict(Q)
            want = cor3(GX, y, Q, recalibrated_gamma(GAMMA, rho, t))
            delta = float(np.abs(got - want).max())
            check(f"(ii) PartialR2(rho={rho}, t={t}) == Cor. 3 at gamma~", delta <= 1e-6, f"max |d| {delta:.2e}")
    off = PartialR2(rho=1.0, recalibrate=False, **common).fit(GX, y).predict(Q)
    on = PartialR2(rho=1.0, recalibrate=True, **common).fit(GX, y).predict(Q)
    check("(ii) rho = 1: t = 0 and t = 1 bit-identical", np.array_equal(off, on))
    # DPI: a sample rho below 1 is noise and must not grow the ball past gamma
    below = PartialR2(rho=0.5, recalibrate=True, **common).fit(GX, y)
    check(
        "(ii) rho = 0.5, t = 1: the budget is the inherited gamma",
        below.budget(GAMMA) == GAMMA,
        f"{below.budget(GAMMA)}",
    )
    check("(ii) rho = 0.5, t = 1: the bounds are the t = 0 bounds", np.array_equal(below.predict(Q), off))
    check("(ii) recalibrated_gamma(g, 0.5, 1) == g", recalibrated_gamma(GAMMA, 0.5, 1.0) == GAMMA)

    names = ["PI", "PI+INV", "PI+IV", "DA+PI", "DA+PI+IV", "PI&DA+PI", "PI&DA+PI+IV"]
    built = MethodRegistry.build_methods(
        names, gamma=GAMMA, epsilon=EPSILON, epsilon_iv=EPSILON_IV, rho=1.7, recalibrate=True, n_jobs=1
    )
    models = {name: build() for name, build in built.items()}
    for name in ("PI", "PI+INV", "PI+IV"):
        check(f"(ii) registry: {name}.rho == 1.0", models[name].rho == 1.0, f"{models[name].rho}")
    for name in ("DA+PI", "DA+PI+IV"):
        check(f"(ii) registry: {name}.rho == 1.7", models[name].rho == 1.7, f"{models[name].rho}")
        check(f"(ii) registry: {name}.recalibrate == 1.0", models[name].recalibrate == 1.0)
    for name in ("PI&DA+PI", "PI&DA+PI+IV"):
        check(f"(ii) registry: {name}.rho == 1.0 before fitting", models[name].rho == 1.0, f"{models[name].rho}")
        model = models[name].fit(X, y, GX=GX, G=GX)
        ratio = model.augmented.sigma_sq / model.baseline.sigma_sq
        check(f"(ii) registry: {name}.rho == branch ratio after fitting", model.rho == ratio, f"{model.rho:.6f}")
        check(f"(ii) registry: {name}.augmented.rho == {name}.rho", model.augmented.rho == model.rho)
        check(f"(ii) registry: {name}.baseline.rho == 1.0", model.baseline.rho == 1.0)
    off = MethodRegistry.build_methods(["DA+PI"], gamma=GAMMA, epsilon=EPSILON, rho=1.7, recalibrate=False)
    check("(ii) registry: recalibrate=False reaches the ball as t = 0", off["DA+PI"]().recalibrate == 0.0)


# ---------------------------------------------------------- (iii) predict-time knob


def leg_iii():
    print("(iii) predict-time knob")
    X, GX, y, Q = synthetic()
    common = dict(gamma=GAMMA, epsilon=EPSILON, mean_match=True, clipy=False, n_jobs=1)
    at_zero = PartialR2(rho=1.7, recalibrate=False, **common).fit(GX, y)
    at_one = PartialR2(rho=1.7, recalibrate=True, **common).fit(GX, y)
    plain = at_zero.predict(Q)
    switched = at_zero.predict(Q, recalibrate=1.0)
    check(
        "(iii) predict(Q, recalibrate=1.0) on a t = 0 model == the t = 1 model",
        np.array_equal(switched, at_one.predict(Q)),
    )
    check("(iii) the knob sticks on the model", at_zero.recalibrate == 1.0)
    check("(iii) predict(Q, recalibrate=0.0) goes back", np.array_equal(at_zero.predict(Q, recalibrate=0.0), plain))
    check("(iii) t = 0 and t = 1 differ on this fixture", not np.array_equal(plain, switched))

    inter = IntersectedPartialR2(recalibrate=False, **common).fit(X, y, GX=GX, G=GX)
    got = inter.predict(Q, recalibrate=1.0)
    base = PartialR2(recalibrate=False, **common).fit(X, y).predict(Q)
    da = PartialR2(rho=inter.rho, recalibrate=True, **common).fit(GX, y).predict(Q)
    want = np.column_stack([np.maximum(base[:, 0], da[:, 0]), np.minimum(base[:, 1], da[:, 1])])
    check(
        "(iii) PI&DA+PI.predict(Q, recalibrate=1.0) == max/min of PI and DA+PI(t = 1)",
        np.allclose(got, want, atol=1e-9),
    )
    check(
        "(iii) the intersection's branches carry t",
        inter.augmented.recalibrate == 1.0 and inter.baseline.recalibrate == 1.0,
    )


# ----------------------------------------------------- (iv), (iv'), (v) sim legs


def leg_iv_identity():
    print("(iv) identity leg: sim, pad off, clipy off")
    data, runner, models, off = sim_models(False, pad=False, clipy=False, mean_match=True)
    _, _, models_on, on = sim_models(True, pad=False, clipy=False, mean_match=True)
    rho = rho_hat(data.X, data.GX, data.y, intercept=True)
    check("(iv) PI bounds bit-identical across the toggle", np.array_equal(off["PI"], on["PI"]))
    ratio = width(off["DA+PI"]) / width(on["DA+PI"])
    check(
        "(iv) width_DA+PI(False) / width_DA+PI(True) == sqrt(rho_hat) per query",
        np.allclose(ratio, np.sqrt(rho), rtol=1e-6),
        f"ratio {ratio.min():.6f}..{ratio.max():.6f} vs sqrt(rho_hat) {np.sqrt(rho):.6f}",
    )
    check(
        "(iv) runner.fit_rho == PI&DA+PI.rho",
        np.isclose(runner.fit_rho(0, data), models_on["PI&DA+PI"].rho, rtol=1e-9),
        f"{runner.fit_rho(0, data):.6f} vs {models_on['PI&DA+PI'].rho:.6f}",
    )
    check("(iv) the standalone DA+PI carries the runner's rho", models_on["DA+PI"].rho == runner.fit_rho(0, data))
    print(
        f"      widths False: PI {width(off['PI']).mean():.4f} DA+PI {width(off['DA+PI']).mean():.4f} "
        f"DA+PI+IV {width(off['DA+PI+IV']).mean():.4f} | True: DA+PI {width(on['DA+PI']).mean():.4f} "
        f"DA+PI+IV {width(on['DA+PI+IV']).mean():.4f} | rho_hat {rho:.5f}"
    )
    leg_v(on, "identity leg")


def leg_iv_configured():
    print("(iv') configured leg: toggles from config.yaml")
    with open(os.path.join(REPO, "config.yaml")) as handle:
        defaults = (yaml.safe_load(handle) or {}).get("defaults") or {}
    toggles = dict(pad=bool(defaults.get("pad", False)), clipy=bool(defaults.get("clipy", True)))
    toggles["mean_match"] = bool(defaults.get("mean_match", True))
    print(f"      toggles {toggles}; config recalibrate {defaults.get('recalibrate')!r}")
    _, _, _, off = sim_models(False, **toggles)
    data_on, _, models_on, on = sim_models(True, **toggles)
    check("(iv') PI bounds bit-identical across the toggle", np.array_equal(off["PI"], on["PI"]))
    for name in ("DA+PI", "DA+PI+IV"):
        w_off, w_on = width(off[name]), width(on[name])
        check(
            f"(iv') {name}(True) <= {name}(False) per query",
            bool(np.all(w_on <= w_off + 1e-9)),
            f"max excess {float(np.max(w_on - w_off)):.2e}",
        )
        check(
            f"(iv') {name}(True) strictly narrower in the mean",
            float(np.nanmean(w_on)) < float(np.nanmean(w_off)),
            f"{np.nanmean(w_on):.4f} vs {np.nanmean(w_off):.4f}",
        )
    model = models_on["DA+PI"]
    query = data_on.X_test[:1]
    clipy = model.clipy
    model.clipy = False
    padded = width(model.predict(query))[0]
    model.pad = False
    unpadded = width(model.predict(query))[0]
    model.pad = toggles["pad"]
    model.clipy = clipy
    want = 2.0 * model.pad_amount if toggles["pad"] else 0.0
    check(
        "(iv') padding still applied as configured",
        np.isclose(padded - unpadded, want, rtol=1e-9, atol=1e-12),
        f"{padded - unpadded:.6g} vs {want:.6g}",
    )
    leg_v(on, "configured leg")


def leg_v(on, label):
    # both intersection pairs: a dropped `augmented.rho` on either one shows up here
    for name, da in (("PI&DA+PI", "DA+PI"), ("PI&DA+PI+IV", "DA+PI+IV")):
        got = on[name]
        want = np.column_stack([np.maximum(on["PI"][:, 0], on[da][:, 0]), np.minimum(on["PI"][:, 1], on[da][:, 1])])
        delta = float(np.nanmax(np.abs(got - want)))
        check(
            f"(v) {name} == max/min of PI and {da} under True ({label})",
            np.allclose(got, want, atol=1e-9),
            f"max |d| {delta:.2e}",
        )


# --------------------------------------------------------------- (vi) the floor


def leg_vi():
    print("(vi) the floor on the same ball")
    # Z = X, not GX: with the design as its own instrument the IV minimiser is
    # the ball centre and every floor is 0, which would make this leg vacuous.
    # A small gamma keeps the minimiser outside the ball, so the floor is
    # positive and moves with the budget.
    X, GX, y, _ = synthetic()
    gamma = 1e-3
    floors = []
    for rho in (1.3, 2.0):
        for t in (0.5, 1.0):
            got = constraint_floor(GX, y, gamma, kind="iv", Z=X, rho=rho, recalibrate=t)
            want = constraint_floor(GX, y, recalibrated_gamma(gamma, rho, t), kind="iv", Z=X)
            floors.append(got)
            check(
                f"(vi) floor(gamma, rho={rho}, t={t}) == floor(gamma~)",
                np.isclose(got, want, rtol=1e-9),
                f"{got:.6g} vs {want:.6g}",
            )
    plain = constraint_floor(GX, y, gamma, kind="iv", Z=X)
    check("(vi) the floors are positive and rise with the recalibration", plain < floors[0] < floors[-1])
    inert = constraint_floor(GX, y, gamma, kind="iv", Z=X, rho=1.0, recalibrate=1.0)
    check("(vi) rho = 1 leaves the floor alone", plain == inert)


# ------------------------------------------------------------- (vii) oracle units


def leg_vii():
    print("(vii) oracle units")
    set_seed(0)
    sem = LinearSimulationSEM(treatment_dimension=8, gamma=1.0)
    da = NullSpaceTranslation(sem.W_XY, kernel_dim=0)
    o = oracle.compute_oracle_parameters(sem, da, n_samples=512, mean_match=True)
    check(
        "(vii) gamma* == bias^2 / sigma^2", np.isclose(o.gamma_star, sem.bias_sq / sem.sigma_sq), f"{o.gamma_star:.6f}"
    )
    want = max(0.0, (o.gamma_star - (o.rho - 1.0)) / o.rho)
    check("(vii) thm1_gamma_min == max(0, (gamma* - (rho - 1)) / rho)", np.isclose(oracle.thm1_gamma_min(o), want))
    for name in ("gamma_star", "compute_oracle_parameters", "thm1_eps_ceiling", "thm1_eps_valid", "thm1_gamma_min"):
        params = inspect.signature(getattr(oracle, name)).parameters
        check(f"(vii) oracle.{name} has no `{OLD}` parameter", OLD not in params)


# ------------------------------------------------------------------ (viii) config


def leg_viii():
    print("(viii) config")
    check(
        "(viii) TOGGLE_KEYS",
        {"recalibrate", "pad", "clipy", "n_jobs", "mean_match"} == TOGGLE_KEYS,
        f"{sorted(TOGGLE_KEYS)}",
    )
    try:
        resolve_dataset_block("simulation", {"seed": 42, "kernel_dim": 0, OLD: False})
        rejected = False
    except ValueError:
        rejected = True
    check(f"(viii) resolve_dataset_block rejects `{OLD}`", rejected)
    accepted = resolve_dataset_block("simulation", {"seed": 42, "kernel_dim": 0, "recalibrate": False})
    check("(viii) resolve_dataset_block accepts `recalibrate`", accepted.get("recalibrate") is False)
    with open(os.path.join(REPO, "config.yaml")) as handle:
        defaults = (yaml.safe_load(handle) or {}).get("defaults") or {}
    check(
        "(viii) config.yaml defaults.recalibrate is a bool",
        isinstance(defaults.get("recalibrate"), bool),
        f"{defaults.get('recalibrate')!r}",
    )


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    os.chdir(REPO)
    leg_i()
    leg_ii()
    leg_iii()
    leg_iv_identity()
    leg_iv_configured()
    leg_vi()
    leg_vii()
    leg_viii()
    print(f"\n{'A38 ALL PASS' if not FAIL else 'A38 FAILURES: ' + chr(10) + chr(10).join('  ' + f for f in FAIL)}")
    sys.exit(bool(FAIL))
