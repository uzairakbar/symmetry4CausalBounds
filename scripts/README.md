# Acceptance gates

Run with `uv run python` (the repo `.venv`), from the repo root.

Gates A2 and A3 compare against SOURCE
(`../doMNIST/symmetry4CausalBoundsDoMNIST`), so they run in two stages: `--source`
under SOURCE's own `.venv`, then `--check` under this repo's `.venv`. A4 keeps the
same two-stage shape but both stages run here: `--dump` freezes, `--check` compares.

| gate | script | what it pins |
|---|---|---|
| A1 | `a1_a2_sem.py --check` | SEM structural laws vs closed form |
| A2 | `a1_a2_sem.py` | SEM draws bit-identical to SOURCE |
| A3 | `a3_da_parity.py` | DA `(GX, G)` bit-identical to SOURCE |
| A4 | `a4_partial_r2_net_regression.py` | `partial_r2_net` bounds + intermediates vs a frozen dump |
| A5 | `a5_njobs_exactness.py` | `n_jobs` changes nothing but wall clock (`--micro` for a 4k/512/8 fixture) |
| A6-A8, A11-A14 | `a6_a14_pipeline.py` | perf fairness, status, JAX≡FD, config, recipe, cost, memory |
| A9 | `a6_a14_pipeline.py --full` | the estimand is the CAUSAL one (needs 1.2M draws) |
| A10 | `a10_partial_r2_regression.py` | `PartialR2` digest; every bound moved with the sigma-scaled ball (see A10 below) |
| A21 | `a6_a14_pipeline.py` | intersection wiring: branch nets, fit ball, `pad`, `n_jobs` |
| A24 | `a24_budget_selection.py` | bisection contract, floor cache, budget-report schema |
| A25 | `a25_floor_guard.py` | closed-form floor vs cvxpy; guard is a no-op when feasible, rescues when not |
| A27 | `a27_domnist_r2.py` | partial_r2_net backend: nesting, h_* membership at gamma*, Lem. 2 band, JAX≡FD, l=2 path (`--micro`, `--band-se`, `--polish-compare`, `--compare-off`) |
| A28 | `a28_mean_match.py` | Lem. 2 slice: classes == an explicit intercept+equality reference, Cor. 3 closed form, floors, coverage |
| A29 | `a29_thm1_ceiling.py` | Thm. 1: eps+ tight at gamma_min, gamma_min == the fitted DA+PI transition on sim, optical reported as a reference |
| A30 | `a30_optical_truth.py` | optical estimand: h_* on Lem. 2's slice, gamma* over span(phi, 1), both epsilon budgets vs the measured defect, lazy data load |
| A31 | `a31_trs_axis.py` | trS sweep axis by the ball in force: tr(S)/k under `recalibrate: true`, rho tr(S)/k under `false`, the label (`TRS_XLABEL`) following the factor on the render and in `trS_axis.pkl`, the one sort in `create_sweep_plot` on the plotted x under both toggles (line, y and CI band reordered together; argsort pinned per dataset and toggle), factors and vline |
| A32 | `a32_figure_style.py` | figure style from the sweep pkls (`--artifacts DIR`; `--save` re-renders every sweep and perf pdf into DIR): major-only tick labels and at least two in-view major ticks on every axes of every figure, the `legend` / `x_color` / `y_color` / `title` / `title_color` keys through `PLOT_CONFIGS` and `ANNOTATE_SWEEP_PLOT`, an unknown key fails at import |
| A33 | `a33_gamma_vline.py` | gamma sweep annotation: no strategy overrides the spec vlines, one unlabelled line per in-view reference and no text, `generic_runner` free of `thm1_gamma_min`, `sweep_record` stores the spec constant |
| A34 | `a34_eps_tol.py` | query sweep tolerance: `eps_tol` 2**-8 from `SimulationConfig` / `OpticalDeviceConfig` reaches the query runners' IV and optical budgets, the param sweeps and floor guards keep `EPS_TOL` 2**-5; the query runners rescale the raw declared gamma by 1/sigma-hat^2 so the fitted PI radius is sqrt(gamma) (1.0 sim, 0.5 optical) |
| A35 | `a35_sim_dim.py` | simulation `treatment_dim`: optional yaml key with the `DatasetDefaults` fallback (the SEM's 32), validated as a positive int, required by `SimulationOrchestrator`, reaches the SEM draw and the DA, and every script states it |
| A36 | `a36_trs_knob.py` | optical trS knob: `RandomPermutation` honours p (bit-identical at the default p = 1, identity rows kept under p < 1), p = s, grid `linspace(0.2, 0.99)`, tr(S)/k rising and pinned on the shipped-chain fixture |
| A37 | `a37_major_ticks.py` | at least two labelled major ticks on every axis: the (1, 2, 5) log fallback with plain labels on the n sweep (200, 500, 1000), every sweep param on both datasets, the query sweep and the perf figure, on synthetic inputs; x is the exact grid plus a 2 % margin (no top-tail clip) with the in-grid reference lines drawn inside the frame, r = 1 on the gamma sweep pkls too (`--artifacts DIR`, read only) |
| A38 | `a38_recalibrate.py` | the recalibrated post-DA budget (SS4.2): every ball at sigma-hat sqrt(gamma~), gamma~ = gamma((1 - t) + t/rho); Cor. 3 closed form at gamma~, the registry hands rho to the standalone DA+ balls only, the predict-time knob, DA+PI(False)/DA+PI(True) == sqrt(rho_hat) on one sim experiment, ordering under the configured toggles, intersection == max/min of its branches, the floor guard on the same ball, oracle units, `TOGGLE_KEYS`; the old token is grepped away and the recipes that still carry it are reported |
| A39 | `a39_recalibrate_sweep.py` | the `recalibrate` sweep (`param: [recalibrate]`): t on linspace(0, 1) re-solved through the predict-time knob on constant data, x = gamma~/gamma = (1 - t) + t/rho_hat from 1 down to 1/rho, PI unchanged and DA+PI narrower at every t per query, intersection == max/min of its branches, width(t)/width(0) == sqrt((1 - t) + t/rho_hat) with pad off, the t = 0 / t = 1 bounds equal to the `recalibrate: false` / `true` toggle runs on the same draw, `recalibrate_axis.pkl`, the render's sort, and the yaml plumbing (toggle and sweep together); `--optical` adds one optical experiment |
| A40 | `a40_eps_star.py` | robustness eps* per dataset: `ROBUSTNESS_EPSILON_TRUE[experiment_name]` reaches the epsilon runner of both orchestrators and the tuned DA lands on it (sim 3.0, optical 5.0), the sim constant clears the post-DA radius sqrt(gamma*) and sigma sqrt(gamma*), on two sim experiments (5 steps) DA+PI coverage dips below 1 at the smallest r, is 1 at r = 1 and never falls under 0.7; the optical epsilon runner's DA is config.yaml's chain plus the `ROBUSTNESS_AUGMENTATION` component (gaussian-noise) while every other strategy's runner, the query runner and the budget DA keep config.yaml's chain, the same dip (lowest of DA+PI and DA+PI+IV) on three optical experiments, and the optical constant stays 5.0 at under 10 std of h*; `--seed` |

`smoke_do_mnist.py` is an end-to-end query-sweep + perf run at reduced scale;
`--full` runs it at the config's own numbers.

## Mean matching (Lem. 2)

Every PI program solves on the mean-matched slice `E_n[h(X)] = E_n[Y]`, the
covariance ball of Lem. 2. `mean_match: false` in `config.yaml`'s `defaults:`
restores the pre-2026-09 uncentred, intercept-free ball; `a10 --mean-match false`
reproduces the pre-change digest byte-for-byte, which is what pins the old path.

The linear backend enforces the slice EXACTLY (it eliminates the intercept by
centring). The `partial_r2_net` backend cannot: the constraint is nonlinear in the
head weights and the solver backtracks along segments, so it enforces a BAND
`|mean_n h - ybar| <= tau` with `tau = 2 sqrt((sigma_hat^2 + b_r2) / n_pi)`, where
`b_r2` is the ball's own budget -- two standard errors of the level under the
sensitivity model's own bound on `Var(U + xi)` (see `MEAN_BAND_SE`; the second term
is the budget, not `sigma_hat^2 gamma`, so the recalibrated budget is the one in the band). It
enters as the PAIR `(m <= tau, -m <= tau)`, linear in `m`; the squared form
`m^2 <= tau^2` is ill-conditioned as the slab thins. The band is a live constraint,
so PI/DA+PI take the full multi-start polish there; `a27 --polish-compare` at
production scale is what would license flipping `SINGLE_POLISH_WITH_BAND` back on.

`a27 --micro --band-se 0.146` is the leg that drives `theta_c` out of the band and
so exercises the slab anchor; run it after any change to the band, because a slab
too thin to travel in narrows the bounds SILENTLY rather than erroring.
`a27 --compare-off` solves the same fixture with `mean_match: false` and prints
what the band cost. Note the two runs differ in anchor, starts and polish policy as
well as in the feasible set, so their widths are two heuristic optima of nested
sets: band-on coming out slightly wider says the band-OFF solve was the looser one.

`gate_band` is written so that deleting the band clause from `_feasible` makes it
FAIL (checked 2026-09-03: 6/6 methods). Keep that property -- the obvious probe,
shifting the level far out and checking it is rejected, is answered by the R^2 ball
long before the band is consulted and passes a model with no band at all.

do-MNIST gates were run at `--micro` scale only for this change. Before trusting
the full figures, re-run at full scale: `a27`, `a27 --polish-compare`, `a5`,
`a4 --dump`/`--check` (the band moves the frozen numbers), `a6_a14_pipeline.py`
and `smoke_do_mnist.py`. Expect the PI family to be ~2x slower per query.

## The optical estimand (A30)

The optical ground truth is FITTED, not declared, so it is the paper's `h_*` only
if it is fitted in the paper's hypothesis class. Two consequences the code now
carries explicitly:

- **The fit keeps its intercept.** Asm. 1 closes the class under constant shifts
  and Lem. 2's set lives on `E[h(X)] = E[Y]`; the intercept-free fit sat 0.333 off
  that slice (8.6 SE of the level) and was excluded from its own identified set.
  `bias_sq` is projected onto `span(phi, 1)` for the same reason -- measuring
  `gamma*` over one class while solving over another is not a rounding error.
  Restoring it moved `bias_sq` 0.40088 -> 0.402549 and `gamma*` 0.66911 -> 0.673778.
- **`sigma_sq` is the whole conditional spread.** It used to be `1 - bias_sq`, i.e.
  `E[Var(U|X)]` alone, dropping the exogenous noise. That understates `sigma^2` and
  so OVERSTATES `gamma* = bias_sq/sigma^2`, leaving the "tightest gamma keeping
  `h_*` inside" loose: 1.8 % on the shipped device (`gamma*` 0.6738 -> 0.6619),
  801x on experiment 6.
- **The invariance budget is measured, not declared -- in BOTH its norms.** The
  paper carries two epsilons for the same defect `W = h_*(X) - h_*(X~)`: SS3.1
  constrains `E_inv(h) <= eps^2` (L2), while SS2.4 defines eps-approximate
  T-invariance by `sup |W| <= eps` and Thm. 3.A's proof uses THAT one pointwise.
  `epsilon_star` is the L2 norm, so it is right for the constraint and no bound at
  all for the pad -- measured here, RMS 0.212 against a sup of 1.230.
  `OpticalDeviceConfig.epsilon = None` takes the measured L2 budget;
  `pad_epsilon = None` takes `oracle.epsilon_pad_star`, a high quantile of `|W|`.
  A quantile and not the sup because under a DA with a Gaussian component the sup
  is INFINITE, so no finite epsilon makes `h_*` eps-approximately T-invariant in
  the SS2.4 sense: what the budget buys is Thm. 3.A with "a.s." weakened to "with
  probability >= the quantile". Whether the published `2**-2` clears the L2 budget
  DEPENDS on the augmentation (measured `eps*`: 0.2121 for `rotation >
  gaussian-noise`, which `config.yaml` ships, 0.2600 for `all`) -- which is the
  reason to measure it rather than declare it. Floats still pin either budget.
- **`gamma` stays DECLARED.** Deliberately unlike `epsilon`: `gamma` is an
  assumption about unobserved confounding, the one quantity the data cannot report,
  and reading it off the oracle would make the sensitivity analysis circular. The
  consequence to read the query panel with is that its `gamma = 2**-1.5 = 0.354`
  sits below `gamma* = 0.662`, so `h_*` is genuinely outside the identified set
  there and PI+INV misses it on a minority of queries -- the assumption being
  violated, which is what the gamma sweep exists to locate.
- **The oracle is pooled over seeded DA draws.** For a fixed-pool SEM the rows
  never change, so a single augmentation draw is not the population quantity the
  figures annotate. This matters most for `rho`: Thm. 1's threshold has
  `d ln / d ln rho ~ -8` where this device sits, so a 3 % draw wobble moves it
  20 %.

A30's pinch-query leg is the one to keep: at `phi(x) = mean(phi)` Cor. 3 collapses
the interval to `{ybar}`, so that single query -- not the coverage average over the
pool, which stayed at 1.000 throughout -- is what a dropped intercept shows up in.
It discriminates only UNPADDED, though: Thm. 3.A's pad is four times the defect.

`a30` gates the CONFIGURED budgets, across every augmentation the repo can run --
not a hard-coded orchestrator. Keep it that way: pinning `epsilon = 2**-3` must
make it fail (checked 2026-09-03, 8 checks across 3 augmentations), and an earlier
version that tested `_epsilon_budget(None)` against `measured_epsilon_star()` was
asserting `x + EPS_TOL >= x` and passed that pin happily.

## Budget selection

`select_domnist_budgets.py` picks `gamma`, `epsilon` and `epsilon_iv` by POPULATION
coverage and writes `artifacts/domnist-budget_report.json`. DEMOTED to a sanity
check: the pipeline consumes oracle `gamma* = bias_sq/sigma_sq` directly (the
`partial_r2_net` ball is the Lemma-2 ball in function space) and A27 gates
membership, so nothing here needs a coverage-selected budget.

Three sequential legs at one target coverage `X` (`--target-coverage`, or
`do_mnist.target_coverage`; 0.95 or 0.99):

1. lowest `gamma` with coverage ≥ `X` on `PI` — **fixed, never re-selected**
2. lowest `epsilon_iv` with coverage ≥ `X` on `DA+PI+IV`, at that `gamma`
3. lowest `epsilon` with coverage ≥ `X` on `PI+INV`, at that `gamma`

One `gamma` for every method, because that is how the pipeline consumes it.
`DA+PI+IV` and `PI+INV` are subsets of `DA+PI`, so if `DA+PI` misses `X` at the
selected `gamma`, no budget can reach it — the leg is marked
`target_reachable: false` and reports the lowest budget attaining the ceiling.

```bash
python scripts/select_domnist_budgets.py            # 1.2M draws, 10k eval, hours
python scripts/select_domnist_budgets.py --smoke    # 60k / 6k / 2k, minutes
python scripts/a24_budget_selection.py artifacts/domnist-budget_report_smoke.json
```

Budgets are conditional on every setting the report records (`pad`, `recalibrate`,
`n_pi`, `net`, ...). Change one and they are stale. Read `warnings[]` first: it flags
an inert budget and a `DA+PI` ceiling below target.

## Two-stage gates

```bash
SRC=../doMNIST/symmetry4CausalBoundsDoMNIST
$SRC/.venv/bin/python scripts/a1_a2_sem.py    --source /tmp/a2.npz
$SRC/.venv/bin/python scripts/a3_da_parity.py --source /tmp/a3.npz

uv run python scripts/a1_a2_sem.py    --check /tmp/a2.npz
uv run python scripts/a3_da_parity.py --check /tmp/a3.npz

uv run python scripts/a4_partial_r2_net_regression.py --dump  ~/scratch/a4
uv run python scripts/a4_partial_r2_net_regression.py --check ~/scratch/a4
```

## A10

Digest the current tree, then the tree without the change, and diff:

```bash
python scripts/a10_partial_r2_regression.py > /tmp/after.json
git stash && python scripts/a10_partial_r2_regression.py > /tmp/before.json && git stash pop
diff /tmp/before.json /tmp/after.json
```

Mean matching moved every bound, so no earlier tree digests the same any more.
What pins the OLD geometry now is the toggle, not an older commit:

```bash
python scripts/a10_partial_r2_regression.py --mean-match false > /tmp/off.json
```

`off.json` was byte-identical to the digest of the last pre-mean-match commit
(checked 2026-09-03 against `purge the scatter experiment type`). That identity
no longer holds: since the `recalibrate` toggle replaced the old sigma toggle every ball has the
sigma-scaled radius, so every digest moved (checked 2026-09-11). Diff two trees
that both carry the sigma-scaled ball; `a38` pins the mechanism itself.
