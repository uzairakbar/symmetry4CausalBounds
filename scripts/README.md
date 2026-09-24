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
| A10 | `a10_partial_r2_regression.py` | `PartialR2` digest; every bound moved with the sigma-scaled ball (see A10 below) |
| A25 | `a25_floor_guard.py` | closed-form floor vs cvxpy; no budget is ever raised (feasible and infeasible cells return the raw oracle budget, an infeasible one logs one INFO BELOW line); an omega knob under its T floor reads all-INFEASIBLE with a gap in the width line; a completeness grep finds nothing of the old guard in `src` or `scripts` |
| A28 | `a28_mean_match.py` | Lem. 2 slice: classes == an explicit intercept+equality reference, Cor. 3 closed form, floors, coverage |
| A29 | `a29_thm1_ceiling.py` | Thm. 1: eps+ tight at gamma_min, gamma_min == the fitted DA+PI transition on sim, optical reported as a reference |
| A30 | `a30_optical_truth.py` | optical estimand: h_* on Lem. 2's slice, gamma* over span(phi, 1), both epsilon budgets vs the measured defect, lazy data load |
| A31 | `a31_omega_axis.py` | omega sweep axis by the ball in force: tr(S)/k under `recalibrate: true`, rho tr(S)/k under `false`, the label (`OMEGA_XLABEL`) following the factor on the render and in `omega_axis.pkl`, the one sort in `create_sweep_plot` on the plotted x under both toggles (line, y and CI band reordered together; argsort pinned per dataset and toggle), factors and vline; the rename is complete (a case-insensitive grep for the retired sweep key over src, recipes, config.yaml and scripts hits only tr(S)/k's own names) and the retired key has no alias |
| A32 | `a32_figure_style.py` | figure style from the sweep and perf pkls (`--artifacts DIR`; `--save` re-renders every sweep pdf and the three perf sweep figures into DIR): major-only tick labels and at least two in-view major ticks on every axes of every figure, the `legend` / `x_color` / `y_color` / `title` / `title_color` keys through `PLOT_CONFIGS` and `ANNOTATE_SWEEP_PLOT`, an unknown key fails at import |
| A33 | `a33_gamma_vline.py` | gamma sweep annotation: no strategy overrides the spec vlines, one unlabelled line per in-view reference and no text, `generic_runner` free of `thm1_gamma_min`, `sweep_record` stores the spec constant |
| A34 | `a34_eps_tol.py` | query sweep tolerance: `eps_tol` 2**-8 from `SimulationConfig` / `OpticalDeviceConfig` reaches the query runners' IV and optical budgets, the param sweeps keep `EPS_TOL` 2**-5; the query runners rescale the raw declared gamma by 1/sigma-hat^2 so the fitted PI radius is sqrt(gamma) (1.0 sim, 0.5 optical) |
| A35 | `a35_sim_dim.py` | simulation `treatment_dim`: optional yaml key with the `DatasetDefaults` fallback (the SEM's 32), validated as a positive int, required by `SimulationOrchestrator`, reaches the SEM draw and the DA, and every script states it |
| A36 | `a36_omega_knob.py` | optical omega knob: `RandomPermutation` honours p (bit-identical at the default p = 1, identity rows kept under p < 1), p = s, grid `linspace(0.2, 0.99)`, tr(S)/k rising and pinned on the shipped-chain fixture |
| A37 | `a37_major_ticks.py` | at least two labelled major ticks on every axis: the (1, 2, 5) log fallback with plain labels on the n sweep (200, 500, 1000), every sweep param on both datasets, the query sweep and the three perf sweep figures, on synthetic inputs; x is the exact grid plus a 2 % margin (no top-tail clip) with the in-grid reference lines drawn inside the frame, r = 1 on the gamma sweep pkls too (`--artifacts DIR`, read only) |
| A38 | `a38_recalibrate.py` | the recalibrated post-DA budget (SS4.2): every ball at sigma-hat sqrt(gamma~), gamma~ = gamma((1 - t) + t/rho); Cor. 3 closed form at gamma~, the registry hands rho to the standalone DA+ balls only, the predict-time knob, DA+PI(False)/DA+PI(True) == sqrt(rho_hat) on one sim experiment, ordering under the configured toggles, intersection == max/min of its branches, the floor on the same ball, oracle units, `TOGGLE_KEYS`; the old token is grepped away and the recipes that still carry it are reported |
| A39 | `a39_recalibrate_sweep.py` | the `recalibrate` sweep (`param: [recalibrate]`): t on linspace(0, 1) re-solved through the predict-time knob on constant data, x = gamma~/gamma = (1 - t) + t/rho_hat from 1 down to 1/rho, PI unchanged and DA+PI narrower at every t per query, intersection == max/min of its branches, width(t)/width(0) == sqrt((1 - t) + t/rho_hat) with pad off, the t = 0 / t = 1 bounds equal to the `recalibrate: false` / `true` toggle runs on the same draw, `recalibrate_axis.pkl`, the render's sort, and the yaml plumbing (toggle and sweep together); `--optical` adds one optical experiment |
| A40 | `a40_eps_star.py` | robustness eps* per dataset: `ROBUSTNESS_EPSILON_TRUE[experiment_name]` reaches the epsilon runner of both orchestrators and the tuned DA lands on it (sim 3.0, optical 5.0), the sim constant clears the post-DA radius sqrt(gamma*) and sigma sqrt(gamma*), on two sim experiments (5 steps) DA+PI coverage dips below 1 at the smallest r, is 1 at r = 1 and never falls under 0.7; the optical epsilon runner's DA is config.yaml's chain plus the `ROBUSTNESS_AUGMENTATION` component (gaussian-noise) while every other strategy's runner, the query runner and the budget DA keep config.yaml's chain, the same dip (lowest of DA+PI and DA+PI+IV) on three optical experiments, and the optical constant stays 5.0 at under 10 std of h*; `--seed` |
| A54 | `a54_cigarettes_runner.py` | the cigarette orchestrator: validation, the declared budget at the solver, the gamma sweep (the INV / baseline width ratio), the recalibrated omega axis, target routing and both replicate schemes, each on the block as given AND on its other instrument declaration (`--config PATH` reads the cigarettes block of another yaml, default `config.yaml`; `--seed`) |
| A56 | `a56_iv_registry.py` | the IV registry, the instrument sets and the display tables; `COPSENS_METHODS` is pinned at ten (the nine plus the do-MNIST-only `ERM+INV`). Its (D) leg runs the digest of the other experiments' query panels and gamma sweep steps, so do-MNIST work runs it with `--skip-digest` |
| A64 | `a64_perf_aggregate.py` | round 12: the `(T)` instrument mode (grammar, display tables, dispatch on fitted attributes, `(T)` equal to `(T,Z)` under an empty set), the perf sweeps (`src/experiments/perf.py`: the cumulation table on `_prepare` call counts, D(eps) on a synthetic record with planted failures, `clip_y` on rendered artists, the simulation perf path end to end at 4 steps on all three perf metrics), and `python -m src.aggregate` on a synthetic tree with blank cells and on the shipped artifacts (`--shipped DIR`); `--only LEG` runs one leg, `--skip-digest` drops (D) |
| A74 | `a74_domnist_sem.py` | the do-MNIST SEM, split and DA: the analytic target, the `h_erm` cell table, Bayes and ERM accuracy on obs and do, the tint round trip, the colour ops exogenous and support-preserving, `amounts`, `identity_params`, `mix_in` (counts, masks, seeds, `frac 0` untouched), the A/B/C partition and `split_key`, `pop_seed != seed + 1`, `centre_error_report`, `bisect_gamma` on a stub; `--nets` adds the `init_seed` coupling (GPU) |
| A75 | `a75_copsens.py` | the copsens backend on a synthetic factor SEM: gamma 0 is the ERM, ordered bounds in [0, 1], gaussian and probit closed forms, the latent model, INV and IV nested in PI and monotone in gamma, `iv_budget`, JAX gradients against finite differences, `n_jobs 4` bit-identical to `n_jobs 1`, `recalibrate`/`rho` reaching the radius, `mean_match` inert |
| A76 | `a76_domnist_parity.py` | two-stage parity against the reference checkout: `--source DIR` under its env dumps the nets, the B arrays and the bounds, `--check DIR` refits the ported classes on the frozen arrays and compares bounds to 5e-4 and intermediates to 1e-6 |
| A77 | `a77_domnist_block.py` | the do-MNIST block, the recipes and the registry, static: validation of every key (`net` a registered net, `inv_recenter` off/on/inv, `erm_inv_tau` positive, `ERM+INV` on do-MNIST only), config.yaml, F1, F2 and the smoke `BLOCK` agree on the pinned values (`inv_recenter: inv`, PI's gamma, `target_coverage` 0.995, `erm_inv_tau` 4e-4), `ERM+INV` listed commented out, `copsens` builds its ten, the selection's shared gamma is PI's alone, the `experiment.query.tint` spec parses and rejects bad digits, ranges and keys, a tint spec on another orchestrator raises, only the `gamma` sweep is wired |
| A78 | `a78_domnist_erm_inv.py` | the ERM+INV centre: PI+INV built on `nets["X"]` / `nets["GX"]` / `nets["INV"]` under off / on / inv, no PI+INV nesting under inv, the augmented Lagrangian's windows split each epoch evenly (20 updates per epoch at 60k and 1.2M), the class defaults equal `DoMNISTConfig`'s AL constants, `train_inv` only when PI+INV runs under inv or `ERM+INV` is listed, the `domnist-pool` head (global average pooling, a 64-unit dense layer) and the flat default; `--nets` (GPU, three 60k draws, about three minutes) adds: the AL fit deterministic at a fixed seed, ERM+INV more invariant than the ERM on the B pairs, the ERM and DA+ERM nets and every B array bit-identical with and without the ERM+INV fit, the `inv_fits` hook training the run's net (same `state_sha1`), `split_key` the partition's sha1 on every path, the flat nets' sha1s pinned at 421549d, and under `net: domnist-pool` all three nets pooled and deterministic with the B arrays unchanged |
| A79 | `a79_domnist_tint.py` | the tint sweep, its figure and the results table (two MNIST loads, no nets, about a minute): the exemplar indices, tints and image sha1s pinned from `develop`; `tinted` (the grid round trip, one ink per row, the exemplar's image); `run_tint_sweep` on a stub runner (MNIST-test image, ATE constant, the result layout, the status split, the density pkl); `tint_stack` on a synthetic tree written out of order (0 at the top, a missing digit absent, blue image left and red right, the failed-tint crosses); `domnist_table` (one row per interval method with its band line, the fit charges, latency = fit + mean solve, n_jobs and the BLAS cap in the comments, `pdflatex` compiles it or `[SKIP]`); the table's gamma checked against the `gamma_selection.json` beside it (same gamma, split and nets); `aggregate.main` on a tree holding only `do_mnist/query/` |
| A70 | `a70_perf_feasibility.py` | the `feasibility` perf metric (the share of the seed_var backends returning a usable bound, per method, step and query): the config (perf only, `normalize` rejected), the formula against `solver_stability`'s failure count on a64's synthetic record, the simulation perf path end to end (PI+INV 0 where refuted and 1 at r = 1, the figure on `CLAMP_YLIM`, seed_var unchanged beside it), no failure marker on the stability figure, the feasible rate stacked under the stability in the aggregate's `epsilon_seed_var.pdf` (no `epsilon_feasibility.pdf`), a completeness grep for the removed marker code, and do-MNIST perf still skipping (source text only); `--only LEG` runs one leg |
| A71 | `a71_empty_cells.py` | empty cells drop out of the coverage rows (D1c): `evaluate_queries` gives coverage NaN iff every query's interval has a NaN bound (exactly when `interval_width` is NaN), under INFEASIBLE and FAILURE alike, the status split untouched; a partly empty cell keeps `coverage`'s number and a point estimate is never NaN; `coverage` itself unchanged; the NaN rules agree on random masks; `_draw_series` draws an all-empty step as a gap, not 0; `--only LEG` runs one leg |
| A72 | `a72_im_ci.py` | Imbens-Manski CIs on the sweep bounds (`im-ci` in `defaults:`): the n and m grids read `sweep_samples` (bit-identical at 16), the critical value's two limits and its equation on synthetic inputs, the unit bootstrap (a row; on the fold sweep a base row with its m augmented copies, carried together, one draw for both groups) and its determinism across pool sizes, `im-ci: 0` bit-identical to the recorded `finite` digest, under 95 the pad at eps* alone (point models, intersection branches and replicates; `results_raw` finite's on the unpadded methods, up to 2 `EPS_TOL` narrower on the padded ones), perf and query padding as before, the config validation, every yaml-reading gate pinning `im-ci` off, and the do-MNIST block forced off (static), perf and query never reaching the helper (sim only), the sim n and m sweeps at sweep_samples 8 on config.yaml's six methods (CI never under raw, the CI excess falling as a power of n, the n readings exact, PI on m within tolerance of the iid rule's, pinned readings), the pkls and the render reading the CI, the Slurm launcher's dry run; `--only LEG`, `--quick` |
| A73 | `a73_tolerance_audit.py` | not a pass/fail gate: the `EPS_TOL` audit under the IM-CI on the n and m sweeps of every dataset block of `--config`, V0 raw / V1 IM-CI with the pre-retirement pad eps* + `EPS_TOL` (an existing run's tree, `--v1`, or run here) against V2 (the pad by eps* alone, now the shipped im-ci sweep; builders patched so the replicates match) and V3 (`EPS_TOL = 0` on the budget modules), all by attribute patches; prints the retire / keep verdict on the pad tolerance (V2 coverage >= min(0.95, V1 - 0.01) and narrower, every DA+ cell) and on the constraint tolerance (V3 raw-infeasible share within 5 points of V1), writes `audit.json` and one V0-V3 figure per (dataset, param); `--reuse` |

A leg that cannot run on the inputs it was given (no shipped tree, no pkls of the
kind it reads, a block with no pair to compare, `--skip-digest`) prints `[SKIP]` and
is counted in the summary line, e.g. `A64 PASS (1 SKIPPED: (vii) no shipped tree)`,
never folded into a silent PASS. Where a synthetic input can stand in, the leg runs
on it instead: `a32` (b) to (f) render a fixture tree written under `TMPROOT` when
`--artifacts` carries none of the pkls they read, and name the tree they used.

`smoke_do_mnist.py` is an end-to-end do-MNIST query sweep at reduced scale (60k
draw, 6k PI rows, 200 queries) on the copsens block (perf logs a warning and skips
there); `--full` runs it at the block's own numbers, `--methods` restricts the list.
A `config.yaml` in the cwd overrides its `BLOCK`, including `experiment`, so a tint
sweep is `experiment: {query: {tint: {digit: [7], sweep_samples: 8}}}` there. Run it
from a scratch cwd: `save` writes `./artifacts`.

`sbatch_sweeps.py` is not a gate: it fans a config's sweeps out over a Slurm job
array, one task per (dataset, sweep param) and one per perf block (do-MNIST blocks
never), each running `src/main.py` unchanged in its own directory with `artifacts/`
and `data/` linked to the repo's, so every task writes into the one shared tree.
Every directive is a flag or an `SBATCH_SWEEPS_*` variable and none has a site
default; `--dry-run` writes the task yamls and `run.sbatch` without submitting. On a
laptop `uv run python src/main.py` stays the whole story, only slower. EXAMPLE, on
PACE ICE: `--partition coc-cpu --account oms-csp --qos coc-ice --env-setup "module
load uv" --out ~/scratch/runs/<name>` (the `--help` epilogue has the full line).

## Mean matching (Lem. 2)

Every PI program solves on the mean-matched slice `E_n[h(X)] = E_n[Y]`, the
covariance ball of Lem. 2. `mean_match: false` in `config.yaml`'s `defaults:`
restores the pre-2026-09 uncentred, intercept-free ball; `a10 --mean-match false`
reproduces the pre-change digest byte-for-byte, which is what pins the old path.

The linear backend enforces the slice EXACTLY (it eliminates the intercept by
centring). The copsens backend (do-MNIST) has no mean-matched slice: its ball lives on
the latent factor scores of a prefit net, so `mean_match` is accepted for the uniform
signature and logged as inert (A75 pins that).

The do-MNIST estimand is analytic since the copsens port (`sem.ate_of`,
`sem.h_star`); the fitted target net and its gates (A4 to A9, A21, A24, A27) went with
the `partial_r2_net` backend they tested.

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

## do-MNIST gamma selection

`select_domnist_gamma.py` picks the block's `gamma` by POPULATION coverage of h_* on
split C: it trains the block's replicate exactly as the run does (nets on A, the
mix-in, the PI balls on the B rows), draws `n_select` rows from C at `seed + 1`,
bisects log gamma per method (PI and DA+PI) to the smallest value whose coverage
reaches `target_coverage`, writes `artifacts/do_mnist/select/gamma_selection.json`
and `coverage.pdf`, and prints one line per method. ONE gamma serves every method:
PI's value (`shared_gamma`, `calibrated_on: PI`) is pasted into
`config.yaml::do_mnist.gamma` by hand, and the run errors without one. DA+PI's own
bisection is a diagnostic; its split-C coverage and width at the shared gamma are
recorded beside it. The shipped value, 0.059352 at `target_coverage` 0.995 on 5,000
rows, gives PI 0.9952 and DA+PI 0.9818 on C (DA+PI alone would need 0.1596). `--target`
overrides the block's target for a one-off. The CopSens gamma is a LATENT budget the Lemma-2 oracle gamma* does not
measure, which is why the sweeps read the declared value.

```bash
D=~/scratch/domnist_runs/port_full && mkdir -p $D && cp config.yaml $D/   # do_mnist block active
(cd $D && PYTHONPATH=$REPO uv run --project $REPO python $REPO/scripts/select_domnist_gamma.py)
```

The selection is conditional on every block key it records (`mix_in`, `n_pi`,
`n_components`, `augmentation`, ...). Change one and it is stale. The results table
(`python -m src.aggregate`) states the calibration only when the
`gamma_selection.json` in the run's `artifacts/do_mnist/select/` selected the run's
gamma on the run's split with the run's `net`; copy in the matching one, or it says the gamma is not from
the selection beside it.

## do-MNIST ERM+INV diagnostics

`diagnose_domnist_erm_inv.py` is not a gate: it checks the ERM+INV centre (PI+INV's
centre under `inv_recenter: inv`) on the full draw, on the GPU, without bound solves.
Through `draw_replicate`'s `inv_fits` hook it fits one net per point of tau in
{1e-3, 4e-4, 1e-4} x epochs in {1, 2} at the exact point where the run fits its own,
and records per point the invariance error on the B pairs (`E_inv_B`, with
`constraint_met` at <= 1.5 tau, which says nothing about fit quality), f-accuracy and
RMSE to h_* on 1,000 split-C rows, the PI+INV constraint value at the centre and its
floor against eps^2, the fit time, the AL trace and the net's `state_sha1`. The
`recommended` entry is the configured net; the run's `run.json` `erm_inv_state_sha1`
must equal its hash. Writes `artifacts/do_mnist/select/erm_inv_diagnostics.json` and
`erm_inv_trace.pdf`; about 6 minutes on an L40S.

```bash
D=~/scratch/domnist_runs/erm_inv_diag && mkdir -p $D && cp recipes/doMnistFigF1.yaml $D/config.yaml
(cd $D && PYTHONPATH=$REPO uv run --frozen --project $REPO python $REPO/scripts/diagnose_domnist_erm_inv.py)
```

On the shipped settings (tau 4e-4, mu0 1e-4, growth 2, 2 epochs) the configured net
reaches E_inv_B = 0.98 tau and f-accuracy 0.825 on C (the ERM: 0.980) in an 18 s fit.

## Two-stage gates

```bash
SRC=../doMNIST/symmetry4CausalBoundsDoMNIST
$SRC/.venv/bin/python scripts/a1_a2_sem.py    --source /tmp/a2.npz
$SRC/.venv/bin/python scripts/a3_da_parity.py --source /tmp/a3.npz

uv run python scripts/a1_a2_sem.py    --check /tmp/a2.npz
uv run python scripts/a3_da_parity.py --check /tmp/a3.npz

# do-MNIST parity against the reference checkout (its env for the source stage, CPU)
SRC=~/close-this/copsens/symmetry4CausalBounds
(cd $SRC && CUDA_VISIBLE_DEVICES= PYTHONPATH=$SRC ~/scratch/uv_envs/symmetry4CausalBounds/bin/python \
    $REPO/scripts/a76_domnist_parity.py --source ~/scratch/a76/fix)
uv run python scripts/a76_domnist_parity.py --check ~/scratch/a76/fix
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
