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
| A25 | `a25_floor_guard.py` | closed-form floor vs cvxpy; no budget is ever raised (feasible and infeasible cells return the raw oracle budget, an infeasible one logs one INFO BELOW line; the sim n fixture's 10 % step of 640 rows is an infeasible cell); an omega knob under its T floor reads all-INFEASIBLE with a gap in the width line; a completeness grep finds nothing of the old guard in `src` or `scripts` |
| A28 | `a28_mean_match.py` | Lem. 2 slice: classes == an explicit intercept+equality reference, Cor. 3 closed form, floors, coverage |
| A29 | `a29_thm1_ceiling.py` | Thm. 1: eps+ tight at gamma_min, gamma_min == the fitted DA+PI transition on sim, optical reported as a reference |
| A30 | `a30_optical_truth.py` | optical estimand: h_* on Lem. 2's slice, gamma* over span(phi, 1), both epsilon budgets vs the measured defect (the sweeps' q0.95 and the query's RMS eps* each in its own norm), lazy data load |
| A31 | `a31_omega_axis.py` | omega sweep axis by the ball in force: tr(S)/k under `recalibrate: true`, rho tr(S)/k under `false`, the label (`OMEGA_XLABEL`) following the factor on the render and in `omega_axis.pkl`, the one sort in `create_sweep_plot` on the plotted x under both toggles (line, y and CI band reordered together; argsort pinned per dataset and toggle), factors and vline; the rename is complete (a case-insensitive grep for the retired sweep key over src, recipes, config.yaml and scripts hits only tr(S)/k's own names) and the retired key has no alias |
| A32 | `a32_figure_style.py` | figure style from the sweep and perf pkls (`--artifacts DIR`; `--save` re-renders every sweep pdf and the three perf sweep figures into DIR): major-only tick labels and at least two in-view major ticks on every axes of every figure, the `legend` / `x_color` / `y_color` / `title` / `title_color` keys through `PLOT_CONFIGS` and `ANNOTATE_SWEEP_PLOT`, an unknown key fails at import |
| A33 | `a33_gamma_vline.py` | gamma sweep annotation: no strategy overrides the spec vlines, one unlabelled line per in-view reference and no text, `generic_runner` free of `thm1_gamma_min`, `sweep_record` stores the spec constant |
| A35 | `a35_sim_dim.py` | simulation `treatment_dim`: optional yaml key with the `DatasetDefaults` fallback (the SEM's 32), validated as a positive int, required by `SimulationOrchestrator`, reaches the SEM draw and the DA, and every script states it |
| A36 | `a36_omega_knob.py` | optical omega knob: `RandomPermutation` honours p (bit-identical at the default p = 1, identity rows kept under p < 1), p = s, grid `linspace(0.2, 0.99)`, tr(S)/k rising and pinned on the shipped-chain fixture |
| A37 | `a37_major_ticks.py` | at least two labelled major ticks on every axis: the n sweep's majors at its log-spaced percentage range's ends (10, 100), the (1, 2, 5) log fallback with plain labels on a pre-ladder n grid (200, 500, 1000), every sweep param on both datasets, the query sweep and the three perf sweep figures, on synthetic inputs; x is the exact grid plus a 2 % margin (no top-tail clip) with the in-grid reference lines drawn inside the frame, r = 1 on the gamma sweep pkls too (`--artifacts DIR`, read only) |
| A38 | `a38_recalibrate.py` | the recalibrated post-DA budget (SS4.2): every ball at sigma-hat sqrt(gamma~), gamma~ = gamma((1 - t) + t/rho); Cor. 3 closed form at gamma~, the registry hands rho to the standalone DA+ balls only, the predict-time knob, DA+PI(False)/DA+PI(True) == sqrt(rho_hat) on one sim experiment, ordering under the configured toggles, intersection == max/min of its branches, the floor on the same ball, oracle units, `TOGGLE_KEYS`; the old token is grepped away and the recipes that still carry it are reported |
| A39 | `a39_recalibrate_sweep.py` | the `recalibrate` sweep (`param: [recalibrate]`): t on linspace(0, 1) re-solved through the predict-time knob on constant data, x = gamma~/gamma = (1 - t) + t/rho_hat from 1 down to 1/rho, PI unchanged and DA+PI narrower at every t per query, intersection == max/min of its branches, width(t)/width(0) == sqrt((1 - t) + t/rho_hat) with pad off, the t = 0 / t = 1 bounds equal to the `recalibrate: false` / `true` toggle runs on the same draw, `recalibrate_axis.pkl`, the render's sort, and the yaml plumbing (toggle and sweep together); `--optical` adds one optical experiment |
| A40 | `a40_eps_star.py` | robustness eps* as one rule: `ROBUSTNESS_EPSILON_RADII` (1.0) reaches the epsilon runner of both orchestrators, each SEM's target is that multiple of its oracle sigma sqrt(gamma*) and the tuned DA lands on it (sim exactly, optical at the recorded pooled read), the sim target is the post-DA ball's radius sigma sqrt(gamma*) = sqrt(bias^2) at the recorded R; on two experiments of the robustness recipe's simulation block (its d, iv and methods, 5 steps) the recorded feasible counts and lowest reading (no dip under the sweeps' App. D T radius r eps*), every DA+ line is 1 at r = 1 (the midpoint) and stays above 0.7 from its first all-feasible r; the optical epsilon runner's DA is config.yaml's chain plus the `ROBUSTNESS_AUGMENTATION` component (gaussian-noise) while every other strategy's runner, the query runner and the budget DA keep config.yaml's chain, every strategy runner's SEMs carry `epsilon_quantile` and budget at that eps* while the query runner keeps the RMS, the optical curve on three experiments (all pinned to device 8, `sweep_devices`) recovers and stays flat at the recorded 1.0 (the device caps the dip), the optical target is the recorded 1 R at under 10 std of h*, and the same q0.95 / RMS split on the simulation and cigarette runners (the cigarette budgets at that eps*, the tuned sim eps* the q0.95 of its W); `--seed` |
| A54 | `a54_cigarettes_runner.py` | the cigarette orchestrator: validation, the declared budget at the solver, the gamma sweep (the INV / baseline width ratio), the recalibrated omega axis, target routing and both replicate schemes, each on the block as given AND on its other instrument declaration (`--config PATH` reads the cigarettes block of another yaml, default `config.yaml`; `--seed`) |
| A56 | `a56_iv_registry.py` | the IV registry, the instrument sets and the display tables; `COPSENS_METHODS` is pinned at ten (the nine plus the do-MNIST-only `ERM+INV`). Its (D) leg runs the digest of the other experiments' query panels and gamma sweep steps, so do-MNIST work runs it with `--skip-digest` |
| A64 | `a64_perf_aggregate.py` | round 12: the `(T)` instrument mode (grammar, display tables, dispatch on fitted attributes, `(T)` equal to `(T,Z)` under an empty set), the perf sweeps (`src/experiments/perf.py`: the cumulation table on `_prepare` call counts, D(eps) on a synthetic record with planted failures, `clip_y` on rendered artists, the simulation perf path end to end at 4 steps on all three perf metrics), and `python -m src.aggregate` on a synthetic tree with blank cells and on the shipped artifacts (`--shipped DIR`); `--only LEG` runs one leg, `--skip-digest` drops (D) |
| A74 | `a74_domnist_sem.py` | the do-MNIST SEM, split and DA: the analytic target, the `h_erm` cell table, Bayes and ERM accuracy on obs and do, the tint round trip, the colour ops exogenous and support-preserving, `amounts`, `identity_params`, `mix_in` (counts, masks, seeds, `frac 0` untouched), the A/B/C partition and `split_key`, `pop_seed != seed + 1`, `centre_error_report`, `bisect_gamma` on a stub; `--nets` adds the `init_seed` coupling (GPU) |
| A75 | `a75_copsens.py` | the copsens backend on a synthetic factor SEM: gamma 0 is the ERM, ordered bounds in [0, 1], gaussian and probit closed forms, the latent model, INV and IV nested in PI and monotone in gamma, `iv_budget`, JAX gradients against finite differences, `n_jobs 4` bit-identical to `n_jobs 1`, `recalibrate`/`rho` reaching the radius, `mean_match` inert, `sigma_model` (None bit-identical, a given net sets sigma2_ to its mean mu(1-mu) on the observed rows, the intersections' branches None) |
| A76 | `a76_domnist_parity.py` | two-stage parity against the reference checkout: `--source DIR` under its env dumps the nets, the B arrays and the bounds, `--check DIR` refits the ported classes on the frozen arrays and compares bounds to 5e-4 and intermediates to 1e-6 |
| A77 | `a77_domnist_block.py` | the do-MNIST block, the recipes and the registry, static: validation of every key (`net` a registered net, `inv_recenter` off/on/inv, `erm_inv_tau` positive, `ERM+INV` on do-MNIST only), config.yaml, F1, F2 and the smoke `BLOCK` agree on the pinned values (`inv_recenter: inv`, PI's gamma, `target_coverage` 0.995, `erm_inv_tau` 4e-4), `ERM+INV` listed commented out, `copsens` builds its ten, the selection's shared gamma is PI's alone, the `experiment.query.tint` spec parses and rejects bad digits, ranges and keys, a tint spec on another orchestrator raises, only the `gamma` sweep is wired |
| A78 | `a78_domnist_erm_inv.py` | the ERM+INV centre: PI+INV built on `nets["X"]` / `nets["GX"]` / `nets["INV"]` under off / on / inv with sigma-hat from `nets["X"]` under off and inv (on keeps the DA+ERM's, Prop. 3), no PI+INV nesting under inv, the augmented Lagrangian's windows split each epoch evenly (20 updates per epoch at 60k and 1.2M), the class defaults equal `DoMNISTConfig`'s AL constants, `train_inv` only when PI+INV runs under inv or `ERM+INV` is listed, the `domnist-pool` head (global average pooling, a 64-unit dense layer) and the flat default, the prescreen's `rho` the nets' squared-error ratio (`net_noise`, equal to `_net_rho`) with its seeded paired bootstrap `rho_band` containing it, the pixel-logistic `rho_linear` kept and `contracts_calibrated` / `da_inert` on the nets' rho; `--nets` (GPU, three 60k draws, about three minutes) adds: the AL fit deterministic at a fixed seed, ERM+INV more invariant than the ERM on the B pairs, the ERM and DA+ERM nets and every B array bit-identical with and without the ERM+INV fit, the `inv_fits` hook training the run's net (same `state_sha1`), `split_key` the partition's sha1 on every path, PI+INV's sigma2_ equal to PI's on the replicate under off and inv (DA+PI's under on), the replicate's `rho` equal to `_net_rho` on its B rows inside its `rho_band`, with `rho_linear` kept, the flat nets' sha1s pinned at 421549d, and under `net: domnist-pool` all three nets pooled and deterministic with the B arrays unchanged |
| A79 | `a79_domnist_tint.py` | the tint sweep, its figure and the results table (two MNIST loads, no nets, about a minute): the exemplar indices, tints and image sha1s pinned from `develop`; `tinted` (the grid round trip, one ink per row, the exemplar's image); `run_tint_sweep` on a stub runner (MNIST-test image, ATE constant, the result layout, the status split, the density pkl); `tint_stack` on a synthetic tree written out of order (0 at the top, a missing digit absent, blue image left and red right, the failed-tint crosses); `domnist_table` (one row per interval method with its band line, the fit charges, latency = fit + mean solve, Omega-hat on the nets' `rho` with a comment naming it, its `rho_band` and `rho_linear`, n_jobs and the BLAS cap in the comments, `pdflatex` compiles it or `[SKIP]`); the table's gamma checked against the `gamma_selection.json` beside it (same gamma, split and nets); `aggregate.main` on a tree holding only `do_mnist/query/` |
| A70 | `a70_perf_feasibility.py` | the `feasibility` perf metric (the share of the seed_var backends returning a usable bound, per method, step and query): the config (perf only, `normalize` rejected), the formula against `solver_stability`'s failure count on a64's synthetic record, the simulation perf path end to end (a method 0 where refuted, the recorded ones under r = 1 (PI+INV and the T family at r = 2^-4 and 2^-2 on the four-octave grid); PI+INV 1 at r = 1, the figure on `CLAMP_YLIM`, seed_var unchanged beside it), no failure marker on the stability figure, the feasible rate stacked under the stability in the aggregate's `epsilon_seed_var.pdf` (no `epsilon_feasibility.pdf`), a completeness grep for the removed marker code, and do-MNIST perf still skipping (source text only); `--only LEG` runs one leg |
| A71 | `a71_empty_cells.py` | empty cells drop out of the coverage rows (D1c): `evaluate_queries` gives coverage NaN iff every query's interval has a NaN bound (exactly when `interval_width` is NaN), under INFEASIBLE and FAILURE alike, the status split untouched; a partly empty cell keeps `coverage`'s number and a point estimate is never NaN; `coverage` itself unchanged; the NaN rules agree on random masks; `_draw_series` draws an all-empty step as a gap, not 0; `--only LEG` runs one leg |
| A80 | `a80_method_labels.py` | method labels, colours and line styles by attributes, no data: `parse_method`'s attributes on every spelling (an intersection's PI baseline on Z in every IV mode, the invalid attribute sets raising, unknown names through the config's did-you-mean), every label row at both `has_z` against TeX composed from the blocks (the two null-Z cells without a legal method raise in the config), the hues on literals with the six hex values equal to seaborn deep, the alphas, the line styles (every point estimate dashed, every interval solid, at both `has_z`) and legend slots (frozen at `has_z` true, member 0 without Z), uniqueness within a block (every recipe, the digest blocks, the default lists, exhaustively) and across blocks (one label, one render) with a mutation of each, the aggregate merge (the render-differing `z_counterpart` pairs exactly the seven cross-dataset pairs, every multi-column recipe paired, validityFig9 folding to 6 entries), do-MNIST's PI+INV row under every `inv_recenter`, the null-Z column titles and "cigarette demand" in merged grids, completeness greps over `src/` (no static table, no typed method name in a displayed string, `has_z=` on every drawing call); `--only LEG`, `--skip-run` |
| A81 | `a81_domnist_seeds.py` | the do-MNIST population across seeds (no MNIST, no nets, seconds): `domnist_seed_table` on synthetic records (one row per interval method in ALL_METHODS order, ATE and the point estimators not rows and their RMSE mean +- SE in a comment, each cell the mean +- std(ddof 1)/sqrt(n finite), coverage highest and width and worst error lowest best, `\bm` the best and `\mathit` the second best, a tie at the printed precision sharing its mark, an all-NaN method `--` and unranked, a partly NaN one over its finite seeds with the count named, the seeds, `target_coverage` and the fixed seeds in the comments, one seed without SE, `pdflatex` compiles it or `[SKIP]`); `evaluate_population(save_outcomes=False)` writing nothing (`save` patched); `_run_seeds` on stub runners (seeds `seed + j`, one fresh runner per further seed at `n_experiments: 1`, each released before the next is built, `seeds.json` holding only the population metrics, `seeds_table.tex` its table); `aggregate.main` re-rendering `do_mnist_seeds_table.tex` from `seeds.json` alone; static: `seed + j` as on the gamma sweep path, `gc.collect()` and `torch.cuda.empty_cache()` in the release |
| A82 | `a82_optical_devices.py` | one optical device per sweep experiment (setup only but one 2-step sweep, a few minutes on 16 cores): the default order is `dataset_index` (8) first then the rest ascending with the odd recordings 6 (brfactor 0, no confounding) and 11 (pure confounding) excluded, i.e. (8, 0, 1, 2, 3, 4, 5, 7, 9, 10), the indices pinned to their files; a ten-experiment gamma sweep builds its SEMs on those ten devices once each (read off a tagging SEM, pools pairwise distinct) with `polys[j]` of `sems[j]`'s own degree (both 1 and 2 present) and `sems[j].f` fitting experiment j's features, eleven experiments raise; the perf-shaped epsilon runner (`n_experiments: 1`) and the query runner sit on device 8 (perf never run); `sweep_devices = (8, 8)` reproduces the single-device gamma sweep bit for bit against the base-class hooks (wall clock aside); simulation and cigarettes keep `polys[j] is poly`, sim draws a fresh SEM per experiment (`W_XY` pairwise distinct), cigarettes (plasmode) keeps one panel, redraws the outcome and splits at `seed + j`; on all ten devices gamma* is finite, every strategy runner's SEMs carry the q0.95, the robustness DA's pooled eps* lands on `ROBUSTNESS_EPSILON_RADII` x sigma sqrt(gamma*) of that device (10%) or clamps at zero strength where the device's floor already exceeds it (a NOTE), and the n ladder's 10% cell keeps n_train > k; per-device degree, gamma*, k, floor, target, strength and eps* reported |
| A83 | `a83_bootstrap_band.py` | the sweep band is the 95% percentile-bootstrap CI of the mean over experiments (no experiment run, seconds): `create_sweep_plot(savefig=False)` on a synthetic (n_steps, 10) record draws a band (read off the axes' fill) equal to `BAND_PERCENTILES` of `bootstrap(y)`, not the raw percentiles and narrower than them at every step; the aggregate `sweep_grid` on a temp tree draws the same band in its coverage cell; static: `create_sweep_plot` defaults to `bootstrapped=True`, `_run_sweeps` never turns it off, every other call site is a known non-sweep one (`_run_perf`, the cigarette width ratio), `create_sweep_plot` and `sweep_grid` draw through `sweep_series` and it calls `bootstrap`; `bootstrap` is deterministic under `BOOTSTRAP_SEED` (another seed moves it), returns `BOOTSTRAP_RESAMPLES` resample means per row inside the row's range, and `BAND_PERCENTILES` is (2.5, 97.5); on the normalised rows (`create_sweep_plot(normalize=True)` and the aggregate's width and worst-error cells) each experiment is divided by its own baseline before the bootstrap and the ratios aggregate geometrically: the band is exp of `BAND_PERCENTILES` of `bootstrap` of the per-experiment log-ratios, the line exp of the mean of those resample means (within 1e-2 relative of the exact geometric mean), the baseline reads exactly 1.0 with no band, and the arithmetic mean of the ratios (the previous rule) and the old ratio of means (bootstrap, then divide by the baseline's per-step mean) are computed as mutations and must fail the comparison; what stays arithmetic: `GEOMETRIC_SWEEP_SUFFIXES` is width and worst_error, the aggregate's coverage row and an unnormalised width figure draw nanmean and percentiles of `bootstrap(y)`, not the geometric rule, and `log_ratios` reads a non-positive or non-finite ratio as NaN |
| A85 | `a85_sum_zero_translation.py` | the optical sweeps run `rotation > hflip > vflip > translate` at rotation / flip probability 0.25 and the query keeps `rotation > hflip > vflip > random-permutation` at 0.5 (three two-step sweeps, a few minutes on 8 cores): translate's basis B is 9 x 8, orthonormal, sum-zero and spans 1-perp; config.yaml and every optical sweep recipe name the chain with `augmentation_p: 0.25`, which the resolved block carries to `OpticalDeviceDA(p=...)`, and the query recipe names the permutation chain with no `augmentation_p` (the DA's `P` 0.5; random-permutation keeps its own 1.0, which `p` never reaches); translate sits outside "all" and carries no invariance error (the robustness sweep's gaussian-noise the only knob); `augmentation_p` outside (0, 1), a bool or a string is a config error and the key is unknown off the optical block; on device 8 every augmented row keeps its pixel sum, T has 11 columns (12 for the query chain), each flip fires at its p (within 0.03), and GX less `TRANSLATION_SCALE` std(X) T[:, 3:] B' is a pixel permutation of each row (also under the omega knob's p); E[GX \| t] = B t by least squares over 40 draws (slope B' and intercept the pool mean within 0.05); the gamma, omega and epsilon sweeps (two experiments, two steps) build their DAs at p 0.25 and hash bit for bit to `DIGEST`, a self-regression digest recorded off leg (v)'s own computation on `DIGEST_NODE` (`--dump FILE` for the arrays); `--only LEG` |
| A86 | `a86_gamma_n.py` | the finite-sample budget gamma_n(d; g) = F^-1_{chi2_d(n g)}(1 - a) / n (no experiment run, seconds): `finite_sample_budget` equals scipy's chi2 / ncx2 quantile over n on a grid, a = 0 returns g exactly (raw), g <= 0 is the central chi-square, increasing in d and g and as a falls, decreasing in n towards g, and the reference values at a = alpha/3 (sim ball 1.1194, optical 0.8428, cigarettes 0.2993 at n 2205, the mean row 0.0062 at n 1843); (ii) off the cvx parameters of PI, PI+INV, PI+IV, DA+PI and PI&DA+PI's branches on a synthetic fixture: the ball s sqrt(gamma_n(k; gamma~)) at alpha/3 with k = d + 1 and n_eff the fit's n_obs (the original samples on an m = 4 tiling), the mean row s sqrt((1 + gamma~) gamma_n(1; 0)), raw the population ball bit for bit with no mean row, no unit cap on either cigarette target (`iv`, `plasmode`), and the IV rows (the declared gamma_z on a non-DA row, gamma~_z(eps) = (eps / s~ + sqrt(gamma_z / rho))^2 on a DA row, the joint row alone on a T + Z fit (`IV_LAYOUT`), each at sqrt(s^2 (1 + gamma~) gamma_n(d; g / (1 + gamma~))), raw s sqrt(g), a DA row moving with epsilon); (iii) the mean row binds at the design mean (ybar +- its radius), padded contains raw, the closed form equals cvxpy, and the IV rows see delta (PI+IV on an off-mean Z keeps its data residual under the threshold, and breaks it with delta dropped); (iv) validity on a small iv = 1 simulation (d 8, n 400, gamma = gamma* 0.25): the PI+IV rows at h* each hold on >= 1 - alpha/3 - 3 SE of 400 draws and jointly on >= 0.95 - 3 SE, and over 250 solved draws the simultaneous coverage of h* at 20 queries is >= 0.95 - 3 SE for PI, PI+IV, DA+PI, DA+PI+IV(T,Z) (0.90 for PI&DA+PI+IV(T,Z)); (v) every +IV interval inside its non-IV counterpart, raw and padded; (vi) `gamma_n` 95 -> alpha 0.05, absent / 0 / false raw (one INFO when absent), `true` / 100 / -5 / "95" raise, config.yaml and every non-do-MNIST recipe at `RECIPE_ALPHA` (0.05, `gamma_n: 95`) but opticalDeviceFig6 (`RAW_RECIPES`, raw), the retired CI key raises, do-MNIST ignores the key, `EPS_TOL` 2^-8 the one tolerance (no per-dataset tolerance or pad field), the optical `_epsilon_budget(None)` exactly `EPS_TOL` over the RMS and over the q0.95 reading, a DA+ pad equal to its epsilon, the optical and cigarette query budgets bit-identical to c29af19, the simulation query at its oracle RMS eps* + 2^-8, the query runners' raw gamma rescaled by 1/sigma-hat^2 (PI radius sqrt(gamma)), do-MNIST's copsens T-as-IV cone at the block epsilon; (vii) one record: `_run_sweeps` writes `<param>_values` / `_results` / `_statuses` / `_axis` only, the aggregate reads `_results`, no CI level on the runner, no pad tolerance on `BoundedSA`, and a grep finds no symbol of the retired bootstrap CI around the sweep bounds in src/, scripts/, recipes/, config.yaml or the READMEs (a77's unknown-key check aside); (viii) a report, never a FAIL: cigarette plasmode coverage over 100 resimulations at the n sweep's 10 % and 100 % cells (220 and 2205 fit rows) and the m base (220 rows, m = 4), per method the mean per-query and the simultaneous coverage against 0.95 - 3 SE (0.90 for an intersection) and the per-query minimum, a WARN under it; (ix) a whole-word grep for every retired symbol of the old IV budgets, the pad tolerance, the bootstrap CI and the unit cap, `py_compile` on every script, `ruff check --select F`, every script with a command line exits 0 on `--help`; (x) the n / m grids and the sim strategies' row counts (205 .. 2048 at 8 steps; m 8 steps at 128 rows), no 16 / 17 literal in either `grid_fn`; (xi) `sbatch_sweeps.py --dry-run`: one resolving yaml per (dataset, param) and perf block, no shared (dataset, subdir, stem), only the asked directives, the task dirs' links, do_mnist never fanned out, no new machine-specific value outside the launcher's example; (xii) on every non-do-MNIST recipe block and config.yaml, every built model's epsilon, ball budget, IV leak and IV radius > 0 (the simulation's gamma_z 2^-8 under `iv: 1`, the cigarettes' 0.0177; optical has no Z); (xiii) `oracle_t_leak`: a non-bool, a `true` off the optical block or beside a sweep raise; off, no leak is measured and the T row is the row formula; on, opticalDeviceFig6 (device 8, raw) reads the oracle T leak 0.014733, the T row's radius exactly leak + 2^-8 (0.018639), and the panel's mean widths DA+PI 1.935, DA+PI+IV(T) 0.971, PI+INV 0.547 (1e-3), h* covered; `--only LEG` |

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

The linear backend enforces the slice EXACTLY on the raw program (it eliminates the
intercept by centring). Under the finite-sample pads (`gamma_n`, `defaults:` of
every shipped recipe) the slice gets its own padded row: the mean may sit off
`E_n[Y]` by delta, inside the ERM ball and within `s sqrt((1 + gamma~) gamma_n(1;
0))`, and the IV rows see delta too (the instruments are not centred). A86
(ii)/(iii) pin it. The copsens backend (do-MNIST) has no mean-matched slice: its
ball lives on the latent factor scores of a prefit net, so `mean_match` is accepted
for the uniform signature and logged as inert (A75 pins that).

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
  The RMS `epsilon_star` is right for the constraint and no bound at all for the
  pad -- measured here, RMS 0.212 against a sup of 1.230. So the sweeps read eps*
  once as `oracle.epsilon_star_q95`, the q0.95 of `|W|` over 8 concatenated seeded
  draws, and use that reading + `EPS_TOL` (2^-8) as both the constraint budget and
  the pad; the query panels keep the RMS + `EPS_TOL` for both. A quantile and not
  the sup because under a DA with a Gaussian component the sup is INFINITE, so no
  finite epsilon makes `h_*` eps-approximately T-invariant in the SS2.4 sense:
  what the pad buys is Thm. 3.A with "a.s." weakened to "with probability >= the
  quantile". `OpticalDeviceConfig.epsilon = None` takes the measured budget. Whether the published `2**-2` clears the L2 budget
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
recorded beside it. The shipped value, 0.062082 at `target_coverage` 0.995 on 5,000
rows with `net: domnist-pool`, gives PI 0.9950 and DA+PI 0.9904 on C (DA+PI alone
would need 0.1219). The flat head's value was 0.059352. `--target`
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

On the shipped settings (tau 4e-4, mu0 1e-4, growth 2, 2 epochs, `net: domnist-pool`)
the configured net reaches E_inv_B = 0.95 tau, f-accuracy 0.959 and RMSE 0.173 on C
(the ERM: 0.982 and 0.124) in a 20 s fit. The flat head (`domnist-fast`) reached
0.98 tau, 0.825 and 0.219 in 18 s.

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
The degrees-of-freedom sigma-hat (SSR / (n - k), `residual_variance`) moved every
bound digest again (checked 2026-09-28): only the statuses and the PI+INV
coverage, approximation error and status counts kept theirs, and serial still
equals parallel.
