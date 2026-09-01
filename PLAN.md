# Plot rescale/clip plan

Rescale and clip sweep/scatter/perf. No re-encoding.

All numbers below measured on the checked-in `artifacts/{simulation,optical_device}/
sweep/gamma_results.pkl` (32 steps, 6 PI methods), **post-`bootstrap()`** — i.e. on
the series the code actually plots — as a 5-seed median.

## Evidence (TARGET)

**The pathology is cross-method dynamic range, not CI blowout.** On
`simulation/gamma_approx_error` the six mean lines span 0.00098-0.0295 (30x), and
four of them (`DA+PI`, `PI`, `PI&DA+PI`, most of `PI_INV`) sit inside
0.00099-0.00117 — 0.7% of the axis. Optical is worse: 0.000109-0.0536 (493x), four
of six inside 0.1% of the axis. Excluding the CI band does not fix this.

- **Measure post-`bootstrap()`, not off the pkl.** `bootstrap()`'s inner
  `.mean(axis=1)` (`data_operations.py:83`) is not NaN-aware, so on NaN-heavy series
  the plotted mean is not the nanmean of the stored data: sim approx_error carries
  115/256 NaN on `PI_INV` and 22/256 on `DA+PI_IV`/`PI&DA+PI_IV`, and `PI_INV` comes
  out with 18 of 32 steps finite (26 of 32 on optical). Measured raw, that sweep reads
  133x; measured as plotted, 30x. Every number in this plan is the latter.
  `bootstrap()` is also unseeded (`np.random.randint`, `data_operations.py:72`), so
  anything derived from it jitters on exactly those series: 5-seed `p98` spread is
  1.93x on sim approx_error and 1.21x on optical approx_error, 1.00x on the other six
  measured series. Consequence for the design: the automatic defaults are stable
  everywhere except the two NaN-heavy approx_error sweeps, which is an argument for
  pinning those two in `PLOT_CONFIGS`, and for the diagnostic to report the multi-seed
  spread rather than one number. The NaN handling in `bootstrap()` is a real bug;
  out of scope here (see Non-goals).

- `create_sweep_plot` computes `global_min/global_max` (`plotting.py:96-98,117-121`)
  and never uses them: the y-limit block is commented out (`plotting.py:174-182`).
  Only `xlim` is set, exactly `[min, max]` (`plotting.py:171-172`).
- Excluding the CI band from the limits is still correct, and is not everywhere
  cosmetic: the upper limit drops 2.2-2.4x on `optical/gamma_approx_error` (~0.13 ->
  ~0.054; the spread is the unseeded bootstrap), and the lower limit moves off 0 on
  both datasets' approx_error (0 -> 1.1e-04 / 9.9e-04) which is what makes a log/asinh
  floor representable at all. On the other six measured series it moves the limits
  <3%. Note the band is a bootstrap CI **of the mean** (`data_operations.py:83`), not
  the raw 2.5/97.5 spread — which is why it is tight nearly everywhere.
- Scales are pure policy, never checked against the data:
  `plt.xscale/yscale` take the caller's literal (`plotting.py:184-185`), from
  `PARAM_SPECS.xscale` (`configs.py:167,193`) and `METRIC_SPECS.yscale`
  (`configs.py:221-230`).
- `asinh` metrics run at matplotlib's default `linear_width=1.0` while
  `approx_error` lives at ~1e-3: that axis is asinh in name and linear in fact.
  `PANEL_CONFIGS` fixes exactly this per row, at `ylim / linear_width = 40`
  (`constants.py:125,126,132`; the fourth asinh row, `constants.py:131`, uses 667).
- `create_scatter_plot` sets **no** limits and **no** scales (`plotting.py:591-605`):
  auto-scale includes the SE crosshairs (`plotting.py:561-565`).
- `create_perf_plot` top panel is the 100%-stacked reliability bars
  (`plotting.py:664-685`); the bottom panel is wall clock + seed SD
  (`plotting.py:687-724`) and exists only when `overlay_metrics` is non-empty
  (`plotting.py:656-662`). Cost y-scale is hardcoded (`plotting.py:701`).
- Plot ids already exist: `fname=f'{param}_{metric}'` (`base.py:533`),
  `fname=f'scatter_{param}_{metric_x}_vs_{metric_y}'` (`base.py:627`), `perf`
  (default arg `plotting.py:634`, called `base.py:601`). All three plot functions
  already receive `experiment` too.
- Precedent for per-experiment per-plot knobs: `PANEL_CONFIGS`
  (`constants.py:123-136`), read as `.get(experiment, {}).get(key, {})`
  (`plotting.py:380,465`); its scale kwargs are resolved at `plotting.py:468-473`.
- `_reject_unknown` exists (`configs.py:263-268`) and `utils -> configs` is acyclic,
  but **not free**: importing `configs` costs 1.42s and pulls cvxpy + sklearn, and
  `utils/__init__.py:6` does `from .constants import *`, so reusing it at module scope
  in `constants.py` would make the leaf constants module drag the whole methods stack.
  Inlined instead — see design 2.
- `constants.py:4` imports `Dict, Literal, List` only; `Any` must be added.
- Dead: `PANEL_PREDICTION_LIMITS`, `PANEL_Y_SCALES` (`constants.py:104-115`) have no
  consumers. Flagged, not touched.

## Scope

`create_sweep_plot`, `create_scatter_plot`, `create_perf_plot` only.
Untouched: `create_query_sweep_plot`, `create_panel_plot`, `create_digit_sweep_plot`,
colours, alphas, line styles, bubble sizing, annotations, draw order, method
registry, runners, saved pkls.

Files: `src/experiments/utils/{plotting.py,constants.py}`, two lines in
`data_operations.py` (the `bootstrap()` fix — see below), plus **2 kwargs** in
`base.py` (scatter axis scales) and one new `scripts/` diagnostic. `configs.py` and
`config.yaml` unchanged.

## Design

1. **`PLOT_CONFIGS` in `constants.py`**, beside `PANEL_CONFIGS`, same shape and same
   lookup idiom. Keyed `experiment -> plot id -> knobs`. The plot id **is** the
   `fname` `base.py` already builds; no new id scheme. Literal below.

2. **`_plot_config(experiment, fname)`** in `plotting.py`: merge
   `PLOT_CONFIGS['*'][id]` under `PLOT_CONFIGS[experiment][id]`, key by key; `{}`
   when absent. Called once at the top of each of the three plot functions — they
   already have both arguments, so `base.py` needs no plumbing. Key validation is a
   3-line loop **inlined in `constants.py`**, at import time, raising `ValueError`
   (not `assert` — stripped under `python -O`). Deliberately *not* `configs.py`'s
   `_reject_unknown`: duplicating a 4-line guard in a leaf module beats inverting the
   layering and adding 1.42s + cvxpy/sklearn to every `utils` import.
   `xlim`/`xscale` are rejected for the `perf` id: categorical x. Unknown *ids* stay
   inert; the diagnostic lists ids no run consumed.

3. **`_finite(*arrays) -> NDArray`**: concatenate, drop non-finite. Empty is a valid
   answer and every caller handles it (no limits, no promotion, scale unchanged).

4. **`_limits(values) -> (lo, hi)`**, from the **point series only** — CI bands and SE
   crosshairs are excluded and clip against the frame, still drawn.
   - `hi = p98(pooled means)`, `lo = min(pooled means)`, then expanded so **every
     series keeps its median inside the frame**. That is the guarantee: a series may
     lose points, it can never go blank.
   - Top-tail only, deliberately. Measured effect of `hi = p98`: sim approx_error
     0.0295 -> 0.0098 (3.03x), optical approx_error 0.0536 -> 0.0109 (4.91x), sim
     worst_error 73.3 -> 60.0 (1.22x), optical worst_error 1.02x, width 1.01-1.09x;
     1-3 points leave the frame per sweep. The median guarantee never binds on this
     data (all 24 series medians already inside).
   - Top-tail only, settled with the reviewer: a symmetric `p2` floor crops `PI_INV`
     alone, 4 of its finite steps on **each** of optical `worst_error` and `width`
     (~12% of the range) — and `PI_INV` is the tightest method there (0.92-11.6 vs
     3.5-17.0 for the rest), so a `p2` floor deletes exactly what those figures exist
     to show. These metrics are errors/widths: small is the signal, large is the
     runaway. `coverage` is bounded so `p98` = max there anyway (measured: limits
     unchanged, both datasets).

5. **`_resolve_scale(scale, values, limits, cfg) -> (scale, scale_kwargs)`**.
   Precedence: `cfg['xscale'|'yscale']` > caller's spec literal > auto-promote; then
   one safety clamp over whatever won.

   - *Auto-promote*, **metric axes only** (sweep y, both scatter axes): if the
     resolved scale is `'linear'`, all pooled means `> 0`, and
     `p99.5/p0.5 >= 100` (2 decades) -> `'log'`.
     Sweep **x is excluded**: `PARAM_SPECS.xscale` defaults to `'log'`
     (`configs.py:167`) and only `trS` opts out to `'linear'` (`configs.py:193`) with
     a comment explaining x is the *measured* expansion — the rule cannot tell an
     author's choice from a dataclass default, and trS is the only axis it could ever
     fire on. Zero upside, real downside.
   - *Measured: this promotion fires on nothing today.* `p99.5/p0.5` post-bootstrap:
     sim `width` 20x, optical `width` 6.3x, optical `worst_error` 13.5x; `coverage`
     has a zero mean so the all-positive gate blocks it (its `max/median` is 1.0 — the
     earlier "boundedness" argument was wrong, the conclusion was not). sim
     `worst_error` 126x and optical `approx_error` 250x clear the bar but are already
     `asinh` by spec. It is a guard for future metrics, not the fix. Keeping 2 decades
     and keeping it inert is deliberate: tuning to 1.5 to catch one figure (sim
     `width`, which spans 0.64-15.7 with no method pinned flat and is legible linear)
     is fitting a threshold to a single observation — the kind of heuristic this plan
     exists to avoid.
   - *The live auto rule is `linear_width`*, and it is where requirement 3 actually
     acts: for `asinh`/`symlog`, `linear_width`/`linthresh` = `cfg` value, else
     **`hi / 40`** from step 4, matching `PANEL_CONFIGS`' own `ylim/linear_width = 40`
     (`constants.py:125,126,132`). Gives 0.000245 / 1.50 on sim approx_error /
     worst_error and 0.000273 / 0.415 on optical — every one lands mid-range, so the
     linear knee sits below the bulk and the top compresses.
     **Rejected: `median(|v|)`** (my previous draft). Measured, it equals the series
     *minimum* on both approx_error sweeps because 4 of 6 methods are pinned flat at
     the floor, which turns the axis into a plain log; and on sim worst_error it gives
     9.4 on data 0.54-73, i.e. near-linear. Wrong in both directions.
   - *Safety clamp*, applied to config- and spec-supplied scales too, so a hand-set
     scale can never crash or blank an axis: `'log'` with any non-positive value ->
     `'asinh'` + `logger.warning`; `'log'` with no positive value -> `'linear'`;
     `linear_width`/`linthresh` always `> 0` (fallback 1.0 / 0.1, preserving
     `plotting.py:471-472`). `cfg` sets `linear_width` but the resolved scale is
     neither asinh nor symlog -> `logger.warning`, ignored.
   - The kwargs half of this is factored out of `plotting.py:468-473` and the panel
     block calls it. Behaviour-identical (same defaults), so `create_panel_plot` stays
     observably untouched.

6. **`_pad(axis, lo, hi, frac=0.05)`**: pad in the axis's own transformed space
   (`axis.get_transform()`, then `.inverted()`). One expression covers
   linear/log/asinh/symlog — no per-scale branch. Non-finite transform or `lo == hi`
   -> return unpadded, let matplotlib expand.

7. **`_rescale(ax, cfg, x_points, y_points, xscale, yscale, pad_x=True)`**, in order:
   (1) `_limits` per axis, (2) apply `cfg['xlim']/['ylim']` element-wise (`None` =
   keep the computed edge, so `ylim: (0, None)` works; `lo >= hi` raises),
   (3) `_resolve_scale` from those limits, (4) `set_xscale/set_yscale(**kwargs)`,
   (5) `_pad`, (6) `set_xlim/set_ylim`.

8. **`create_sweep_plot`**: the scalar `global_min`/`global_max` accumulators
   (`plotting.py:96-98,117-121`) are replaced by an `all_means` list, appended at
   `plotting.py:111` where `mean_error` is already computed — the only new state.
   Then replace `plotting.py:170-185` with
   `_rescale(plt.gca(), cfg, x_values, all_means, xscale, yscale, pad_x=False)`.
   **x-limits stay exactly `[min(x), max(x)]`** unless `cfg['xlim']` overrides —
   padding them would visibly widen every existing sweep (gamma: 0.25-4.0 -> 0.218-4.6
   at 5% on a log axis). Pad y only. `hide_legend` already exists
   (`plotting.py:188`): OR it with `cfg.get('legend', True) is False`; a string value
   passes through to the existing `legend_loc` (`plotting.py:58`). `fname` fallback
   moves to the top of the function so the config id and the filename cannot diverge
   (`plotting.py:214` currently derives it inside `if savefig:`).
   The `vlines` block (`plotting.py:151-162`) **moves below** `_rescale` and gates on
   the resolved xlim instead of the raw data range, so a narrowing `cfg['xlim']`
   cannot leave the Thm. 1 label (`plotting.py:159-162`) anchored off-frame. Safe to
   move: the `axvline` carries an explicit `zorder=0` (`plotting.py:157`), so draw
   order is unchanged. Nothing else moves; pyplot state-machine style kept.

9. **`create_scatter_plot`**: add `xscale`/`yscale` params (default `'linear'`), call
   `_rescale` with the stacked `mx`/`my` means before the legend
   (`plotting.py:598-603`), padding both axes (matplotlib's own 5% margin applies
   today, so this is not a visible change on its own). Gate the existing
   `if handles:` legend on `cfg`; same `fname` fix at `plotting.py:609`.
   `base.py:620-628` passes `xscale=spec_x.yscale, yscale=spec_y.yscale` — the only
   edit outside `utils/`, and it is DRY: `METRIC_SPECS` stays the single home of
   per-metric scale policy, as the sweep already treats it.

10. **`create_perf_plot`: `bars` toggle.** `bars = cfg.get('bars', True)`, shipped
    `False` via the `'*'` block, **no per-experiment override**.
    - `bars=True` -> today's figure, unchanged.
    - `bars=False`, overlays present -> **single panel**: today's bottom panel
      promoted. `fig, ax_cost = plt.subplots()` with `ax = None`; skip
      `plotting.py:665-685` (rates, bars, reliability legend), set
      `axis_for_labels = ax_cost` at `plotting.py:688`, skip
      `fig.align_ylabels` (`plotting.py:732-733`). No new encoding.
      **`seed_var` is kept, not dropped.** The promoted panel carries both series
      exactly as the bottom panel does today: wall clock on `ax_cost` (log, left) and
      the across-seed width SD on its twin (`plotting.py:705-716`, right), each in its
      own units. Two lines whenever `n_experiments >= 2`; `base.py:568-573` still
      drops `seed_var` at `n_experiments == 1` because the SD is undefined there, not
      because the series is unwanted. Nothing about the `seed_var` path is removed, so
      bumping `n_experiments` later is a config edit and nothing more.
    - `bars=False`, no overlays -> nothing to draw: `logger.warning`, return.
      `perf.pkl` is still written (`base.py:600`).
    - **How the override applies:** perf does *not* call `_rescale` — its x is
      categorical and it has no x point series. It resolves the scale on the
      wall-clock series only and applies the result to `ax_cost`; x is untouched.
      `plotting.py:701`'s `'log'` becomes that call's default (kept local — see
      design 2, no `METRIC_SPECS` import) and `cfg['yscale']`/`cfg['ylim']` override
      it. It also does **not** use `_limits`' top-tail clip: on a per-method cost
      line the slowest method *is* the result, not a runaway to be cropped, so the
      limits are plain min/max. Default behaviour unchanged.
    - **Consequence, stated plainly:** `do_mnist` is the only live block
      (`config.yaml:48-95`; simulation `16-46` and optical `97-127` are commented out)
      and runs `n_experiments: 1` (`config.yaml:50`), so `seed_var` is dropped
      (`base.py:568-573`) leaving `overlay = ['wall_clock']`. With `bars=False` the
      shipped perf figure is **one 6-point wall-clock line over a categorical x** —
      thin, and shipped that way on purpose (Decisions 1): raising `n_experiments` is
      a compute decision, not a plotting one. It becomes the intended two-line figure
      the moment `n_experiments >= 2`, with no code change. Reliability survives in
      `perf.pkl`, in the >99%-infeasible warning (`base.py:592-598`), and as the
      `coverage` sweep metric / scatter axis (`configs.py:227`). Overrule with
      `{'do_mnist': {'perf': {'bars': True}}}`.
    - **Override mapping, honestly:** `xlim`/`xscale` are meaningless on a categorical
      x and are rejected at import. `yscale`/`ylim` bind to the **cost (left,
      wall-clock) axis** only; the bar axis is `0-100` by construction
      (`plotting.py:678`) and the SD twin keeps its own units. `legend: False` hides
      both legends.

## Bootstrap fix (prerequisite)

`bootstrap()` (`data_operations.py:42-93`) carries two bugs. Both are fixed here, not
filed: the auto-derived limits and `linear_width` are computed *from* the bootstrapped
means, so an unseeded, NaN-dropping bootstrap makes this change's own defaults
unreproducible. It is a prerequisite, not a drive-by.

1. **Not NaN-aware.** `_bootstrap_sample(...).mean(axis=1)` (`:83`) returns NaN if a
   resample draws even one NaN replicate. `PI_INV` carries 115/256 NaN on the
   simulation gamma sweep, so it survives at only 18 of 32 steps — a method that HAS a
   finite mean at a step is drawn as a gap. -> `np.nanmean`, with the all-NaN-slice
   warning suppressed (matching `plotting.py:109-111`, which already does this).
   An all-NaN step stays NaN; that one is honest.
2. **Unseeded.** `np.random.randint` (`:72`) draws from the global stream, so the band
   — and everything this plan derives from it — moves run to run: 5-seed `p98` spread
   1.93x on sim approx_error, 1.21x on optical. -> a local
   `np.random.default_rng(seed)`, `seed` a keyword defaulting to module-level
   `BOOTSTRAP_SEED = 0`.

**Consequences, stated up front:**
- Plotted means change on NaN-carrying series only (`PI_INV`, `DA+PI_IV`,
  `PI&DA+PI_IV`); NaN-free series are bit-identical apart from the reseed.
- Gaps close: methods reappear at steps where they were silently dropped. This moves
  the pooled ranges, so **every measured number in this plan is a pre-fix baseline**
  and the `PLOT_CONFIGS` values must be chosen from a post-fix
  `scripts/diag_plot_scaling.py` run. The plan's entries are illustrative regardless.
- A local RNG no longer advances the global stream. `_run_sweeps` plots between
  sweeps (`base.py:527-544`), so in a multi-param run the later sweeps' data shifts
  once. A one-time re-baseline, in exchange for plots that stop moving.
- `bootstrap()` has one other caller (`create_sweep_plot`, `plotting.py:87`), so the
  blast radius is the sweep band and nothing else.

## Config surface

`src/experiments/utils/constants.py`, immediately after `PANEL_CONFIGS`:

```python
# Per-plot rescale/clip overrides. Keyed experiment -> plot id, where the id is the
# `fname` base.py already builds: '<param>_<metric>' (base.py:533),
# 'scatter_<param>_<mx>_vs_<my>' (base.py:627), 'perf' (base.py:601). The id has no
# '_sweep' suffix -- plotting.py:215 appends that when writing the pdf, so
# 'gamma_approx_error' -> gamma_approx_error_sweep.pdf.
# '*' applies to every experiment; a named entry wins key by key.
#   xlim/ylim      (lo, hi); None on either end = keep the auto edge
#   xscale/yscale  'linear' | 'log' | 'symlog' | 'asinh'. Two keys, not
#                  PANEL_CONFIGS' single 'scale': these plots scale both axes.
#   linear_width   asinh only; linthresh symlog only. Default: upper limit / 40.
#   legend         False = hide, True = auto, str = matplotlib loc (legend_loc)
#   bars           perf only; False retires the stacked reliability bars
_PLOT_KEYS: set = {'xlim', 'ylim', 'xscale', 'yscale',
                   'linear_width', 'linthresh', 'legend'}
_PLOT_KEYS_PERF: set = (_PLOT_KEYS - {'xlim', 'xscale'}) | {'bars'}  # categorical x

PLOT_CONFIGS: Dict[str, Dict[str, Dict[str, Any]]] = {
    '*': {
        'perf': {'bars': False},                     # stacked bars retired
    },
    'simulation': {
        # pinned: bootstrap() is unseeded and this series is NaN-heavy, so the
        # automatic p98 swings 1.93x across seeds
        'gamma_approx_error': {'ylim': (None, 0.012)},
        'trS_worst_error':    {'linear_width': 0.25},
        'scatter_n_width_vs_coverage': {'ylim': (0.0, 1.02), 'legend': 'lower right'},
    },
}

for _exp, _plots in PLOT_CONFIGS.items():      # typos must not be silent no-ops
    for _id, _cfg in _plots.items():           # inlined, not configs._reject_unknown
        _bad = set(_cfg) - (_PLOT_KEYS_PERF if _id == 'perf' else _PLOT_KEYS)
        if _bad:
            raise ValueError(f'PLOT_CONFIGS[{_exp!r}][{_id!r}]: unknown key(s) {sorted(_bad)}.')
```

Needs `Any` added to `constants.py:4` (`from typing import Dict, Literal, List`).

The per-experiment entries are illustrative; real values get set after the
diagnostic prints what the automatic rules pick on the real pkls.

## Defaults & backward compat

Someone who sets nothing gets, on the measured sweeps:

1. **Sweep y-limits clipped at `p98` of the pooled means.** sim `gamma_approx_error`
   top 0.0295 -> 0.0098 (3.03x), optical 0.0536 -> 0.0109 (4.91x), sim `worst_error`
   1.22x, the rest 1.01-1.09x. 1-3 points leave the frame; no series goes blank
   (median guarantee).
2. **CI bands and SE crosshairs excluded from the limits** and clipped by the frame.
   Material on optical `approx_error` (2.2-2.4x on the upper limit) and on the lower
   limit of both `approx_error` sweeps (0 -> ~1e-4); <3% elsewhere.
3. **`asinh` axes get `linear_width = hi/40`** instead of 1.0: 0.000245 (sim
   approx_error), 1.50 (sim worst_error), 0.000273 / 0.415 (optical). Real shape
   change on the four asinh sweeps — this is the rescale being asked for, and the
   change the user is most likely to want tuned per plot. Note `cfg['ylim']` feeds
   `hi` (step 7 order), so pinning a limit also moves `linear_width`: desirable
   coupling, but pin `linear_width` alongside if that is not wanted.
4. Scatter limits from the mean points; scatter axes now honour `METRIC_SPECS.yscale`
   (previously always linear). The padding itself is a no-op only on linear axes,
   where it matches matplotlib's own 5% margin.
5. Perf: stacked bars off; the shipped `do_mnist` figure becomes a single wall-clock
   line (see 10 above). It also loses `sharex` and `height_ratios=[2.2, 1]`
   (`plotting.py:657-658`), so the cost panel goes from a 1/3.2 slice to full height.
   Its ylabel stays at `FS_TICK` (`plotting.py:701`) — undersized next to `FS_LABEL`
   now that it is the only panel; left alone deliberately, since font sizing is
   re-encoding, not rescaling.
6. **Sweep bands and means change on NaN-carrying series** — the `bootstrap()` fix.
   Gaps in `PI_INV`/`DA+PI_IV`/`PI&DA+PI_IV` close, and the band stops moving between
   runs. NaN-free series are unaffected apart from the reseed.
7. **Unchanged:** sweep x-limits (still exactly `[min, max]`), every keyword argument's
   meaning and default, figure size, colours, styles, file paths, saved pkls. No new
   dependency, `config.yaml` untouched. Auto scale *promotion* fires on nothing in the
   current repo.

## Verification

No `tests/` directory here and none is proposed. Matching `scripts/`
(`diag_trs_sweep.py`: build inputs, print a table, print verdict lines):

**`scripts/diag_plot_scaling.py`** (new).
1. Loads the real `artifacts/*/sweep/*_results.pkl` that already exist and prints, per
   (experiment, param, metric): pooled mean range, resolved scale, `linear_width`,
   resolved limits, points clipped per series. Measured **post-`bootstrap()` over
   several seeds**, reporting the spread — a single number is misleading on the
   NaN-heavy series (1.93x seed-to-seed on sim approx_error), and the spread is what
   says which plots need pinning in `PLOT_CONFIGS`. This is how the round-1 numbers
   were caught; it costs nothing and is how the config entries get chosen.
2. Synthesises the cases the artifacts do not cover — all-NaN series, flat series,
   zero-crossing, 6-decade positive, a single step, a single method, a CI 100x the
   mean spread — and renders all three plot functions into `artifacts/_diag/`.
3. Verdicts: no exception **and** no `logger.error` — the three plot functions swallow
   exceptions (`plotting.py:218,612,740`), so this needs an explicit
   `logger.add(sink, level='ERROR')` sink or the diagnostic passes green on a total
   failure; every series keeps its median inside the final limits; a resolved `log`
   scale never coexists with a non-positive point value; resolved `lo < hi`.
4. Lists `PLOT_CONFIGS` ids no plot consumed (stale/typo'd ids).

**End-to-end**: uncomment `simulation` in `config.yaml` with
`sweep: {param: [gamma, epsilon, trS], metric: [approx_error, worst_error]}`,
`scatter`, `perf`, `n_experiments: 8`; `.venv/bin/python -m src.main`; eyeball
`artifacts/simulation/{sweep,scatter,perf}/`. Revert `config.yaml` before committing.
Repeat with the checked-in `do_mnist` block for the `bars=False` path.

## Rejected from SOURCE

- `experiment.plot:` block in root `config.yaml` + `resolve_plot_scales` in
  `configs.py` — the user explicitly rejected plot fine-tuning in the root config.
- `fig, ax = plt.subplots()` rewrite of `create_sweep_plot`/`create_scatter_plot` —
  churn; `plt.gca()` reaches the same axes.
- Two-pass CI/line z-ordering rewrite — re-encoding, not rescaling.
- Scatter re-encoding: rank bubbles -> uniform points, endpoint labels removed,
  connecting lines added, title reworded — all out of scope.
- `_masked_ci` — matplotlib already drops non-positive vertices on a log axis.
- Deleted docstrings/comments across both functions — they carry the *why*.
- `tests/` directory — no such thing in this repo.
- Silent `log -> linear` fallback — blanks the series when data straddles zero;
  `asinh` + a warning keeps it visible.

## Non-goals

Adaptive CI alpha, CI epsilon-clipping, blanket auto-scale that overrides author
intent, per-experiment branches in code, label repulsion, colour/alpha/linewidth
changes, rc overhaul, a second YAML file, touching query/panel/digit plots, deleting
the dead `PANEL_PREDICTION_LIMITS`/`PANEL_Y_SCALES`.

## As implemented — post-fix measurements

`scripts/diag_plot_scaling.py`, on the checked-in gamma sweeps, after the
`bootstrap()` fix. These supersede the pre-fix numbers above:

| experiment | metric | pooled mean range | scale | knee | upper limit | clipped | seed jitter |
|---|---|---|---|---|---|---|---|
| simulation | approx_error | 0.00098 .. 0.1407 | asinh | 0.00195 | 0.0778 (1.81x) | 3 | 1.00x |
| simulation | worst_error | 0.542 .. 73.3 | asinh | 1.50 | 59.96 (1.22x) | 4 | 1.00x |
| simulation | width | 0.444 .. 15.71 | linear | — | 14.43 (1.09x) | 4 | 1.00x |
| simulation | coverage | 0 .. 0.9994 | linear | — | unchanged | 4 | 1.00x |
| optical | approx_error | 0.000108 .. 0.1036 | asinh | 0.000611 | 0.0244 (4.24x) | 4 | 1.04x |
| optical | worst_error | 0.570 .. 16.96 | asinh | 0.415 | 16.6 (1.02x) | 4 | 1.00x |
| optical | width | 0.464 .. 6.871 | linear | — | 6.806 (1.01x) | 4 | 1.00x |

Two things the fix changed, both as predicted:
- **Seed jitter collapsed from 1.93x to 1.00-1.04x.** The `nanmean` fix is what did
  it: a NaN replicate no longer nulls the whole bootstrap draw. Nothing needs pinning
  in `PLOT_CONFIGS` for stability, so the shipped map carries only `perf.bars`.
- **The gaps closed.** simulation `approx_error` reads 0.00098..0.1407 again (the
  30x pre-fix figure was `DA+PI_IV`'s top steps being silently dropped), so the
  cross-method spread is back to ~143x — which is what the `linear_width` and `p98`
  rules are there to handle.

Scale promotion fired on nothing, as designed. No series blanked, no `log` resolved
over non-positive data, no logged plot failure, no stale config id.

## Follow-ups (not in this change)

- Top-clipping assumes small-is-signal. A future larger-is-better metric needs an
  explicit `ylim`; `coverage` is safe only because it saturates near 1.
- The `FS_TICK` cost ylabel and the lost `height_ratios` on a bars-off perf figure
  (Defaults 5) are cosmetic and deliberately untouched.

## Decisions (recommended; overrule freely)

1. **`bars: False` globally, and do not couple it to `n_experiments`.** The live
   `do_mnist` perf figure becomes one 6-point wall-clock line. `n_experiments` is a
   compute decision (`config.yaml:51`: 1.2M SEM draws plus net training) and doubling
   it to improve a figure is a large hidden cost inside a plotting change. Ship the
   thin figure — it is the honest, visible argument for `n_experiments >= 2` the next
   time there is compute, and it becomes the two-line wall-clock/seed-SD figure the
   moment that happens, with no code change. Reliability survives in `perf.pkl`
   (`base.py:600`), the >99%-infeasible warning (`base.py:592-598`), and `coverage`
   (`configs.py:227`).
2. **Top-tail-only clipping** (`lo = min`, `hi = p98`) — settled, see design 4.
3. **Scale promotion stays at 2 decades and stays inert** — see design 5.
