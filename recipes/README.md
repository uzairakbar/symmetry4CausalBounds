# Recipes

One recipe per figure. Each file carries a `defaults:` block and one dataset block
per dataset the figure covers (`simulation`, `optical_device`, `cigarettes`), and
each block carries exactly ONE experiment type: `query:`, `sweep:` or `perf:`. Run
one by copying its blocks into `config.yaml` and calling `python src/main.py`; gates
read them in place.

```bash
python - <<'PY'
import yaml
config = yaml.safe_load(open("recipes/validityFig9.yaml"))
print(list(config))
PY
```

The cigarette experiment is split by TARGET as well as by type: the query figures
run the restricted-2sls target (`target: iv`), every sweep and both perf metrics run
the plasmode (`target: plasmode`), whose confounding is of known strength so
coverage means something.

## Output paths

A run writes to `artifacts/<dataset>/<experiment type>/`, with nothing from the
recipe's name in the path. The sweep recipes differ in `param` and the perf recipes
in `metric`, so their files never meet, but `simulationFig5` and `ivSimulationFig5`
are both simulation query runs and both write `artifacts/simulation/query/`: the
second run overwrites the first. Run one, move its `query/` aside, then run the
other. It is the only collision among the twelve, and `python -m src.aggregate`
does not see it (it reads `sweep/` and `perf/` only).

## Old -> new

The twelve files were restructured, not renamed. The old ones were single or double
block and bundled several experiment types per file; the new ones are multi-dataset
and split by type, so most old files fan out across several new ones and the two
cigarette files fan in.

| new recipe | blocks | type | where its content came from |
|---|---|---|---|
| `simulationFig5.yaml` | simulation | query | `simulation_fig5.yaml`, plus `treatment_dim: 32` and `iv: 4`; `DA+IV` / `DA+PI+IV` respelled `(T)` |
| `ivSimulationFig5.yaml` | simulation | query | `iv_fig13.yaml`, query half |
| `opticalDeviceFig6.yaml` | optical | query | `optical-device_fig6.yaml`; `DA+IV` / `DA+PI+IV` respelled `(T)` |
| `cigarettesFig7.yaml` | cigarettes | query | `neighbour-price_fig12.yaml`; the query half of `cigarettes_fig11.yaml` |
| `validityFig9.yaml` | sim, optical, cigarettes | sweep `gamma` | `validity_fig7.yaml`, whose one metric `coverage` broadens to four; the `gamma` step of `iv_fig13.yaml`, `cigarettes_fig11.yaml` and `cigarettes-plasmode_fig12b.yaml` |
| `sharpnessInformativenessFig10.yaml` | sim, optical, cigarettes | sweep `omega` | `sharpness_fig7.yaml` and `informativeness_fig7.yaml`, merged and inverted (omega is now the x-axis, not the metric); the `omega` step of the two cigarette files |
| `robustnessFig11.yaml` | sim, optical, cigarettes | sweep `epsilon` | `robustness_fig8.yaml`, whose three metrics broaden to four; the `epsilon` step of `iv_fig13.yaml` and the two cigarette files |
| `recalibrationFig12.yaml` | sim, optical, cigarettes | sweep `recalibrate` | the `recalibrate` step of `cigarettes_fig11.yaml` and `cigarettes-plasmode_fig12b.yaml`; new on sim and optical |
| `nEfficiencyFig13.yaml` | sim, optical, cigarettes | sweep `n` | `n-efficiency_fig9.yaml`, whose three metrics broaden to four; the `n` step of `iv_fig13.yaml` and the two cigarette files |
| `mEfficiencyFig14.yaml` | sim, optical, cigarettes | sweep `m` | `m-efficiency_fig10.yaml`, whose three metrics broaden to four; the `m` step of `iv_fig13.yaml` and the two cigarette files |
| `latencyFig15.yaml` | sim, optical, cigarettes | perf `wall_clock` | the `wall_clock` half of `cigarettes-plasmode_fig12b.yaml`'s perf block; new on sim and optical |
| `stabilityFig16.yaml` | sim, optical, cigarettes | perf `seed_var`, `feasibility` | the `seed_var` half of the same perf block; new on sim and optical. `feasibility` (the share of backends returning a usable bound) is new everywhere and reads the same backend runs |

Read the other way:

| old recipe | went to |
|---|---|
| `simulation_fig5.yaml` | `simulationFig5` |
| `optical-device_fig6.yaml` | `opticalDeviceFig6` |
| `sharpness_fig7.yaml` | `sharpnessInformativenessFig10` |
| `informativeness_fig7.yaml` | `sharpnessInformativenessFig10` |
| `validity_fig7.yaml` | `validityFig9` |
| `robustness_fig8.yaml` | `robustnessFig11` |
| `n-efficiency_fig9.yaml` | `nEfficiencyFig13` |
| `m-efficiency_fig10.yaml` | `mEfficiencyFig14` |
| `cigarettes_fig11.yaml` | `cigarettesFig7` (query) + the six sweep recipes, on the plasmode target |
| `neighbour-price_fig12.yaml` | `cigarettesFig7` |
| `cigarettes-plasmode_fig12b.yaml` | the six sweep recipes + `latencyFig15` + `stabilityFig16` |
| `iv_fig13.yaml` | `ivSimulationFig5` (query) + `validityFig9`, `robustnessFig11`, `sharpnessInformativenessFig10`, `nEfficiencyFig13`, `mEfficiencyFig14` (sweeps) |

Four things changed across the board, not per file:

- **`calibrate: false` -> `recalibrate: true`.** Eight of the twelve old recipes
  (`simulation_fig5`, `optical-device_fig6`, `sharpness_fig7`, `informativeness_fig7`,
  `validity_fig7`, `robustness_fig8`, `n-efficiency_fig9`, `m-efficiency_fig10`)
  carried the retired `calibrate` key, which `resolve_dataset_block` rejects outright
  (`Unknown key(s) ['calibrate']`), so those eight had not been loadable for a while;
  the four cigarette and IV files already carried `recalibrate: true`. Every new
  recipe carries `recalibrate: true` and `normalize: true`.
- **Metrics broadened.** Every sweep now reports
  `[coverage, approx_error, worst_error, width]`. The old files reported one
  (`validity_fig7`) or three (`robustness_fig8`, `n-efficiency_fig9`,
  `m-efficiency_fig10`).
- **`DA+IV` / `DA+PI+IV` respelled `(T)` on the two headline query recipes.** Same
  estimator, the mode written out; see the grammar below.
- **`im-ci: 95` on the six sweep recipes.** Every sweep metric now reads the 95%
  Imbens-Manski CI around each bound (a bootstrap of each method's own fitted rows)
  instead of the raw bound, and the raw record sits beside it in
  `{param}_results_raw.pkl`; `im-ci: 0` gives the raw numbers back. Under the CI the
  DA+ pad is eps* alone (the `EPS_TOL` it carried is the CI's job now; a73), so the
  raw record's DA+ widths are 2 `EPS_TOL` narrower than an `im-ci: 0` run's. The query
  and perf recipes carry no key: neither path reads it.

Two sweep params did not survive: the old `worst_error` and `width` x-axes of
`informativeness_fig7` / `sharpness_fig7` are now metrics on the `omega` x-axis. The
two perf recipes drop the old `repeats: 3`, which `configs.py` defaults to 3 anyway.

## Method spellings

A `DA+` IV method takes an instrument mode: bare (or `"DA+PI+IV(T,Z)"`, quoted in a
flow list, or the comma splits it) constrains BOTH instruments, each at its own
radius; `(Z)` the configured instrument alone, `(T)` the DA parameter alone. The
mode is legal on `IV_MODE_METHODS` only (`src/experiments/utils/constants.py`), so
the non-DA baselines are spelled `PI+IV` and `PI+INV+IV` with no suffix.
