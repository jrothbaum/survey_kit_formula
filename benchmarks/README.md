# Benchmarks

`polars_formula` vs [formulaic](https://github.com/matthewwardrop/formulaic)
vs R's `model.matrix()`. 10 formulas × 20K/100K/400K rows, all 3 tools;
`survey_kit_shaped` and `wide_columns` also run at 800K/1.6M/2.4M rows for
`ours`/`r` only (`formulaic` already fails at 400K on similarly-shaped
formulas, so it's skipped at larger sizes).

## Running

```bash
uv run python benchmarks/generate_data.py   # writes benchmarks/data/n*.csv
uv run python benchmarks/benchmark.py       # writes benchmarks/results.csv
```

Requires `uv`, R (`Rscript` on PATH), Linux `/proc`. Each cell runs as its
own subprocess under a live 8GB RSS cap (`benchmark.py`'s watchdog); a cell
that exceeds it gets killed and marked `exceeded_memory_cap`, not crashed.
`build_seconds` times only the model-matrix construction (after the CSV
read); `run_ours.py` times `get_model_frame` (the Polars-native result),
not `get_model_matrix` (`= get_model_frame(...).to_numpy()`, a separate,
column-count-proportional cost).

## Formulas

| name | formula | cols |
|---|---|---|
| numeric | `~ x1` | 2 |
| single_cat_low | `~ A` | 3 |
| add_numeric_cat | `~ x1 + A` | 4 |
| interact_numeric_cat | `~ x1:A` | 4 |
| two_cat_low | `~ A + B` | 5 |
| four_cat_interact_low | `~ A:B:C:D` | 82 |
| numeric_cat_cross | `~ x1*x2*A*B` | 36 |
| high_card_interact | `~ Ahi:Bhi` (60×45 levels) | 2701 |
| survey_kit_shaped | `~ 1+x1+x2+x2*x1*Ahi` | 240 |
| wide_columns | `~ Vhi:Ahi` (150×60 levels) | 9001 |

`A`/`B`/`C`/`D`: 3 levels. `Ahi`/`Bhi`/`Vhi`: 60/45/150 levels.

## Results (build_seconds / peak RSS MB)

n=20,000:

| formula | ours | formulaic | r |
|---|---|---|---|
| numeric | 0.027 / 110 | 0.002 / 164 | 0.001 / 68 |
| single_cat_low | 0.025 / 101 | 0.005 / 155 | 0.003 / 68 |
| add_numeric_cat | 0.008 / 116 | 0.004 / 152 | 0.003 / 68 |
| interact_numeric_cat | 0.029 / 116 | 0.004 / 152 | 0.003 / 69 |
| two_cat_low | 0.029 / 99 | 0.006 / 152 | 0.003 / 68 |
| four_cat_interact_low | 0.026 / 99 | 0.031 / 170 | 0.009 / 68 |
| numeric_cat_cross | 0.029 / 123 | 0.012 / 153 | 0.006 / 69 |
| high_card_interact | 0.052 / 155 | 0.404 / 983 | 0.238 / 430 |
| survey_kit_shaped | 0.038 / 155 | 0.052 / 228 | 0.025 / 69 |
| wide_columns | 0.156 / 224 | 1.377 / 2903 | 0.680 / 1394 |

n=100,000:

| formula | ours | formulaic | r |
|---|---|---|---|
| numeric | 0.009 / 123 | 0.003 / 160 | 0.003 / 92 |
| single_cat_low | 0.012 / 102 | 0.011 / 164 | 0.018 / 127 |
| add_numeric_cat | 0.030 / 115 | 0.012 / 160 | 0.019 / 110 |
| interact_numeric_cat | 0.022 / 137 | 0.012 / 162 | 0.020 / 106 |
| two_cat_low | 0.020 / 107 | 0.019 / 158 | 0.019 / 119 |
| four_cat_interact_low | 0.030 / 111 | 0.116 / 285 | 0.115 / 156 |
| numeric_cat_cross | 0.026 / 162 | 0.045 / 216 | 0.032 / 128 |
| high_card_interact | 0.177 / 265 | 1.869 / 4298 | 1.009 / 2105 |
| survey_kit_shaped | 0.046 / 314 | 0.195 / 512 | 0.173 / 278 |
| wide_columns | 0.625 / 484 | **capped @8GB** / 8276 | 3.306 / 6993 |

n=400,000:

| formula | ours | formulaic | r |
|---|---|---|---|
| numeric | 0.012 / 164 | 0.006 / 187 | 0.003 / 201 |
| single_cat_low | 0.016 / 177 | 0.036 / 186 | 0.012 / 201 |
| add_numeric_cat | 0.032 / 170 | 0.040 / 209 | 0.017 / 201 |
| interact_numeric_cat | 0.025 / 174 | 0.040 / 212 | 0.017 / 201 |
| two_cat_low | 0.022 / 160 | 0.072 / 197 | 0.022 / 201 |
| four_cat_interact_low | 0.076 / 348 | 0.432 / 626 | 0.184 / 419 |
| numeric_cat_cross | 0.041 / 231 | 0.166 / 411 | 0.096 / 267 |
| high_card_interact | 1.169 / 676 | **capped @8GB** / 8250 | **capped @8GB** / 8267 |
| survey_kit_shaped | 0.092 / 806 | 0.749 / 1663 | 0.372 / 835 |
| wide_columns | 3.823 / 1376 | **capped @8GB** / 8218 | **error** (alloc 26.8GB) |

## Large-scale tier (ours / r only)

| formula | n | ours | r |
|---|---|---|---|
| survey_kit_shaped | 800,000 | 0.170 / 1683 | 0.739 / 1739 |
| survey_kit_shaped | 1,600,000 | 0.324 / 3341 | 1.402 / 3220 |
| survey_kit_shaped | 2,400,000 | 0.473 / 4895 | 2.348 / 5465 |
| wide_columns | 800,000 | 7.392 / 2543 | **error** |
| wide_columns | 1,600,000 | 14.505 / 4886 | **error** |
| wide_columns | 2,400,000 | 23.366 / 7223 | **error** |

`r`'s `wide_columns` error is `model.matrix()` itself refusing the
allocation (`cannot allocate vector of size 26.8 Gb`) — R exits cleanly
before RSS climbs, not a watchdog kill.

## Known caveat

`formulaic` and R disagree on column count for a bare multi-way
categorical interaction with no lower-order terms (`A:B:C:D`: 81 cols for
formulaic, 82 for R and `ours`) — formulaic prunes to true full rank per
scoped sub-block, R's own heuristic (`model.c`) doesn't catch this case
and leaves one redundant column. `polars_formula` matches R here, so
comparisons on that formula shape aren't perfectly apples-to-apples.

## Files

- `generate_data.py` — writes `data/n*.csv`.
- `run_ours.py`, `run_formulaic.py`, `run_r.R` — per-tool runners.
- `benchmark.py` — orchestrator + RSS watchdog.
- `results.csv`, `run_log.txt` — latest run.
