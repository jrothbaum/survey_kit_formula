# Benchmarks

Time and peak-RAM comparison of `polars_formula` against
[formulaic](https://github.com/matthewwardrop/formulaic) and R's
`model.matrix()`, across 9 formulas and 3 dataset sizes.

Full write-up with charts: see the published report (ask for the link, or
regenerate per below).

## Running

```bash
uv run python benchmarks/generate_data.py   # writes benchmarks/data/n*.csv
uv run python benchmarks/benchmark.py       # writes benchmarks/results.csv
```

Requires `uv`, R (`Rscript` on `PATH`), and Linux `/proc` (the memory
watchdog reads `/proc/*/stat` and `/proc/<pid>/status`; it will not run
correctly on macOS/Windows).

Each cell (tool × formula × row-count) runs as its own subprocess via
`run_ours.py`, `run_formulaic.py`, or `run_r.R`. `benchmark.py` is the
orchestrator: it builds the command, runs it under a live RSS cap, and
records wall time, build time (model-matrix construction only, excluding
process/import startup), peak RSS, and output column count.

## The memory cap, and why it's not optional

An earlier, uncapped version of this suite let one cell (a high-cardinality
categorical interaction at large N) grow unboundedly and froze the host
machine hard enough to need a reboot. `benchmark.py` now enforces a hard
8GB RSS ceiling on every cell, live, not just after the fact:

- `systemd-run --scope -p MemoryMax=` was tried first and silently did
  **not** enforce anything in this sandbox (no cgroup delegation available) —
  worse than no guard at all, since it looks like protection while providing
  none. Don't rely on it without first verifying enforcement with a
  deliberate over-limit test.
- `ulimit -v` (virtual memory) is a real kernel rlimit and does enforce, but
  it's the wrong metric: Python/NumPy/Polars reserve far more virtual
  address space than they resident-use, so it kills legitimate builds well
  under the intended resident-memory limit.
- What's actually running: a polling watchdog (`_run_with_rss_cap` in
  `benchmark.py`) that sums real resident memory (`VmRSS`) across the whole
  process *group* (found via `/proc/*/stat`, not the unreliable
  `/proc/<pid>/task/<pid>/children`) every 50ms, and `os.killpg`s the group
  the instant it crosses the cap.

If you modify the watchdog, re-verify it before trusting it on a real run:
confirm a tiny cap kills a trivial allocation, confirm a real cap lets a
real build through, and confirm a real cap kills a real over-limit build —
each with `free -h` open to watch actual system memory, not just the
watchdog's own accounting.

Three cells (`high_card_interact` — `~ Ahi:Bhi`, 60×45 levels — at
n=400,000, for all three tools) hit this cap on the last full run and were
killed cleanly; this is expected, not a bug.

## Formula battery

Based on formulaic's own benchmark suite (`A`/`B`/`C`/`D`, 3-level
categoricals), extended with:

- `Ahi`/`Bhi` — 60- and 45-level categoricals, to stress the interaction
  path this library exists for. formulaic's own low-cardinality columns
  never exercise it.
- `survey_kit_shaped` (`~ 1 + x1 + x2 + x2*x1*Ahi`) — shaped like the
  production formula that originally motivated this project.

patsy is not included; the comparison here is specifically against
formulaic and R.

## Known caveat: formulaic and R disagree on column count for one shape

For a bare multi-way categorical interaction with an intercept and no
lower-order terms (`A:B:C:D`, `Ahi:Bhi`), formulaic produces one fewer
column than R (e.g. 81 vs 82 for `A:B:C:D`). formulaic's
`model_spec.structure` shows why: it decomposes the interaction into
separately-reduced "scoped" sub-blocks and reaches the true minimal
full-rank representation. R's own algorithm (read directly from
`model.c`, not assumed) is a simpler heuristic that doesn't catch this
case and leaves one redundant column. `polars_formula` matches R's
heuristic, not formulaic's, since matching R is the stated goal — but it
means a time/RAM comparison on this formula shape isn't quite
apples-to-apples: formulaic is doing measurably less work because it's
computing a smaller matrix.

## Files

- `generate_data.py` — writes `data/n{20000,100000,400000}.csv`. Numeric
  columns `x1`, `x2`; categoricals `A`/`B`/`C`/`D` (3 levels) and
  `Ahi`/`Bhi` (60/45 levels, prefixed `"L"` so CSV readers don't
  re-infer them as integers).
- `run_ours.py`, `run_formulaic.py`, `run_r.R` — per-tool runners, each
  prints a JSON result line.
- `benchmark.py` — orchestrator: formula/size matrix, subprocess
  invocation, the RSS watchdog, CSV output (written incrementally, after
  every cell).
- `results.csv` — latest run's results (3 tools × 9 formulas × 3 sizes).
- `run_log.txt` — stdout log of the latest run.
