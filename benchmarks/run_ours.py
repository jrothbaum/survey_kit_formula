#!/usr/bin/env python3
"""Runs one (data, formula) cell for survey_kit_formula. Meant to be invoked as
its own subprocess (see benchmark.py) wrapped in `/usr/bin/time -v`, so peak
RSS reflects this one build in isolation. Prints a single JSON line to
stdout with the *internal* build time (excludes import + CSV read, which
`/usr/bin/time -v`'s wall-clock figure does include -- the orchestrator
reports both).

Times `get_model_frame` (the Polars-native result), not `get_model_matrix`
-- `get_model_matrix` is just `get_model_frame(...).to_numpy()`, and that
conversion is a real, separately-payable cost (proportional to column
count) that a caller only incurs if they actually want a dense numpy
array. Including it here would conflate "how fast is the build" with "how
fast is exporting to numpy afterward", and formulaic/R's own equivalents
(a pandas/numpy array, an R matrix) don't have a separate Polars-native
intermediate to compare against in the first place. Row/col counts are
read straight off the DataFrame's own `.shape` -- no `.to_numpy()` call
anywhere in the timed path.
"""

from __future__ import annotations

import argparse
import json
import time

import polars as pl

from survey_kit_formula.terms.spec import ModelSpec

_CATEGORICAL_COLS = ("A", "B", "C", "D", "Ahi", "Bhi", "Vhi")


def _assert_categoricals_are_strings(df: pl.DataFrame) -> None:
    # CSV round-tripping silently re-infers digit-strings as integers --
    # caught a real bug from exactly this once, see generate_data.py.
    for col in _CATEGORICAL_COLS:
        if col in df.columns and df.schema[col] != pl.String:
            raise TypeError(f"expected {col!r} to read back as String, got {df.schema[col]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--formula", required=True)
    args = ap.parse_args()

    df = pl.read_csv(args.data)
    _assert_categoricals_are_strings(df)

    t0 = time.perf_counter()
    spec = ModelSpec.from_formula(args.formula, df)
    mf = spec.get_model_frame(df)
    build_seconds = time.perf_counter() - t0

    print(json.dumps({"build_seconds": build_seconds, "rows": mf.shape[0], "cols": mf.shape[1]}))


if __name__ == "__main__":
    main()
