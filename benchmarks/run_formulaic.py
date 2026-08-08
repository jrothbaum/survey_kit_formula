#!/usr/bin/env python3
"""Runs one (data, formula) cell for formulaic. Meant to be invoked via
`uv run --with formulaic --with pandas --with pyarrow python run_formulaic.py`
as its own subprocess (see benchmark.py), wrapped in `/usr/bin/time -v`.
Pandas, not polars: formulaic's own benchmark suite and the overwhelming
majority of real-world usage (including the survey_kit code this whole
project traces back to) run it against pandas -- that's the representative
comparison, not formulaic's newer/less-exercised polars support.

Times `model_matrix()` through to a raw numpy array (`np.asarray(mm)`),
matching what polars_formula returns, since that's the actually-comparable
end state (a design matrix ready for a solver), not formulaic's intermediate
`ModelMatrix` wrapper.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

import formulaic

_CATEGORICAL_COLS = ("A", "B", "C", "D", "Ahi", "Bhi", "Vhi")


def _assert_categoricals_are_strings(df: pd.DataFrame) -> None:
    # See run_ours.py: CSV round-tripping silently re-infers digit-strings
    # as integers, which would make these numeric instead of categorical.
    for col in _CATEGORICAL_COLS:
        if col in df.columns and not pd.api.types.is_string_dtype(df[col]):
            raise TypeError(f"expected {col!r} to read back as string, got {df[col].dtype}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--formula", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    _assert_categoricals_are_strings(df)

    t0 = time.perf_counter()
    mm = formulaic.model_matrix(args.formula, df)
    arr = np.asarray(mm)
    build_seconds = time.perf_counter() - t0

    print(json.dumps({"build_seconds": build_seconds, "rows": arr.shape[0], "cols": arr.shape[1]}))


if __name__ == "__main__":
    main()
