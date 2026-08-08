#!/usr/bin/env python3
"""Writes one CSV per row-count in `SIZES`, shared across all three tools
(polars_formula, formulaic, R) so every benchmark cell runs on byte-identical
data. CSV, not Parquet: it's the one format all three read natively without
extra optional dependencies (R would need the `arrow` package for Parquet).

Columns, modeled on formulaic's own benchmark data generator
(benchmarks/benchmark.py in matthewwardrop/formulaic) plus two additions
(`Ahi`, `Bhi`) that formulaic's own suite doesn't cover: formulaic's A/B/C/D
are only 3 levels each, which never stresses the categorical x categorical
interaction path this project exists for. Ahi/Bhi are deliberately
higher-cardinality to exercise that.

    x1, x2       : numeric (standard normal)
    A, B, C, D   : categorical, 3 levels each ("a".."l")
    Ahi          : categorical, 60 levels
    Bhi          : categorical, 45 levels
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

SIZES = [50_000, 500_000, 2_000_000]
SEED = 0
DATA_DIR = Path(__file__).parent / "data"


def make_df(n: int, seed: int = SEED) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    low_card_letters = list("abcdefghijkl")  # A:a,b,c  B:d,e,f  C:g,h,i  D:j,k,l
    return pl.DataFrame(
        {
            "x1": rng.standard_normal(n),
            "x2": rng.standard_normal(n),
            "A": rng.choice(low_card_letters[0:3], size=n),
            "B": rng.choice(low_card_letters[3:6], size=n),
            "C": rng.choice(low_card_letters[6:9], size=n),
            "D": rng.choice(low_card_letters[9:12], size=n),
            # "L" prefix is deliberate: plain digit-strings ("0".."59") get
            # silently re-inferred as *integers* on CSV round-trip by both
            # polars and pandas read_csv, which would make Ahi/Bhi numeric
            # instead of categorical in the Python tools while R (which
            # explicitly forces these columns to factor()) stayed correct
            # -- a benchmark-harness bug that would have quietly made the
            # very case this project cares about most look trivially fast.
            "Ahi": ["L" + s for s in rng.integers(0, 60, size=n).astype(str)],
            "Bhi": ["L" + s for s in rng.integers(0, 45, size=n).astype(str)],
        }
    )


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    sizes = SIZES if len(sys.argv) == 1 else [int(a) for a in sys.argv[1:]]
    for n in sizes:
        path = DATA_DIR / f"n{n}.csv"
        make_df(n).write_csv(path)
        print(f"wrote {path} ({n:,} rows)")


if __name__ == "__main__":
    main()
