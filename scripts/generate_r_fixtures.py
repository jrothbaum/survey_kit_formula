#!/usr/bin/env python3
"""Regenerates tests/parity/fixtures/model_matrix/*.csv from live R.

Run this whenever `FIXTURE_CASES` in tests/materialize/cases.py changes, or
the R side of a comparison needs refreshing:

    uv run python scripts/generate_r_fixtures.py

Requires R (`Rscript`) on PATH. The generated fixtures are committed so
`tests/materialize/test_matrix.py`'s fixture-based tests get real R-parity
coverage in CI without requiring R to be installed there — only local
regeneration needs it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tests"))

from materialize.cases import BOOL_COLS, DF, FACTOR_COLS, FIXTURE_CASES  # noqa: E402
from parity.r_oracle import R_AVAILABLE, r_model_matrix  # noqa: E402

FIXTURES_DIR = REPO_ROOT / "tests" / "parity" / "fixtures" / "model_matrix"


def main() -> None:
    if not R_AVAILABLE:
        print("Rscript not found on PATH; cannot regenerate fixtures.", file=sys.stderr)
        raise SystemExit(1)

    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    for slug, formula in FIXTURE_CASES:
        mat = r_model_matrix(formula, DF, factor_cols=FACTOR_COLS, bool_cols=BOOL_COLS)
        out_path = FIXTURES_DIR / f"{slug}.csv"
        np.savetxt(out_path, mat, delimiter=",")
        print(f"{slug:30s} {formula:40s} -> {out_path.relative_to(REPO_ROOT)} {mat.shape}")


if __name__ == "__main__":
    main()
