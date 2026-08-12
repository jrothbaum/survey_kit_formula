"""Marginality tests, cross-checked against real R's `model.matrix` via the
`assign` attribute (exact per-term column counts) — this is the single
highest-risk piece of the whole project, so every case here talks to real R
rather than trusting a from-memory/from-docs reading of the algorithm.
"""

from __future__ import annotations

import polars as pl
import pytest

from parity.r_oracle import R_AVAILABLE, r_term_column_counts
from survey_kit_formula.parser import parse_formula
from survey_kit_formula.terms.marginality import compute_marginality, term_column_count

requires_r = pytest.mark.skipif(not R_AVAILABLE, reason="R is not installed")

# name -> nlevels (factor) or None (numeric)
CASES = [
    ("y ~ a", {"a": 2}),
    ("y ~ a + b", {"a": 2, "b": 3}),
    ("y ~ a:b", {"a": 2, "b": 3}),
    ("y ~ 0 + a:b", {"a": 2, "b": 3}),
    ("y ~ a + a:b", {"a": 2, "b": 3}),
    ("y ~ b + a:b", {"a": 2, "b": 3}),
    ("y ~ a*b", {"a": 2, "b": 3}),
    ("y ~ 0 + a*b", {"a": 2, "b": 3}),
    ("y ~ a:b:c", {"a": 2, "b": 3, "c": 2}),
    ("y ~ a + b + a:b:c", {"a": 2, "b": 3, "c": 2}),
    ("y ~ a:b + a:c", {"a": 2, "b": 3, "c": 2}),
    ("y ~ a:b + b:c", {"a": 2, "b": 3, "c": 2}),
    ("y ~ 0 + a", {"a": 2}),
    ("y ~ a - 1", {"a": 4}),
    ("y ~ a:b:dd + a:b:c", {"a": 2, "b": 2, "c": 2, "dd": 2}),  # subset-of-larger-term case
    ("y ~ x + a", {"x": None, "a": 3}),
    ("y ~ x + a + x:a", {"x": None, "a": 3}),
    ("y ~ 0 + x + a", {"x": None, "a": 3}),
    ("y ~ 0 + x:a", {"x": None, "a": 3}),
    ("y ~ a*b*c", {"a": 2, "b": 2, "c": 2}),
    ("y ~ 0 + a*b*c", {"a": 2, "b": 2, "c": 2}),
]


@requires_r
@pytest.mark.parametrize("formula,column_specs", CASES)
def test_column_counts_match_r(formula, column_specs):
    schema = pl.Schema(
        {name: (pl.Enum([f"L{i}" for i in range(1, n + 1)]) if n is not None else pl.Float64) for name, n in column_specs.items()}
    )
    schema = pl.Schema({"y": pl.Float64, **schema})

    parsed = parse_formula(formula)
    result = compute_marginality(parsed.rhs, schema, parsed.intercept)

    nlevels = {v: column_specs[v.name] for v in result.dataclasses if result.dataclasses[v].is_factor}
    ours_counts = [term_column_count(p, result.dataclasses, nlevels) for p in result.term_plans]

    r_intercept, r_counts = r_term_column_counts(formula, column_specs)

    assert parsed.intercept == r_intercept
    assert ours_counts == r_counts, f"{formula}: {ours_counts} != {r_counts}"
