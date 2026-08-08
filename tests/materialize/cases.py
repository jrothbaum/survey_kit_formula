"""Shared (formula, data) battery for R parity checks — used both by the
live-R test (`test_matrix.py`) and the fixture generator
(`scripts/generate_r_fixtures.py`), so the two can't silently drift apart.
"""

from __future__ import annotations

import polars as pl

DF = pl.DataFrame(
    {
        "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        "x": [1.0, 2.5, 3.0, 4.2, 5.0, 6.1, 7.0, 8.3, 9.0, 10.5, 11.0, 12.7],
        "z": [2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0, 7.0, 10.0, 9.0, 12.0, 11.0],
        "a": ["a1", "a2", "a1", "a2", "a1", "a2", "a1", "a2", "a1", "a2", "a1", "a2"],
        "b": ["b1", "b1", "b2", "b2", "b3", "b3", "b1", "b2", "b3", "b1", "b2", "b3"],
        "flag": [True, False, True, False, True, False, True, False, True, False, True, False],
    }
)

FACTOR_COLS = ("a", "b")
BOOL_COLS = ("flag",)

# (slug, formula) -- slug must be filesystem-safe, used as the fixture filename.
FIXTURE_CASES = [
    ("simple_numeric", "y ~ x"),
    ("two_numeric", "y ~ x + z"),
    ("single_factor", "y ~ a"),
    ("two_factors_additive", "y ~ a + b"),
    ("factor_crossing", "y ~ a * b"),
    ("factor_interaction_only", "y ~ a:b"),
    ("no_intercept_factor", "y ~ 0 + a"),
    ("no_intercept_crossing", "y ~ 0 + a * b"),
    ("numeric_and_factor", "y ~ x + a"),
    ("numeric_factor_interaction", "y ~ x + a + x:a"),
    ("boolean_main_effect", "y ~ flag"),
    ("boolean_and_numeric", "y ~ flag + x"),
    ("log_transform", "y ~ log(x)"),
    ("scale_transform", "y ~ scale(x)"),
    ("I_sum", "y ~ I(x + z)"),
    ("I_product_literal", "y ~ I(x * 2)"),
    ("three_way_mixed", "y ~ a + b + a:b:x"),
    ("poly_degree3", "y ~ poly(x, degree = 3)"),
    ("bs_df4", "y ~ bs(x, df = 4)"),
    ("C_treatment_base", "y ~ C(a, treatment, base = 2)"),
    ("survey_kit_shaped", "y ~ 1 + x + z + z*x*C(a)"),
    ("sin_cos_transform", "y ~ sin(x) + cos(z)"),
    ("abs_transform", "y ~ abs(x)"),
    ("polym_2var_degree2", "y ~ poly(x, z, degree = 2)"),
    ("polym_2var_raw", "y ~ poly(x, z, degree = 2, raw = TRUE)"),
    ("polym_with_other_terms", "y ~ a + poly(x, z, degree = 2)"),
]
