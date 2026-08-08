"""`get_model_frame` — the Polars-DataFrame counterpart to
`get_model_matrix`: same values and column order, but each column packed
to the tightest dtype that represents it exactly instead of every column
paying float64's 8 bytes/row regardless of content.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from materialize.cases import DF, FIXTURE_CASES
from polars_formula.terms.spec import ModelSpec


@pytest.mark.parametrize("slug,formula", FIXTURE_CASES)
def test_model_frame_values_match_model_matrix(slug, formula):
    spec = ModelSpec.from_formula(formula, DF)
    mm = spec.get_model_matrix(DF)
    mf = spec.get_model_frame(DF)
    assert mf.shape == mm.shape
    np.testing.assert_allclose(mf.to_numpy().astype(np.float64), mm, atol=1e-8)


def test_model_frame_column_count_matches_names():
    spec = ModelSpec.from_formula("y ~ x + a", DF)
    mf = spec.get_model_frame(DF)
    assert mf.width == spec.total_columns
    assert len(set(mf.columns)) == mf.width  # no accidental name collisions


def test_dummy_columns_are_boolean():
    spec = ModelSpec.from_formula("y ~ a", DF)  # default treatment contrast, base a1
    mf = spec.get_model_frame(DF)
    assert mf.schema["aa2"] == pl.Boolean  # single contrast column for 2-level `a`


def test_full_dummy_no_intercept_is_boolean():
    spec = ModelSpec.from_formula("y ~ 0 + a", DF)
    mf = spec.get_model_frame(DF)
    assert mf.schema["aa1"] == pl.Boolean
    assert mf.schema["aa2"] == pl.Boolean


def test_column_names_match_r():
    # Verified directly against `colnames(model.matrix(...))` in real R --
    # see the naming-convention checks run while implementing this.
    spec = ModelSpec.from_formula("y ~ x + a + a:b", DF)
    mf = spec.get_model_frame(DF)
    assert mf.columns == [
        "(Intercept)",
        "x",
        "aa2",
        "aa1:bb2",
        "aa2:bb2",
        "aa1:bb3",
        "aa2:bb3",
    ]


def test_intercept_is_small_int_not_boolean():
    spec = ModelSpec.from_formula("y ~ x", DF)
    mf = spec.get_model_frame(DF)
    assert mf.schema["(Intercept)"] == pl.Int8
    assert mf["(Intercept)"].to_list() == [1] * DF.height


def test_continuous_column_stays_float64():
    spec = ModelSpec.from_formula("y ~ x", DF)
    mf = spec.get_model_frame(DF)
    assert mf.schema["x"] == pl.Float64


def test_integer_sourced_numeric_column_is_packed_small():
    df = DF.with_columns(pl.Series("cnt", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]))
    spec = ModelSpec.from_formula("y ~ cnt", df)
    mf = spec.get_model_frame(df)
    assert mf.schema["cnt"] == pl.Int8
    assert mf["cnt"].to_list() == df["cnt"].to_list()


def test_sum_contrast_is_integer_not_boolean():
    spec = ModelSpec.from_formula("y ~ C(a, sum)", DF)
    mf = spec.get_model_frame(DF)
    assert mf.columns[1] == "C(a, sum)1"
    col = mf.columns[1]  # the single contrast column for 2-level `a`
    assert mf.schema[col] in (pl.Int8, pl.Int16, pl.Int32, pl.Int64)
    assert set(mf[col].to_list()) == {1, -1}


def test_poly_contrast_stays_float():
    spec = ModelSpec.from_formula("y ~ poly(x, degree = 2)", DF)
    mf = spec.get_model_frame(DF)
    poly_cols = [c for c in mf.columns if c.startswith("poly(")]
    assert poly_cols
    for c in poly_cols:
        assert mf.schema[c] == pl.Float64


def test_null_companion_column_is_boolean():
    df = pl.DataFrame({"y": [1.0, 2.0, 3.0], "x": [1.0, None, 3.0]})
    spec = ModelSpec.from_formula("y ~ x", df, null_dummy=True)
    mf = spec.get_model_frame(df)
    assert mf.schema["x_isnull"] == pl.Boolean
    assert mf["x_isnull"].to_list() == [False, True, False]


def test_interaction_column_with_continuous_is_not_boolean():
    spec = ModelSpec.from_formula("y ~ x + a + x:a", DF)
    mf = spec.get_model_frame(DF)
    interact_cols = [c for c in mf.columns if ":" in c]
    assert interact_cols
    for c in interact_cols:
        assert mf.schema[c] == pl.Float64


def test_declared_variable_order_in_interaction_names():
    # a's levels must vary fastest (first-declared) -- verified against
    # real R: `~a:b` gives aa1:bb1, aa2:bb1, aa1:bb2, aa2:bb2, not the
    # reverse.
    spec = ModelSpec.from_formula("y ~ a:b", DF)
    mf = spec.get_model_frame(DF)
    interact = [c for c in mf.columns if ":" in c]
    assert interact[0].split(":")[0] == "aa1"
    assert interact[1].split(":")[0] == "aa2"
    assert interact[0].split(":")[1] == interact[1].split(":")[1] == "bb1"


def test_get_model_frame_reused_on_new_data():
    train = DF.head(8)
    test = DF.tail(4)
    spec = ModelSpec.from_formula("y ~ x + a", train)
    train_mf = spec.get_model_frame(train)
    test_mf = spec.get_model_frame(test)
    assert train_mf.height == 8
    assert test_mf.height == 4
    assert train_mf.schema == test_mf.schema
