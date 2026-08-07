from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from polars_formula.terms.spec import ModelSpec

DF = pl.DataFrame(
    {
        "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "x": [1.0, 2.0, None, 4.0, 5.0, 6.0],
        "a": ["a1", "a2", "a1", None, "a1", "a2"],
        "clean": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    }
)


def test_null_raises_by_default_factor():
    with pytest.raises(ValueError, match="contains nulls"):
        ModelSpec.from_formula("y ~ a", DF)


def test_null_raises_by_default_numeric():
    with pytest.raises(ValueError, match="contains nulls"):
        ModelSpec.from_formula("y ~ x", DF)


def test_null_dummy_no_raise():
    spec = ModelSpec.from_formula("y ~ x + a", DF, null_dummy=True)
    mm = spec.get_model_matrix(DF)
    assert mm.shape == (6, spec.total_columns)


def test_null_dummy_companion_columns_tracked():
    spec = ModelSpec.from_formula("y ~ x + a", DF, null_dummy=True)
    from polars_formula.parser.ast_nodes import Identifier

    assert set(spec.null_companions) == {Identifier("x"), Identifier("a")}
    assert spec.total_columns == 1 + 1 + 1 + 2  # intercept + x + a(1 contrast) + 2 companions


def test_null_dummy_no_companion_for_clean_variable():
    spec = ModelSpec.from_formula("y ~ clean", DF, null_dummy=True)
    assert spec.null_companions == []
    assert spec.total_columns == 2  # intercept + clean, no companion


def test_numeric_null_filled_to_zero_default():
    spec = ModelSpec.from_formula("y ~ 0 + x", DF, null_dummy=True)
    mm = spec.get_model_matrix(DF)
    # column order: [x, companion(x)]
    np.testing.assert_allclose(mm[:, 0], [1.0, 2.0, 0.0, 4.0, 5.0, 6.0])
    np.testing.assert_allclose(mm[:, 1], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0])


def test_numeric_null_custom_fill_value():
    spec = ModelSpec.from_formula("y ~ 0 + x", DF, null_dummy=True, null_fill=-999.0)
    mm = spec.get_model_matrix(DF)
    np.testing.assert_allclose(mm[:, 0], [1.0, 2.0, -999.0, 4.0, 5.0, 6.0])


def test_factor_null_row_is_all_zero_plus_companion_flag():
    spec = ModelSpec.from_formula("y ~ 0 + a", DF, null_dummy=True)
    mm = spec.get_model_matrix(DF)
    # a has 2 levels (a1, a2), no intercept -> DUMMY coding, full 2 columns
    # row 3 (index 3) has a=None -> both dummy columns should be 0
    np.testing.assert_allclose(mm[3, :2], [0.0, 0.0])
    # companion column (last) should be 1 for row 3, 0 elsewhere
    companion = mm[:, -1]
    expected = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    np.testing.assert_allclose(companion, expected)


def test_scale_null_propagates_correctly_not_via_log_of_zero():
    # log(0) would be -inf if nulls were filled *before* the transform --
    # correct behavior is null propagates through the expression and only
    # the final null result gets filled.
    df = pl.DataFrame({"y": [1.0, 2.0, 3.0], "x": [1.0, None, 3.0]})
    spec = ModelSpec.from_formula("y ~ 0 + log(x)", df, null_dummy=True)
    mm = spec.get_model_matrix(df)
    assert mm[1, 0] == 0.0  # filled, not -inf or NaN
    np.testing.assert_allclose(mm[0, 0], np.log(1.0))
    np.testing.assert_allclose(mm[2, 0], np.log(3.0))


def test_unexpected_null_at_apply_time_still_raises_for_non_companion_var():
    train = DF.select("y", "clean")  # 'clean' has no nulls in training
    spec = ModelSpec.from_formula("y ~ clean", train, null_dummy=True)
    assert spec.null_companions == []
    new_df = pl.DataFrame({"y": [1.0], "clean": [None]})
    with pytest.raises(ValueError, match="contains nulls"):
        spec.get_model_matrix(new_df)


def test_companion_var_tolerates_nulls_at_apply_time_too():
    spec = ModelSpec.from_formula("y ~ x", DF, null_dummy=True)
    new_df = pl.DataFrame({"y": [1.0, 2.0], "x": [None, 5.0]})
    mm = spec.get_model_matrix(new_df)
    assert mm.shape == (2, spec.total_columns)
    assert mm[0, 1] == 0.0  # filled x
    assert mm[:, -1].tolist() == [1.0, 0.0]  # companion flags


def test_unseen_level_still_raises_even_with_null_dummy():
    train = DF.drop_nulls("a")
    spec = ModelSpec.from_formula("y ~ a", train, null_dummy=True)
    new_df = pl.DataFrame({"y": [1.0], "a": ["a3"]})
    with pytest.raises(ValueError, match="not seen"):
        spec.get_model_matrix(new_df)
