"""`ns()` (natural cubic splines) parity tests against real R. The
QR-based null-space projection that enforces the "linear beyond the
boundary knots" constraint has the same LAPACK-vs-LINPACK sign-convention
risk that `poly()`'s orthogonalization does — verified numerically against
R rather than assumed, same discipline as the rest of this project.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from parity.r_oracle import R_AVAILABLE, r_model_matrix, r_ns_matrix, r_ns_predict_outside
from survey_kit_formula.dispatch.poly_bs import NaturalSplineState, apply_ns, fit_ns
from survey_kit_formula.terms.spec import ModelSpec

requires_r = pytest.mark.skipif(not R_AVAILABLE, reason="R is not installed")

X = np.array([1.0, 2.5, 3.0, 4.2, 5.0, 6.1, 7.0, 8.3, 9.0, 10.5, 11.0, 12.7])


@requires_r
@pytest.mark.parametrize("df", [3, 4, 5, 6])
def test_ns_matches_r_df(df):
    ours, _state = fit_ns(X, df=df)
    theirs = r_ns_matrix(X, df=df)
    np.testing.assert_allclose(ours, theirs, atol=1e-8)


@requires_r
def test_ns_matches_r_intercept():
    ours, _state = fit_ns(X, df=5, intercept=True)
    theirs = r_ns_matrix(X, df=5, intercept=True)
    np.testing.assert_allclose(ours, theirs, atol=1e-8)


@requires_r
def test_ns_matches_r_default_no_df():
    # no df/knots given -> zero interior knots, just the 2-column natural
    # linear basis
    ours, _state = fit_ns(X)
    theirs = r_ns_matrix(X)
    np.testing.assert_allclose(ours, theirs, atol=1e-8)


def test_ns_apply_reproduces_fit():
    fitted, state = fit_ns(X, df=5)
    reapplied = apply_ns(X, state)
    np.testing.assert_allclose(fitted, reapplied, atol=1e-10)


def test_ns_predict_on_new_data():
    _fitted, state = fit_ns(X, df=5)
    new_x = np.array([2.0, 4.0, 6.0, 8.0])
    out = apply_ns(new_x, state)
    assert out.shape == (4, 5)


@requires_r
def test_ns_na_row_passthrough_matches_r():
    x_na = np.array([1.0, 2.0, 3.0, np.nan, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    ours, state = fit_ns(x_na, df=5)
    theirs = r_ns_matrix(x_na, df=5, allow_na=True)
    assert np.isnan(ours[3]).all()
    np.testing.assert_allclose(ours, theirs, atol=1e-8, equal_nan=True)
    assert state.boundary_knots == (1.0, 12.0)


def test_ns_all_nan_raises():
    with pytest.raises(ValueError, match="all values are missing"):
        fit_ns(np.full(5, np.nan), df=3)


def test_ns_out_of_range_warns_not_raises():
    _fitted, state = fit_ns(X, df=5)
    with pytest.warns(UserWarning, match="ill-conditioned"):
        out = apply_ns(np.array([-100.0]), state)
    assert out.shape == (1, 5)


@requires_r
def test_ns_out_of_range_matches_r():
    newx = np.array([-5.0, 0.0, 1.0, 6.0, 12.7, 15.0, 20.0])
    _fitted, state = fit_ns(X, df=5)
    with pytest.warns(UserWarning, match="some 'x' values beyond boundary knots may cause ill-conditioned bases"):
        ours = apply_ns(newx, state)
    theirs = r_ns_predict_outside(X, newx, df=5)
    np.testing.assert_allclose(ours, theirs, atol=1e-6)


# --- formula-level integration ---

DF = pl.DataFrame({"y": X.tolist(), "x": X.tolist(), "a": ["a1", "a2"] * 6})


@requires_r
def test_ns_explicit_knots_matches_r():
    formula = "y ~ ns(x, knots=[3.0, 6.0, 9.0])"
    spec = ModelSpec.from_formula(formula, DF)
    ours = spec.get_model_matrix(DF)
    r_formula = "y ~ ns(x, knots=c(3.0, 6.0, 9.0))"
    theirs = r_model_matrix(r_formula, DF, factor_cols=("a",), bool_cols=())
    np.testing.assert_allclose(ours, theirs, atol=1e-8)


@requires_r
def test_formula_level_ns_matches_r():
    formula = "y ~ ns(x, df = 4)"
    spec = ModelSpec.from_formula(formula, DF)
    ours = spec.get_model_matrix(DF)
    theirs = r_model_matrix(formula, DF, factor_cols=(), bool_cols=())
    np.testing.assert_allclose(ours, theirs, atol=1e-8)


@requires_r
def test_formula_level_ns_with_other_terms_matches_r():
    formula = "y ~ a + ns(x, df = 3)"
    spec = ModelSpec.from_formula(formula, DF)
    ours = spec.get_model_matrix(DF)
    theirs = r_model_matrix(formula, DF, factor_cols=("a",), bool_cols=())
    np.testing.assert_allclose(ours, theirs, atol=1e-8)


def test_formula_level_ns_state():
    spec = ModelSpec.from_formula("y ~ ns(x, df = 4)", DF)
    (var, ns) = next(iter(spec.numerics.items()))
    assert var.name == "ns"
    assert isinstance(ns.ns_state, NaturalSplineState)
    assert ns.width == 4
    assert spec.total_columns == 1 + 4


def test_formula_level_ns_predict_reuse():
    # train covers the full x range so `test`'s values fall within the
    # fitted Boundary.knots -- out-of-range extrapolation is covered
    # separately (test_ns_out_of_range_matches_r), not what this test is
    # checking.
    train = DF
    test = DF[3:6]
    spec = ModelSpec.from_formula("y ~ ns(x, df = 4)", train)
    train_mm = spec.get_model_matrix(train)
    test_mm = spec.get_model_matrix(test)
    assert train_mm.shape == (12, spec.total_columns)
    assert test_mm.shape == (3, spec.total_columns)


def test_formula_level_ns_null_dummy():
    df = pl.DataFrame({"y": [1.0, 2.0, 3.0, 4.0, 5.0], "x": [1.0, None, 3.0, 4.0, 5.0]})
    spec = ModelSpec.from_formula("y ~ ns(x, df = 3)", df, null_dummy=True)
    mm = spec.get_model_matrix(df)
    assert mm.shape == (5, spec.total_columns)
    assert mm[1, -1] == 1.0
