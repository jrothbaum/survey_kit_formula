"""Multivariate `poly(x1, x2, ..., degree=D)` / R's `polym()` — a total-
degree-filtered tensor product of each variable's own basis, not a plain
cross. `_poly_exponent_combos`'s ordering (which column is which) was
derived by reading R's `expand.grid` convention and then verified directly
against `colnames(polym(...))` before trusting it — see
`dispatch/poly_bs.py`'s docstring.
"""

from __future__ import annotations

import numpy as np
import pytest
import polars as pl

from parity.r_oracle import R_AVAILABLE, r_polym_matrix
from survey_kit_formula.dispatch.poly_bs import (
    MultivariatePolyState,
    _poly_exponent_combos,
    apply_polym,
    fit_polym,
)
from survey_kit_formula.terms.spec import ModelSpec

requires_r = pytest.mark.skipif(not R_AVAILABLE, reason="R is not installed")

X = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
Y = np.array([2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0, 7.0])
Z = np.array([1.0, 3.0, 2.0, 5.0, 4.0, 7.0, 6.0, 8.0])


def test_exponent_combos_2var_degree2():
    assert _poly_exponent_combos(2, 2) == ((1, 0), (2, 0), (0, 1), (1, 1), (0, 2))


def test_exponent_combos_3var_degree2():
    assert _poly_exponent_combos(3, 2) == (
        (1, 0, 0),
        (2, 0, 0),
        (0, 1, 0),
        (1, 1, 0),
        (0, 2, 0),
        (0, 0, 1),
        (1, 0, 1),
        (0, 1, 1),
        (0, 0, 2),
    )


@requires_r
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_polym_2var_matches_r(degree):
    ours, _state = fit_polym([X, Y], degree=degree)
    theirs = r_polym_matrix([X, Y], degree=degree)
    np.testing.assert_allclose(ours, theirs, atol=1e-8)


@requires_r
def test_polym_3var_matches_r():
    ours, _state = fit_polym([X, Y, Z], degree=2)
    theirs = r_polym_matrix([X, Y, Z], degree=2)
    np.testing.assert_allclose(ours, theirs, atol=1e-8)


@requires_r
def test_polym_raw_matches_r():
    ours, _state = fit_polym([X, Y], degree=2, raw=True)
    theirs = r_polym_matrix([X, Y], degree=2, raw=True)
    np.testing.assert_allclose(ours, theirs, atol=1e-8)


def test_polym_apply_reproduces_fit():
    fitted, state = fit_polym([X, Y], degree=2)
    reapplied = apply_polym([X, Y], state)
    np.testing.assert_allclose(fitted, reapplied, atol=1e-10)


def test_polym_predict_on_new_data():
    _fitted, state = fit_polym([X, Y], degree=2)
    new_x = np.array([2.0, 4.0, 6.0])
    new_y = np.array([1.0, 5.0, 3.0])
    out = apply_polym([new_x, new_y], state)
    assert out.shape == (3, 5)


def test_polym_requires_at_least_two_variables():
    with pytest.raises(ValueError):
        fit_polym([X], degree=2)


def test_polym_requires_equal_length():
    with pytest.raises(ValueError):
        fit_polym([X, Y[:-1]], degree=2)


def test_polym_wrong_variable_count_at_apply_raises():
    _fitted, state = fit_polym([X, Y], degree=2)
    with pytest.raises(ValueError):
        apply_polym([X], state)


# --- formula-level integration ---

DF = pl.DataFrame({"y": X.tolist(), "x": X.tolist(), "z": Z.tolist(), "a": ["a1", "a2"] * 4})


def test_formula_level_multivariate_poly_state():
    spec = ModelSpec.from_formula("y ~ poly(x, z, degree = 2)", DF)
    (var, ns) = next(iter(spec.numerics.items()))
    assert var.name == "poly"
    assert isinstance(ns.poly_state, MultivariatePolyState)
    assert ns.width == 5
    assert spec.total_columns == 1 + 5


def test_formula_level_multivariate_poly_predict_reuse():
    train = DF.head(6)
    test = DF.tail(2)
    spec = ModelSpec.from_formula("y ~ poly(x, z, degree = 2)", train)
    train_mm = spec.get_model_matrix(train)
    test_mm = spec.get_model_matrix(test)
    assert train_mm.shape == (6, spec.total_columns)
    assert test_mm.shape == (2, spec.total_columns)


def test_formula_level_multivariate_poly_null_dummy():
    df = pl.DataFrame({"y": [1.0, 2.0, 3.0, 4.0], "x": [1.0, None, 3.0, 4.0], "z": [1.0, 2.0, 3.0, 4.0]})
    spec = ModelSpec.from_formula("y ~ poly(x, z, degree = 2)", df, null_dummy=True)
    from survey_kit_formula.parser.ast_nodes import Call

    assert Call("poly", "x, z, degree = 2") in spec.null_companions
    mm = spec.get_model_matrix(df)
    assert mm.shape == (4, spec.total_columns)
    assert mm[1, -1] == 1.0  # companion flag for the null row
