import math

import numpy as np
import polars as pl
import pytest

from survey_kit_formula.dispatch import (
    STRUCTURAL_RESERVED,
    UnknownFormulaFunction,
    contrast_override,
    eval_I,
    is_offset,
    is_registered,
    polars_expr_for_call,
)
from survey_kit_formula.parser.ast_nodes import Call
from survey_kit_formula.parser.tokenizer import FormulaSyntaxError

DF = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "z": [10.0, 20.0, 30.0, 40.0]})
ANGLES = pl.DataFrame({"x": [0.1, 0.5, 0.9, 0.3]})


@pytest.mark.parametrize(
    "call,expected",
    [
        (Call("log", "x"), np.log([1.0, 2.0, 3.0, 4.0])),
        (Call("sqrt", "x"), np.sqrt([1.0, 2.0, 3.0, 4.0])),
        (Call("exp", "x"), np.exp([1.0, 2.0, 3.0, 4.0])),
        (Call("log1p", "x"), np.log1p([1.0, 2.0, 3.0, 4.0])),
    ],
)
def test_polars_backed_math_functions(call, expected):
    expr = polars_expr_for_call(call)
    out = DF.select(expr).to_series().to_numpy()
    np.testing.assert_allclose(out, expected)


@pytest.mark.parametrize(
    "call,expected",
    [
        (Call("sin", "x"), np.sin([0.1, 0.5, 0.9, 0.3])),
        (Call("cos", "x"), np.cos([0.1, 0.5, 0.9, 0.3])),
        (Call("tan", "x"), np.tan([0.1, 0.5, 0.9, 0.3])),
        (Call("asin", "x"), np.arcsin([0.1, 0.5, 0.9, 0.3])),
        (Call("acos", "x"), np.arccos([0.1, 0.5, 0.9, 0.3])),
        (Call("atan", "x"), np.arctan([0.1, 0.5, 0.9, 0.3])),
        (Call("sinh", "x"), np.sinh([0.1, 0.5, 0.9, 0.3])),
        (Call("cosh", "x"), np.cosh([0.1, 0.5, 0.9, 0.3])),
        (Call("tanh", "x"), np.tanh([0.1, 0.5, 0.9, 0.3])),
        (Call("abs", "x"), np.abs([0.1, 0.5, 0.9, 0.3])),
        (Call("sign", "x"), np.sign([0.1, 0.5, 0.9, 0.3])),
    ],
)
def test_elementary_math_functions(call, expected):
    expr = polars_expr_for_call(call)
    out = ANGLES.select(expr).to_series().to_numpy()
    np.testing.assert_allclose(out, expected)


def test_floor_and_ceiling():
    df = pl.DataFrame({"x": [1.2, -1.2, 2.7, -2.7]})
    floor_out = df.select(polars_expr_for_call(Call("floor", "x"))).to_series().to_numpy()
    ceil_out = df.select(polars_expr_for_call(Call("ceiling", "x"))).to_series().to_numpy()
    np.testing.assert_allclose(floor_out, [1.0, -2.0, 2.0, -3.0])
    np.testing.assert_allclose(ceil_out, [2.0, -1.0, 3.0, -2.0])


def test_scale_matches_r_default():
    # R's scale(x): (x - mean(x)) / sd(x), sd = sample sd (ddof=1)
    expr = polars_expr_for_call(Call("scale", "x"))
    out = DF.select(expr).to_series().to_numpy()
    x = np.array([1.0, 2.0, 3.0, 4.0])
    expected = (x - x.mean()) / x.std(ddof=1)
    np.testing.assert_allclose(out, expected)


def test_center():
    expr = polars_expr_for_call(Call("center", "x"))
    out = DF.select(expr).to_series().to_numpy()
    x = np.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(out, x - x.mean())


def test_unknown_function_raises():
    with pytest.raises(UnknownFormulaFunction):
        polars_expr_for_call(Call("bogus", "x"))


def test_structural_reserved_not_polars_backed():
    with pytest.raises(ValueError):
        polars_expr_for_call(Call("factor", "x"))


def test_wrong_arity_raises():
    with pytest.raises(FormulaSyntaxError):
        polars_expr_for_call(Call("log", "x, y"))


def test_non_identifier_argument_raises():
    with pytest.raises(FormulaSyntaxError):
        polars_expr_for_call(Call("log", "x + 1"))


def test_is_registered():
    assert is_registered("log")
    assert is_registered("factor")
    assert is_registered("I")
    assert not is_registered("bogus")


def test_structural_reserved_set():
    assert STRUCTURAL_RESERVED == {"factor", "ordered", "C", "offset"}


def test_eval_I_arithmetic():
    namespace = {"x": np.array([1.0, 2.0, 3.0]), "z": np.array([10.0, 20.0, 30.0])}
    out = eval_I(Call("I", "x + z"), namespace)
    np.testing.assert_allclose(out, [11.0, 22.0, 33.0])


def test_eval_I_uses_allowed_numpy_functions():
    namespace = {"x": np.array([1.0, 2.0, 3.0])}
    out = eval_I(Call("I", "log(x) + 1"), namespace)
    np.testing.assert_allclose(out, np.log([1.0, 2.0, 3.0]) + 1)


def test_eval_I_no_builtin_access():
    namespace = {"x": np.array([1.0])}
    with pytest.raises(Exception):
        eval_I(Call("I", "__import__('os')"), namespace)


def test_is_offset():
    assert is_offset(Call("offset", "z"))
    assert not is_offset(Call("log", "z"))


def test_contrast_override_present():
    assert contrast_override(Call("C", "x, contr.sum")) == ("contr.sum", {})


def test_contrast_override_absent():
    assert contrast_override(Call("C", "x")) is None
    assert contrast_override(Call("factor", "x")) is None
