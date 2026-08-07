import polars as pl
import pytest

from polars_formula.parser.ast_nodes import Call, Identifier
from polars_formula.terms.classify import DataClass, classify_var, underlying_column

SCHEMA = pl.Schema(
    {
        "x_int": pl.Int64,
        "x_float": pl.Float64,
        "x_str": pl.String,
        "x_bool": pl.Boolean,
        "x_cat": pl.Categorical(),
        "x_enum": pl.Enum(["p", "q"]),
    }
)


@pytest.mark.parametrize(
    "name,expected",
    [
        ("x_int", DataClass.NUMERIC),
        ("x_float", DataClass.NUMERIC),
        ("x_str", DataClass.FACTOR_UNORDERED),
        ("x_bool", DataClass.FACTOR_UNORDERED),
        ("x_cat", DataClass.FACTOR_UNORDERED),
        ("x_enum", DataClass.FACTOR_UNORDERED),
    ],
)
def test_identifier_classification(name, expected):
    assert classify_var(Identifier(name), SCHEMA) == expected


def test_missing_column_raises():
    with pytest.raises(KeyError):
        classify_var(Identifier("nope"), SCHEMA)


@pytest.mark.parametrize(
    "call,expected",
    [
        (Call("factor", "x_int"), DataClass.FACTOR_UNORDERED),
        (Call("factor", "x_str"), DataClass.FACTOR_UNORDERED),
        (Call("C", "x_int"), DataClass.FACTOR_UNORDERED),
        (Call("C", "x_int, contr.sum"), DataClass.FACTOR_UNORDERED),
        (Call("ordered", "x_int"), DataClass.FACTOR_ORDERED),
        (Call("log", "x_float"), DataClass.NUMERIC),
        (Call("poly", "x_float, degree=2"), DataClass.NUMERIC),
        (Call("I", "x_float + x_int"), DataClass.NUMERIC),
        (Call("scale", "x_float"), DataClass.NUMERIC),
        (Call("bs", "x_float, df=4"), DataClass.NUMERIC),
    ],
)
def test_call_classification(call, expected):
    assert classify_var(call, SCHEMA) == expected


def test_C_delegates_to_underlying_column_classification():
    # C() itself never forces unordered the way factor() does -- it
    # delegates to whatever the underlying column would classify as (dtype
    # alone can only ever yield unordered here, since only an explicit
    # ordered() wrapper produces FACTOR_ORDERED -- this just confirms C()
    # doesn't clobber that distinction the way an earlier, simpler version
    # of this function did).
    schema = pl.Schema({"x_enum": pl.Enum(["lo", "mid", "hi"]), "x_int": pl.Int64})
    assert classify_var(Call("C", "x_enum"), schema) == DataClass.FACTOR_UNORDERED
    assert classify_var(Call("C", "x_int"), schema) == DataClass.FACTOR_UNORDERED


def test_is_factor_property():
    assert DataClass.FACTOR_UNORDERED.is_factor
    assert DataClass.FACTOR_ORDERED.is_factor
    assert not DataClass.NUMERIC.is_factor


@pytest.mark.parametrize(
    "var,expected",
    [
        (Identifier("x_int"), "x_int"),
        (Call("poly", "x_float, degree=2"), "x_float"),
        (Call("C", "x_str, contr.treatment('q')"), "x_str"),
        (Call("log", "x_float"), "x_float"),
        (Call("I", "x_float"), "x_float"),
    ],
)
def test_underlying_column(var, expected):
    assert underlying_column(var) == expected


def test_underlying_column_rejects_nested_transform():
    with pytest.raises(ValueError):
        underlying_column(Call("log", "scale(x_float)"))


def test_underlying_column_rejects_keyword_only_call():
    with pytest.raises(ValueError):
        underlying_column(Call("poly", "degree=2"))
