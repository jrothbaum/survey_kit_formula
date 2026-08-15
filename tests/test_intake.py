"""narwhals-backed intake/outtake: native Polars stays untouched by
narwhals; pandas/PyArrow (and, via narwhals, any other dataframe library)
get converted to Polars for processing -- pruned to just the columns the
formula needs *before* a lazy source is collected -- and a `model_frame`/
`get_model_frame` result comes back out in that same native form.
"""

from __future__ import annotations

import narwhals.stable.v2 as nw
import polars as pl
import pytest

pd = pytest.importorskip("pandas")
pa = pytest.importorskip("pyarrow")
duckdb = pytest.importorskip("duckdb")

from survey_kit_formula import ModelSpec, model_frame, model_matrix, required_columns  # noqa: E402

DATA = {"y": [1.0, 2.0, 3.0, 4.0], "x1": [1.0, 2.0, 3.0, 4.0], "x2": ["a", "b", "a", "b"]}


def _expected_frame() -> pl.DataFrame:
    return model_frame("y ~ x1 + x2", pl.DataFrame(DATA))


@pytest.mark.parametrize(
    "make_native, expected_type",
    [
        (lambda: pd.DataFrame(DATA), pd.DataFrame),
        (lambda: pa.table(DATA), pa.Table),
        (lambda: pl.DataFrame(DATA), pl.DataFrame),
    ],
)
def test_model_frame_round_trips_native_type(make_native, expected_type):
    native = make_native()
    result = model_frame("y ~ x1 + x2", native)
    assert isinstance(result, expected_type)

    if isinstance(result, pd.DataFrame):
        got = pl.from_pandas(result)
    elif isinstance(result, pa.Table):
        got = pl.from_arrow(result)
    else:
        got = result
    assert got.equals(_expected_frame())


def test_model_matrix_always_numpy_regardless_of_input():
    import numpy as np

    assert isinstance(model_matrix("y ~ x1 + x2", pd.DataFrame(DATA)), np.ndarray)
    assert isinstance(model_matrix("y ~ x1 + x2", pa.table(DATA)), np.ndarray)


def test_required_columns_resolves_dot_against_pandas_schema():
    assert required_columns("y ~ .", pd.DataFrame(DATA)) == ["y", "x1", "x2"]


def test_model_frame_round_trips_duckdb():
    """DuckDB is narwhals' canonical lazy-only backend (it has no eager
    mode at all), so a DuckDB relation in must come back out as a DuckDB
    relation too -- exercising the from_arrow(...).lazy(kind) fallback
    path, not just the eager pandas/pyarrow/polars reconstruction."""
    src = pl.DataFrame(DATA)  # noqa: F841 -- referenced by name in the SQL below
    rel = duckdb.sql("select * from src")

    result = model_frame("y ~ x1 + x2", rel)

    assert isinstance(result, duckdb.DuckDBPyRelation)
    assert result.pl().equals(_expected_frame())


def test_duckdb_source_is_pruned_before_collecting():
    """Same guarantee as the narwhals-lazy-polars case above, but against
    a real lazy-only engine: a column that raises if evaluated must stay
    untouched when the formula doesn't reference it. Built as a DuckDB UDF
    on a DuckDB table (rather than routing through a Polars column) so
    this tests DuckDB's own projection pushdown, not Polars/DuckDB Arrow
    interop quirks."""

    def poison(_x: float) -> float:
        raise RuntimeError("poison materialized")

    con = duckdb.connect()
    con.create_function("poison_udf", poison, ["DOUBLE"], "DOUBLE")
    con.sql("create table base as select * from (values (1.0, 1.0), (2.0, 2.0)) as t(y, x1)")
    rel = con.sql("select y, x1, poison_udf(x1) as poison from base")

    result = model_matrix("y ~ x1", rel)
    assert result.shape == (2, 2)

    with pytest.raises(duckdb.Error, match="poison materialized"):
        model_matrix("y ~ x1 + poison", rel)


def test_lazy_narwhals_source_is_pruned_before_collecting():
    """Columns the formula doesn't reference must never be computed, not
    just "not returned" -- proven with a column whose expression raises if
    it's ever actually materialized."""
    poisoned = pl.LazyFrame({"y": [1.0, 2.0], "x1": [1.0, 2.0]}).with_columns(
        pl.col("x1")
        .map_batches(
            lambda s: (_ for _ in ()).throw(RuntimeError("poison materialized")),
            return_dtype=pl.Float64,
        )
        .alias("poison")
    )
    wrapped = nw.from_native(poisoned)

    result = model_matrix("y ~ x1", wrapped)
    assert result.shape == (2, 2)

    with pytest.raises(RuntimeError, match="poison materialized"):
        model_matrix("y ~ x1 + poison", wrapped)


def test_fit_once_reapply_pattern_round_trips_pandas():
    train = pd.DataFrame(DATA)
    test = pd.DataFrame({"y": [5.0], "x1": [5.0], "x2": ["b"]})

    spec = ModelSpec.from_formula("y ~ x1 + x2", train)
    result = spec.get_model_frame(test)

    assert isinstance(result, pd.DataFrame)
    assert pl.from_pandas(result).to_dicts() == [{"(Intercept)": 1, "x1": 5, "x2b": True}]
