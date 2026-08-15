"""Polars/narwhals intake and outtake: convert arbitrary supported input to
an eager Polars DataFrame -- pruned to just the needed columns *before* any
full materialization when the source is lazy -- and convert a computed
Polars result back to the input's native library.

Native `pl.DataFrame`/`pl.LazyFrame` never touch narwhals; that path is
exactly as fast as it always was. Anything else goes through narwhals
(a hard dependency, see pyproject.toml).
"""

from __future__ import annotations

from types import ModuleType
from typing import Any, List, Optional, Sequence, Tuple

import narwhals.stable.v2 as nw
import polars as pl

# The input's own native module (pandas, pyarrow, duckdb, ...), as reported
# by narwhals -- not a fixed set of names we recognize up front. `None`
# means the input was already Polars, so no conversion is needed either way.
NativeKind = Optional[ModuleType]


def schema_names(data: Any) -> List[str]:
    """Column names only, without materializing any rows."""
    if isinstance(data, (pl.DataFrame, pl.LazyFrame)):
        return list(data.collect_schema().keys())
    frame = nw.from_native(data)
    schema = frame.collect_schema() if isinstance(frame, nw.LazyFrame) else frame.schema
    return list(schema.keys())


def to_polars(data: Any, columns: Optional[Sequence[str]]) -> Tuple[pl.DataFrame, NativeKind]:
    """Materialize `data` as an eager Polars DataFrame. When `columns` is
    given and `data` is lazy (a Polars LazyFrame, or any lazy engine
    narwhals recognizes), the column selection happens *before* collecting,
    so columns outside `columns` are never read from the source. Returns
    the input's native module too, so a result can later be handed back in
    the same form via `from_polars`."""
    if isinstance(data, pl.DataFrame):
        return (data.select(columns) if columns is not None else data), None
    if isinstance(data, pl.LazyFrame):
        lf = data.select(columns) if columns is not None else data
        return lf.collect(), None

    frame = nw.from_native(data)
    if columns is not None:
        frame = frame.select(columns)

    # Captured before collecting: a lazy-only backend's `.collect()`
    # defaults to converting into pyarrow/pandas (e.g. DuckDB -> PyArrow),
    # which would make the input's real native module unrecoverable here.
    native_module = nw.get_native_namespace(frame)
    if isinstance(frame, nw.LazyFrame):
        frame = frame.collect()

    if native_module.__name__ == "polars":
        return frame.to_native(), None
    return pl.from_arrow(frame.to_arrow()), native_module


def from_polars(result: pl.DataFrame, kind: NativeKind) -> Any:
    """Convert a computed Polars result back to `kind`, the native module
    `to_polars` reported for the original input. `kind=None` means the
    input was already Polars, so `result` is returned unchanged.

    Otherwise this asks narwhals to build a frame of that same module's
    type from `result`'s data. Most backends (pandas, PyArrow, Modin,
    cuDF, ...) construct eagerly; lazy-only engines (DuckDB, Dask,
    PySpark, ...) build eagerly via PyArrow first and then convert to that
    backend's lazy frame -- narwhals' own documented route for that case."""
    if kind is None:
        return result
    try:
        return nw.from_arrow(result.to_arrow(), backend=kind).to_native()
    except ValueError:
        return nw.from_arrow(result.to_arrow(), backend="pyarrow").lazy(kind).to_native()
