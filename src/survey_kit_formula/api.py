"""Public entry points."""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from ._intake import from_polars, schema_names, to_polars
from .parser.parser import parse_formula
from .terms.columns import required_columns_from_parsed
from .terms.spec import ModelSpec


def model_matrix(formula: str, data: Any) -> np.ndarray:
    """Build a design matrix from `formula` and `data`, returned as a NumPy
    array of float64 values.

    `data` can be a Polars DataFrame or LazyFrame, a pandas DataFrame, a
    PyArrow Table, or any other dataframe type supported by narwhals.

    To apply the same formula to more than one dataset -- e.g. fit on
    training data, then transform test data the same way -- use
    `ModelSpec` instead of calling this repeatedly."""
    pl_df, _kind = to_polars(data, required_columns(formula, data))
    return ModelSpec.from_formula(formula, pl_df).get_model_matrix(pl_df)


def model_frame(formula: str, data: Any) -> Any:
    """Same as `model_matrix`, but returns a dataframe in the same format
    as `data` (pandas in, pandas out; Polars in, Polars out; PyArrow in,
    PyArrow out; ...) instead of a NumPy array. Each column uses a compact
    dtype -- e.g. a boolean column for a dummy/indicator variable -- rather
    than one dense float64 array.

    To apply the same formula to more than one dataset, use `ModelSpec`
    instead of calling this repeatedly."""
    cols = required_columns(formula, data)
    pl_df, kind = to_polars(data, cols)
    result = ModelSpec.from_formula(formula, pl_df).get_model_frame(pl_df)
    return from_polars(result, kind)


def required_columns(formula: str, data: Optional[Any] = None) -> List[str]:
    """Return the column names `formula` reads from the data, in the order
    they first appear (response variable first). `poly(x1, x2, degree=2)`
    returns both `x1` and `x2`; `I(x + log(y))` returns `x` and `y`.

    If the formula uses `.` (meaning "every other column"), pass `data` --
    a `pl.Schema`, a DataFrame, a LazyFrame, or anything else narwhals
    supports -- so the full set of columns can be resolved."""
    parsed = parse_formula(formula)
    names = None
    if data is not None:
        names = list(data.keys()) if hasattr(data, "keys") else schema_names(data)
    return required_columns_from_parsed(parsed, names)
