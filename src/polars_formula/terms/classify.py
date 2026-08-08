"""Resolves each atomic variable in a formula to R's `dataClasses` concept:
numeric, unordered factor, or ordered factor. This is what the marginality
algorithm (Phase 3) needs before it can decide contrasts-vs-full-dummy for
any factor.

Classification rules (validated against real R's `model.matrix`, which
auto-coerces raw character *and logical* columns to factors — it is not
just `pl.String`/`pl.Categorical`/`pl.Enum` that count):

- `pl.String`, `pl.Categorical`, `pl.Enum`, `pl.Boolean` -> unordered factor
  (Polars `Enum`'s inherent category order is for efficient storage/sort,
  not a statement of ordinality, so it still defaults to unordered — same
  as R, where reading a character column never produces an *ordered*
  factor without saying so explicitly).
- everything else (numeric dtypes) -> numeric
- `factor(x, ...)` / `C(x, ...)` -> forces unordered factor regardless of
  the underlying dtype
- `ordered(x, ...)` -> forces ordered factor
- any other function call (`log(x)`, `poly(x, degree=2)`, `I(x + y)`,
  `scale(x)`, `bs(x, df=4)`, ...) -> numeric. R's model.matrix treats the
  output of an arbitrary function call as numeric (possibly multi-column,
  e.g. `poly()`/`bs()`) — it is never subject to the factor contrast/
  full-dummy decision, only actual factor variables are.
"""

from __future__ import annotations

from enum import Enum

import polars as pl

from ..parser.args import split_args
from ..parser.ast_nodes import Call, Identifier, Var


class DataClass(Enum):
    NUMERIC = "numeric"
    FACTOR_UNORDERED = "factor"
    FACTOR_ORDERED = "ordered"

    @property
    def is_factor(self) -> bool:
        return self is not DataClass.NUMERIC


_STRUCTURAL_ORDERED = {"ordered"}


def classify_var(v: Var, schema: pl.Schema) -> DataClass:
    if isinstance(v, Identifier):
        return _classify_identifier(v.name, schema)
    if isinstance(v, Call):
        return _classify_call(v, schema)
    raise TypeError(f"unknown Var type: {type(v)!r}")  # pragma: no cover - defensive


def underlying_column(v: Var) -> str:
    """The single data column this variable ultimately reads from, e.g.
    `poly(x, degree=2)` -> `"x"`. Used to look up dtype/levels. Raises if
    the call's first argument isn't a bare identifier (e.g. nested calls
    like `log(scale(x))`, not needed by any real survey_kit formula and
    out of scope for v1)."""
    if isinstance(v, Identifier):
        return v.name
    if isinstance(v, Call):
        args = split_args(v.raw_args)
        positional = [a for a in args if a.keyword is None]
        if not positional:
            raise ValueError(f"{v.name}(...) has no positional argument to classify")
        first = positional[0].raw_value.strip()
        if not _looks_like_bare_identifier(first):
            raise ValueError(
                f"{v.name}({v.raw_args}) does not have a bare column name as its first "
                "argument; nested transforms are not supported in v1"
            )
        return first
    raise TypeError(f"unknown Var type: {type(v)!r}")  # pragma: no cover - defensive


def referenced_columns(v: Var) -> list:
    """All positional bare-identifier arguments, e.g. `poly(x1, x2,
    degree=2)` -> `["x1", "x2"]`. Only `poly()` currently has more than one
    (R has no multivariate `bs()`/`log()`/etc.); everything else's list is
    just `[underlying_column(v)]`."""
    if isinstance(v, Identifier):
        return [v.name]
    if isinstance(v, Call):
        args = split_args(v.raw_args)
        positional = [a.raw_value.strip() for a in args if a.keyword is None]
        if not positional:
            raise ValueError(f"{v.name}(...) has no positional argument to classify")
        for text in positional:
            if not _looks_like_bare_identifier(text):
                raise ValueError(
                    f"{v.name}({v.raw_args}) does not have bare column names as its positional "
                    "arguments; nested transforms are not supported in v1"
                )
        return positional
    raise TypeError(f"unknown Var type: {type(v)!r}")  # pragma: no cover - defensive


def _looks_like_bare_identifier(text: str) -> bool:
    from ..parser.tokenizer import _IDENT_CONT, _IDENT_START

    if not text:
        return False
    if text[0] not in _IDENT_START:
        return False
    return all(c in _IDENT_CONT for c in text)


def _classify_identifier(name: str, schema: pl.Schema) -> DataClass:
    if name not in schema:
        raise KeyError(f"column {name!r} referenced in formula not found in data schema")
    return _classify_dtype(schema[name])


def _classify_dtype(dtype: pl.DataType) -> DataClass:
    if dtype == pl.String or dtype == pl.Boolean or isinstance(dtype, (pl.Categorical, pl.Enum)):
        return DataClass.FACTOR_UNORDERED
    return DataClass.NUMERIC


def _classify_call(v: Call, schema: pl.Schema) -> DataClass:
    if v.name == "factor":
        return DataClass.FACTOR_UNORDERED
    if v.name in _STRUCTURAL_ORDERED:
        return DataClass.FACTOR_ORDERED
    if v.name == "C":
        # R's C() calls as.factor(object), which preserves an already-
        # ordered factor's ordered-ness when no explicit contrast is given
        # (ordered-ness only matters for that no-override default case —
        # see dispatch/reserved.py:contrast_override). Only an explicit
        # `ordered(x)` wrapper forces ordered; plain `factor(x)` always
        # forces unordered, matching R's own `factor()`.
        underlying = _classify_identifier(underlying_column(v), schema)
        return DataClass.FACTOR_ORDERED if underlying is DataClass.FACTOR_ORDERED else DataClass.FACTOR_UNORDERED
    return DataClass.NUMERIC
