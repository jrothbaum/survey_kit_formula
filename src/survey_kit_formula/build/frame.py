"""`model.frame` equivalent: resolves each `ModelSpec` variable to either a
Polars expression (bare identifiers and dispatch-registered functions like
`log`/`sqrt`/`scale`/... — these have no need to ever leave Polars) or a
precomputed numpy array (`I()`/`poly()`/`bs()`/`ns()` — the only transforms
that need fit-time numpy/scipy state with no Polars-expression equivalent).
Factor variables become small per-level lookup tables, joined against the
main data in `build/matrix.py` rather than resolved to integer codes here.

Level/null validation happens eagerly, up front, independent of how a
factor ends up materialized (join or otherwise): `pl.Enum` cast raises on
a value outside the factor's known levels, exactly the "new data has a
level never seen at fit time" error R would also raise. Only variables
that had nulls in the *training* data (`spec.null_companions`, set by
`null_dummy=True` at `ModelSpec.from_formula` time) tolerate nulls here. A
variable with no companion column allocated still raises on any null, at
any point — the spec's structure is fixed at fit time, so there's nowhere
for an unexpected null to go.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import polars as pl

from ..dispatch import apply_bs, apply_ns, apply_poly, eval_I, polars_expr_for_call
from ..dispatch.poly_bs import MultivariatePolyState, apply_polym
from ..parser.ast_nodes import Call, Identifier, Var
from ..terms.classify import referenced_columns, underlying_column
from ..terms.spec import FactorSpec, ModelSpec


def _check_null(series: pl.Series, allow_null: bool, label: str) -> None:
    if series.null_count() > 0 and not allow_null:
        raise ValueError(
            f"{label} contains nulls, but had none when this ModelSpec was fit "
            "(and null_dummy wasn't used) -- no companion indicator column was allocated for it"
        )


def validate_factor_values(df: pl.DataFrame, fs: FactorSpec, allow_null: bool) -> None:
    """Raises on either an unexpected null or a level outside `fs.levels`
    -- checked up front, independent of materialization strategy, since a
    join silently produces a null for an unmatched key rather than raising
    the way R (and this project's own contract) requires."""
    series = df[fs.column]
    _check_null(series, allow_null, f"column {fs.column!r}")
    if df.schema[fs.column] == pl.Boolean:
        return  # only False/True are possible; always a known level
    str_levels = [str(l) for l in fs.levels]
    try:
        series.cast(pl.String).cast(pl.Enum(str_levels))
    except pl.exceptions.InvalidOperationError as e:
        raise ValueError(
            f"column {fs.column!r} has a level not seen when the model spec was fit "
            f"(known levels: {fs.levels}): {e}"
        ) from e


def compute_numpy_only_numerics(df: pl.DataFrame, spec: ModelSpec) -> Dict[Var, np.ndarray]:
    """`I()`/`poly()`/`bs()`/`ns()` -- the only numeric transforms without
    a native Polars expression, because they need fit-time numpy/scipy
    state (QR coefficients, spline knots) or arbitrary numpy-eval text.
    Everything else (bare identifiers, log/sqrt/scale/... dispatch
    functions) is built as a Polars expression directly in
    `build/matrix.py` instead, and never touches numpy at all."""
    companions = set(spec.null_companions)
    out: Dict[Var, np.ndarray] = {}
    for var, ns in spec.numerics.items():
        if isinstance(var, Identifier):
            continue
        assert isinstance(var, Call)
        if var.name not in ("I", "poly", "bs", "ns"):
            continue
        allow_null = var in companions
        null_fill = spec.null_fill

        if var.name == "I":
            namespace = {
                name: df[name].cast(pl.Float64).to_numpy()
                for name in df.columns
                if df.schema[name].is_numeric()
            }
            arr = eval_I(var, namespace)
            out[var] = np.asarray(arr, dtype=np.float64).reshape(-1, 1)
        elif var.name == "poly":
            cols = referenced_columns(var)
            for col in cols:
                _check_null(df[col], allow_null, f"poly({var.raw_args})")
            if isinstance(ns.poly_state, MultivariatePolyState):
                xs = [df[c].cast(pl.Float64).fill_null(null_fill).to_numpy() for c in cols]
                out[var] = apply_polym(xs, ns.poly_state)
            else:
                x = df[cols[0]].cast(pl.Float64).fill_null(null_fill).to_numpy()
                out[var] = apply_poly(x, ns.poly_state)
        elif var.name == "bs":
            col = underlying_column(var)
            _check_null(df[col], allow_null, f"bs({var.raw_args})")
            x = df[col].cast(pl.Float64).fill_null(null_fill).to_numpy()
            out[var] = apply_bs(x, ns.bs_state)
        elif var.name == "ns":
            col = underlying_column(var)
            _check_null(df[col], allow_null, f"ns({var.raw_args})")
            x = df[col].cast(pl.Float64).fill_null(null_fill).to_numpy()
            out[var] = apply_ns(x, ns.ns_state)
    return out


def numeric_var_expr(var: Var, df: pl.DataFrame, allow_null: bool, null_fill: float) -> pl.Expr:
    """Polars-expression form for the variables that have one: bare
    identifiers and dispatch-registered function calls (log/sqrt/scale/
    sin/...). Stays lazy all the way to the build's single final
    `.collect()` -- no numpy round-trip for these at all, which covers
    most of what a typical formula actually does."""
    if isinstance(var, Identifier):
        _check_null(df[var.name], allow_null, f"column {var.name!r}")
        return pl.col(var.name).cast(pl.Float64).fill_null(null_fill)
    assert isinstance(var, Call)
    col = underlying_column(var)
    _check_null(df[col], allow_null, f"{var.name}({var.raw_args})")
    return polars_expr_for_call(var).fill_null(null_fill)


def factor_lookup_table(fs: FactorSpec, base: np.ndarray, col_names: List[str]) -> pl.DataFrame:
    """One row per known level, columns named `col_names` holding this
    factor's block under whichever coding `base` represents (`eye(k)` for
    full dummy, `fs.contrast_matrix` for contrasts) -- joined against the
    main data on the factor's own column in `build/matrix.py` to
    materialize its per-row block without ever computing an integer code
    array or fancy-indexing into a (n_levels, ...) matrix per row.

    Takes the final column names directly rather than building with
    placeholder names and renaming afterward -- a `DataFrame.rename()`
    call turned out to route through its own lazy-collect round trip
    (traced empirically), a fixed ~15-20ms tax paid regardless of this
    table's size (always tiny, one row per level) that's pure waste to
    pay at all."""
    cols: Dict[str, list] = {"__pf_level": list(fs.levels)}
    for j, name in enumerate(col_names):
        cols[name] = base[:, j].tolist()
    return pl.DataFrame(cols)


def fs_or_ns_columns(var: Var, spec: ModelSpec) -> List[str]:
    if var in spec.factors:
        return [spec.factors[var].column]
    if isinstance(var, Identifier):
        return [var.name]
    return referenced_columns(var)
