import polars as pl

from ..parser.args import split_args
from ..parser.ast_nodes import Call
from . import polars_fns  # noqa: F401 - import for registration side effects
from .reserved import contrast_override, eval_I, is_offset
from .registry import (
    Backend,
    DispatchEntry,
    STATEFUL_RESERVED,
    STRUCTURAL_RESERVED,
    UnknownFormulaFunction,
    is_registered,
    resolve,
)
from .poly_bs import (
    BSplineState,
    MultivariatePolyState,
    PolyState,
    apply_bs,
    apply_poly,
    apply_polym,
    fit_bs,
    fit_poly,
    fit_polym,
)


def polars_expr_for_call(call: Call) -> pl.Expr:
    entry = resolve(call)
    if entry.backend is not Backend.POLARS:
        raise ValueError(f"{call.name!r} is not a Polars-backed dispatch entry")
    return entry.build_expr(split_args(call.raw_args))


__all__ = [
    "Backend",
    "DispatchEntry",
    "STATEFUL_RESERVED",
    "STRUCTURAL_RESERVED",
    "UnknownFormulaFunction",
    "BSplineState",
    "MultivariatePolyState",
    "PolyState",
    "apply_bs",
    "apply_poly",
    "apply_polym",
    "fit_bs",
    "fit_poly",
    "fit_polym",
    "contrast_override",
    "eval_I",
    "is_offset",
    "is_registered",
    "resolve",
    "polars_expr_for_call",
]
