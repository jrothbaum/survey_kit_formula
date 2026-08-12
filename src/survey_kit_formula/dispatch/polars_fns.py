"""Value dispatch-table entries that stay entirely inside Polars-expression
land — the result is a `pl.Expr`, so extraction later is a free
`.to_numpy(zero_copy_only=True)` rather than an eager numpy computation.

A closed list: what real survey_kit formulas use (`log`, `scale`, `center`)
plus R's elementary math functions (`sin`, `cos`, `abs`, ...), all of which
have a direct Polars `Expr` method. Anything not registered here (or in
`reserved.py`) is a parse-time error — see `registry.py`.
"""

from __future__ import annotations

from typing import List

import polars as pl

from ..parser.args import Arg
from ..parser.tokenizer import FormulaSyntaxError, _IDENT_CONT, _IDENT_START
from .registry import register_polars


def _single_column(args: List[Arg], fn_name: str) -> str:
    positional = [a for a in args if a.keyword is None]
    if len(positional) != 1 or len(args) != 1:
        raise FormulaSyntaxError(
            f"{fn_name}(...) expects exactly one positional column argument, got {args!r}"
        )
    text = positional[0].raw_value.strip()
    if not text or text[0] not in _IDENT_START or not all(c in _IDENT_CONT for c in text):
        raise FormulaSyntaxError(f"{fn_name}(...) expects a bare column name, got {text!r}")
    return text


def _log(args: List[Arg]) -> pl.Expr:
    return pl.col(_single_column(args, "log")).log()


def _log1p(args: List[Arg]) -> pl.Expr:
    return pl.col(_single_column(args, "log1p")).log1p()


def _sqrt(args: List[Arg]) -> pl.Expr:
    return pl.col(_single_column(args, "sqrt")).sqrt()


def _exp(args: List[Arg]) -> pl.Expr:
    return pl.col(_single_column(args, "exp")).exp()


def _scale(args: List[Arg]) -> pl.Expr:
    col = pl.col(_single_column(args, "scale"))
    return (col - col.mean()) / col.std()


def _center(args: List[Arg]) -> pl.Expr:
    col = pl.col(_single_column(args, "center"))
    return col - col.mean()


def _unary_math(fn_name: str, expr_method: str):
    def build(args: List[Arg]) -> pl.Expr:
        return getattr(pl.col(_single_column(args, fn_name)), expr_method)()

    return build


register_polars("log", _log)
register_polars("log1p", _log1p)
register_polars("sqrt", _sqrt)
register_polars("exp", _exp)
register_polars("scale", _scale)
register_polars("center", _center)

# Elementary math functions, matching R's own names (asin/acos/atan and
# ceiling, not numpy/Polars' arcsin/arccos/arctan and ceil) -- each just
# wraps the equivalent Polars Expr method, so these stay lazy too.
for _r_name, _polars_method in [
    ("sin", "sin"),
    ("cos", "cos"),
    ("tan", "tan"),
    ("asin", "arcsin"),
    ("acos", "arccos"),
    ("atan", "arctan"),
    ("sinh", "sinh"),
    ("cosh", "cosh"),
    ("tanh", "tanh"),
    ("abs", "abs"),
    ("floor", "floor"),
    ("ceiling", "ceil"),
    ("sign", "sign"),
]:
    register_polars(_r_name, _unary_math(_r_name, _polars_method))
