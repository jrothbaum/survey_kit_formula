"""`ModelSpec` — R's `terms` object equivalent: a compact, reusable plan
built once from (formula, training data). Structure (marginality decisions,
resolved contrasts, factor levels, `poly()`/`bs()` basis state) is fixed at
`from_formula()` time and independent of row count, so it can be reapplied
to new data via `get_model_matrix()` (Phase 8) without re-deriving anything
— matching survey_kit's actual `model_spec.get_model_matrix(new_df)` usage.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import polars as pl

from ..contrasts.base import CONTRAST_FUNCTIONS, default_contrast
from ..dispatch.poly_bs import (
    BSplineState,
    MultivariatePolyState,
    NaturalSplineState,
    PolyState,
    fit_bs,
    fit_ns,
    fit_poly,
    fit_polym,
)
from ..dispatch.registry import UnknownFormulaFunction, is_registered
from ..dispatch.reserved import contrast_override, is_offset
from ..parser.args import split_args
from ..parser.ast_nodes import Call, Identifier, Term, TermList, Var, var_term
from ..parser.parser import parse_formula
from .classify import DataClass, classify_var, referenced_columns, underlying_column
from .marginality import Coding, compute_marginality

_TRUE_STRINGS = {"true", "t"}


@dataclass
class FactorSpec:
    var: Var
    column: str
    levels: List[object]
    ordered: bool
    contrast_matrix: np.ndarray  # (n_levels, n_levels - 1)


@dataclass
class NumericSpec:
    var: Var
    width: int
    poly_state: Optional[Union[PolyState, MultivariatePolyState]] = None
    bs_state: Optional[BSplineState] = None
    ns_state: Optional[NaturalSplineState] = None


@dataclass
class PlannedTerm:
    term: Term
    coding: Dict[Var, Coding]
    n_columns: int


@dataclass
class ModelSpec:
    formula: str
    lhs: Optional[Var]
    intercept: bool
    terms: List[PlannedTerm]
    factors: Dict[Var, FactorSpec]
    numerics: Dict[Var, NumericSpec]
    offsets: List[Var]
    total_columns: int
    null_dummy: bool = False
    null_fill: float = 0.0
    null_companions: List[Var] = field(default_factory=list)

    @classmethod
    def from_formula(
        cls,
        formula: str,
        df: Union[pl.DataFrame, pl.LazyFrame],
        *,
        null_dummy: bool = False,
        null_fill: float = 0.0,
    ) -> "ModelSpec":
        """`null_dummy=True` opts into treating nulls as data rather than an
        error: any factor or numeric variable that has nulls in `df` gets
        filled to `null_fill` (default 0) plus a companion 0/1 "was this
        null" indicator column appended after the regular terms. Variables
        with no nulls in `df` are unaffected and stay strict — a null
        showing up for one of *those* at `get_model_matrix()` time still
        raises, since no companion column was allocated for it and the
        spec's structure (`total_columns`) is fixed at fit time. Default
        (`null_dummy=False`) matches the original strict behavior: any null
        anywhere raises."""
        schema = df.collect_schema() if isinstance(df, pl.LazyFrame) else df.schema
        if isinstance(df, pl.LazyFrame):
            df = df.collect()

        parsed = parse_formula(formula)
        rhs = _expand_dot(parsed.rhs, parsed.lhs, schema)
        rhs, offset_vars = _extract_offsets(rhs)

        marg = compute_marginality(rhs, schema, parsed.intercept)

        factors: Dict[Var, FactorSpec] = {}
        numerics: Dict[Var, NumericSpec] = {}
        null_companions: List[Var] = []
        for v, dc in marg.dataclasses.items():
            if dc.is_factor:
                fs, has_nulls = _build_factor_spec(v, dc, df, null_dummy)
                factors[v] = fs
            else:
                ns, has_nulls = _build_numeric_spec(v, df, null_dummy, null_fill)
                numerics[v] = ns
            if has_nulls:
                null_companions.append(v)

        planned_terms: List[PlannedTerm] = [
            PlannedTerm(term=plan.term, coding=plan.coding, n_columns=_term_columns(plan, factors, numerics))
            for plan in marg.term_plans
        ]

        total = (1 if parsed.intercept else 0) + sum(t.n_columns for t in planned_terms) + len(null_companions)

        return cls(
            formula=formula,
            lhs=parsed.lhs,
            intercept=parsed.intercept,
            terms=planned_terms,
            factors=factors,
            numerics=numerics,
            offsets=offset_vars,
            total_columns=total,
            null_dummy=null_dummy,
            null_fill=null_fill,
            null_companions=null_companions,
        )

    def get_model_matrix(self, df: Union[pl.DataFrame, pl.LazyFrame]) -> np.ndarray:
        """Reapply this spec's fixed structure (marginality decisions,
        contrasts, factor levels, poly()/bs() basis state) to `df` — the
        `model_spec.get_model_matrix(new_df)` pattern. The `build` import is
        deferred to call time (not module top-level) because `build`
        depends on `terms` (this module) for `ModelSpec`/`FactorSpec`/etc.,
        so importing it eagerly here would be circular."""
        from ..build.matrix import build_model_matrix

        if isinstance(df, pl.LazyFrame):
            df = df.collect()
        return build_model_matrix(self, df)


def _expand_dot(rhs: TermList, lhs: Optional[Var], schema: pl.Schema) -> TermList:
    dot_term = None
    for t in rhs:
        if t.order == 1:
            v = next(iter(t.vars))
            if isinstance(v, Identifier) and v.name == ".":
                dot_term = t
                break
    if dot_term is None:
        return rhs

    referenced = set()
    for t in rhs:
        for v in t.vars:
            if isinstance(v, Identifier):
                if v.name != ".":
                    referenced.add(v.name)
            elif isinstance(v, Call):
                try:
                    referenced.add(underlying_column(v))
                except ValueError:
                    pass
    if isinstance(lhs, Identifier):
        referenced.add(lhs.name)

    remaining = [name for name in schema.keys() if name not in referenced]

    result = TermList()
    for t in rhs:
        if t is dot_term:
            for name in remaining:
                result.add(var_term(Identifier(name)))
        else:
            result.add(t)
    return result


def _extract_offsets(rhs: TermList):
    remaining = TermList()
    offsets: List[Var] = []
    for t in rhs:
        if t.order == 1:
            v = next(iter(t.vars))
            if isinstance(v, Call) and is_offset(v):
                offsets.append(v)
                continue
        remaining.add(t)
    return remaining, offsets


def _build_factor_spec(v: Var, dc: DataClass, df: pl.DataFrame, null_dummy: bool) -> Tuple[FactorSpec, bool]:
    col = underlying_column(v)
    dtype = df.schema[col]
    explicit_levels, scores = _parse_levels_and_scores(v)

    has_nulls = df[col].null_count() > 0
    if has_nulls and not null_dummy:
        raise ValueError(
            f"column {col!r} contains nulls; pass null_dummy=True to ModelSpec.from_formula "
            "to fill them (default 0) with a companion null-indicator column, instead of erroring"
        )

    if explicit_levels is not None:
        if dtype != pl.Boolean:
            non_null = df[col].cast(pl.String).drop_nulls()
            outside = non_null.filter(~non_null.is_in(explicit_levels)).unique().to_list()
            if outside:
                raise ValueError(
                    f"column {col!r} has values not present in the given levels={explicit_levels!r}: "
                    f"{outside!r}. Explicit levels must cover every observed value -- use "
                    "null_dummy=True separately for actual missing data, not to paper over a mismatch."
                )
        levels = list(explicit_levels)
    else:
        levels = _extract_levels(df, col, dtype)

    ordered = dc is DataClass.FACTOR_ORDERED
    n = len(levels)
    if n < 2:
        raise ValueError(f"column {col!r} has fewer than 2 levels; cannot build contrasts")

    contrast_name: Optional[str] = None
    contrast_kwargs: Dict[str, int] = {}
    if isinstance(v, Call) and v.name == "C":
        override = contrast_override(v)
        if override is not None:
            contrast_name, raw_kwargs = override
            contrast_kwargs = {k: int(val) for k, val in raw_kwargs.items()}

    if contrast_name is not None:
        matrix = CONTRAST_FUNCTIONS[contrast_name](n, **contrast_kwargs)
    else:
        matrix = default_contrast(n, ordered, scores=scores)

    return FactorSpec(var=v, column=col, levels=levels, ordered=ordered, contrast_matrix=matrix), has_nulls


def _extract_levels(df: pl.DataFrame, col: str, dtype: pl.DataType) -> List[object]:
    if isinstance(dtype, pl.Enum):
        return list(dtype.categories)
    if dtype == pl.Boolean:
        return [False, True]
    return df[col].cast(pl.String).drop_nulls().unique().sort().to_list()


def _parse_levels_and_scores(v: Var) -> Tuple[Optional[List[str]], Optional[List[float]]]:
    """`factor(x, levels=[...])` / `ordered(x, levels=[...], scores=[...])`
    -- Python list-literal syntax (parsed with `ast.literal_eval`, not a
    general `eval`), matching R's `factor()`/`ordered()`/`contr.poly()`
    parameters of the same names. `scores` only makes sense for
    `ordered()` (it feeds `contr.poly`'s scores, R's own restriction)."""
    if not isinstance(v, Call) or v.name not in ("factor", "ordered"):
        return None, None
    _positional, kwargs = _split_call_args(v)

    levels: Optional[List[str]] = None
    if "levels" in kwargs:
        levels = [str(x) for x in _literal_eval_list(v, "levels", kwargs["levels"])]

    scores: Optional[List[float]] = None
    if "scores" in kwargs:
        if v.name != "ordered":
            raise ValueError(f"{v.name}({v.raw_args}): 'scores' is only meaningful for ordered(...)")
        scores = [float(x) for x in _literal_eval_list(v, "scores", kwargs["scores"])]

    return levels, scores


def _literal_eval_list(v: Call, arg_name: str, raw: str) -> list:
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"{v.name}({v.raw_args}): could not parse {arg_name}={raw!r} as a Python list: {e}") from e
    if not isinstance(parsed, (list, tuple)):
        raise ValueError(f"{v.name}({v.raw_args}): {arg_name}= must be a list, got {raw!r}")
    return list(parsed)


def _build_numeric_spec(
    v: Var, df: pl.DataFrame, null_dummy: bool, null_fill: float
) -> Tuple[NumericSpec, bool]:
    if isinstance(v, Identifier):
        has_nulls = _check_numeric_nulls(v.name, df, null_dummy)
        return NumericSpec(var=v, width=1), has_nulls

    assert isinstance(v, Call)
    if v.name == "poly":
        return _build_poly_spec(v, df, null_dummy, null_fill)
    if v.name == "bs":
        return _build_bs_spec(v, df, null_dummy, null_fill)
    if v.name == "ns":
        return _build_ns_spec(v, df, null_dummy, null_fill)
    if v.name == "I":
        # I()'s raw text can reference multiple columns arbitrarily, so
        # there's no single underlying column to null-check generically --
        # out of scope for null_dummy in v1. A null column referenced
        # inside I(...) still just propagates NaN silently, unchanged from
        # before.
        return NumericSpec(var=v, width=1), False
    if not is_registered(v.name):
        raise UnknownFormulaFunction(f"{v.name!r} is not a recognized formula function")
    has_nulls = _check_numeric_nulls(underlying_column(v), df, null_dummy, context=f"{v.name}({v.raw_args})")
    return NumericSpec(var=v, width=1), has_nulls


def _check_numeric_nulls(col: str, df: pl.DataFrame, null_dummy: bool, context: Optional[str] = None) -> bool:
    has_nulls = df[col].null_count() > 0
    if has_nulls and not null_dummy:
        label = context or f"column {col!r}"
        raise ValueError(
            f"{label} contains nulls; pass null_dummy=True to ModelSpec.from_formula "
            "to fill them (default 0) with a companion null-indicator column, instead of erroring"
        )
    return has_nulls


def _split_call_args(v: Call):
    args = split_args(v.raw_args)
    positional = [a for a in args if a.keyword is None]
    kwargs = {a.keyword: a.raw_value.strip() for a in args if a.keyword is not None}
    return positional, kwargs


def _as_bool(text: str) -> bool:
    return text.strip().lower() in _TRUE_STRINGS or text.strip() == "TRUE"


def _build_poly_spec(v: Call, df: pl.DataFrame, null_dummy: bool, null_fill: float) -> Tuple[NumericSpec, bool]:
    """`poly(x, degree=D)` (univariate) or `poly(x1, x2, ..., degree=D)` /
    R's `polym()` (multivariate, >=2 positional columns) -- R dispatches on
    argument count at call time (`poly`'s own source: if `...` holds more
    than a lone scalar degree, it delegates to `polym`); we dispatch the
    same way, except `degree` is always required as a keyword in v1, so
    there's no "positional degree vs. second variable" ambiguity to
    resolve the way R's own dispatch has to."""
    positional, kwargs = _split_call_args(v)
    if not positional:
        raise ValueError(f"poly({v.raw_args}) needs at least one column as a positional argument")
    if "degree" not in kwargs:
        raise ValueError(f"poly({v.raw_args}): 'degree' must be given as a keyword argument in v1")
    cols = [p.raw_value.strip() for p in positional]
    degree = int(kwargs["degree"])
    raw = _as_bool(kwargs["raw"]) if "raw" in kwargs else False

    has_nulls = False
    for col in cols:
        if _check_numeric_nulls(col, df, null_dummy, context=f"poly({v.raw_args})"):
            has_nulls = True

    if len(cols) == 1:
        x = df[cols[0]].cast(pl.Float64).fill_null(null_fill).to_numpy()
        _, state = fit_poly(x, degree=degree, raw=raw)
        return NumericSpec(var=v, width=degree, poly_state=state), has_nulls

    xs = [df[c].cast(pl.Float64).fill_null(null_fill).to_numpy() for c in cols]
    fitted, mstate = fit_polym(xs, degree=degree, raw=raw)
    return NumericSpec(var=v, width=fitted.shape[1], poly_state=mstate), has_nulls


def _knots_kwarg(v: Call, kwargs: Dict[str, str]) -> Optional[List[float]]:
    if "knots" not in kwargs:
        return None
    return [float(k) for k in _literal_eval_list(v, "knots", kwargs["knots"])]


def _build_bs_spec(v: Call, df: pl.DataFrame, null_dummy: bool, null_fill: float) -> Tuple[NumericSpec, bool]:
    positional, kwargs = _split_call_args(v)
    if not positional:
        raise ValueError(f"bs({v.raw_args}) needs a column as its first argument")
    col = positional[0].raw_value.strip()
    has_nulls = _check_numeric_nulls(col, df, null_dummy, context=f"bs({v.raw_args})")
    df_arg = int(kwargs["df"]) if "df" in kwargs else None
    knots = _knots_kwarg(v, kwargs)
    degree = int(kwargs.get("degree", 3))
    intercept = _as_bool(kwargs["intercept"]) if "intercept" in kwargs else False
    x = df[col].cast(pl.Float64).fill_null(null_fill).to_numpy()
    fitted, state = fit_bs(x, df=df_arg, knots=knots, degree=degree, intercept=intercept)
    return NumericSpec(var=v, width=fitted.shape[1], bs_state=state), has_nulls


def _build_ns_spec(v: Call, df: pl.DataFrame, null_dummy: bool, null_fill: float) -> Tuple[NumericSpec, bool]:
    positional, kwargs = _split_call_args(v)
    if not positional:
        raise ValueError(f"ns({v.raw_args}) needs a column as its first argument")
    col = positional[0].raw_value.strip()
    has_nulls = _check_numeric_nulls(col, df, null_dummy, context=f"ns({v.raw_args})")
    df_arg = int(kwargs["df"]) if "df" in kwargs else None
    knots = _knots_kwarg(v, kwargs)
    intercept = _as_bool(kwargs["intercept"]) if "intercept" in kwargs else False
    x = df[col].cast(pl.Float64).fill_null(null_fill).to_numpy()
    fitted, state = fit_ns(x, df=df_arg, knots=knots, intercept=intercept)
    return NumericSpec(var=v, width=fitted.shape[1], ns_state=state), has_nulls


def _term_columns(plan, factors: Dict[Var, FactorSpec], numerics: Dict[Var, NumericSpec]) -> int:
    total = 1
    for v in plan.term.vars:
        if v in factors:
            k = len(factors[v].levels)
            total *= k if plan.coding[v] is Coding.DUMMY else (k - 1)
        else:
            total *= numerics[v].width
    return total
