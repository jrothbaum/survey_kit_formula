"""`model.matrix`/`model.frame` equivalent. `build_model_frame` is the
primary builder, and it builds the *entire* result as a single lazy Polars
query -- one `.select()` of aliased, dtype-cast expressions, collected
exactly once. `build_model_matrix` is just `build_model_frame(...).to_numpy()`
— Polars' own upcast-to-common-dtype conversion, matching R's own
`model.matrix`, which is always a plain double matrix.

Factor variables never become an integer code array fancy-indexed into a
contrast matrix (the previous design, and formulaic/patsy's usual
approach). Instead, each distinct (factor, coding) pair used anywhere in
the spec gets a small per-level lookup table (`build/frame.py`'s
`factor_lookup_table`, `n_levels` rows) `.join()`ed against the data
*once*; every term that needs that factor's block then just references
the joined columns by name. Numeric identifiers and dispatch-registered
functions (`log`, `sqrt`, `scale`, ...) are plain Polars expressions the
whole way through. Interactions are elementwise-multiplication
expressions between two variables' own column expressions, following the
exact same fold order as the old Kronecker product so column values and
names are unchanged. Only `poly()`/`bs()`/`ns()`/`I()` -- the transforms
with no Polars-expression equivalent -- get precomputed via numpy and
injected as literal columns before the lazy plan runs.

The payoff of collecting once, as native Polars expressions the whole
way, isn't just skipping a redundant scan (the previous fix already did
that) -- it's skipping the numpy<->Polars round trip *entirely* for most
columns. Profiled on `~1+x1+x2+x2*x1*Ahi` (a 60-level categorical crossed
three ways, 240 output columns, n=400,000): the eager fancy-indexing +
per-column-pack design took ~1.3s; this design takes ~0.08s. The gain
comes from never leaving Polars' own columnar representation to build an
intermediate numpy array in the first place -- both the interaction
products and the final dtype casts happen inside the query engine, not as
a Python loop calling into numpy and then copying the result back into
Polars column buffers one at a time.

Dtype selection is still *structural*, not a value scan over each n-row
output column, for the same reason as before: a factor's dummy/contrast
block is exactly 0/1 or another small closed set of values *by
construction* (from the (n_levels, ...) lookup table, not the n-row data),
so classifying it costs O(n_levels). A numeric variable's own column still
needs one real value scan to classify, but only once per distinct
variable (cached), not once per interaction column it ends up combined
into -- `_combine_kind` composes two variables' classifications
algebraically (bool*bool stays bool; int*int bounds via corner products)
without ever touching a materialized column.

Column names match R's own `colnames(model.matrix(...))` (verified against
real R, not assumed): a factor's own deparsed call text (or bare column
name) as a prefix, then per-column suffixes that depend on how it's coded
-- the concatenated level label for full-dummy or treatment/SAS-style
contrast columns ("aa2"), a plain 1-based index for sum/helmert/custom
contrasts ("a1", "a2"), R's ".L"/".Q"/".C"/"^4" suffixes for the default
ordered-factor contr.poly, and R's own multivariate-poly exponent-tuple
suffixes ("1.0", "1.1", ...) for `poly(x, z, degree=...)`. Interaction
columns join each variable's own label with ":", in declared term order,
regardless of which variable actually varies fastest in the underlying
fold.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Tuple

import numpy as np
import polars as pl

from ..dispatch.poly_bs import MultivariatePolyState
from ..parser.ast_nodes import Identifier, Var
from ..terms.marginality import Coding
from ..terms.spec import FactorSpec, ModelSpec, NumericSpec, PlannedTerm
from .frame import (
    _check_null,
    compute_numpy_only_numerics,
    factor_lookup_table,
    fs_or_ns_columns,
    numeric_var_expr,
    validate_factor_values,
)

_FactorKey = Tuple[Var, Coding]


def build_model_frame(spec: ModelSpec, df: pl.DataFrame) -> pl.DataFrame:
    null_companion_set = set(spec.null_companions)
    names = column_names(spec)
    counter = [0]

    def fresh(k: int) -> List[str]:
        i = counter[0]
        counter[0] += 1
        return [f"__pf_{i}_{j}" for j in range(k)]

    # Null-count checks are cheap (no collect involved) and need to raise
    # before any real work happens -- kept eager. The "value not in
    # fs.levels" check is folded into the main build's own single collect
    # instead of its own separate one (see the join loop below and the
    # post-collect check at the end of this function).
    for var, fs in spec.factors.items():
        _check_null(df[fs.column], var in null_companion_set, f"column {fs.column!r}")

    numeric_cols: Dict[Var, List[str]] = {}
    numeric_kinds: Dict[Var, List["_Kind"]] = {}

    numpy_only = compute_numpy_only_numerics(df, spec)
    if numpy_only:
        lit_cols: Dict[str, pl.Series] = {}
        for var, arr in numpy_only.items():
            cnames = fresh(arr.shape[1])
            numeric_cols[var] = cnames
            numeric_kinds[var] = [_classify_array(arr[:, j]) for j in range(arr.shape[1])]
            for j, cname in enumerate(cnames):
                lit_cols[cname] = pl.Series(cname, arr[:, j])
        df = df.with_columns(list(lit_cols.values()))

    # Bare identifiers are the only native-expression vars worth actually
    # classifying (log/sqrt/scale/sin/... essentially never come out
    # exactly integral, so those are just treated as float -- see below).
    # Classifying them individually (one `.to_numpy()` call each) each pays
    # its own dispatch/collect overhead; one batched `.select()` across all
    # of them at once is markedly cheaper (~4x, measured) for the same
    # reason the rest of this module avoids per-column round trips.
    id_vars = [v for v in spec.numerics if isinstance(v, Identifier) and v not in numeric_cols]
    id_kinds: Dict[Var, "_Kind"] = {}
    if id_vars:
        id_arr = df.select([pl.col(v.name).fill_null(spec.null_fill) for v in id_vars]).to_numpy()
        id_kinds = {v: _classify_array(id_arr[:, i]) for i, v in enumerate(id_vars)}

    native_exprs: List[pl.Expr] = []
    for var in spec.numerics:
        if var in numeric_cols:
            continue  # already handled via compute_numpy_only_numerics
        allow_null = var in null_companion_set
        cname = fresh(1)[0]
        numeric_cols[var] = [cname]
        numeric_kinds[var] = [id_kinds[var]] if var in id_kinds else [_FLOAT_KIND]
        native_exprs.append(numeric_var_expr(var, df, allow_null, spec.null_fill).alias(cname))

    lf = df.lazy()
    if native_exprs:
        lf = lf.with_columns(native_exprs)

    factor_cols: Dict[_FactorKey, List[str]] = {}
    factor_kinds: Dict[_FactorKey, List["_Kind"]] = {}
    # (flag column name, FactorSpec, allow_null) -- one per non-boolean
    # factor actually joined, checked once against the main build's own
    # result below instead of via a separate up-front validation collect.
    invalid_flags: List[Tuple[str, FactorSpec, bool]] = []
    for planned in spec.terms:
        for var in planned.term.vars:
            if var not in spec.factors:
                continue
            key = (var, planned.coding[var])
            if key in factor_cols:
                continue
            fs = spec.factors[var]
            base = np.eye(len(fs.levels)) if planned.coding[var] is Coding.DUMMY else fs.contrast_matrix
            cnames = fresh(base.shape[1])
            lookup = factor_lookup_table(fs, base, cnames)
            factor_cols[key] = cnames
            factor_kinds[key] = _factor_kinds(fs, base, var in null_companion_set)

            if df.schema[fs.column] == pl.Boolean:
                lf = lf.join(lookup.lazy(), left_on=fs.column, right_on="__pf_level", how="left", maintain_order="left")
            else:
                key_col = fresh(1)[0]
                lf = lf.with_columns(pl.col(fs.column).cast(pl.String).alias(key_col))
                lf = lf.join(lookup.lazy(), left_on=key_col, right_on="__pf_level", how="left", maintain_order="left")
                # A left-joined row with no match is either a genuine null
                # key (fine, zeroed out below) or a value outside
                # `fs.levels` (an error) -- the join alone can't tell
                # these apart, so flag "non-null key, no match" here,
                # before the fill_null below erases the distinction, and
                # check it against the main build's own result rather
                # than paying for a separate validation collect first.
                # (Boolean columns skip this: every non-null bool value is
                # a known level by construction.)
                flag_name = fresh(1)[0]
                lf = lf.with_columns((pl.col(fs.column).is_not_null() & pl.col(cnames[0]).is_null()).alias(flag_name))
                invalid_flags.append((flag_name, fs, var in null_companion_set))
            # Zero out any unmatched row -- correct for a genuine null key
            # (the null_dummy contract) and harmless for an invalid-value
            # key (already flagged above; the whole build raises before
            # this value would ever be returned to the caller).
            lf = lf.with_columns([pl.col(c).fill_null(0.0) for c in cnames])

    final_exprs: List[pl.Expr] = []
    idx = 0
    if spec.intercept:
        final_exprs.append(pl.lit(1, dtype=pl.Int8).alias(names[idx]))
        idx += 1

    for planned in spec.terms:
        cols: List[pl.Expr] = None
        kinds: List["_Kind"] = None
        for var in planned.term.vars:
            if var in spec.factors:
                key = (var, planned.coding[var])
                vcols = [pl.col(c) for c in factor_cols[key]]
                vkinds = factor_kinds[key]
            else:
                vcols = [pl.col(c) for c in numeric_cols[var]]
                vkinds = numeric_kinds[var]
            # New variable goes first (the "slow"/outer operand) and the
            # running block second (the "fast"/inner operand): R's
            # convention is that the *first*-declared variable in a term
            # varies fastest across the interaction's columns (confirmed
            # against real R -- e.g. `a:b` produces a1:b1, a2:b1, a1:b2,
            # a2:b2, ... not a1:b1, a1:b2, ...). Getting this backwards
            # still spans the same column space, just permuted, which is
            # exactly the kind of bug that only shows up as a numeric
            # mismatch against R, never as a shape or rank difference.
            if cols is None:
                cols, kinds = vcols, vkinds
            else:
                cols = [nc * oc for nc in vcols for oc in cols]
                kinds = [_combine_kind(nk, ok) for nk in vkinds for ok in kinds]
        term_names = names[idx : idx + len(cols)]
        idx += len(cols)
        final_exprs.extend(_cast_expr(c, k).alias(name) for c, k, name in zip(cols, kinds, term_names))

    for var in spec.null_companions:
        src_cols = fs_or_ns_columns(var, spec)
        expr = pl.col(src_cols[0]).is_null()
        for c in src_cols[1:]:
            expr = expr | pl.col(c).is_null()
        final_exprs.append(expr.alias(names[idx]))
        idx += 1

    flag_names = [f for f, _fs, _allow in invalid_flags]
    result = lf.select(final_exprs + [pl.col(f) for f in flag_names]).collect()
    if flag_names and result.select(pl.any_horizontal([pl.col(f).any() for f in flag_names])).item():
        # Something's invalid -- fall back to the slower precise per-column
        # check purely to produce an exact, specific error message; the
        # fast path above already paid for detecting *that* something's
        # wrong, cheaply, as part of the one collect every build pays for
        # regardless.
        for _flag, fs, allow_null in invalid_flags:
            validate_factor_values(df, fs, allow_null)
        raise AssertionError("internal: a validity flag was set but no factor failed re-validation")  # pragma: no cover
    return result.drop(flag_names) if flag_names else result


def build_model_matrix(spec: ModelSpec, df: pl.DataFrame) -> np.ndarray:
    return build_model_frame(spec, df).to_numpy()


@dataclass(frozen=True)
class _Kind:
    """A column's dtype classification, cheap to compute and cheap to
    combine across a fold without ever touching the full n-row result.
    `bool_ok=True` implies `int_ok=True` (0/1 is a whole number)."""

    bool_ok: bool
    int_ok: bool
    lo: float = 0.0
    hi: float = 0.0


_FLOAT_KIND = _Kind(bool_ok=False, int_ok=False)


def _classify_array(col: np.ndarray) -> _Kind:
    """The one real value scan in this module -- used only on a factor's
    small (n_levels, ...) contrast matrix (cheap regardless of row count)
    and once per distinct numeric variable's own column (cached, so a
    variable referenced by several terms is scanned only once)."""
    if np.isin(col, (0.0, 1.0)).all():
        return _Kind(bool_ok=True, int_ok=True, lo=0.0, hi=1.0)
    if np.isfinite(col).all() and np.array_equal(col, np.round(col)):
        return _Kind(bool_ok=False, int_ok=True, lo=float(col.min()), hi=float(col.max()))
    return _FLOAT_KIND


def _combine_kind(a: "_Kind", b: "_Kind") -> "_Kind":
    """Classification of the elementwise product of two columns, from
    their own classifications alone -- 0/1 times 0/1 is always 0/1; two
    integral columns' product is bounded (not always tightly) by the
    corner products of their own bounds, which is enough to pick a safe
    smallest int width without inspecting the actual product values."""
    if not (a.int_ok and b.int_ok):
        return _FLOAT_KIND
    if a.bool_ok and b.bool_ok:
        return _Kind(bool_ok=True, int_ok=True, lo=0.0, hi=1.0)
    corners = (a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi)
    return _Kind(bool_ok=False, int_ok=True, lo=min(corners), hi=max(corners))


def _factor_kinds(fs: FactorSpec, base: np.ndarray, has_null_companion: bool) -> List["_Kind"]:
    kinds = []
    for j in range(base.shape[1]):
        col = base[:, j]
        if has_null_companion:
            # a null/unmatched join row is filled to exactly 0.0, which
            # can introduce a value not otherwise present in this
            # contrast column (e.g. a helmert/sum column that never
            # naturally takes 0).
            col = np.append(col, 0.0)
        kinds.append(_classify_array(col))
    return kinds


_PL_INT_DTYPES: Tuple[Tuple[pl.DataType, int, int], ...] = (
    (pl.Int8, np.iinfo(np.int8).min, np.iinfo(np.int8).max),
    (pl.Int16, np.iinfo(np.int16).min, np.iinfo(np.int16).max),
    (pl.Int32, np.iinfo(np.int32).min, np.iinfo(np.int32).max),
    (pl.Int64, np.iinfo(np.int64).min, np.iinfo(np.int64).max),
)


def _cast_expr(expr: pl.Expr, kind: "_Kind") -> pl.Expr:
    if kind.bool_ok:
        return expr.cast(pl.Boolean)
    if kind.int_ok:
        for pl_dtype, lo, hi in _PL_INT_DTYPES:
            if kind.lo >= lo and kind.hi <= hi:
                return expr.cast(pl_dtype)
        return expr.cast(pl.Int64)
    return expr


_EQ_RE = re.compile(r"\s*=\s*")
_COMMA_RE = re.compile(r"\s*,\s*")


def _normalize_call_args(raw_args: str) -> str:
    """R's colnames use the *deparsed* call (`degree = 3`, `, ` between
    args), not necessarily the literal source text (`degree=3`) the user
    typed -- approximated here without a full R-grammar-aware deparser."""
    text = _EQ_RE.sub(" = ", raw_args)
    return _COMMA_RE.sub(", ", text)


def _deparse(var: Var) -> str:
    """The prefix every column name for this variable is built from --
    "a", "C(a, base = 2)", "poly(x, degree = 3)" -- matching R's own use of
    the term's deparsed call text (not just the underlying column name) as
    the naming prefix."""
    if isinstance(var, Identifier):
        return var.name
    return f"{var.name}({_normalize_call_args(var.raw_args)})"


_POLY_SUFFIXES = {1: ".L", 2: ".Q", 3: ".C"}


def _poly_suffix(pos_1indexed: int) -> str:
    return _POLY_SUFFIXES.get(pos_1indexed, f"^{pos_1indexed}")


def _level_label(level: object) -> str:
    # R's factor levels stringify as-is, but a logical column's levels
    # print as "TRUE"/"FALSE" (R's own capitalization), not Python's
    # "True"/"False".
    if level is True:
        return "TRUE"
    if level is False:
        return "FALSE"
    return str(level)


def _is_one_hot_columns(cm: np.ndarray) -> bool:
    """True iff every column has exactly one 1.0 and is otherwise all
    0.0 -- the structural signature of contr.treatment/contr.SAS (a
    column-deleted identity matrix) regardless of `base=`, distinguishing
    them from contr.sum/contr.helmert (which always contain a -1) without
    needing to special-case `base=` separately."""
    if cm.size == 0:
        return True
    if not np.isin(cm, (0.0, 1.0)).all():
        return False
    return bool(((cm == 1.0).sum(axis=0) == 1).all())


def _var_labels(var: Var, coding: Dict[Var, Coding], factors: Dict[Var, FactorSpec], numerics: Dict[Var, NumericSpec]) -> List[str]:
    if var in factors:
        fs = factors[var]
        prefix = _deparse(var)
        if coding[var] is Coding.DUMMY:
            return [f"{prefix}{_level_label(level)}" for level in fs.levels]
        cm = fs.contrast_matrix
        if fs.contrast_name == "contr.poly":
            return [f"{prefix}{_poly_suffix(j + 1)}" for j in range(cm.shape[1])]
        if _is_one_hot_columns(cm):
            kept_levels = [int(np.argmax(cm[:, j])) for j in range(cm.shape[1])]
            return [f"{prefix}{_level_label(fs.levels[i])}" for i in kept_levels]
        # contr.sum/contr.helmert/other custom contrasts: no per-column
        # level correspondence, so R itself falls back to a plain index.
        return [f"{prefix}{j + 1}" for j in range(cm.shape[1])]

    ns = numerics[var]
    prefix = _deparse(var)
    if ns.width == 1:
        return [prefix]
    if isinstance(ns.poly_state, MultivariatePolyState):
        # R's polym()/multivariate poly() suffix: per-variable exponents
        # dot-joined, e.g. "1.0", "0.1", "1.1" for degree-2 in 2 variables.
        return [f"{prefix}{'.'.join(str(e) for e in combo)}" for combo in ns.poly_state.combos]
    return [f"{prefix}{j + 1}" for j in range(ns.width)]


def _term_names(planned: PlannedTerm, factors: Dict[Var, FactorSpec], numerics: Dict[Var, NumericSpec]) -> List[str]:
    label_lists = [_var_labels(v, planned.coding, factors, numerics) for v in planned.term.vars]
    # Matches the fold's order: the first-declared variable varies
    # fastest. `itertools.product` varies its *last* argument fastest, so
    # feed the label lists in reverse and reverse each combo back before
    # joining, to print names in declared order ("A2:B2", not "B2:A2").
    return [":".join(reversed(combo)) for combo in product(*reversed(label_lists))]


def column_names(spec: ModelSpec) -> List[str]:
    """One name per column of `build_model_frame`'s output, in the same
    order, matching R's `colnames(model.matrix(...))` for every case R
    itself distinguishes. The one exception is the null-companion
    indicator column (`{var}_isnull`) -- a `survey_kit_formula`-specific
    opt-in feature (`null_dummy=True`) with no R equivalent to match."""
    names: List[str] = []
    if spec.intercept:
        names.append("(Intercept)")
    for planned in spec.terms:
        names.extend(_term_names(planned, spec.factors, spec.numerics))
    for var in spec.null_companions:
        label = spec.factors[var].column if var in spec.factors else _deparse(var)
        names.append(f"{label}_isnull")
    return names
