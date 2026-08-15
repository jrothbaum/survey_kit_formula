"""Formula -> source-column resolution, independent of any particular
dataframe library. `expand_dot`/`extract_offsets` are the same rewrite
steps `ModelSpec.from_formula` needs before it can build factors/numerics;
`required_columns_from_parsed` is the public `required_columns()` entry
point's implementation. Both live here (rather than in `terms/spec.py` or
`api.py`) so each side can import this leaf module without importing the
other -- `api.py` already imports `ModelSpec` from `terms/spec.py`, so
`terms/spec.py` importing back from `api.py` would be circular.
"""

from __future__ import annotations

import ast
from typing import List, Optional, Sequence, Tuple

from ..dispatch.numpy_fns import _ALLOWED_FUNCS
from ..dispatch.reserved import is_offset
from ..parser.ast_nodes import Call, Identifier, ParsedFormula, TermList, Var, var_term
from .classify import referenced_columns, underlying_column


def has_dot(rhs: TermList) -> bool:
    for t in rhs:
        if t.order == 1:
            v = next(iter(t.vars))
            if isinstance(v, Identifier) and v.name == ".":
                return True
    return False


def expand_dot(rhs: TermList, lhs: Optional[Var], schema_names: Sequence[str]) -> TermList:
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

    remaining = [name for name in schema_names if name not in referenced]

    result = TermList()
    for t in rhs:
        if t is dot_term:
            for name in remaining:
                result.add(var_term(Identifier(name)))
        else:
            result.add(t)
    return result


def extract_offsets(rhs: TermList) -> Tuple[TermList, List[Var]]:
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


def _var_source_columns(v: Var) -> List[str]:
    if isinstance(v, Identifier):
        return [v.name]
    if v.name == "I":
        return _extract_I_columns(v.raw_args)
    if v.name == "C":
        # `C(x, contr, base=...)`'s second positional argument is a bare
        # contrast-function name (e.g. `treatment`), not a column --
        # `referenced_columns` would otherwise treat it as one. Only
        # `poly()` has more than one genuine column among its positional
        # arguments (multivariate `poly(x1, x2, degree=n)`).
        return [underlying_column(v)]
    return referenced_columns(v)


class _NameCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.seen: set = set()
        self.names: List[str] = []

    def visit_Name(self, node: ast.Name) -> None:
        if node.id not in _ALLOWED_FUNCS and node.id not in self.seen:
            self.seen.add(node.id)
            self.names.append(node.id)


def _extract_I_columns(raw_expr: str) -> List[str]:
    # A NodeVisitor walks fields in declaration order (pre-order DFS), which
    # matches left-to-right source order for ordinary expressions -- unlike
    # `ast.walk`'s breadth-first order, which doesn't.
    collector = _NameCollector()
    collector.visit(ast.parse(raw_expr, mode="eval"))
    return collector.names


def required_columns_from_parsed(parsed: ParsedFormula, schema_names: Optional[Sequence[str]]) -> List[str]:
    """The source columns `parsed` reads, in first-appearance order
    (response first). `schema_names` is only consulted if the formula uses
    `.` ("every other column"); pass `None` when unavailable and a formula
    without `.` still resolves fine."""
    rhs = parsed.rhs

    if has_dot(rhs):
        if schema_names is None:
            raise ValueError(
                "formula uses '.' (all other columns); the full column set is needed to resolve it"
            )
        rhs = expand_dot(rhs, parsed.lhs, schema_names)

    rhs, offset_vars = extract_offsets(rhs)

    columns: List[str] = []
    seen: set = set()

    def add_all(names: List[str]) -> None:
        for name in names:
            if name not in seen:
                seen.add(name)
                columns.append(name)

    if parsed.lhs is not None:
        add_all(_var_source_columns(parsed.lhs))
    for v in rhs.flatten_vars():
        add_all(_var_source_columns(v))
    for v in offset_vars:
        add_all(_var_source_columns(v))
    return columns
