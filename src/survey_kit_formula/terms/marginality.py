"""R's full-rank / marginality algorithm — decides, per term and per factor
within that term, whether to code it with contrasts (k-1 columns) or a full
dummy expansion (k columns).

This is a direct translation of `TermCode()` in R's own
`src/library/stats/src/model.c` (read from the R 4.6.1 source, not
reconstructed from documentation), plus the separate no-intercept
adjustment applied afterwards in `modelmatrix()` in the same file. Two
things are easy to get backwards without the source in hand:

1. `TermCode`'s "is the margin already covered" check is `margin ⊆
   preceding_term`, i.e. a margin is covered by any *earlier* term whose
   variable set *contains* it — not just an exact match. E.g. in
   `a:b:d + a:b:c`, the margin `{a,b}` of `c` (in the second term) is
   covered by the first term `{a,b,d}` even though it's a proper subset,
   not an equal set.

2. The no-intercept adjustment does not upgrade one factor per term. It
   scans terms in order, and within each term scans variables in
   first-appearance order across the *whole* formula, and upgrades only
   the *first* (term, factor) pair it finds — one upgrade, total, across
   the entire model. This is why `y ~ 0 + a*b` ends up with `a` in full
   dummy but `b` still in contrasts, not both in full dummy.

Both were verified against real R (`model.matrix`'s `assign` attribute) in
`tests/terms/test_marginality.py`, not just read from source.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List

import polars as pl

from ..parser.ast_nodes import Term, TermList, Var
from .classify import DataClass, classify_var


class Coding(Enum):
    CONTRASTS = 1
    DUMMY = 2


@dataclass(frozen=True)
class TermPlan:
    term: Term
    coding: Dict[Var, Coding] = field(default_factory=dict)


@dataclass
class MarginalityResult:
    term_plans: List[TermPlan]
    dataclasses: Dict[Var, DataClass]


def compute_marginality(rhs: TermList, schema: pl.Schema, intercept: bool) -> MarginalityResult:
    ordered_terms = rhs.sorted_by_order()
    all_vars = _unique_vars_in_order(ordered_terms)
    dataclasses = {v: classify_var(v, schema) for v in all_vars}

    codings: List[Dict[Var, Coding]] = []
    for idx, term in enumerate(ordered_terms):
        preceding = ordered_terms[:idx]
        codings.append(_term_code(term, preceding))

    if not intercept:
        _apply_no_intercept_adjustment(ordered_terms, codings, all_vars, dataclasses)

    term_plans = [TermPlan(term=t, coding=c) for t, c in zip(ordered_terms, codings)]
    return MarginalityResult(term_plans=term_plans, dataclasses=dataclasses)


def _term_code(term: Term, preceding: List[Term]) -> Dict[Var, Coding]:
    coding: Dict[Var, Coding] = {}
    for v in term.vars:
        margin = term.var_set - {v}
        if not margin:
            coding[v] = Coding.CONTRASTS
        elif any(margin <= t.var_set for t in preceding):
            coding[v] = Coding.CONTRASTS
        else:
            coding[v] = Coding.DUMMY
    return coding


def _apply_no_intercept_adjustment(
    ordered_terms: List[Term],
    codings: List[Dict[Var, Coding]],
    all_vars: List[Var],
    dataclasses: Dict[Var, DataClass],
) -> None:
    for term, term_coding in zip(ordered_terms, codings):
        for v in all_vars:
            if v in term.vars and dataclasses[v].is_factor:
                term_coding[v] = Coding.DUMMY
                return


def _unique_vars_in_order(terms: List[Term]) -> List[Var]:
    seen = set()
    out: List[Var] = []
    for t in terms:
        for v in t.vars:
            if v not in seen:
                seen.add(v)
                out.append(v)
    return out


def term_column_count(
    plan: TermPlan,
    dataclasses: Dict[Var, DataClass],
    nlevels: Dict[Var, int],
    numeric_width: Dict[Var, int] | None = None,
) -> int:
    """The number of columns a term contributes, computed purely from the
    marginality decision plus level/width counts — no data rows involved.
    `nlevels` must have an entry for every factor `Var` in the term;
    `numeric_width` defaults missing entries to 1 (the common case: a plain
    numeric column or a single-column transform like `log(x)`)."""
    numeric_width = numeric_width or {}
    total = 1
    for v in plan.term.vars:
        if dataclasses[v].is_factor:
            k = nlevels[v]
            total *= k if plan.coding[v] is Coding.DUMMY else (k - 1)
        else:
            total *= numeric_width.get(v, 1)
    return total
