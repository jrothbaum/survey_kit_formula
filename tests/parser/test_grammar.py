"""Grammar / term-algebra tests.

Every case here is cross-checked against real R's `terms()` (see
`tests/parity/r_oracle.py`) rather than trusting hardcoded expectations —
the precedence rules were derived empirically against R in the first place,
so re-deriving them from memory here would just risk the same mistake
twice.
"""

from __future__ import annotations

import pytest

from parity.r_oracle import R_AVAILABLE, r_term_structure
from survey_kit_formula.parser import Identifier, parse_formula
from survey_kit_formula.parser.ast_nodes import Term

requires_r = pytest.mark.skipif(not R_AVAILABLE, reason="R is not installed")


def _our_term_structure(formula: str):
    parsed = parse_formula(formula)
    terms = [_term_varnames(t) for t in parsed.rhs.sorted_by_order()]
    return terms, parsed.intercept


def _term_varnames(t: Term):
    names = set()
    for v in t.vars:
        assert isinstance(v, Identifier), "oracle comparison only covers plain-identifier formulas"
        names.add(v.name)
    return frozenset(names)


FORMULAS = [
    "y ~ a + b",
    "y ~ a + b:c",
    "y ~ a:b + c",
    "y ~ a:b*c",
    "y ~ a*b:c",
    "y ~ (a+b+c)^2",
    "y ~ (a+b+c)^3",
    "y ~ a/b",
    "y ~ a/b/c",
    "y ~ b %in% a",
    "y ~ x - 1",
    "y ~ a*b*c",
    "y ~ -1 + x",
    "y ~ a - a:b",
    "y ~ (a+b)*(c+d)",
    "y ~ a:b:c",
    "y ~ a+b*c",
    "y ~ a/b*c",
    "y ~ a*b/c",
    "y ~ a:b^2",
    "y ~ (a:b)^2",
    "y ~ a %in% b:c",
    "y ~ a*b*c*d",
    "y ~ a %in% b * c",
    "y ~ a + b %in% c",
    "y ~ a %in% b + c",
    "y ~ a:b %in% c",
    "y ~ a - b:c",
    "~ a + b",
    "y ~ x + 0",
    "y ~ 0 + x",
    "y ~ a + b - a",
]


@requires_r
@pytest.mark.parametrize("formula", FORMULAS)
def test_matches_r(formula):
    ours_terms, ours_intercept = _our_term_structure(formula)
    r_terms, r_intercept = r_term_structure(formula)
    assert ours_terms == r_terms, f"{formula}: {ours_terms} != {r_terms}"
    assert ours_intercept == r_intercept, f"{formula}: intercept {ours_intercept} != {r_intercept}"


def test_one_sided_no_lhs():
    parsed = parse_formula("~ a + b")
    assert parsed.lhs is None


def test_lhs_captured():
    parsed = parse_formula("y ~ x")
    assert parsed.lhs == Identifier("y")


def test_lhs_function_call():
    parsed = parse_formula("log(y) ~ x")
    from survey_kit_formula.parser import Call

    assert parsed.lhs == Call("log", "y")


def test_call_raw_args_untouched_by_grammar():
    # the '+' inside I(...) must NOT be treated as a formula operator
    parsed = parse_formula("y ~ a + I(b + c)")
    terms = list(parsed.rhs)
    assert len(terms) == 2
    from survey_kit_formula.parser import Call

    assert any(t.vars == (Call("I", "b + c"),) for t in terms)


def test_default_intercept_true():
    parsed = parse_formula("y ~ x")
    assert parsed.intercept is True


def test_removing_absent_term_is_noop():
    parsed = parse_formula("y ~ a - b:c")
    assert [frozenset(v.name for v in t.vars) for t in parsed.rhs] == [frozenset({"a"})]
