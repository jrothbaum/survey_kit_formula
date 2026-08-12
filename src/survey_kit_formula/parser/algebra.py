"""R's term-set algebra: `+ - * : ^ / %in%`.

Every rule here was validated empirically against real R (`terms()`), not
just against R's prose docs — see the parity tests in
`tests/parser/test_grammar.py`. Two rules are easy to get backwards:

- `:` is a per-pair *distribute*: crossing `(a+b):(c+d)` yields four
  separate terms `a:c, a:d, b:c, b:d`.
- `/` and `%in%` instead *flatten* the "grouping" side to a single term
  (the union of every variable across every one of its terms) before
  crossing. `a*b/c` is `(a+b+a:b) + c:{a,b}` = `a + b + a:b + a:b:c` — a
  single 3-way term, not `a:c + b:c + a:b:c`. This is what makes nesting
  mean "within the joint groups", not "within each factor separately".
"""

from __future__ import annotations

from .ast_nodes import Term, TermList, Var, var_term


def cross(a: TermList, b: TermList) -> TermList:
    """Per-pair distribute: `:` and the interaction part of `*`.

    `Term(ta.vars + tb.vars)` (tuple concatenation, not a set union):
    `Term.__init__` dedupes while preserving this left-to-right order,
    which is what determines interaction column layout later in
    `build/matrix.py` — see `Term`'s docstring for why that has to be
    deterministic rather than frozenset-iteration-order dependent.
    """
    result = TermList()
    for ta in a:
        for tb in b:
            result.add(Term(ta.vars + tb.vars))
    return result


def star(a: TermList, b: TermList) -> TermList:
    """`a*b` = `a + b + a:b`."""
    return a.union(b).union(cross(a, b))


def power(a: TermList, n: int) -> TermList:
    """`a^n` = `a * a * ... * a` (n times), per R's own docs example
    (`(a+b+c)^2` is defined as identical to `(a+b+c)*(a+b+c)`)."""
    if n < 1:
        raise ValueError(f"formula power must be a positive integer, got {n}")
    result = a
    for _ in range(2, n + 1):
        result = star(result, a)
    return result


def _flatten_term(x: TermList) -> TermList:
    return TermList([Term(x.flatten_vars())]) if len(x) else TermList()


def in_op(y: TermList, x: TermList) -> TermList:
    """`y %in% x` = each term of `y` crossed with the *flattened* variable
    set of `x` (nesting within the joint groups of `x`, not each factor of
    `x` separately)."""
    return cross(y, _flatten_term(x))


def nest(x: TermList, y: TermList) -> TermList:
    """`x / y` = `x + (y %in% x)` — R's own defining identity."""
    return x.union(in_op(y, x))


def var_termlist(v: Var) -> TermList:
    return TermList([var_term(v)])
