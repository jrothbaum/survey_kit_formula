"""AST node types for parsed R-style formulas.

A `Term` is an interaction: a set of atomic variable expressions (`Var`).
A `TermList` is an ordered, deduplicated collection of `Term`s — R's term
algebra (`+ - * : ^ / %in%`) operates entirely on `TermList`s.

Every function call (`poly(x, degree=2)`, `I(x + y)`, `log(x)`, ...) is
captured as a `Call` with its argument list kept as *raw, unparsed text*.
The formula-level operators only ever see whatever is outside of a call's
parentheses, so `I(x + y)` is naturally a single opaque atom to this layer —
the `+` inside never reaches the term algebra. Argument structure (splitting
on top-level commas, positional vs. keyword) is resolved later by whichever
dispatch/reserved-form handler owns that function name.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Iterable, Iterator, List, Tuple, Union


@dataclass(frozen=True)
class Identifier:
    """A bare variable reference, e.g. `x`, `income`, or the special `.`."""

    name: str

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return self.name


@dataclass(frozen=True)
class Call:
    """A function-call atom, e.g. `poly(x, degree = 2)`.

    `raw_args` is the untouched source text between the outer parentheses.
    """

    name: str
    raw_args: str

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{self.name}({self.raw_args})"


Var = Union[Identifier, Call]


class Term:
    """An interaction: the `Var`s combined by `:`.

    `vars` is an order-preserving, deduplicated tuple, *not* a frozenset —
    that order is what later determines the column layout of this term's
    contribution to the design matrix (`build/matrix.py`'s row-wise
    Kronecker product iterates `term.vars` directly). A plain frozenset
    would make column order depend on Python's per-process hash-randomized
    iteration order: internally consistent within one run, but
    nondeterministic *across* runs and liable to silently disagree with
    R's fixed (declaration-order) column layout on some runs and not
    others — exactly the kind of bug that looks like a flaky test.

    Term *identity* (equality, hashing, dedup in `TermList`, the marginality
    algorithm's subset checks) is still set-based — `a:b` and `b:a` are the
    same term — via the `var_set` property and custom `__eq__`/`__hash__`.
    """

    __slots__ = ("vars",)

    def __init__(self, vars: Iterable[Var]):
        seen: set[Var] = set()
        ordered: List[Var] = []
        for v in vars:
            if v not in seen:
                seen.add(v)
                ordered.append(v)
        self.vars = tuple(ordered)

    @property
    def var_set(self) -> FrozenSet[Var]:
        return frozenset(self.vars)

    @property
    def order(self) -> int:
        return len(self.vars)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Term):
            return NotImplemented
        return self.var_set == other.var_set

    def __hash__(self) -> int:
        return hash(self.var_set)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return ":".join(repr(v) for v in self.vars)


def var_term(v: Var) -> Term:
    return Term((v,))


class TermList:
    """Ordered, deduplicated collection of `Term`s.

    Order matters: R breaks ties within the same interaction `order` by
    insertion sequence (see `?formula` term-generation examples), so this
    preserves append order rather than behaving like a plain set.
    """

    __slots__ = ("_terms", "_seen")

    def __init__(self, terms: Iterable[Term] | None = None):
        self._terms: List[Term] = []
        self._seen: set[FrozenSet[Var]] = set()
        if terms is not None:
            for t in terms:
                self.add(t)

    def add(self, term: Term) -> None:
        if term.var_set not in self._seen:
            self._seen.add(term.var_set)
            self._terms.append(term)

    def discard(self, term: Term) -> None:
        if term.var_set in self._seen:
            self._seen.discard(term.var_set)
            self._terms = [t for t in self._terms if t.var_set != term.var_set]

    def union(self, other: "TermList") -> "TermList":
        result = TermList(self._terms)
        for t in other:
            result.add(t)
        return result

    def difference(self, other: "TermList") -> "TermList":
        result = TermList(self._terms)
        for t in other:
            result.discard(t)
        return result

    def flatten_vars(self) -> Tuple[Var, ...]:
        """Every distinct variable across every term, in first-appearance
        order — used by `/` and `%in%` to build the "container" term they
        nest within. Order-preserving for the same reason `Term.vars` is:
        it becomes column layout, not just set membership."""
        seen: set[Var] = set()
        out: List[Var] = []
        for t in self._terms:
            for v in t.vars:
                if v not in seen:
                    seen.add(v)
                    out.append(v)
        return tuple(out)

    def sorted_by_order(self) -> List[Term]:
        return sorted(self._terms, key=lambda t: t.order)

    def __iter__(self) -> Iterator[Term]:
        return iter(self._terms)

    def __len__(self) -> int:
        return len(self._terms)

    def __contains__(self, term: Term) -> bool:
        return term.var_set in self._seen

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TermList):
            return NotImplemented
        return self._seen == other._seen

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return " + ".join(repr(t) for t in self._terms) or "<empty>"


@dataclass
class ParsedFormula:
    """Result of parsing a formula string."""

    lhs: Var | None
    rhs: TermList
    intercept: bool
