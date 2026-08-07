"""Recursive-descent parser for R-style formula strings.

Operator precedence (tightest to loosest), validated empirically against
real R (`terms()`) rather than assumed — formulas are parsed with R's
*ordinary* expression-operator precedence, not a formula-specific grammar:

    ^            (right-hand side must be a positive integer literal)
    unary - +
    :
    %in%
    *  /
    +  -         (binary; also where bare `1`/`0` toggle the intercept)
    ~            (lowest; separates response from predictors)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from .algebra import cross, in_op, nest, power as power_op, star, var_termlist
from .ast_nodes import Call, Identifier, ParsedFormula, Term, TermList, Var
from .tokenizer import FormulaSyntaxError, Token, tokenize

_TIGHT_AFTER_NUMBER = {"CARET", "COLON", "PCT_IN", "STAR", "SLASH"}


class _Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def peek(self, offset: int = 0) -> Token:
        idx = self.pos + offset
        if idx >= len(self.tokens):
            return self.tokens[-1]
        return self.tokens[idx]

    def advance(self) -> Token:
        tok = self.tokens[self.pos]
        if tok.kind != "EOF":
            self.pos += 1
        return tok

    def expect(self, kind: str) -> Token:
        tok = self.peek()
        if tok.kind != kind:
            raise FormulaSyntaxError(f"expected {kind} but found {tok.kind} at position {tok.pos}")
        return self.advance()

    # ---- grammar levels, tightest to loosest ------------------------

    def parse_atom(self) -> TermList:
        tok = self.peek()
        if tok.kind == "LPAREN":
            self.advance()
            terms, _intercept = self.parse_additive()
            self.expect("RPAREN")
            return terms
        if tok.kind == "IDENT":
            self.advance()
            name = tok.value
            if name == ".":
                return var_termlist(Identifier("."))
            return var_termlist(Identifier(name))
        if tok.kind == "CALL":
            self.advance()
            name, raw_args = tok.value
            return var_termlist(Call(name, raw_args))
        if tok.kind == "NUMBER" and tok.value in (0, 1):
            # Reached only when 0/1 is combined via a tighter operator than
            # +/- (e.g. `1:x`), since the additive level intercepts the
            # common standalone `1`/`0`/`-1` cases before recursing here.
            # Pragmatic fallback: contributes no term.
            self.advance()
            return TermList()
        raise FormulaSyntaxError(f"unexpected token {tok.kind} at position {tok.pos}")

    def parse_power(self) -> TermList:
        base = self.parse_atom()
        if self.peek().kind == "CARET":
            self.advance()
            exp_tok = self.expect("NUMBER")
            if exp_tok.value < 1:
                raise FormulaSyntaxError(
                    f"formula '^' exponent must be a positive integer, got {exp_tok.value} at position {exp_tok.pos}"
                )
            return power_op(base, exp_tok.value)
        return base

    def parse_unary(self) -> TermList:
        if self.peek().kind == "MINUS":
            self.advance()
            self.parse_power()
            # Pragmatic fallback for unary '-' outside the additive
            # top level (e.g. `a : -b`): contributes no term. The common
            # `-1` intercept-removal idiom is handled at the additive
            # level, before this path is ever reached.
            return TermList()
        if self.peek().kind == "PLUS":
            self.advance()
            return self.parse_power()
        return self.parse_power()

    def parse_interaction(self) -> TermList:
        result = self.parse_unary()
        while self.peek().kind == "COLON":
            self.advance()
            rhs = self.parse_unary()
            result = cross(result, rhs)
        return result

    def parse_nesting(self) -> TermList:
        result = self.parse_interaction()
        while self.peek().kind == "PCT_IN":
            self.advance()
            rhs = self.parse_interaction()
            result = in_op(result, rhs)
        return result

    def parse_multiplicative(self) -> TermList:
        result = self.parse_nesting()
        while self.peek().kind in ("STAR", "SLASH"):
            op = self.advance().kind
            rhs = self.parse_nesting()
            result = star(result, rhs) if op == "STAR" else nest(result, rhs)
        return result

    def _is_standalone_intercept_number(self) -> bool:
        return self.peek(1).kind not in _TIGHT_AFTER_NUMBER

    def parse_additive(self) -> Tuple[TermList, bool]:
        terms = TermList()
        intercept = True

        op = "PLUS"
        if self.peek().kind in ("PLUS", "MINUS"):
            op = self.advance().kind

        terms, intercept = self._apply_additive_operand(terms, intercept, op)

        while self.peek().kind in ("PLUS", "MINUS"):
            op = self.advance().kind
            terms, intercept = self._apply_additive_operand(terms, intercept, op)

        return terms, intercept

    def _apply_additive_operand(self, terms: TermList, intercept: bool, op: str) -> Tuple[TermList, bool]:
        tok = self.peek()
        if tok.kind == "NUMBER" and tok.value in (0, 1) and self._is_standalone_intercept_number():
            self.advance()
            if tok.value == 0:
                return terms, False
            return terms, op != "MINUS"

        operand = self.parse_multiplicative()
        if op == "PLUS":
            return terms.union(operand), intercept
        return terms.difference(operand), intercept

    # ---- entry point --------------------------------------------------

    def parse_formula(self) -> ParsedFormula:
        lhs: Optional[Var] = None
        if self.peek().kind != "TILDE":
            lhs_terms = self.parse_multiplicative()
            self.expect("TILDE")
            lhs = _extract_single_var(lhs_terms)
        else:
            self.advance()

        rhs_terms, intercept = self.parse_additive()

        if self.peek().kind != "EOF":
            tok = self.peek()
            raise FormulaSyntaxError(f"unexpected token {tok.kind} at position {tok.pos}")

        return ParsedFormula(lhs=lhs, rhs=rhs_terms, intercept=intercept)


def _extract_single_var(terms: TermList) -> Var:
    term_list = list(terms)
    if len(term_list) != 1 or term_list[0].order != 1:
        raise FormulaSyntaxError(
            "the left-hand side of a formula must be a single variable or function call "
            "(e.g. `y`, `log(y)`, `cbind(y1, y2)`)"
        )
    return next(iter(term_list[0].vars))


def parse_formula(text: str) -> ParsedFormula:
    tokens = tokenize(text)
    return _Parser(tokens).parse_formula()
