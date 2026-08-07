"""Tokenizer for R-style formula strings.

Function calls are lexed as a single CALL token whose argument text is
captured verbatim (balanced-paren, quote-aware scan) rather than tokenized —
see `ast_nodes.py` for why: it's what makes `I(x + y)` and friends "just
work" without the formula grammar ever seeing the `+` inside.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

_IDENT_START = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ._")
_IDENT_CONT = _IDENT_START | set("0123456789")
_DIGITS = set("0123456789")


class FormulaSyntaxError(ValueError):
    pass


@dataclass(frozen=True)
class Token:
    kind: str
    value: object = None
    pos: int = 0

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{self.kind}({self.value!r})" if self.value is not None else self.kind


def _scan_balanced_args(s: str, start: int) -> tuple[str, int]:
    """`s[start]` is the character right after an opening '('. Returns
    (raw_args_text, index_just_past_the_matching_close_paren)."""
    depth = 1
    i = start
    n = len(s)
    while i < n:
        c = s[i]
        if c in ("'", '"'):
            quote = c
            i += 1
            while i < n and s[i] != quote:
                if s[i] == "\\" and i + 1 < n:
                    i += 1
                i += 1
            if i >= n:
                raise FormulaSyntaxError(f"unterminated string literal starting at {start}")
            i += 1
            continue
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return s[start:i], i + 1
        i += 1
    raise FormulaSyntaxError(f"unbalanced parentheses starting at position {start}")


def tokenize(formula: str) -> List[Token]:
    tokens: List[Token] = []
    i = 0
    n = len(formula)
    while i < n:
        c = formula[i]

        if c.isspace():
            i += 1
            continue

        if c == "~":
            tokens.append(Token("TILDE", pos=i))
            i += 1
            continue
        if c == "+":
            tokens.append(Token("PLUS", pos=i))
            i += 1
            continue
        if c == "-":
            tokens.append(Token("MINUS", pos=i))
            i += 1
            continue
        if c == ":":
            tokens.append(Token("COLON", pos=i))
            i += 1
            continue
        if c == "^":
            tokens.append(Token("CARET", pos=i))
            i += 1
            continue
        if c == "*":
            tokens.append(Token("STAR", pos=i))
            i += 1
            continue
        if c == "/":
            tokens.append(Token("SLASH", pos=i))
            i += 1
            continue
        if c == "(":
            tokens.append(Token("LPAREN", pos=i))
            i += 1
            continue
        if c == ")":
            tokens.append(Token("RPAREN", pos=i))
            i += 1
            continue
        if c == ",":
            tokens.append(Token("COMMA", pos=i))
            i += 1
            continue

        if c == "%":
            j = formula.index("%", i + 1) if "%" in formula[i + 1 :] else -1
            if j == -1:
                raise FormulaSyntaxError(f"unterminated %op% at position {i}")
            op_name = formula[i + 1 : j]
            if op_name != "in":
                raise FormulaSyntaxError(
                    f"unsupported special operator %{op_name}% at position {i}; only %in% is supported"
                )
            tokens.append(Token("PCT_IN", pos=i))
            i = j + 1
            continue

        if c == "`":
            j = formula.index("`", i + 1)
            if j == -1:
                raise FormulaSyntaxError(f"unterminated backtick identifier at position {i}")
            name = formula[i + 1 : j]
            i = j + 1
            # a backtick-quoted name may still be a function call
            k = i
            while k < n and formula[k].isspace():
                k += 1
            if k < n and formula[k] == "(":
                raw_args, after = _scan_balanced_args(formula, k + 1)
                tokens.append(Token("CALL", (name, raw_args), pos=i))
                i = after
            else:
                tokens.append(Token("IDENT", name, pos=i))
            continue

        if c in _DIGITS:
            start = i
            while i < n and formula[i] in _DIGITS:
                i += 1
            tokens.append(Token("NUMBER", int(formula[start:i]), pos=start))
            continue

        if c in _IDENT_START:
            start = i
            while i < n and formula[i] in _IDENT_CONT:
                i += 1
            name = formula[start:i]
            k = i
            while k < n and formula[k].isspace():
                k += 1
            if k < n and formula[k] == "(":
                raw_args, after = _scan_balanced_args(formula, k + 1)
                tokens.append(Token("CALL", (name, raw_args), pos=start))
                i = after
            else:
                tokens.append(Token("IDENT", name, pos=start))
            continue

        raise FormulaSyntaxError(f"unexpected character {c!r} at position {i}")

    tokens.append(Token("EOF", pos=n))
    return tokens
