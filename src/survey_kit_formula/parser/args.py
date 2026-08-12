"""Splits a `Call`'s raw argument text into positional/keyword pieces.

This is deliberately *not* a full Python-expression parser — it only needs
to find top-level commas and `name=value` boundaries, respecting nested
parens/brackets/quotes so that e.g. `C(x, contr.treatment(base=2))` or
`poly(x, degree=2)` split correctly. What each argument's raw text *means*
is entirely up to the caller (a reserved-form handler in `terms/classify.py`
or a dispatch-table entry) — this module only does structural splitting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .tokenizer import FormulaSyntaxError, _IDENT_CONT, _IDENT_START


@dataclass(frozen=True)
class Arg:
    keyword: Optional[str]
    raw_value: str


def split_args(raw: str) -> List[Arg]:
    parts = _split_top_level_commas(raw)
    args: List[Arg] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        keyword, value = _split_keyword(part)
        args.append(Arg(keyword=keyword, raw_value=value.strip()))
    return args


def _split_top_level_commas(raw: str) -> List[str]:
    parts: List[str] = []
    depth = 0
    start = 0
    i = 0
    n = len(raw)
    while i < n:
        c = raw[i]
        if c in ("'", '"'):
            quote = c
            i += 1
            while i < n and raw[i] != quote:
                if raw[i] == "\\" and i + 1 < n:
                    i += 1
                i += 1
            if i >= n:
                raise FormulaSyntaxError(f"unterminated string literal in arguments: {raw!r}")
            i += 1
            continue
        if c in "([":
            depth += 1
        elif c in ")]":
            depth -= 1
        elif c == "," and depth == 0:
            parts.append(raw[start:i])
            start = i + 1
        i += 1
    parts.append(raw[start:])
    return parts


def _split_keyword(part: str) -> tuple[Optional[str], str]:
    i = 0
    n = len(part)
    while i < n and part[i].isspace():
        i += 1
    if i >= n or part[i] not in _IDENT_START:
        return None, part
    start = i
    while i < n and part[i] in _IDENT_CONT:
        i += 1
    name = part[start:i]
    j = i
    while j < n and part[j].isspace():
        j += 1
    if j < n and part[j] == "=" and (j + 1 >= n or part[j + 1] != "="):
        return name, part[j + 1 :]
    return None, part
