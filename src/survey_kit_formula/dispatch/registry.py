"""The closed whitelist of formula function names.

Every `Call` atom in a parsed formula falls into exactly one bucket:

- **structural reserved forms** (`factor`, `ordered`, `C`, `offset`) — these
  don't produce a value themselves; they change how their target variable
  is classified/coded (`terms/classify.py`, `terms/marginality.py`) or are
  excluded from the design matrix entirely (`offset`).
- **value dispatch table entries** (`log`, `sqrt`, `scale`, `center`, ...)
  — Polars-expression-backed where possible, so the result stays lazy and
  the eventual `.to_numpy(zero_copy_only=True)` extraction is free.
- **`I(...)`** — the arithmetic escape hatch; not in the value dispatch
  table because its "argument" is arbitrary raw text, not a fixed
  signature. Handled directly in `reserved.py`.

Anything else is a parse-time error — this is a closed system by design
(see the project's design discussion): no `eval()` of arbitrary formula
terms, no silent fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List

import polars as pl

from ..parser.args import Arg
from ..parser.ast_nodes import Call


class Backend(Enum):
    POLARS = "polars"
    NUMPY = "numpy"


@dataclass(frozen=True)
class DispatchEntry:
    name: str
    backend: Backend
    build_expr: Callable[[List[Arg]], pl.Expr] | None = None


STRUCTURAL_RESERVED = frozenset({"factor", "ordered", "C", "offset"})

# `poly`/`bs`/`ns`: numeric, but "stateful" (fit on training data, must be
# reapplied identically to new data) — handled directly by
# `dispatch/poly_bs.py` + `ModelSpec`, not through the simple
# List[Arg] -> pl.Expr registry below.
STATEFUL_RESERVED = frozenset({"poly", "bs", "ns"})

# Populated by polars_fns.py / numpy_fns.py at import time.
_REGISTRY: Dict[str, DispatchEntry] = {}


class UnknownFormulaFunction(ValueError):
    pass


def register_polars(name: str, build_expr: Callable[[List[Arg]], pl.Expr]) -> None:
    if name in _REGISTRY or name in STRUCTURAL_RESERVED or name == "I":
        raise ValueError(f"formula function {name!r} is already registered")
    _REGISTRY[name] = DispatchEntry(name=name, backend=Backend.POLARS, build_expr=build_expr)


def resolve(call: Call) -> DispatchEntry:
    entry = _REGISTRY.get(call.name)
    if entry is None:
        raise UnknownFormulaFunction(
            f"{call.name!r} is not a recognized formula function "
            f"(known: {sorted(_REGISTRY) + ['I'] + sorted(STRUCTURAL_RESERVED) + sorted(STATEFUL_RESERVED)})"
        )
    return entry


def is_registered(name: str) -> bool:
    return name in _REGISTRY or name in STRUCTURAL_RESERVED or name in STATEFUL_RESERVED or name == "I"
