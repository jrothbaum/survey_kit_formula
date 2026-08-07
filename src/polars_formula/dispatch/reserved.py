"""Structural reserved forms: `I()`, `factor()`/`ordered()`/`C()`, `offset()`.

Unlike the value dispatch table (`polars_fns.py`), these don't just compute
a column — they change how their target is classified (already handled by
`terms/classify.py`) or are excluded from the design matrix entirely.
`reserved.py`'s job is narrower: evaluate `I(...)`'s raw arithmetic text,
and extract the pieces of a `C(...)` call that Phase 6 (contrasts) needs.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Tuple

import numpy as np

from ..contrasts.base import resolve_contrast_name
from ..parser.args import split_args
from ..parser.ast_nodes import Call
from .numpy_fns import safe_eval_numeric


def eval_I(call: Call, namespace: Mapping[str, np.ndarray]) -> np.ndarray:
    assert call.name == "I", call.name
    return safe_eval_numeric(call.raw_args, namespace)


def is_offset(call: Call) -> bool:
    return call.name == "offset"


def contrast_override(call: Call) -> Optional[Tuple[str, Dict[str, str]]]:
    """Mirrors R's actual `C(object, contr, how.many, ...)` calling
    convention (`src/library/stats/R/C.R`): the 2nd positional argument is
    a contrast function, possibly given as a bare shorthand name
    (`C(x, treatment)`, `C(x, sum)`, ...); extra *keyword* arguments (e.g.
    `base=2`) are forwarded to that contrast function.

    Note this is `C(x, contr.treatment, base=2)`, not `C(x,
    contr.treatment(base=2))` — the latter isn't valid R either (R's
    `contr.treatment` has no default for its first argument, so calling it
    directly inside the formula, before the factor's levels are known,
    would raise "argument 'n' is missing").

    Returns `(resolved_contrast_name, forwarded_kwargs)`, or `None` for
    `C(x)` / `factor(x)` / `ordered(x)` / any call other than `C`.
    """
    if call.name != "C":
        return None
    args = split_args(call.raw_args)
    positional = [a for a in args if a.keyword is None]
    if len(positional) < 2:
        return None
    raw_name = positional[1].raw_value.strip()
    kwargs = {a.keyword: a.raw_value.strip() for a in args if a.keyword is not None}
    return resolve_contrast_name(raw_name), kwargs
