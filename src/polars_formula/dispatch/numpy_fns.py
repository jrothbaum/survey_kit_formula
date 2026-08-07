"""Shared helper for numpy-backed evaluation — currently only used by
`I(...)` (`reserved.py`), which is the one place arbitrary arithmetic text
has to be evaluated rather than dispatched by name.

`eval()` here is deliberately boxed in: no builtins, a fixed numpy-function
namespace, plus whatever data columns the caller supplies. Formula strings
are assumed to come from the calling code itself (an internal calibration
tool), not from untrusted external input — this is the "closed dispatch
table" design decision, not a general-purpose sandboxed-eval story.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

_ALLOWED_FUNCS = {
    "log": np.log,
    "log1p": np.log1p,
    "log2": np.log2,
    "log10": np.log10,
    "sqrt": np.sqrt,
    "exp": np.exp,
    "abs": np.abs,
    "where": np.where,
    "minimum": np.minimum,
    "maximum": np.maximum,
}


def safe_eval_numeric(raw_expr: str, namespace: Mapping[str, np.ndarray]) -> np.ndarray:
    globals_ = {"__builtins__": {}, **_ALLOWED_FUNCS}
    try:
        result = eval(raw_expr, globals_, dict(namespace))  # noqa: S307 - boxed namespace, see module docstring
    except NameError as e:
        raise NameError(f"I({raw_expr!r}): {e}") from e
    return np.asarray(result, dtype=np.float64)
