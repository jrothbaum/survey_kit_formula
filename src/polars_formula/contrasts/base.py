"""R's five base contrast functions, translated directly from
`src/library/stats/R/contrast.R` and `contr.poly.R` (R 4.6.1) — not
reconstructed from memory. `contr.treatment`'s `base` matches R exactly: a
1-indexed *position* among the factor's levels, not a level name (R itself
requires this; picking a level by name means releveling the factor first,
not passing a string to `contr.treatment`).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .._numeric_core import fit_poly


def _check_min_levels(n: int) -> None:
    if n < 2:
        raise ValueError(f"contrasts not defined for {n - 1} degrees of freedom")


def contr_treatment(n: int, base: int = 1) -> np.ndarray:
    _check_min_levels(n)
    if base < 1 or base > n:
        raise ValueError("baseline group number out of range")
    return np.delete(np.eye(n), base - 1, axis=1)


def contr_sas(n: int) -> np.ndarray:
    return contr_treatment(n, base=n)


def contr_sum(n: int) -> np.ndarray:
    _check_min_levels(n)
    cont = np.eye(n)[:, :-1].copy()
    cont[-1, :] = -1.0
    return cont


def contr_helmert(n: int) -> np.ndarray:
    _check_min_levels(n)
    cont = np.full((n, n - 1), -1.0)
    for r in range(n):
        for c in range(n - 1):
            if c <= r - 2:
                cont[r, c] = 0.0
            elif c == r - 1:
                cont[r, c] = c + 1
    return cont


def contr_poly(n: int, scores: Optional[Sequence[float]] = None) -> np.ndarray:
    """R's default is equally-spaced integer scores `1:n`. Pass `scores=`
    for unequally-spaced ordinal levels (R's `contr.poly(n, scores=...)`),
    e.g. levels lo/mid/hi at scores [1, 2, 10] rather than [1, 2, 3]."""
    _check_min_levels(n)
    if scores is None:
        score_arr = np.arange(1, n + 1, dtype=np.float64)
    else:
        score_arr = np.asarray(scores, dtype=np.float64)
        if len(score_arr) != n:
            raise ValueError(f"'scores' argument is of the wrong length ({len(score_arr)} != {n})")
        if len(set(score_arr.tolist())) != n:
            raise ValueError("'scores' must all be different numbers")
    Z, _state = fit_poly(score_arr, degree=n - 1)
    return Z


# R's C() shorthand: `C(x, treatment)` / `C(x, sum)` / ... (bare name) maps
# to the full `contr.*` function name — see `switch()` in R's `C.R`.
SHORTHAND_ALIASES = {
    "poly": "contr.poly",
    "helmert": "contr.helmert",
    "sum": "contr.sum",
    "treatment": "contr.treatment",
    "SAS": "contr.SAS",
}

CONTRAST_FUNCTIONS = {
    "contr.treatment": contr_treatment,
    "contr.sum": contr_sum,
    "contr.helmert": contr_helmert,
    "contr.poly": contr_poly,
    "contr.SAS": contr_sas,
}


def resolve_contrast_name(name: str) -> str:
    """`treatment` -> `contr.treatment`; `contr.sum` -> `contr.sum`
    (already fully qualified, passed through)."""
    return SHORTHAND_ALIASES.get(name, name)


def default_contrast(n: int, ordered: bool, scores: Optional[Sequence[float]] = None) -> np.ndarray:
    """R's `getOption("contrasts")` default: `contr.treatment` for
    unordered factors, `contr.poly` for ordered ones."""
    return contr_poly(n, scores=scores) if ordered else contr_treatment(n, base=1)
