"""`poly()` and `bs()` — the two "stateful" numeric transforms: unlike
`log`/`scale`/etc, applying them to *new* data (predict-time reuse) must
reproduce the exact basis fit on the *original* data, not recompute a basis
from scratch. Each has a `fit_*` (training data -> array + state) and
`apply_*` (state + new data -> array) pair; `ModelSpec` (Phase 7) is
responsible for storing the state and calling `apply_*` on new data.

Translated directly from R's own source (`stats::poly`,
`src/library/stats/R/contr.poly.R`; `splines::bs`,
`src/library/splines/R/splines.R` in R 4.6.1) rather than reconstructed
from memory — `poly()`'s orthogonal-polynomial recurrence in particular has
enough numerical subtlety (QR sign conventions) that it's easy to get a
basis that spans the same space as R's but doesn't match column-for-column.
Both are checked against real R output in `tests/terms/test_poly_bs.py`.

`bs()` v1 scope: the common path only — `x` within `Boundary.knots`, no
missing values. R's out-of-range linear-extrapolation path
(`warn.outside`/pivot handling) is not implemented; out-of-range values
raise rather than silently extrapolating.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.interpolate import BSpline

from .._numeric_core import PolyState, apply_poly, fit_poly

__all__ = [
    "PolyState",
    "fit_poly",
    "apply_poly",
    "BSplineState",
    "fit_bs",
    "apply_bs",
]


@dataclass(frozen=True)
class BSplineState:
    degree: int
    interior_knots: np.ndarray
    boundary_knots: Tuple[float, float]
    intercept: bool

    @property
    def all_knots(self) -> np.ndarray:
        order = self.degree + 1
        return np.sort(
            np.concatenate(
                (
                    np.repeat(self.boundary_knots[0], order),
                    self.interior_knots,
                    np.repeat(self.boundary_knots[1], order),
                )
            )
        )


def fit_bs(
    x: np.ndarray,
    df: Optional[int] = None,
    knots: Optional[List[float]] = None,
    degree: int = 3,
    intercept: bool = False,
    boundary_knots: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, BSplineState]:
    x = np.asarray(x, dtype=np.float64)
    if np.any(np.isnan(x)):
        raise ValueError("missing values are not allowed in bs() (v1 scope)")
    order = degree + 1
    if order <= 1:
        raise ValueError("'degree' must be integer >= 1")

    bknots = boundary_knots if boundary_knots is not None else (float(x.min()), float(x.max()))
    if np.any(x < bknots[0]) or np.any(x > bknots[1]):
        raise ValueError(
            "bs(): values outside Boundary.knots are not supported in v1 "
            "(R's out-of-range linear-extrapolation path is not implemented)"
        )

    if knots is None and df is not None:
        n_interior = df - order + (0 if intercept else 1)
        if n_interior < 0:
            n_interior = 0
        if n_interior > 0:
            probs = np.linspace(0.0, 1.0, n_interior + 2)[1:-1]
            interior = np.quantile(x, probs)
        else:
            interior = np.array([], dtype=np.float64)
    elif knots is not None:
        interior = np.sort(np.asarray(knots, dtype=np.float64))
    else:
        interior = np.array([], dtype=np.float64)

    state = BSplineState(degree=degree, interior_knots=interior, boundary_knots=bknots, intercept=intercept)
    return apply_bs(x, state), state


def apply_bs(x: np.ndarray, state: BSplineState) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if np.any(np.isnan(x)):
        raise ValueError("missing values are not allowed in bs() (v1 scope)")
    lo, hi = state.boundary_knots
    if np.any(x < lo) or np.any(x > hi):
        raise ValueError("bs(): values outside Boundary.knots are not supported in v1")

    t = state.all_knots
    order = state.degree + 1
    n_basis = len(t) - order
    design = BSpline.design_matrix(x, t, state.degree, extrapolate=False).toarray()
    assert design.shape == (len(x), n_basis)
    if not state.intercept:
        design = design[:, 1:]
    return design
