"""`poly()`, `bs()` and `ns()` — the "stateful" numeric transforms: unlike
`log`/`scale`/etc, applying them to *new* data (predict-time reuse) must
reproduce the exact basis fit on the *original* data, not recompute a basis
from scratch. Each has a `fit_*` (training data -> array + state) and
`apply_*` (state + new data -> array) pair; `ModelSpec` (Phase 7) is
responsible for storing the state and calling `apply_*` on new data.

Translated directly from R's own source (`stats::poly`,
`src/library/stats/R/contr.poly.R`; `splines::bs`/`splines::ns`,
`src/library/splines/R/splines.R` in R 4.6.1) rather than reconstructed
from memory — `poly()`'s orthogonal-polynomial recurrence in particular has
enough numerical subtlety (QR sign conventions) that it's easy to get a
basis that spans the same space as R's but doesn't match column-for-column.
All are checked against real R output in `tests/terms/test_poly_bs.py`,
`tests/terms/test_ns.py`.

Out-of-range `x` (beyond `Boundary.knots`) does not raise, matching R:
both emit R's own warning and extrapolate via a local Taylor expansion of
the basis functions around a pivot point near the boundary (`bs()`: full
`degree`-order expansion, pivoted slightly inside the boundary since basis
derivatives are evaluated at a knot of multiplicity `degree+1` there;
`ns()`: exactly linear, degree 1, pivoted *at* the boundary itself, since a
natural spline is constructed to be linear beyond the boundary by
definition — see `_taylor_extrapolate`).

Missing values also match R: `bs()`/`ns()` drop NA rows before computing
anything (default `Boundary.knots`, interior-knot quantiles, the basis
itself), then reinsert a full NaN row per dropped position in the output —
never raise, never silently zero-fill. `poly()` differs (matches R there
too): `raw=FALSE` (its default) raises `"missing values are not allowed in
poly()"` immediately, since R's own `stats::poly` does exactly that;
`raw=TRUE` just lets NaN propagate through the literal power computation,
also matching R.

Note `ModelSpec`/`terms/spec.py` layers its own stricter default on top of
all this for formula use (any null raises unless `null_dummy=True`, which
fills to a constant plus a companion indicator column, not R's NaN-row
passthrough) — see that module's docstring. The functions here are the
faithful-to-R numeric core; `ModelSpec` intentionally chooses a different
default policy for the reasons discussed when that feature was built.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from itertools import product as _product
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.interpolate import BSpline

from .._numeric_core import PolyState, apply_poly, fit_poly

_OUTSIDE_WARNING = "some 'x' values beyond boundary knots may cause ill-conditioned bases"


def _taylor_extrapolate(
    x_out: np.ndarray, all_knots: np.ndarray, basis_degree: int, taylor_degree: int, pivot: float
) -> np.ndarray:
    """R's `bs()`/`ns()` out-of-range mechanism: a `taylor_degree`-order
    Taylor expansion of each of the underlying degree-`basis_degree`
    B-spline basis functions around `pivot`, evaluated at `x_out`. `bs()`
    calls this with `taylor_degree = basis_degree` (a full local polynomial
    match); `ns()` calls it with `taylor_degree = 1` (exactly linear — a
    natural spline is constructed to be linear beyond the boundary by
    definition, so a linear extrapolation from the boundary is exact, not
    approximate, and no inward pivot shift is needed either)."""
    n_basis = len(all_knots) - (basis_degree + 1)
    orders = np.arange(taylor_degree + 1)
    tt = np.empty((taylor_degree + 1, n_basis))
    for j in range(n_basis):
        coef = np.zeros(n_basis)
        coef[j] = 1.0
        spline = BSpline(all_knots, coef, basis_degree)
        for d in orders:
            tt[d, j] = spline.derivative(nu=int(d))(pivot)
    scalef = np.array([math.factorial(int(d)) for d in orders], dtype=np.float64)
    tt_scaled = tt / scalef[:, None]
    dx = x_out - pivot
    powers = dx[:, None] ** orders[None, :]
    return powers @ tt_scaled

__all__ = [
    "PolyState",
    "fit_poly",
    "apply_poly",
    "MultivariatePolyState",
    "fit_polym",
    "apply_polym",
    "BSplineState",
    "fit_bs",
    "apply_bs",
    "NaturalSplineState",
    "fit_ns",
    "apply_ns",
]


@dataclass(frozen=True)
class MultivariatePolyState:
    degree: int
    raw: bool
    variable_states: Tuple[PolyState, ...]
    combos: Tuple[Tuple[int, ...], ...]


def _poly_exponent_combos(nd: int, degree: int) -> Tuple[Tuple[int, ...], ...]:
    """R's `polym()`: `expand.grid(rep(list(0:degree), nd))` (first variable
    fastest-varying), filtered to combinations whose exponents sum to a
    total degree in `(0, degree]`. Verified directly against R's own
    `colnames(polym(...))` output for 2- and 3-variable cases, not just
    read from the R source — `expand.grid`'s fastest-axis convention is
    easy to get backwards (it's the opposite of `itertools.product`'s)."""
    all_combos = (tuple(reversed(t)) for t in _product(range(degree + 1), repeat=nd))
    return tuple(c for c in all_combos if 0 < sum(c) <= degree)


def fit_polym(xs: Sequence[np.ndarray], degree: int, raw: bool = False) -> Tuple[np.ndarray, MultivariatePolyState]:
    """`poly(x1, x2, ..., degree=D)` / R's `polym()`: a total-degree-filtered
    tensor product of each variable's own orthogonal (or raw) 1D basis, not
    a plain cross of them -- e.g. degree=2 in 2 variables gives the 5 terms
    {x1, x1^2, x2, x1*x2, x2^2}, not all 9 combinations of degree 0-2."""
    if len(xs) < 2:
        raise ValueError("fit_polym needs at least 2 variables; use fit_poly for a single variable")
    lengths = {len(x) for x in xs}
    if len(lengths) != 1:
        raise ValueError("polym: all variables must have the same length")

    variable_states: List[PolyState] = []
    augmented: List[np.ndarray] = []
    for x in xs:
        x = np.asarray(x, dtype=np.float64)
        Z, state = fit_poly(x, degree=degree, raw=raw)
        variable_states.append(state)
        augmented.append(np.column_stack([np.ones(len(x)), Z]))

    combos = _poly_exponent_combos(len(xs), degree)
    result = _combine_polym(augmented, combos)
    state = MultivariatePolyState(degree=degree, raw=raw, variable_states=tuple(variable_states), combos=combos)
    return result, state


def apply_polym(xs: Sequence[np.ndarray], state: MultivariatePolyState) -> np.ndarray:
    if len(xs) != len(state.variable_states):
        raise ValueError(f"polym: expected {len(state.variable_states)} variables, got {len(xs)}")
    augmented = []
    for x, vstate in zip(xs, state.variable_states):
        x = np.asarray(x, dtype=np.float64)
        Z = apply_poly(x, vstate)
        augmented.append(np.column_stack([np.ones(len(x)), Z]))
    return _combine_polym(augmented, state.combos)


def _combine_polym(augmented: List[np.ndarray], combos: Tuple[Tuple[int, ...], ...]) -> np.ndarray:
    n = augmented[0].shape[0]
    out = np.empty((n, len(combos)), dtype=np.float64)
    for j, combo in enumerate(combos):
        col = np.ones(n)
        for var_idx, exponent in enumerate(combo):
            col = col * augmented[var_idx][:, exponent]
        out[:, j] = col
    return out


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


def _bs_design_matrix(x: np.ndarray, all_knots: np.ndarray, degree: int, boundary_knots: Tuple[float, float]) -> np.ndarray:
    order = degree + 1
    n_basis = len(all_knots) - order
    lo, hi = boundary_knots
    below = x < lo
    above = x > hi
    outside = below | above
    if not outside.any():
        return BSpline.design_matrix(x, all_knots, degree, extrapolate=False).toarray()

    warnings.warn(_OUTSIDE_WARNING, stacklevel=3)
    basis = np.zeros((len(x), n_basis))
    e = 0.25  # R's bs(): "in theory anything in (0,1); was (implicitly) 0 in R <= 3.2.2"
    if below.any():
        pivot = (1 - e) * lo + e * all_knots[order]
        basis[below] = _taylor_extrapolate(x[below], all_knots, degree, degree, pivot)
    if above.any():
        pivot = (1 - e) * hi + e * all_knots[-(order + 1)]
        basis[above] = _taylor_extrapolate(x[above], all_knots, degree, degree, pivot)
    inside = ~outside
    if inside.any():
        basis[inside] = BSpline.design_matrix(x[inside], all_knots, degree, extrapolate=False).toarray()
    return basis


def fit_bs(
    x: np.ndarray,
    df: Optional[int] = None,
    knots: Optional[List[float]] = None,
    degree: int = 3,
    intercept: bool = False,
    boundary_knots: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, BSplineState]:
    x = np.asarray(x, dtype=np.float64)
    order = degree + 1
    if order <= 1:
        raise ValueError("'degree' must be integer >= 1")

    # R's bs(): drops NA rows before computing anything -- including the
    # default `Boundary.knots = range(x)` and interior-knot quantiles, both
    # of which would otherwise themselves become NaN -- then reinserts a
    # full NaN row per dropped position in the output (see apply_bs). Only
    # the clean subset informs knot placement.
    x_clean = x[~np.isnan(x)]
    if x_clean.size == 0:
        raise ValueError("bs(): all values are missing")

    bknots = boundary_knots if boundary_knots is not None else (float(x_clean.min()), float(x_clean.max()))
    outside = (x_clean < bknots[0]) | (x_clean > bknots[1])

    if knots is None and df is not None:
        n_interior = df - order + (0 if intercept else 1)
        if n_interior < 0:
            n_interior = 0
        if n_interior > 0:
            probs = np.linspace(0.0, 1.0, n_interior + 2)[1:-1]
            interior = np.quantile(x_clean[~outside], probs)
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
    nax = np.isnan(x)
    if not nax.any():
        design = _bs_design_matrix(x, state.all_knots, state.degree, state.boundary_knots)
    else:
        clean = _bs_design_matrix(x[~nax], state.all_knots, state.degree, state.boundary_knots)
        design = np.full((len(x), clean.shape[1]), np.nan)
        design[~nax] = clean
    if not state.intercept:
        design = design[:, 1:]
    return design


@dataclass(frozen=True)
class NaturalSplineState:
    """`ns()` is always cubic (order 4) -- R's `ns()` has no `degree`
    argument, unlike `bs()`. `projection` is the frozen null-space basis
    (see `fit_ns`) that enforces "linear beyond the boundary knots"; it's
    part of the basis definition, computed once from training data, and
    reused as-is on new data -- never recomputed."""

    interior_knots: np.ndarray
    boundary_knots: Tuple[float, float]
    intercept: bool
    projection: np.ndarray  # (n_raw_basis, n_raw_basis - 2)

    @property
    def all_knots(self) -> np.ndarray:
        order = 4
        return np.sort(
            np.concatenate(
                (
                    np.repeat(self.boundary_knots[0], order),
                    self.interior_knots,
                    np.repeat(self.boundary_knots[1], order),
                )
            )
        )


def _ns_raw_basis(x: np.ndarray, all_knots: np.ndarray) -> np.ndarray:
    return BSpline.design_matrix(x, all_knots, 3, extrapolate=False).toarray()


def _ns_design_matrix(x: np.ndarray, all_knots: np.ndarray, boundary_knots: Tuple[float, float]) -> np.ndarray:
    lo, hi = boundary_knots
    below = x < lo
    above = x > hi
    outside = below | above
    if not outside.any():
        return _ns_raw_basis(x, all_knots)

    warnings.warn(_OUTSIDE_WARNING, stacklevel=3)
    n_basis = len(all_knots) - 4
    basis = np.zeros((len(x), n_basis))
    # Pivot *at* the boundary itself (unlike bs()'s inward shift): a
    # natural spline is exactly linear beyond the boundary by construction,
    # so a linear Taylor expansion right at the boundary is exact, not an
    # approximation that needs pivoting away from the ill-conditioned knot.
    if below.any():
        basis[below] = _taylor_extrapolate(x[below], all_knots, basis_degree=3, taylor_degree=1, pivot=lo)
    if above.any():
        basis[above] = _taylor_extrapolate(x[above], all_knots, basis_degree=3, taylor_degree=1, pivot=hi)
    inside = ~outside
    if inside.any():
        basis[inside] = _ns_raw_basis(x[inside], all_knots)
    return basis


def _ns_boundary_second_derivatives(all_knots: np.ndarray, boundary_knots: Tuple[float, float]) -> np.ndarray:
    """R's `splineDesign(Aknots, Boundary.knots, ord=4, derivs=c(2,2))`:
    the 2nd derivative of every cubic B-spline basis function, evaluated at
    the two boundary knots -- a (2, n_basis) matrix. `scipy.interpolate`
    has no direct `splineDesign(derivs=)` equivalent, so this evaluates
    each basis function's own `BSpline.derivative(nu=2)` individually (a
    loop over basis functions, not over rows of data -- cheap, n_basis is
    always small)."""
    n_basis = len(all_knots) - 4
    boundary = np.asarray(boundary_knots, dtype=np.float64)
    cols = []
    for j in range(n_basis):
        coef = np.zeros(n_basis)
        coef[j] = 1.0
        d2 = BSpline(all_knots, coef, 3).derivative(nu=2)
        cols.append(d2(boundary))
    return np.column_stack(cols)


def _ns_interior_knots(x: np.ndarray, outside: np.ndarray, df: Optional[int], knots, intercept: bool):
    if knots is None and df is not None:
        n_interior = df - 1 - (1 if intercept else 0)
        if n_interior < 0:
            n_interior = 0
        if n_interior > 0:
            probs = np.linspace(0.0, 1.0, n_interior + 2)[1:-1]
            return np.quantile(x[~outside], probs)
        return np.array([], dtype=np.float64)
    if knots is not None:
        return np.sort(np.asarray(knots, dtype=np.float64))
    return np.array([], dtype=np.float64)


def fit_ns(
    x: np.ndarray,
    df: Optional[int] = None,
    knots: Optional[List[float]] = None,
    intercept: bool = False,
    boundary_knots: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, NaturalSplineState]:
    x = np.asarray(x, dtype=np.float64)

    # Same NA handling as fit_bs: drop before computing default boundary
    # knots / interior-knot quantiles (which would otherwise themselves
    # become NaN), then reinsert full NaN rows in the output via apply_ns.
    x_clean = x[~np.isnan(x)]
    if x_clean.size == 0:
        raise ValueError("ns(): all values are missing")

    bknots = boundary_knots if boundary_knots is not None else (float(x_clean.min()), float(x_clean.max()))
    outside = (x_clean < bknots[0]) | (x_clean > bknots[1])

    interior = _ns_interior_knots(x_clean, outside, df, knots, intercept)
    order = 4
    all_knots = np.sort(
        np.concatenate((np.repeat(bknots[0], order), interior, np.repeat(bknots[1], order)))
    )

    const = _ns_boundary_second_derivatives(all_knots, bknots)
    if not intercept:
        const = const[:, 1:]

    # Null-space projection: functions in the span of the raw basis whose
    # 2nd derivative vanishes at both boundaries are exactly `basis @ v`
    # for v in the null space of `const`. QR of `const.T` (n_raw_basis x 2)
    # gives a full orthonormal Q whose last (n_raw_basis - 2) columns span
    # that null space -- same "QR as an orthonormal-basis tool" pattern as
    # `fit_poly`, and the same LAPACK-vs-LINPACK sign-convention risk that
    # mattered there; verified column-for-column against real R rather
    # than assumed, in tests/terms/test_ns.py. This doesn't depend on the
    # data values at all, only on the knots, so it's computed before the
    # actual per-row basis (which apply_ns below handles, including NA).
    Q, _R = np.linalg.qr(const.T, mode="complete")
    projection = Q[:, 2:]

    state = NaturalSplineState(
        interior_knots=interior, boundary_knots=bknots, intercept=intercept, projection=projection
    )
    return apply_ns(x, state), state


def apply_ns(x: np.ndarray, state: NaturalSplineState) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    nax = np.isnan(x)
    if not nax.any():
        basis = _ns_design_matrix(x, state.all_knots, state.boundary_knots)
    else:
        clean = _ns_design_matrix(x[~nax], state.all_knots, state.boundary_knots)
        basis = np.full((len(x), clean.shape[1]), np.nan)
        basis[~nax] = clean
    if not state.intercept:
        basis = basis[:, 1:]
    return basis @ state.projection
