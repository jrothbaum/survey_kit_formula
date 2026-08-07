"""Orthogonal polynomial basis fit — the numerical core shared by `poly()`
(`dispatch/poly_bs.py`) and `contr.poly()` (`contrasts/base.py`).

Deliberately a standalone top-level module with zero dependencies on any
other `polars_formula` subpackage (not even `terms`, where it previously
lived): `dispatch.reserved` needs `contrasts.base`, and `terms.spec` needs
both `contrasts` and `dispatch` — so anything both `contrasts` and
`dispatch` depend on has to sit outside all three, or importing any one of
their submodules can trigger the others' `__init__.py` mid-initialization
and fail. Import `polars_formula._numeric_core` directly, never through
`terms`.

Translated directly from R's `stats::poly` (`src/library/stats/R/
contr.poly.R` in R 4.6.1), including the QR sign-canonicalization needed to
match R's LINPACK-based `qr()` column-for-column rather than just up to a
sign flip — see `tests/terms/test_poly_bs.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class PolyState:
    degree: int
    raw: bool = False
    alpha: np.ndarray = None
    norm2: np.ndarray = None


def fit_poly(x: np.ndarray, degree: int, raw: bool = False) -> Tuple[np.ndarray, PolyState]:
    x = np.asarray(x, dtype=np.float64)
    if degree < 1:
        raise ValueError("'degree' must be at least 1")

    if raw:
        # R's poly(x, degree, raw=TRUE): literal powers, no orthogonalization
        # and nothing to freeze -- apply_poly recomputes directly from x.
        Z = x[:, None] ** np.arange(1, degree + 1)[None, :]
        return Z, PolyState(degree=degree, raw=True)

    if np.any(np.isnan(x)):
        raise ValueError("missing values are not allowed in poly()")
    if degree >= len(np.unique(x)):
        raise ValueError("'degree' must be less than number of unique points")

    xbar = x.mean()
    xc = x - xbar
    X = xc[:, None] ** np.arange(degree + 1)[None, :]

    Q, R = np.linalg.qr(X)
    # Canonicalize the QR sign convention (LAPACK's dgeqrf, used by numpy,
    # does not guarantee a positive R-diagonal the way R's LINPACK-based
    # qr() does) so the resulting basis matches R column-for-column, not
    # just up to a per-column sign flip.
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1.0
    Q = Q * signs[None, :]
    rdiag = np.diag(R) * signs

    Z = Q * rdiag[None, :]
    norm2_raw = np.sum(Z**2, axis=0)
    alpha = (np.sum(xc[:, None] * Z**2, axis=0) / norm2_raw + xbar)[:degree]
    norm2 = np.concatenate(([1.0], norm2_raw))

    Zn = Z / np.sqrt(norm2[1:])[None, :]
    Zn = Zn[:, 1:]
    return Zn, PolyState(degree=degree, alpha=alpha, norm2=norm2)


def apply_poly(x: np.ndarray, state: PolyState) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    degree = state.degree
    if state.raw:
        return x[:, None] ** np.arange(1, degree + 1)[None, :]
    alpha, norm2 = state.alpha, state.norm2
    n = len(x)
    Z = np.ones((n, degree + 1))
    if degree >= 1:
        Z[:, 1] = x - alpha[0]
    for i in range(2, degree + 1):
        Z[:, i] = (x - alpha[i - 1]) * Z[:, i - 1] - (norm2[i] / norm2[i - 1]) * Z[:, i - 2]
    Zn = Z / np.sqrt(norm2[1:])[None, :]
    return Zn[:, 1:]
