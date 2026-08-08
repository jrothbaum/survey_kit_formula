"""`poly()`/`bs()` parity tests against real R — the QR sign convention in
`poly()` and the knot-placement/basis convention in `bs()` are exactly the
kind of thing that silently diverges without checking against real output.
"""

from __future__ import annotations

import numpy as np
import pytest

from parity.r_oracle import R_AVAILABLE, r_bs_matrix, r_bs_predict_outside, r_poly_matrix
from polars_formula.dispatch.poly_bs import apply_bs, apply_poly, fit_bs, fit_poly

requires_r = pytest.mark.skipif(not R_AVAILABLE, reason="R is not installed")

X = np.array([1.0, 2.5, 3.0, 4.2, 5.0, 6.1, 7.0, 8.3, 9.0, 10.5, 11.0, 12.7])


@requires_r
@pytest.mark.parametrize("degree", [1, 2, 3, 4])
def test_poly_matches_r(degree):
    ours, _state = fit_poly(X, degree)
    theirs = r_poly_matrix(X, degree)
    np.testing.assert_allclose(ours, theirs, atol=1e-8)


@requires_r
def test_poly_predict_matches_r_refit_basis():
    # R's poly() re-fit on the same data == the stored-state reapplication
    # on that same data (sanity check that apply_poly reproduces fit_poly).
    degree = 3
    fitted, state = fit_poly(X, degree)
    reapplied = apply_poly(X, state)
    np.testing.assert_allclose(fitted, reapplied, atol=1e-10)


@requires_r
def test_poly_predict_on_new_data():
    degree = 3
    _fitted, state = fit_poly(X, degree)
    new_x = np.array([2.0, 4.0, 6.0, 8.0])
    ours = apply_poly(new_x, state)
    # R equivalent: predict(poly(X, degree=3), newdata = new_x)
    # emulate via the oracle by fitting poly on X then using stats:::predict.poly
    theirs = _r_poly_predict(X, new_x, degree)
    np.testing.assert_allclose(ours, theirs, atol=1e-8)


def _r_poly_predict(x_train, x_new, degree):
    import subprocess
    import tempfile

    import numpy as np

    script = r"""
args <- commandArgs(trailingOnly = TRUE)
xtr <- scan(args[1], quiet = TRUE)
xnew <- scan(args[2], quiet = TRUE)
degree <- as.integer(args[3])
p <- poly(xtr, degree = degree)
Z <- predict(p, xnew)
write.csv(unclass(Z), args[4], row.names = FALSE)
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".R", delete=False) as sf:
        sf.write(script)
        script_path = sf.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f1:
        f1.write("\n".join(str(v) for v in x_train))
        xtr_path = f1.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f2:
        f2.write("\n".join(str(v) for v in x_new))
        xnew_path = f2.name
    out_path = tempfile.NamedTemporaryFile(suffix=".csv", delete=False).name
    result = subprocess.run(
        ["Rscript", script_path, xtr_path, xnew_path, str(degree), out_path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    return np.loadtxt(out_path, delimiter=",", skiprows=1, ndmin=2)


@requires_r
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_poly_raw_matches_r(degree):
    ours, state = fit_poly(X, degree, raw=True)
    assert state.raw is True
    theirs = _r_poly_raw(X, degree)
    np.testing.assert_allclose(ours, theirs)
    reapplied = apply_poly(X, state)
    np.testing.assert_allclose(ours, reapplied)


def _r_poly_raw(x, degree):
    import subprocess
    import tempfile

    import numpy as np

    script = r"""
args <- commandArgs(trailingOnly = TRUE)
x <- scan(args[1], quiet = TRUE)
degree <- as.integer(args[2])
Z <- poly(x, degree = degree, raw = TRUE)
write.csv(unclass(Z), args[3], row.names = FALSE)
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".R", delete=False) as sf:
        sf.write(script)
        script_path = sf.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f1:
        f1.write("\n".join(str(v) for v in x))
        x_path = f1.name
    out_path = tempfile.NamedTemporaryFile(suffix=".csv", delete=False).name
    result = subprocess.run(
        ["Rscript", script_path, x_path, str(degree), out_path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    return np.loadtxt(out_path, delimiter=",", skiprows=1, ndmin=2)


def test_poly_rejects_too_few_unique_points():
    with pytest.raises(ValueError):
        fit_poly(np.array([1.0, 1.0, 1.0, 2.0]), degree=3)


@requires_r
@pytest.mark.parametrize("df", [4, 5, 6])
def test_bs_matches_r_df(df):
    ours, _state = fit_bs(X, df=df)
    theirs = r_bs_matrix(X, df=df)
    np.testing.assert_allclose(ours, theirs, atol=1e-8)


@requires_r
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_bs_matches_r_degree(degree):
    ours, _state = fit_bs(X, df=5, degree=degree)
    theirs = r_bs_matrix(X, df=5, degree=degree)
    np.testing.assert_allclose(ours, theirs, atol=1e-8)


@requires_r
def test_bs_matches_r_intercept():
    ours, _state = fit_bs(X, df=5, intercept=True)
    theirs = r_bs_matrix(X, df=5, intercept=True)
    np.testing.assert_allclose(ours, theirs, atol=1e-8)


@requires_r
def test_bs_explicit_knots_matches_r():
    ours, _state = fit_bs(X, knots=[3.0, 6.0, 9.0])
    theirs = _r_bs_knots(X, [3.0, 6.0, 9.0])
    np.testing.assert_allclose(ours, theirs, atol=1e-8)


def _r_bs_knots(x, knots):
    import subprocess
    import tempfile

    import numpy as np

    script = r"""
args <- commandArgs(trailingOnly = TRUE)
x <- scan(args[1], quiet = TRUE)
knots <- scan(args[2], quiet = TRUE)
library(splines)
Z <- bs(x, knots = knots)
write.csv(unclass(Z), args[3], row.names = FALSE)
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".R", delete=False) as sf:
        sf.write(script)
        script_path = sf.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f1:
        f1.write("\n".join(str(v) for v in x))
        x_path = f1.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f2:
        f2.write("\n".join(str(v) for v in knots))
        knots_path = f2.name
    out_path = tempfile.NamedTemporaryFile(suffix=".csv", delete=False).name
    result = subprocess.run(
        ["Rscript", script_path, x_path, knots_path, out_path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    return np.loadtxt(out_path, delimiter=",", skiprows=1, ndmin=2)


@requires_r
def test_bs_apply_reproduces_fit():
    ours, state = fit_bs(X, df=5)
    reapplied = apply_bs(X, state)
    np.testing.assert_allclose(ours, reapplied, atol=1e-10)


@requires_r
def test_bs_na_row_passthrough_matches_r():
    # R's bs()/ns() drop NA rows before computing anything (including knot
    # placement), fit on the complete cases, then reinsert a full NaN row
    # at the original position -- not an error, not zero-filled.
    x_na = np.array([1.0, 2.0, 3.0, np.nan, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    ours, state = fit_bs(x_na, df=5)
    theirs = r_bs_matrix(x_na, df=5, allow_na=True)
    assert np.isnan(ours[3]).all()
    np.testing.assert_allclose(ours, theirs, atol=1e-8, equal_nan=True)
    # knot placement itself must exclude the NA row, not just the output
    assert state.boundary_knots == (1.0, 12.0)


def test_bs_all_nan_raises():
    with pytest.raises(ValueError, match="all values are missing"):
        fit_bs(np.full(5, np.nan), df=3)


def test_bs_out_of_range_warns_not_raises():
    _fitted, state = fit_bs(X, df=5)
    with pytest.warns(UserWarning, match="ill-conditioned"):
        out = apply_bs(np.array([-100.0]), state)
    assert out.shape == (1, 5)


@requires_r
def test_bs_out_of_range_matches_r():
    # matches R's own warning text exactly, not just "warns something"
    newx = np.array([-5.0, 0.0, 1.0, 6.0, 12.7, 15.0, 20.0])
    _fitted, state = fit_bs(X, df=5)
    with pytest.warns(UserWarning, match="some 'x' values beyond boundary knots may cause ill-conditioned bases"):
        ours = apply_bs(newx, state)
    theirs = r_bs_predict_outside(X, newx, df=5)
    np.testing.assert_allclose(ours, theirs, atol=1e-6)
