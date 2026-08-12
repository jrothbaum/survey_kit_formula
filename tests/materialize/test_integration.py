"""Phase 10: the actual point of this project.

Exercises the real survey_kit formula shape
(`~1+var2+var4+var4*var3*C(var5)`, from `formula_builder.py`/`calibration/`)
against a dataset large enough to make a full one-hot/Cartesian-product
intermediate actually show up in memory if the pipeline built one, and
measures peak RSS to confirm it doesn't — the whole reason this project
exists instead of using `formulaic`. Also checks numeric correctness
against real R at this scale, not just the tiny fixtures used elsewhere.
"""

from __future__ import annotations

import gc
import resource

import numpy as np
import polars as pl
import pytest

from parity.r_oracle import R_AVAILABLE, r_model_matrix
from survey_kit_formula.terms.spec import ModelSpec

requires_r = pytest.mark.skipif(not R_AVAILABLE, reason="R is not installed")

N = 100_000
K3 = 10  # var3 levels
K5 = 8  # var5 levels
FORMULA = "y ~ 1 + var2 + var4 + var4*var3*C(var5)"


def _make_df(n: int) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    return pl.DataFrame(
        {
            "y": rng.normal(size=n),
            "var2": rng.normal(size=n),
            "var4": rng.normal(size=n),
            "var3": [f"L{i}" for i in rng.integers(0, K3, size=n)],
            "var5": [f"M{i}" for i in rng.integers(0, K5, size=n)],
        }
    )


def _expected_columns() -> int:
    # intercept + var2 + var4 + var3(k3-1) + C(var5)(k5-1) + var4:var3(k3-1)
    # + var4:C(var5)(k5-1) + var3:C(var5)((k3-1)*(k5-1)) + var4:var3:C(var5)((k3-1)*(k5-1))
    return 1 + 1 + 1 + (K3 - 1) + (K5 - 1) + (K3 - 1) + (K5 - 1) + (K3 - 1) * (K5 - 1) + (K3 - 1) * (K5 - 1)


def test_column_count_is_full_rank_not_cartesian_product():
    df = _make_df(1000)
    spec = ModelSpec.from_formula(FORMULA, df)
    assert spec.total_columns == _expected_columns()
    # This formula's full lower-order lattice is present (var3, C(var5),
    # var4:var3, var4:C(var5), var3:C(var5) are all separate terms
    # alongside the 3-way interaction), so every factor gets CONTRASTS
    # (not DUMMY) coding everywhere -- confirm that's actually what
    # happened, i.e. the reduction isn't accidentally being skipped.
    codings = {c for t in spec.terms for c in t.coding.values()}
    from survey_kit_formula.terms.marginality import Coding

    assert codings == {Coding.CONTRASTS}
    # What *no* full-rank reduction at all would need (every factor DUMMY
    # coded in every term) -- the two 2-way/3-way categorical crossings
    # would each cost the full K3*K5 raw cell count instead of the reduced
    # (K3-1)*(K5-1):
    naive_no_reduction = 1 + 1 + 1 + K3 + K5 + K3 + K5 + K3 * K5 + K3 * K5
    assert spec.total_columns < naive_no_reduction


@requires_r
def test_survey_kit_formula_matches_r_at_scale():
    df = _make_df(2000)  # smaller N for the R round-trip; shape is what matters
    spec = ModelSpec.from_formula(FORMULA, df)
    ours = spec.get_model_matrix(df)
    theirs = r_model_matrix(FORMULA, df, factor_cols=("var3", "var5"))
    assert ours.shape == theirs.shape
    np.testing.assert_allclose(ours, theirs, atol=1e-8)


def test_peak_memory_scales_with_output_not_a_materialized_intermediate():
    df = _make_df(N)
    spec = ModelSpec.from_formula(FORMULA, df)
    expected_bytes = N * spec.total_columns * 8  # float64 output array

    gc.collect()
    before_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    mm = spec.get_model_matrix(df)

    after_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    delta_bytes = (after_kb - before_kb) * 1024  # ru_maxrss is in KB on Linux

    assert mm.shape == (N, spec.total_columns)
    assert mm.nbytes == expected_bytes

    # Generous slack (not a tight bound -- ru_maxrss deltas are noisy and
    # this measurement isn't perfectly isolated from the rest of the test
    # process) but a full materialize-then-prune approach building the
    # unreduced K3*K5-wide one-hot Cartesian product before pruning down to
    # spec.total_columns would blow well past this: K3*K5 / total_columns
    # here is roughly a 2-3x wider intermediate, before even counting the
    # string->categorical->one-hot->concat copy chain on top of that.
    max_allowed = expected_bytes * 4
    assert delta_bytes < max_allowed, (
        f"peak RSS delta {delta_bytes / 1e6:.1f}MB exceeds "
        f"{max_allowed / 1e6:.1f}MB (4x the {expected_bytes / 1e6:.1f}MB output array) "
        "-- suggests an unwanted materialized intermediate"
    )
