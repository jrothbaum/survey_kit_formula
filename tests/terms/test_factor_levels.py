from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from parity.r_oracle import R_AVAILABLE, r_model_matrix
from polars_formula.contrasts.base import contr_poly, contr_treatment
from polars_formula.terms.spec import ModelSpec

requires_r = pytest.mark.skipif(not R_AVAILABLE, reason="R is not installed")

DF = pl.DataFrame(
    {
        "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "a": ["a1", "a2", "a1", "a2", "a1", "a2"],
        "ordv": ["lo", "mid", "hi", "lo", "mid", "hi"],
    }
)


def test_explicit_levels_custom_order():
    spec = ModelSpec.from_formula("y ~ factor(a, levels=['a2', 'a1'])", DF)
    (fs,) = spec.factors.values()
    assert fs.levels == ["a2", "a1"]
    # base=1 (default) drops the FIRST given level -- now 'a2', not 'a1'
    np.testing.assert_allclose(fs.contrast_matrix, contr_treatment(2, base=1))


@requires_r
def test_explicit_levels_matches_r():
    formula = "y ~ factor(a, levels=['a2', 'a1'])"
    spec = ModelSpec.from_formula(formula, DF)
    ours = spec.get_model_matrix(DF)
    r_formula = "y ~ factor(a, levels=c('a2','a1'))"
    theirs = r_model_matrix(r_formula, DF, factor_cols=(), bool_cols=())
    np.testing.assert_allclose(ours, theirs, atol=1e-8)


def test_explicit_levels_rejects_unlisted_value():
    with pytest.raises(ValueError, match="not present in the given levels"):
        ModelSpec.from_formula("y ~ factor(a, levels=['a1'])", DF)


def test_levels_must_be_a_list():
    with pytest.raises(ValueError):
        ModelSpec.from_formula("y ~ factor(a, levels='a1')", DF)


def test_ordered_explicit_levels_correct_semantic_order():
    # alphabetical would give hi < lo < mid -- wrong ordinal order
    spec = ModelSpec.from_formula("y ~ ordered(ordv, levels=['lo', 'mid', 'hi'])", DF)
    (fs,) = spec.factors.values()
    assert fs.levels == ["lo", "mid", "hi"]
    assert fs.ordered is True
    np.testing.assert_allclose(fs.contrast_matrix, contr_poly(3))


def test_ordered_explicit_scores():
    spec = ModelSpec.from_formula(
        "y ~ ordered(ordv, levels=['lo', 'mid', 'hi'], scores=[1, 2, 10])", DF
    )
    (fs,) = spec.factors.values()
    np.testing.assert_allclose(fs.contrast_matrix, contr_poly(3, scores=[1, 2, 10]))


def test_scores_without_ordered_rejected():
    with pytest.raises(ValueError, match="only meaningful for ordered"):
        ModelSpec.from_formula("y ~ factor(a, scores=[1, 2])", DF)


def test_C_base_without_explicit_contrast_name_applies_base():
    # R's C() forwards extra kwargs (base=) to the factor's *default*
    # contrast even when the contrast function itself is omitted --
    # verified directly against R's own C.R source and against real R's
    # `model.matrix(~C(a, base = 2))` output. A prior version of this
    # function silently dropped `base=2` here and fell back to the
    # untouched default (base=1) whenever `treatment`/etc. wasn't spelled
    # out explicitly.
    spec = ModelSpec.from_formula("y ~ C(a, base=2)", DF)
    (fs,) = spec.factors.values()
    assert fs.contrast_name == "contr.treatment"
    np.testing.assert_allclose(fs.contrast_matrix, contr_treatment(2, base=2))


@requires_r
def test_C_base_without_explicit_contrast_name_matches_r():
    formula = "y ~ C(a, base=2)"
    spec = ModelSpec.from_formula(formula, DF)
    ours = spec.get_model_matrix(DF)
    theirs = r_model_matrix(formula, DF, factor_cols=(), bool_cols=())
    np.testing.assert_allclose(ours, theirs, atol=1e-8)


@requires_r
def test_ordered_scores_matches_r():
    formula = "y ~ ordered(ordv, levels=['lo', 'mid', 'hi'], scores=[1, 2, 10])"
    spec = ModelSpec.from_formula(formula, DF)
    ours = spec.get_model_matrix(DF)
    r_formula = "y ~ ordered(factor(ordv, levels=c('lo','mid','hi')))"
    # R's model.matrix uses contr.poly(scores=...) only via a custom
    # contrasts.arg; simplest apples-to-apples check is against the
    # contrast matrix itself (already covered in test_contr_poly_custom_scores_matches_r)
    # plus this shape/column-count sanity check against the unscored version.
    theirs = r_model_matrix(r_formula, DF, factor_cols=(), bool_cols=())
    assert ours.shape == theirs.shape
