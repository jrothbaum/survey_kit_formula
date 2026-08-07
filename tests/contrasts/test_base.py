from __future__ import annotations

import numpy as np
import pytest

from parity.r_oracle import R_AVAILABLE, r_contr_poly_scores, r_contrast_matrix
from polars_formula.contrasts.base import (
    contr_helmert,
    contr_poly,
    contr_sas,
    contr_sum,
    contr_treatment,
    resolve_contrast_name,
)
from polars_formula.dispatch.reserved import contrast_override
from polars_formula.parser.ast_nodes import Call

requires_r = pytest.mark.skipif(not R_AVAILABLE, reason="R is not installed")


@requires_r
@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_contr_treatment_matches_r(n):
    ours = contr_treatment(n)
    theirs = r_contrast_matrix("contr.treatment", n)
    np.testing.assert_allclose(ours, theirs)


@requires_r
@pytest.mark.parametrize("n,base", [(3, 2), (4, 3), (5, 1), (5, 5)])
def test_contr_treatment_base_matches_r(n, base):
    ours = contr_treatment(n, base=base)
    theirs = r_contrast_matrix("contr.treatment", n, base=base)
    np.testing.assert_allclose(ours, theirs)


@requires_r
def test_contr_sas_matches_r():
    for n in (2, 3, 5):
        np.testing.assert_allclose(contr_sas(n), r_contrast_matrix("contr.SAS", n))


@requires_r
def test_contr_sum_matches_r():
    for n in (2, 3, 4, 5):
        np.testing.assert_allclose(contr_sum(n), r_contrast_matrix("contr.sum", n))


@requires_r
def test_contr_helmert_matches_r():
    for n in (2, 3, 4, 5, 6):
        np.testing.assert_allclose(contr_helmert(n), r_contrast_matrix("contr.helmert", n))


@requires_r
def test_contr_poly_matches_r():
    for n in (2, 3, 4, 5, 6):
        np.testing.assert_allclose(contr_poly(n), r_contrast_matrix("contr.poly", n), atol=1e-8)


@requires_r
def test_contr_poly_custom_scores_matches_r():
    scores = [1, 2, 10]
    ours = contr_poly(len(scores), scores=scores)
    theirs = r_contr_poly_scores(scores)
    np.testing.assert_allclose(ours, theirs, atol=1e-8)


def test_contr_poly_scores_wrong_length_raises():
    with pytest.raises(ValueError):
        contr_poly(3, scores=[1, 2])


def test_contr_poly_scores_must_be_distinct():
    with pytest.raises(ValueError):
        contr_poly(3, scores=[1, 2, 2])


def test_contr_treatment_out_of_range_base():
    with pytest.raises(ValueError):
        contr_treatment(3, base=4)


def test_contr_functions_require_min_two_levels():
    for fn in (contr_treatment, contr_sum, contr_helmert, contr_poly):
        with pytest.raises(ValueError):
            fn(1)


@pytest.mark.parametrize(
    "shorthand,expected",
    [
        ("treatment", "contr.treatment"),
        ("sum", "contr.sum"),
        ("helmert", "contr.helmert"),
        ("poly", "contr.poly"),
        ("SAS", "contr.SAS"),
        ("contr.sum", "contr.sum"),
    ],
)
def test_resolve_shorthand(shorthand, expected):
    assert resolve_contrast_name(shorthand) == expected


def test_contrast_override_shorthand_bare_name():
    assert contrast_override(Call("C", "x, treatment")) == ("contr.treatment", {})


def test_contrast_override_with_forwarded_kwarg():
    assert contrast_override(Call("C", "x, treatment, base=2")) == ("contr.treatment", {"base": "2"})


def test_contrast_override_fully_qualified_name():
    assert contrast_override(Call("C", "x, contr.sum")) == ("contr.sum", {})


def test_contrast_override_none_without_second_arg():
    assert contrast_override(Call("C", "x")) is None
