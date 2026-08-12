from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from survey_kit_formula.parser.ast_nodes import Call, Identifier
from survey_kit_formula.terms.marginality import Coding
from survey_kit_formula.terms.spec import ModelSpec

DF = pl.DataFrame(
    {
        "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "a": ["a1", "a2", "a1", "a2", "a1", "a2", "a1", "a2"],
        "b": ["b1", "b1", "b2", "b2", "b3", "b3", "b1", "b2"],
    }
)


def test_simple_numeric_and_intercept():
    spec = ModelSpec.from_formula("y ~ x", DF)
    assert spec.intercept is True
    assert spec.total_columns == 2  # intercept + x
    assert len(spec.terms) == 1
    assert spec.terms[0].n_columns == 1


def test_factor_levels_extracted_and_sorted():
    spec = ModelSpec.from_formula("y ~ a", DF)
    (var, fs) = next(iter(spec.factors.items()))
    assert var == Identifier("a")
    assert fs.levels == ["a1", "a2"]
    assert fs.column == "a"
    assert fs.ordered is False
    assert fs.contrast_matrix.shape == (2, 1)


def test_total_columns_matches_marginality():
    # y ~ a + b : a has 2 levels (contrast->1 col), b has 3 levels (contrast->2 cols)
    spec = ModelSpec.from_formula("y ~ a + b", DF)
    assert spec.total_columns == 1 + 1 + 2  # intercept + a + b


def test_interaction_no_main_effects_full_rank():
    spec = ModelSpec.from_formula("y ~ a:b", DF)
    # No preceding terms to cover either margin, so both a (2 levels) and b
    # (3 levels) get full dummy: 2*3=6 (matches R -- see
    # tests/terms/test_marginality.py's "y ~ a:b" case).
    assert spec.total_columns == 1 + 6


def test_no_intercept_upgrades_first_factor():
    spec = ModelSpec.from_formula("y ~ 0 + a", DF)
    assert spec.intercept is False
    assert spec.total_columns == 2  # full dummy, both levels


def test_offset_excluded_from_columns():
    spec = ModelSpec.from_formula("y ~ x + offset(x)", DF)
    assert len(spec.offsets) == 1
    assert spec.offsets[0] == Call("offset", "x")
    assert spec.total_columns == 2  # intercept + x, offset excluded
    assert all(t.term != next(iter(spec.offsets)) for t in spec.terms)


def test_dot_expansion():
    spec = ModelSpec.from_formula("y ~ .", DF)
    referenced_names = set()
    for t in spec.terms:
        for v in t.term.vars:
            assert isinstance(v, Identifier)
            referenced_names.add(v.name)
    assert referenced_names == {"x", "a", "b"}  # not y (response), not "."


def test_poly_spec_built_and_state_stored():
    spec = ModelSpec.from_formula("y ~ poly(x, degree=2)", DF)
    (var, ns) = next(iter(spec.numerics.items()))
    assert isinstance(var, Call) and var.name == "poly"
    assert ns.width == 2
    assert ns.poly_state is not None
    assert spec.total_columns == 1 + 2


def test_bs_spec_built_and_state_stored():
    spec = ModelSpec.from_formula("y ~ bs(x, df=4)", DF)
    (var, ns) = next(iter(spec.numerics.items()))
    assert var.name == "bs"
    assert ns.width == 4
    assert ns.bs_state is not None
    assert spec.total_columns == 1 + 4


def test_dispatch_registered_function_is_numeric_width_one():
    spec = ModelSpec.from_formula("y ~ log(x)", DF)
    (var, ns) = next(iter(spec.numerics.items()))
    assert var.name == "log"
    assert ns.width == 1


def test_unknown_function_raises():
    with pytest.raises(Exception):
        ModelSpec.from_formula("y ~ bogus(x)", DF)


def test_C_with_contrast_override():
    spec = ModelSpec.from_formula("y ~ C(a, treatment, base=2)", DF)
    (var, fs) = next(iter(spec.factors.items()))
    # base=2 -> drop the 2nd level (a2) from the identity matrix
    expected = np.delete(np.eye(2), 1, axis=1)
    np.testing.assert_allclose(fs.contrast_matrix, expected)


def test_lazyframe_input_accepted():
    spec = ModelSpec.from_formula("y ~ x + a", DF.lazy())
    assert spec.total_columns == 1 + 1 + 1


def test_factor_coding_recorded_per_term():
    spec = ModelSpec.from_formula("y ~ a + a:b", DF)
    term_a = next(t for t in spec.terms if t.term.order == 1)
    term_ab = next(t for t in spec.terms if t.term.order == 2)
    a_var = Identifier("a")
    assert term_a.coding[a_var] is Coding.CONTRASTS
    assert term_ab.coding[a_var] is Coding.DUMMY
