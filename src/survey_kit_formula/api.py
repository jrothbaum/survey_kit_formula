"""Public entry points."""

from __future__ import annotations

from typing import Union

import numpy as np
import polars as pl

from .terms.spec import ModelSpec


def model_matrix(formula: str, data: Union[pl.DataFrame, pl.LazyFrame]) -> np.ndarray:
    """One-shot equivalent of R's `model.matrix(formula, data)`. For the
    fit-once/reapply-to-new-data pattern (survey_kit's
    `model_spec.get_model_matrix(new_df)` usage), build a `ModelSpec`
    directly and keep it around instead of calling this repeatedly."""
    return ModelSpec.from_formula(formula, data).get_model_matrix(data)
