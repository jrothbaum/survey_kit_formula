from .api import model_matrix
from .parser import parse_formula
from .terms.spec import ModelSpec

__all__ = ["parse_formula", "ModelSpec", "model_matrix"]
