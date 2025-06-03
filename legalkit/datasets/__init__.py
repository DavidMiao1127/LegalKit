"""
Datasets module: register supported benchmark datasets.
"""

from . import LawBench
from . import LexEval
from .utils import clean_prediction

__all__ = ["LawBench", "LexEval", "clean_prediction"]
