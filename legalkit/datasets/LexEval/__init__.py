from typing import List
from .dataset import LexEvalDataset
from .generator import Generator
from .evaluator import Evaluator

__all__ = ["LexEvalDataset", "Generator", "Evaluator", "load_tasks"]

def load_tasks(sub_tasks: List[str] = None):
    """
    Factory to load tasks for verdict_pred using hard-coded relative path.
    """
    ds = LexEvalDataset(sub_tasks)
    return ds.load_data()