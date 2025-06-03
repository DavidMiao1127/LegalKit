from typing import List
from .dataset import CaseGenDataset
from .generator import Generator
from .evaluator import Evaluator

__all__ = ["CaseGenDataset", "Generator", "Evaluator", "load_tasks"]

def load_tasks(sub_tasks: List[str] = None):
    """
    Factory to load tasks for verdict_pred using hard-coded relative path.
    """
    ds = CaseGenDataset(sub_tasks)
    return ds.load_data()