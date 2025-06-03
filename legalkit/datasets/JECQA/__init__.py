from typing import List
from .dataset import JECQADataset
from .generator import Generator
from .evaluator import Evaluator

__all__ = ["JECQADataset", "Generator", "Evaluator", "load_tasks"]

def load_tasks(sub_tasks: List[str] = None):
    """
    Factory to load tasks for verdict_pred using hard-coded relative path.
    """
    ds = JECQADataset(sub_tasks)
    return ds.load_data()