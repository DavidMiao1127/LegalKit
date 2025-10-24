from typing import List
from .dataset import CAIL2025_Task1_Dataset
from .generator import Generator
from .evaluator import Evaluator

__all__ = ["CAIL2025_Task1_Dataset", "Generator", "Evaluator", "load_tasks"]

def load_tasks(sub_tasks: List[str] = None):
    """
    Factory to load tasks for verdict_pred using hard-coded relative path.
    """
    ds = CAIL2025_Task1_Dataset(sub_tasks)
    return ds.load_data()