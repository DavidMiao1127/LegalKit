from abc import ABC, abstractmethod
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class Task:
    id: str
    records: List[Dict]

class BaseDataset(ABC):
    """Abstract base class for datasets."""

    @abstractmethod
    def load_data(self):
        """Load and return the dataset."""
        pass

class BaseEvaluator(ABC):
    """Abstract base class for evaluators."""
    @abstractmethod
    def evaluate(self, task_id: str, records: List[Dict], predictions: Dict[int, str]) -> Dict[str, float]:
        pass