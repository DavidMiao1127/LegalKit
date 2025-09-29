from abc import ABC, abstractmethod
from typing import List, Dict
from dataclasses import dataclass
from typing import Optional

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
    
    def __init__(self) -> None:
        # Avoid forcing subclasses to call super().__init__ by using setattr fallback
        if not hasattr(self, "_judge_runner"):
            self._judge_runner = None

    @abstractmethod
    def evaluate(self, task_id: str, records: List[Dict], predictions: Dict[int, str]) -> Dict[str, float]:
        pass

    def configure_judge(self, judge_runner, **kwargs) -> None:
        self._judge_runner = judge_runner

    @property
    def judge_runner(self):
        return getattr(self, "_judge_runner", None)

    def supports_llm_judge(self) -> bool:
        return False

    def has_judge(self) -> bool:
        return self.judge_runner is not None

    def evaluate_with_judge(self, task_id: str, records: List[Dict], predictions: Dict[int, str]) -> Optional[Dict[str, float]]:
        """Optional judge-assisted evaluation hook."""
        raise NotImplementedError("LLM judge evaluation not implemented for this evaluator")