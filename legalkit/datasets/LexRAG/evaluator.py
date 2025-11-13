from typing import Dict, List
from legalkit.datasets.base import BaseEvaluator


class Evaluator(BaseEvaluator):
	"""
	Placeholder Evaluator for LexRAG. Returns empty metrics for now.
	"""
	def evaluate(self, task_id: str, records: List[Dict], predictions: Dict[int, str]) -> Dict[str, float]:
		return {}
