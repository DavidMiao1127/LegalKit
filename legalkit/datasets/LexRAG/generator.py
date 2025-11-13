from typing import List, Tuple


class Generator:
	"""
	Placeholder Generator for LexRAG. Will be implemented in a later step.
	"""
	def __init__(self, model) -> None:
		self.model = model

	def generate(self, task_id, batch) -> Tuple[List[str], List[str]]:
		raise NotImplementedError("LexRAG Generator will be implemented subsequently.")
