from typing import Dict

class Generator:
    """
    Prompt builder and inference for verdict_pred.
    """
    def __init__(self, model):
        self.model = model

    def generate(self, task_id: str, record: Dict) -> str:
        prompt = f"{record['prompt']}"
        return self.model.generate(prompt)
