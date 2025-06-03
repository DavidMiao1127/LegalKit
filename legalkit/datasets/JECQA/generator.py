from typing import List, Dict, Tuple

class Generator:
    """
    Prompt builder and inference for verdict_pred.
    """
    def __init__(self, model):
        self.model = model

    def generate(
        self,
        task_id: str,
        records: List[Dict]
    ) -> Tuple[List[str], List[str]]:
        prompts = [
            f"{rec['instruction']}\n{rec['statement']}\n{rec['option_list']}"
            for rec in records
        ]
        generated_list = self.model.generate(prompts)
        return prompts, generated_list
    
    
