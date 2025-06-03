from typing import List, Dict, Tuple

class Generator:
    def __init__(self, model):
        self.model = model

    def generate(
        self,
        task_id: str,
        records: List[Dict]
    ) -> Tuple[List[str], List[str]]:
        prompts = [
            f"{rec['instruction']}\n{rec['question']}"
            for rec in records
        ]
        generated_list = self.model.generate(prompts)
        return prompts, generated_list

