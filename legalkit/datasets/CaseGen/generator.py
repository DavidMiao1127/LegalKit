from typing import Dict, List, Tuple, Union

class Generator:
    def __init__(self, model):
        self.model = model

    def generate(self, task_id: str, record_or_records: Union[Dict, List[Dict]]) -> Union[str, Tuple[List[str], List[str]]]:
        if isinstance(record_or_records, list):
            records: List[Dict] = record_or_records
            prompts: List[str] = []
            for rec in records:
                # Prefer 'prompt', fallback to common fields
                text = rec.get('prompt') or rec.get('input') or rec.get('question') or ''
                prompts.append(str(text))
            preds: List[str] = self.model.generate(prompts)
            return prompts, preds

        rec: Dict = record_or_records
        text = rec.get('prompt') or rec.get('input') or rec.get('question') or ''
        prompt = str(text)
        return self.model.generate(prompt)
