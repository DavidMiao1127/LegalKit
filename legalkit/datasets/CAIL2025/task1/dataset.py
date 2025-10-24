import os
import json
from typing import List, Dict
from legalkit.datasets.base import BaseDataset, Task

class CAIL2025_Task1_Dataset(BaseDataset):
    """
    Loader for CAIL2025 Task1 choice questions.
    Looks under data/CAIL2025/task1/*.json
    Each item schema example:
    {
        "id": "3_8586",
        "statement": "...题干...",
        "option_list": {"A": "...", "B": "...", ...},
        "answer": ["A", "B"] or "A",
        "subject": "...",
        "type": "0|1"  # optional
    }
    """
    def __init__(self, sub_tasks: List[str] = None):
        # dataset root relative to this file
        pkg_dir = os.path.dirname(__file__)
        self.tasks_dir = os.path.abspath(
            os.path.join(pkg_dir, '..', '..', '..', '..', 'data', 'CAIL2025', 'task1')
        )
        self.sub_tasks = sub_tasks

    def load_data(self) -> List[Task]:
        tasks: List[Task] = []
        for fname in sorted(os.listdir(self.tasks_dir)):
            if not fname.endswith('.json'):
                continue
            task_id = os.path.splitext(fname)[0]
            if self.sub_tasks and task_id not in self.sub_tasks:
                continue
            path = os.path.join(self.tasks_dir, fname)
            with open(path, 'r', encoding='utf-8') as f:
                first_line = f.readline()
                f.seek(0)
                if first_line.strip().startswith('{') and not first_line.strip().startswith('['):
                    items = [json.loads(line) for line in f if line.strip()]
                else:
                    items = json.load(f)
            records = []
            for idx, item in enumerate(items):
                # Normalize answer to keep raw (list or str); evaluator will handle both
                answer = item.get('answer', '')
                # Build record. Use numeric id within task for compatibility with runner storage.
                records.append({
                    'id': idx,
                    'source_id': item.get('id'),
                    'statement': item.get('statement', ''),
                    'option_list': item.get('option_list', {}),
                    'subject': item.get('subject', ''),
                    'type': item.get('type'),
                    'answer': answer,
                })
            tasks.append(Task(id=task_id, records=records))
        return tasks