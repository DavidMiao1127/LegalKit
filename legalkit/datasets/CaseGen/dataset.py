import os
import json
from typing import List, Dict
from legalkit.datasets.base import BaseDataset, Task

class CaseGenDataset(BaseDataset):
    """
    Loader for lawbench dataset.
    Scans relative path: data/lawbench/zero_shot/*.json
    """
    def __init__(self, sub_tasks: List[str] = None):
        # dataset root relative to this file
        pkg_dir = os.path.dirname(__file__)
        self.tasks_dir = os.path.abspath(
            os.path.join(pkg_dir, '..', '..', '..', 'data', 'CaseGen', 'prompt')
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
                rec_id = item.get('id', idx)
                prompt = item.get('prompt', '')

                records.append({
                    'id': rec_id,
                    'prompt': prompt,
                    'defense': item.get('defense', ''),
                    'fact': item.get('fact', ''),
                    'reasoning': item.get('reasoning', ''),
                    'judgment': item.get('judgment', ''),
                })
            tasks.append(Task(id=task_id, records=records))
        return tasks