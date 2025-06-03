import os
import json
from typing import List, Dict
from legalkit.datasets.base import BaseDataset, Task

class LawBenchDataset(BaseDataset):
    """
    Loader for lawbench dataset.
    Scans relative path: data/lawbench/zero_shot/*.json
    """
    def __init__(self, sub_tasks: List[str] = None):
        # dataset root relative to this file
        pkg_dir = os.path.dirname(__file__)
        self.tasks_dir = os.path.abspath(
            os.path.join(pkg_dir, '..', '..', '..', 'data', 'lawbench', 'zero_shot')
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
            items = json.load(open(path, 'r', encoding='utf-8'))
            records = []
            for idx, item in enumerate(items):
                records.append({
                    'id': idx,
                    'instruction': item.get('instruction', ''),
                    'question': item.get('question', ''),
                    'answer': item.get('answer', '')
                })
            tasks.append(Task(id=task_id, records=records))
        return tasks