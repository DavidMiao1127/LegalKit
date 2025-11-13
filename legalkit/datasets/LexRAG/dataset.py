import os
import json
from typing import List, Dict
from legalkit.datasets.base import BaseDataset, Task

class LexRAGDataset(BaseDataset):
    def __init__(self, sub_tasks: List[str] = None):
        pkg_dir = os.path.dirname(__file__)
        self.tasks_dir = os.path.abspath(
            os.path.join(pkg_dir, '..', '..', '..', 'data', 'LexRAG')
        )
        self.sub_tasks = sub_tasks

    def load_data(self) -> List[Task]:
        tasks: List[Task] = []
        # Map sub-task id to its source question file
        subtask_to_file = {
            "current_question": "current_question.jsonl",
            "prefix_question": "prefix_question.jsonl",
            "prefix_question_answer": "prefix_question_answer.jsonl",
            "suffix_question": "suffix_question.jsonl",
        }

        # Which subtasks to include
        wanted = set(self.sub_tasks) if self.sub_tasks else set(subtask_to_file.keys())

        # Prefer retrieval-enriched files when available (generic per-dataset tag)
        retrieval_tag = os.getenv("LEGALKIT_RETRIEVAL_TAG_LexRAG")  # e.g., bm25_pyserini, qld, bge, gte, openai_model
        retrieval_dir_env = os.getenv("LEGALKIT_RETRIEVAL_DIR_LexRAG")

        for task_id, qfile in subtask_to_file.items():
            if task_id not in wanted:
                continue

            stem = os.path.splitext(os.path.basename(qfile))[0]
            prefer_path = None
            if retrieval_tag:
                # First prefer run_output retrieval dir if provided
                if retrieval_dir_env:
                    candidate = os.path.join(retrieval_dir_env, f"retrieval_{stem}_{retrieval_tag}.jsonl")
                    if os.path.exists(candidate):
                        prefer_path = candidate
                # Fallback to dataset-local retrieval dir (for dev convenience)
                if not prefer_path:
                    candidate = os.path.join(self.tasks_dir, "retrieval", f"retrieval_{stem}_{retrieval_tag}.jsonl")
                    if os.path.exists(candidate):
                        prefer_path = candidate

            src_path = prefer_path or os.path.join(self.tasks_dir, qfile)
            if not os.path.exists(src_path):
                # Skip silently if missing
                continue

            records: List[Dict] = []
            with open(src_path, 'r', encoding='utf-8') as f:
                idx = 0
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        item = json.loads(line)
                    except Exception:
                        continue
                    convs = item.get('conversation', [])
                    for t_idx, conv in enumerate(convs):
                        qobj = conv.get('question', {})
                        question_text = qobj.get('content', '')
                        recall = qobj.get('recall', [])
                        answer = conv.get('assistant', '')
                        records.append({
                            'id': idx,
                            'case_id': item.get('id', None),
                            'turn_id': t_idx,
                            'question': question_text,
                            'recall': recall,
                            'answer': answer,
                        })
                        idx += 1

            tasks.append(Task(id=task_id, records=records))

        return tasks