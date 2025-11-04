import os
import json
import time
import glob
from threading import Lock

class StorageManager:
    """
    Manages prediction storage per subtask and per GPU.
    Directory structure under run_root:
      run_root/
        predict/
          <subtask>/
            <subtask>_<gpu_id>.json
    """
    def __init__(self, run_root: str, model: str, subtask: str, worker_id: int):
        self.run_root = run_root
        self.subtask = subtask
        self.worker_id = worker_id
        self.model = model
        self.subtask_dir = os.path.join(run_root, model.replace("/", "_"), 'predict', subtask)
        self.init_flag = os.path.join(self.subtask_dir, '.init')
        self.lock = Lock()
        self.existing_preds = {}

    def init(self, check_existing=False):
        # Called by GPU 0 to create the subtask directory
        os.makedirs(self.subtask_dir, exist_ok=True)
        # Create a flag file to signal other GPUs
        open(self.init_flag, 'w').close()
        
        # If resuming, load all existing predictions
        if check_existing:
            self.existing_preds = self.load_existing_predictions(self.run_root, self.model, self.subtask)
            print(f"Loaded {len(self.existing_preds)} existing predictions for subtask {self.subtask}")

    def wait_until_initialized(self):
        # GPUs wait until the directory is ready
        while not os.path.exists(self.init_flag):
            time.sleep(0.1)

    def _file_path(self) -> str:
        return os.path.join(self.subtask_dir, f"{self.subtask}_{self.worker_id}.json")

    def save_pred(self, record_id: int, prediction: str, prompt: str, answer=''):
        path = self._file_path()
        with self.lock:
            data = []
            if os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                except Exception:
                    data = []
            data.append({"id": record_id, "prompt": prompt, "prediction": prediction, "gold":answer})
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

    def is_done(self, record_id: int) -> bool:
        # First check in existing predictions (if resuming)
        if record_id in self.existing_preds:
            return True
            
        # Then check in current worker's file
        path = self._file_path()
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return any(rec.get('id') == record_id for rec in data)
            except Exception:
                return False
        return False

    @staticmethod
    def load_existing_predictions(run_root: str, model: str, subtask: str) -> dict:
        """Load all existing predictions for a subtask across all GPU workers"""
        preds = {}
        subtask_dir = os.path.join(run_root, model.replace("/", "_"), 'predict', subtask)
        
        if not os.path.exists(subtask_dir):
            return preds
            
        # Find all prediction files for this subtask across all GPUs
        pattern = os.path.join(subtask_dir, f"{subtask}_*.json")
        pred_files = glob.glob(pattern)
        
        for pred_file in pred_files:
            if os.path.exists(pred_file):
                try:
                    with open(pred_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        for rec in data:
                            rid = rec.get('id')
                            if rid is not None:
                                preds[rid] = rec.get('prediction', '')
                except Exception as e:
                    print(f"Warning: Could not load predictions from {pred_file}: {e}")
        
        return preds

    @staticmethod
    def load_predictions(run_root: str, model: str, subtask: str) -> dict:
        preds = {}
        subtask_dir = os.path.join(run_root, model.replace("/", "_"), 'predict', subtask)
        if not os.path.isdir(subtask_dir):
            return preds
            
        # Updated to match the correct file naming pattern
        for fname in os.listdir(subtask_dir):
            if fname.endswith('.json') and fname.startswith(f"{subtask}_"):
                path = os.path.join(subtask_dir, fname)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        arr = json.load(f)
                except Exception:
                    continue
                for rec in arr:
                    rid = rec.get('id')
                    if rid is not None:
                        preds[rid] = rec.get('prediction', '')
        return preds

    @staticmethod
    def append_judge_result(
        run_root: str,
        model: str,
        subtask: str,
        entry: dict
    ) -> None:
        """Append a single judge evaluation record to the designated judge log."""
        judge_dir = os.path.join(run_root, model.replace("/", "_"), 'judge', subtask)
        os.makedirs(judge_dir, exist_ok=True)

        payload_path = os.path.join(judge_dir, 'judge_results.json')
        print(f"Appending judge result to {payload_path}")
        data = []
        if os.path.exists(payload_path):
            try:
                with open(payload_path, 'r', encoding='utf-8') as f:
                    data = json.load(f) or []
            except Exception:
                data = []

        if not isinstance(data, list):
            data = []

        data.append(entry)

        with open(payload_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)