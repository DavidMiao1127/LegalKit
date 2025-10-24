from importlib import import_module
import os
from typing import Dict, Iterable, List, Optional, Tuple

from legalkit.datasets.base import Task

__all__ = [
    "load_tasks",
    "Generator",
    "Evaluator",
    "load_json_predictions",
]


# Module-level routing map: task_id -> submodule name (e.g., "task1")
_TASK_TO_SUBMODULE: Dict[str, str] = {}


def _pkg_dir() -> str:
    return os.path.dirname(__file__)


def _discover_subpackages() -> List[str]:
    pkgs: List[str] = []
    base = _pkg_dir()
    for name in sorted(os.listdir(base)):
        path = os.path.join(base, name)
        if os.path.isdir(path) and os.path.isfile(os.path.join(path, "__init__.py")):
            pkgs.append(name)
    return pkgs


def _parse_subtask_specs(
    sub_tasks: Optional[List[str]],
    available_modules: List[str],
) -> Tuple[Dict[str, Optional[List[str]]], List[str]]:

    selection: Dict[str, Optional[List[str]]]= {}
    legacy_filters: List[str] = []
    if not sub_tasks:
        return selection, legacy_filters

    module_set = set(available_modules)
    for s in sub_tasks:
        if not s:
            continue
        mod = None
        inner = None
        if ":" in s:
            mod, inner = s.split(":", 1)
        elif "/" in s:
            mod, inner = s.split("/", 1)
        elif s in module_set:
            mod = s

        if mod and mod in module_set:
            if inner is None or inner == "":
                selection[mod] = None
            else:
                selection.setdefault(mod, []).append(inner)
        else:
            # legacy filter (applies to all submodules)
            legacy_filters.append(s)

    return selection, legacy_filters


def load_tasks(sub_tasks: Optional[List[str]] = None) -> List[Task]:
    """
    Aggregate tasks from subpackages under this dataset.
    Populates routing so that Generator/Evaluator can dispatch by task_id.
    """
    global _TASK_TO_SUBMODULE
    _TASK_TO_SUBMODULE = {}

    submodules = _discover_subpackages()
    selection, legacy_filters = _parse_subtask_specs(sub_tasks, submodules)

    # If user didn't specify any modules, include all
    modules_to_load: Iterable[str] = selection.keys() if selection else submodules

    all_tasks: List[Task] = []
    for mod_name in modules_to_load:
        mod = import_module(f"legalkit.datasets.CAIL2025.{mod_name}")
        # Determine inner filters for this module
        inner_filters = selection.get(mod_name)
        if inner_filters is None:
            # Either "all" or module-only selection; pass legacy filters if any
            inner = legacy_filters if legacy_filters else None
        else:
            inner = inner_filters

        # Delegate load
        mod_tasks: List[Task] = []
        if hasattr(mod, "load_tasks"):
            mod_tasks = mod.load_tasks(sub_tasks=inner)
        elif hasattr(mod, "Dataset"):
            # Optional compatibility: a class named Dataset with load_data()
            ds = mod.Dataset(sub_tasks=inner)
            mod_tasks = ds.load_data()
        else:
            raise AttributeError(
                f"Submodule CAIL2025.{mod_name} must provide load_tasks(sub_tasks) or Dataset.load_data()."
            )

        for t in mod_tasks:
            tid = str(t.id)
            # If duplicate id across submodules, prefix with module name to ensure uniqueness
            if tid in _TASK_TO_SUBMODULE and _TASK_TO_SUBMODULE[tid] != mod_name:
                new_tid = f"{mod_name}:{tid}"
                try:
                    t.id = new_tid  # mutate dataclass instance
                except Exception:
                    t = Task(id=new_tid, records=t.records)
                tid = new_tid
            _TASK_TO_SUBMODULE[tid] = mod_name
            all_tasks.append(t)

    return all_tasks


class Generator:
    """
    Wrapper Generator that dispatches to the proper submodule based on task_id.
    Supports both batch (List[Dict]) -> (prompts, preds) and single-record -> pred forms.
    """

    def __init__(self, model):
        self._generators: Dict[str, object] = {}
        for mod_name in _discover_subpackages():
            mod = import_module(f"legalkit.datasets.CAIL2025.{mod_name}")
            if hasattr(mod, "Generator"):
                self._generators[mod_name] = mod.Generator(model)

    def _resolve(self, task_id: str):
        mod_name = _TASK_TO_SUBMODULE.get(str(task_id))
        if not mod_name:
            # Fallback: if only one generator, use it; else raise
            if len(self._generators) == 1:
                return next(iter(self._generators.values()))
            raise KeyError(f"No submodule routing found for task_id='{task_id}'. Did you call load_tasks()?")
        gen = self._generators.get(mod_name)
        if not gen:
            raise KeyError(f"Generator for submodule '{mod_name}' not initialized.")
        return gen

    def generate(self, task_id, records):
        gen = self._resolve(task_id)
        # Batch mode
        if isinstance(records, (list, tuple)):
            return gen.generate(task_id, records)
        # Single-record compatibility: wrap and unwrap
        prompts, preds = gen.generate(task_id, [records])
        return preds[0] if preds else ""


class Evaluator:
    """
    Wrapper Evaluator that dispatches evaluation to the submodule owning the task_id.
    """

    def __init__(self):
        self._evaluators: Dict[str, object] = {}
        self._judge_runner = None
        for mod_name in _discover_subpackages():
            mod = import_module(f"legalkit.datasets.CAIL2025.{mod_name}")
            if hasattr(mod, "Evaluator"):
                self._evaluators[mod_name] = mod.Evaluator()

    # The runner will call this if available
    def configure_judge(self, judge_runner, **kwargs) -> None:
        self._judge_runner = judge_runner
        # Propagate config to sub-evaluators if they implement it
        for ev in self._evaluators.values():
            if hasattr(ev, "configure_judge"):
                ev.configure_judge(judge_runner, **kwargs)

    def evaluate(self, task_id: str, records, predictions, origin_prompts=None):
        mod_name = _TASK_TO_SUBMODULE.get(str(task_id))
        if not mod_name:
            if len(self._evaluators) == 1:
                ev = next(iter(self._evaluators.values()))
            else:
                raise KeyError(f"No evaluator routing found for task_id='{task_id}'. Did you call load_tasks()?")
        else:
            ev = self._evaluators.get(mod_name)
            if not ev:
                raise KeyError(f"Evaluator for submodule '{mod_name}' not initialized.")

        # Support both signatures: some evaluators accept origin_prompts as kwarg or ignore extras
        try:
            return ev.evaluate(task_id, records, predictions, origin_prompts)
        except TypeError:
            return ev.evaluate(task_id, records, predictions)


def load_json_predictions(path: str) -> Dict[str, Dict[int, str]]:
    """
    Load predictions from a JSON file for offline eval.

    Expected common format for CAIL2025 task1:
      JSON Lines (each line one JSON object): {"id": "0", "model_answer": "ACD"}

    Also supports a JSON array with objects of the same schema.

    Returns a mapping: {"__default__": {id_int: answer_str}}
    so main.py can evaluate current dataset's tasks and pick default when task id key isn't present.
    """
    import json
    import os

    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON prediction file not found: {path}")

    def parse_entries(entries) -> Dict[int, str]:
        mapping: Dict[int, str] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            rid = entry.get("id")
            ans = entry.get("model_answer")
            if ans is None:
                ans = entry.get("prediction")
            if rid is None or ans is None:
                continue
            try:
                rid_int = int(rid)
            except Exception:
                # Try to split like "3_8586" -> use incremental index? We fallback to skip, to avoid mismatches.
                continue
            mapping[rid_int] = str(ans)
        return mapping

    # Try to detect JSON lines vs JSON array
    preds: Dict[int, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        first = f.readline()
        f.seek(0)
        if first.strip().startswith("{") and not first.strip().startswith("["):
            # JSON lines
            entries = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    entries.append(obj)
                except Exception:
                    continue
            preds = parse_entries(entries)
        else:
            # JSON array or object
            try:
                data = json.load(f)
            except Exception as e:
                raise ValueError(f"Failed to parse JSON from {path}: {e}")

            if isinstance(data, list):
                preds = parse_entries(data)
            elif isinstance(data, dict):
                # Allow {id: answer} or {id: {model_answer:..}}
                mapping: Dict[int, str] = {}
                for k, v in data.items():
                    try:
                        rid_int = int(k)
                    except Exception:
                        continue
                    if isinstance(v, str):
                        mapping[rid_int] = v
                    elif isinstance(v, dict):
                        ans = v.get("model_answer") or v.get("prediction")
                        if ans is not None:
                            mapping[rid_int] = str(ans)
                preds = mapping
            else:
                preds = {}

    return {"__default__": preds}
