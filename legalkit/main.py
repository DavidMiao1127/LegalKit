import os
import json
import argparse
import yaml
import torch
from tqdm import tqdm
from importlib import import_module
import torch.multiprocessing as mp
from multiprocessing import Barrier, get_context
import torch.distributed as dist
import gc
from datetime import datetime
from typing import Any, Dict, Iterable, List

from legalkit.models import build_model
from legalkit.storage import StorageManager
from legalkit.judge import JudgeConfig, LLMJudgeRunner

JSON_MODEL_PREFIX = "json::"
JSON_DEFAULT_TASK_KEY = "__default__"

def parse_args():
    parser = argparse.ArgumentParser(
        description="LegalKit Strategy Runner",
        argument_default=argparse.SUPPRESS
    )
    parser.add_argument("--config", type=str, help="YAML config file path")
    parser.add_argument("--models", nargs='+', help="Model specs: local paths, hf shortcuts, or api labels")
    parser.add_argument("--datasets", nargs='+', help="Dataset names")
    parser.add_argument("-a", "--accelerator", choices=["vllm", "lmdeploy", "sglang"], help="Acceleration backend")
    parser.add_argument("--num_workers", type=int, help="Number of parallel workers (data parallelism)")
    parser.add_argument("--tensor_parallel", type=int, help="Tensor parallelism degree (model parallelism)")
    parser.add_argument("--task", choices=["infer", "eval", "all"], help="Phase to run")
    parser.add_argument("-r", "--resume", type=str, help="Path to existing run directory to resume")
    parser.add_argument("--output_dir", type=str, default="./run_output", help="Directory for new outputs")
    parser.add_argument("--temperature", type=float, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, help="Top-p sampling")
    parser.add_argument("--max_tokens", type=int, help="Maximum number of tokens to generate")
    parser.add_argument("--sub_tasks", nargs='+', help="Sub-tasks to run")
    parser.add_argument("--batch_size", type=int, help="Batch size of generation")
    parser.add_argument("--repetition_penalty", type=float, help="Repetition penalty for generation")
    parser.add_argument("--api_url", type=str, help="URL of api generation")
    parser.add_argument("--api_key", type=str, help="URL key of api generation")
    parser.add_argument("--json_eval", action="store_true", help="Use pre-generated JSON predictions for evaluation")
    parser.add_argument("--json_paths", nargs='+', help="JSON prediction files (use dataset=path for multiple datasets)")
    parser.add_argument("--json_model_label", type=str, help="Model label to use when running JSON-only evaluation")
    parser.add_argument("--judge", type=str, help="Model spec for LLM-as-Judge evaluation")
    parser.add_argument("--judge_accelerator", choices=["vllm", "lmdeploy"], help="Acceleration backend for judge model")
    parser.add_argument("--judge_tensor_parallel", type=int, help="Tensor parallelism degree for judge model")
    parser.add_argument("--judge_batch_size", type=int, help="Batch size when querying the judge model")
    parser.add_argument("--judge_temperature", type=float, help="Sampling temperature for judge model")
    parser.add_argument("--judge_top_p", type=float, help="Top-p sampling for judge model")
    parser.add_argument("--judge_max_tokens", type=int, help="Maximum number of tokens to generate for judge model")
    parser.add_argument("--judge_repetition_penalty", type=float, help="Repetition penalty for judge model")
    parser.add_argument("--judge_api_url", type=str, help="Override API URL for judge model")
    parser.add_argument("--judge_api_key", type=str, help="Override API key for judge model")
    parser.add_argument("--few_shot", action="store_true", help="Enable few-shot prompting")
    return parser.parse_args()

def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def merge_config_with_args(cfg_dict, args):
    merged = cfg_dict.copy()
    for k, v in vars(args).items():
        if v is not None and k != "config":
            merged[k] = v
    return merged

def _format_metrics(metrics: dict) -> str:
    """Return a compact human-readable string for a metrics dict.

    Behaviors:
    - If 'score' exists, show it as primary.
    - Otherwise, list available scalar metrics sorted by key.
    - For numeric values, format to 4 decimal places; for dict/list show a short repr.
    """
    if not metrics:
        return '{}'
    # Prefer primary 'score' field if present
    if 'score' in metrics:
        try:
            val = float(metrics['score'])
            return f"score={val:.4f}"
        except Exception:
            return f"score={metrics['score']}"

    parts = []
    for k in sorted(metrics.keys()):
        v = metrics[k]
        if v is None:
            parts.append(f"{k}=None")
        elif isinstance(v, (int, float)):
            try:
                parts.append(f"{k}={float(v):.4f}")
            except Exception:
                parts.append(f"{k}={v}")
        elif isinstance(v, str):
            short = v if len(v) < 60 else (v[:57] + '...')
            parts.append(f"{k}='{short}'")
        else:
            # For lists/dicts and other complex objects show shortened json
            try:
                s = json.dumps(v, ensure_ascii=False)
                short = s if len(s) < 120 else (s[:117] + '...')
                parts.append(f"{k}={short}")
            except Exception:
                parts.append(f"{k}={str(v)}")
    return ', '.join(parts)

def parse_json_path_specs(specs: List[str], datasets: List[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    default_path = None
    for spec in specs:
        if "=" in spec:
            ds_name, path = spec.split("=", 1)
            ds_name = ds_name.strip()
            if not ds_name:
                raise ValueError(f"Invalid dataset label in json path spec '{spec}'")
            mapping[ds_name] = path.strip()
        else:
            if default_path is not None:
                raise ValueError("Multiple JSON paths provided without dataset labels; please use 'dataset=path' format.")
            default_path = spec

    if default_path:
        if not datasets or len(datasets) != 1:
            raise ValueError("A single unlabeled JSON path can only be used when exactly one dataset is specified.")
        mapping[datasets[0]] = default_path

    if not mapping:
        raise ValueError("No valid JSON path provided for evaluation.")

    return mapping


def _parse_prediction_entries(entries: Any, source: str) -> Dict[int, str]:
    result: Dict[int, str] = {}
    if isinstance(entries, list):
        iterator: Iterable = entries
    elif isinstance(entries, dict):
        iterator = entries.items()
    else:
        raise ValueError(f"Unsupported prediction entry type in {source}: {type(entries)}")

    if isinstance(entries, list):
        for entry in iterator:
            if not isinstance(entry, dict):
                raise ValueError(f"Entries in {source} must be objects containing 'id' and 'prediction'.")
            if "id" not in entry or "prediction" not in entry:
                raise ValueError(f"Each entry in {source} must contain 'id' and 'prediction'.")
            rid = entry["id"]
            try:
                rid_int = int(rid)
            except (TypeError, ValueError):
                raise ValueError(f"Record id '{rid}' in {source} is not an integer.") from None
            result[rid_int] = entry["prediction"]
        return result

    # entries is dict
    for key, value in iterator:
        if isinstance(value, str):
            rid = key
            pred = value
        elif isinstance(value, dict):
            rid = value.get("id", key)
            pred = value.get("prediction")
            if pred is None:
                raise ValueError(f"Missing 'prediction' for record '{key}' in {source}.")
        else:
            raise ValueError(f"Unsupported prediction record type in {source}: {type(value)}")
        try:
            rid_int = int(rid)
        except (TypeError, ValueError):
            raise ValueError(f"Record id '{rid}' in {source} is not an integer.") from None
        result[rid_int] = pred
    return result


def load_default_json_predictions(path: str) -> Dict[str, Dict[int, str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON prediction file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse JSON predictions from {path}: {exc}") from exc

    if isinstance(data, list):
        return {JSON_DEFAULT_TASK_KEY: _parse_prediction_entries(data, path)}

    if isinstance(data, dict):
        predictions: Dict[str, Dict[int, str]] = {}

        if "tasks" in data and isinstance(data["tasks"], dict):
            for task_id, records in data["tasks"].items():
                predictions[str(task_id)] = _parse_prediction_entries(records, path)
            return predictions

        if "task" in data and "predictions" in data:
            task_id = str(data.get("task", JSON_DEFAULT_TASK_KEY))
            predictions[task_id] = _parse_prediction_entries(data["predictions"], path)
            return predictions

        try:
            single_task = _parse_prediction_entries(data, path)
        except ValueError:
            pass
        else:
            return {JSON_DEFAULT_TASK_KEY: single_task}

        for key, value in data.items():
            predictions[str(key)] = _parse_prediction_entries(value, path)
        if predictions:
            return predictions

    raise ValueError(f"Unsupported JSON prediction format in {path}")

def wrap_accelerator(model, accel, num_workers, tensor_parallel, gen_cfg, worker_id):
    if not accel:
        return model
    mod = import_module(f"legalkit.accelerator.{accel}_backend")
    return getattr(mod, f"{accel.upper()}Accelerator")(
        model,
        num_workers=num_workers,
        tensor_parallel=tensor_parallel,
        gen_cfg=gen_cfg,
        worker_id=worker_id
    )

def discover_models(spec):
    """discover models from a directory or a list of model paths"""
    models = []
    if not os.path.exists(spec):
        print(f"Warning: Path '{spec}' does not exist")
        return models
        
    if os.path.isfile(os.path.join(spec, "config.json")):
        models.append({"model_type": "local", "model_path": spec})
    else:
        for item in os.listdir(spec):
            subdir = os.path.join(spec, item)
            if os.path.isdir(subdir) and os.path.isfile(os.path.join(subdir, "config.json")):
                models.append({"model_type": "local", "model_path": subdir})
                
    if models:
        print(f"Discovered {len(models)} models in '{spec}':")
        for i, mcfg in enumerate(models):
            print(f"  [{i+1}] {mcfg['model_path']}")
    else:
        print(f"Warning: No valid models found in '{spec}'")
        
    return models

def run_worker(worker_id, num_workers, merged_args, cfg, run_root, barrier):
    os.environ['LOCAL_RANK'] = str(worker_id)
    os.environ['RANK'] = str(worker_id)
    print(f"Worker {worker_id} started with PID {os.getpid()}")
    
    is_resuming = 'resume' in merged_args and merged_args['resume']
    
    models = merged_args["models"]
    datasets = merged_args["datasets"]
    accelerator = merged_args.get("accelerator")
    task_phase = merged_args.get("task", "all")
    temperature = merged_args.get("temperature", 1.0)
    top_p = merged_args.get("top_p", 1.0)
    max_tokens = merged_args.get("max_tokens", 8192)
    sub_tasks = merged_args.get("sub_tasks")
    tensor_parallel = merged_args.get("tensor_parallel", 1)
    batch_size = merged_args.get("batch_size", 1)
    rep_penalty = merged_args.get("repetition_penalty", 1)
    gen_cfg = {"temperature": temperature, "top_p": top_p, "max_tokens": max_tokens, "rep_penalty": rep_penalty}
    api_url = merged_args.get("api_url")
    api_key = merged_args.get("api_key")
    json_eval_mode = bool(merged_args.get("json_eval"))
    json_inputs: Dict[str, str] = merged_args.get("json_inputs", {})
    json_model_label = merged_args.get("json_model_label") or "json_eval"
    requires_inference = (task_phase in ("infer", "all")) and not json_eval_mode
    judge_cfg_dict = merged_args.get("judge_config")
    judge_debug = bool(merged_args.get("judge_debug", False))
    # Set generic few-shot environment flag per worker
    if merged_args.get("few_shot"):
        os.environ["FEW_SHOT"] = "1"
    else:
        os.environ.pop("FEW_SHOT", None)
    judge_runner = None
    if worker_id == 0 and judge_cfg_dict:
        judge_runner = LLMJudgeRunner(JudgeConfig(**judge_cfg_dict), worker_id=worker_id)
    
    for spec in models:
        if dist.is_initialized():
            dist.destroy_process_group()

        if isinstance(spec, str) and spec.startswith(JSON_MODEL_PREFIX):
            label = spec[len(JSON_MODEL_PREFIX):] or json_model_label
            mcfgs = [{"model_type": "json", "model_name": label}]
        elif os.path.isdir(spec):
            mcfgs = discover_models(spec)
            if not mcfgs:
                mcfgs = [{"model_type": "local", "model_path": spec}]
        elif spec.startswith("hf:"):
            mcfgs = [{"model_type": "hf", "model_name": spec.split("hf:")[1]}]
        elif spec.startswith("api:"):
            mcfgs = [{"model_type": "api", "model_name": spec.split("api:")[1], "api_url": api_url, "api_key": api_key}]
        else:
            mcfgs = [{"model_type": "local", "model_path": spec}]
            
        for mcfg in mcfgs:
            model = None
            model_id = None

            if mcfg["model_type"] == "json":
                model_id = mcfg.get("model_name") or json_model_label
                if requires_inference:
                    raise ValueError("JSON-only evaluation specs do not support inference.")

            elif mcfg["model_type"] == "local":
                model_id = mcfg['model_path']
                if requires_inference:
                    mcfg['gen_cfg'] = gen_cfg
                    mcfg['worker_id'] = worker_id
                    if accelerator:
                        mcfg['device'] = "meta"
                        model = build_model(**mcfg)
                        model = wrap_accelerator(model, accelerator, num_workers, tensor_parallel, gen_cfg, worker_id)
                    else:
                        mcfg['device'] = "cuda"
                        model = build_model(**mcfg)

            elif mcfg["model_type"] == "hf":
                model_id = mcfg['model_name']
                if requires_inference:
                    mcfg['gen_cfg'] = gen_cfg
                    mcfg['worker_id'] = worker_id
                    mcfg['device'] = "cuda"
                    model = build_model(**mcfg)

            elif mcfg["model_type"] == "api":
                model_id = mcfg['model_name']
                if requires_inference:
                    mcfg['gen_cfg'] = gen_cfg
                    model = build_model(**mcfg)

            if not model_id:
                model_id = json_model_label
            model_id = str(model_id)

            for ds in datasets:
                ds_mod = import_module(f"legalkit.datasets.{ds}")
                tasks = ds_mod.load_tasks(sub_tasks=sub_tasks)
                dataset_json_cache = None
                if json_eval_mode:
                    json_path = json_inputs.get(ds)
                    if not json_path:
                        raise ValueError(f"JSON evaluation requested but no file provided for dataset '{ds}'.")
                    if hasattr(ds_mod, "load_json_predictions"):
                        dataset_json_cache = ds_mod.load_json_predictions(json_path)
                    else:
                        dataset_json_cache = load_default_json_predictions(json_path)
                    if not isinstance(dataset_json_cache, dict):
                        raise ValueError(f"Dataset '{ds}' JSON loader must return a dict mapping task ids to predictions.")
                
                # If "all", do per-subtask inference + eval
                if task_phase == "all":
                    results = {} if worker_id == 0 else None
                    evaluator = ds_mod.Evaluator() if worker_id == 0 else None
                    if evaluator and judge_runner and hasattr(evaluator, "configure_judge"):
                        evaluator.configure_judge(
                            judge_runner,
                            dataset=ds,
                            run_root=run_root,
                            model_id=model_id,
                            debug=judge_debug
                        )
                    for task in tasks:
                        if requires_inference:
                            storage = StorageManager(run_root, model_id, task.id, worker_id)
                            if worker_id == 0:
                                storage.init(check_existing=is_resuming)
                            else:
                                storage.wait_until_initialized()
                                if is_resuming:
                                    storage.existing_preds = storage.load_existing_predictions(run_root, model_id, task.id)

                            generator = ds_mod.Generator(model)

                            if is_resuming:
                                records = [rec for rec in task.records if rec['id'] not in storage.existing_preds]
                                if worker_id == 0:
                                    print(f"Resuming task {task.id}: {len(records)}/{len(task.records)} records left to process")
                            else:
                                records = task.records

                            assigned_recs = [
                                rec for idx, rec in enumerate(records)
                                if idx % num_workers == worker_id
                            ]

                            loop = (
                                tqdm(
                                    range(0, len(assigned_recs), batch_size),
                                    desc=f"Gen {task.id}",
                                )
                                if worker_id == 0
                                else range(0, len(assigned_recs), batch_size)
                            )

                            for i in loop:
                                batch = assigned_recs[i: i + batch_size]
                                prompts, preds = generator.generate(task.id, batch)
                                for rec, prompt, pred in zip(batch, prompts, preds):
                                    if 'answer' in rec:
                                        storage.save_pred(rec['id'], pred, prompt, rec['answer'])
                                    else:
                                        storage.save_pred(rec['id'], pred, prompt)

                        barrier.wait()
                        # Evaluation (only worker 0)
                        if worker_id == 0:
                            if json_eval_mode:
                                task_key = task.id if isinstance(task.id, str) else str(task.id)
                                preds = dataset_json_cache.get(task_key) if dataset_json_cache is not None else None
                                if preds is None and dataset_json_cache is not None:
                                    preds = dataset_json_cache.get(JSON_DEFAULT_TASK_KEY)
                                if preds is None:
                                    raise ValueError(f"No predictions found in JSON file for dataset '{ds}' task '{task.id}'.")
                            else:
                                preds = StorageManager.load_predictions(run_root, model_id, task.id)
                            score = evaluator.evaluate(task.id, task.records, preds)
                            results[task.id] = score
                            print(f"Result {task.id}: {_format_metrics(score)}")

                    barrier.wait()

                    # Write final results once
                    if worker_id == 0:
                        import json
                        result_path = os.path.join(run_root, model_id.replace("/","_"), 'result.json')
                        with open(result_path, 'w', encoding='utf-8') as f:
                            json.dump(results, f, ensure_ascii=False, indent=2)
                        print(f"Results saved to {result_path}")
                    
                    barrier.wait()

                else:
                    # Inference
                    if requires_inference and task_phase in ("infer",):
                        for task in tasks:
                            storage = StorageManager(run_root, task.id, worker_id)
                            if worker_id == 0:
                                storage.init(check_existing=is_resuming)
                            else:
                                storage.wait_until_initialized()
                                if is_resuming:
                                    storage.existing_preds = storage.load_existing_predictions(run_root, model_id, task.id)
                            
                            generator = ds_mod.Generator(model)
                            
                            if is_resuming:
                                # Filter out records that have already been processed
                                records = [rec for rec in task.records if rec['id'] not in storage.existing_preds]
                                if worker_id == 0:
                                    print(f"Resuming task {task.id}: {len(records)}/{len(task.records)} records left to process")
                            else:
                                records = task.records
                                
                            for idx, rec in enumerate(
                                    tqdm(records, desc=f"Gen {task.id}", disable=(worker_id != 0))
                            ):
                                if idx % num_workers != worker_id:
                                    continue
                                pred = generator.generate(task.id, rec)
                                storage.save_pred(rec['id'], pred)
                            
                            barrier.wait()
                                
                    # Evaluation
                    if task_phase in ("eval",) and worker_id == 0:
                        evaluator = ds_mod.Evaluator()
                        if judge_runner and hasattr(evaluator, "configure_judge"):
                            evaluator.configure_judge(
                                judge_runner,
                                dataset=ds,
                                run_root=run_root,
                                model_id=model_id,
                                debug=judge_debug
                            )
                        results = {}
                        for task in tqdm(tasks, desc="Eval", leave=False):
                            if json_eval_mode:
                                task_key = task.id if isinstance(task.id, str) else str(task.id)
                                preds = dataset_json_cache.get(task_key) if dataset_json_cache is not None else None
                                if preds is None and dataset_json_cache is not None:
                                    preds = dataset_json_cache.get(JSON_DEFAULT_TASK_KEY)
                                if preds is None:
                                    raise ValueError(f"No predictions found in JSON file for dataset '{ds}' task '{task.id}'.")
                            else:
                                preds = StorageManager.load_predictions(run_root, model_id, task.id)
                            if not preds:
                                print(f"Warning: No predictions found for task '{task.id}' (dataset {ds}, model {model_id}).")
                            results[task.id] = evaluator.evaluate(task.id, task.records, preds)
                        import json
                        model_dir = os.path.join(run_root, model_id.replace("/","_"))
                        os.makedirs(model_dir, exist_ok=True)
                        result_path = os.path.join(model_dir, 'result.json')
                        with open(result_path, 'w', encoding='utf-8') as f:
                            json.dump(results, f, ensure_ascii=False, indent=2)
                        print(f"Results saved to {result_path}")

            try:
                if hasattr(model, "close"):
                    model.close()
            except Exception as e:
                print(f"[Worker {worker_id}] Error during model.close(): {e}")
            
            try:
                del model
            except:
                pass

            gc.collect()
            torch.cuda.empty_cache()
            barrier.wait()

    if judge_runner:
        judge_runner.close()

    if dist.is_initialized():
        dist.destroy_process_group()

def main():
    args = parse_args()
    cfg = {}
    if hasattr(args, 'config'):
        cfg = load_config(args.config)
    merged_args = merge_config_with_args(cfg.get('args', {}), args)

    datasets = merged_args.get('datasets')
    if isinstance(datasets, str):
        datasets = [datasets]
        merged_args['datasets'] = datasets
    json_eval_mode = bool(merged_args.get('json_eval'))
    merged_args['json_eval'] = json_eval_mode
    if json_eval_mode:
        json_paths = merged_args.get('json_paths')
        if json_paths is None:
            raise ValueError("--json_paths must be provided when --json_eval is set.")
        if isinstance(json_paths, str):
            json_paths = [json_paths]
            merged_args['json_paths'] = json_paths
        if not datasets:
            raise ValueError("--datasets must be specified when using --json_eval.")
        json_inputs = parse_json_path_specs(json_paths, datasets)
        merged_args['json_inputs'] = json_inputs
        merged_args['task'] = 'eval'
        if not merged_args.get('models'):
            label = merged_args.get('json_model_label') or 'json_eval'
            merged_args['models'] = [f"{JSON_MODEL_PREFIX}{label}"]
        else:
            merged_args['models'] = list(merged_args['models'])
    else:
        merged_args['json_inputs'] = {}

    judge_spec = merged_args.get('judge')
    if judge_spec:
        judge_gen_cfg = {
            "temperature": merged_args.get('judge_temperature', merged_args.get('temperature', 0.0)),
            "top_p": merged_args.get('judge_top_p', merged_args.get('top_p', 1.0)),
            "max_tokens": merged_args.get('judge_max_tokens', merged_args.get('max_tokens', 512)),
            "rep_penalty": merged_args.get('judge_repetition_penalty', merged_args.get('repetition_penalty', 1.0)),
        }
        judge_config = {
            "model_spec": judge_spec,
            "accelerator": merged_args.get('judge_accelerator'),
            "tensor_parallel": int(merged_args.get('judge_tensor_parallel', 1)),
            "batch_size": int(merged_args.get('judge_batch_size', 4 if json_eval_mode else merged_args.get('batch_size', 1) or 1)),
            "gen_cfg": judge_gen_cfg,
            "api_url": merged_args.get('judge_api_url', merged_args.get('api_url')),
            "api_key": merged_args.get('judge_api_key', merged_args.get('api_key')),
        }
        merged_args['judge_config'] = judge_config
    else:
        merged_args['judge_config'] = None

    if not merged_args.get('datasets'):
        raise ValueError("--datasets must be provided (CLI or config).")

    if not merged_args.get('models'):
        raise ValueError("--models must be provided unless --json_eval is used.")

    if merged_args.get('resume'):
        run_root = merged_args['resume']
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_root = os.path.join(merged_args.get('output_dir', 'run_output'), timestamp)
        os.makedirs(run_root, exist_ok=True)
        with open(os.path.join(run_root, 'config.yaml'), 'w', encoding='utf-8') as cf:
            cfg_args = merged_args.copy()
            yaml.safe_dump({'args': cfg_args, 'cfg': cfg}, cf, allow_unicode=True)

    num_workers = int(merged_args.get('num_workers', 1))
    tensor_parallel = int(merged_args.get('tensor_parallel', 1))
    task_phase = merged_args.get('task', 'all')

    requires_inference = (task_phase in ('infer', 'all')) and not json_eval_mode
    has_real_models = any(not str(spec).startswith(JSON_MODEL_PREFIX) for spec in merged_args.get('models', []))
    if requires_inference and has_real_models:
        # Validate that we have enough GPUs for the requested parallelism
        total_gpus_required = num_workers * tensor_parallel
        available_gpus = torch.cuda.device_count()

        if total_gpus_required > available_gpus:
            raise ValueError(f"Not enough GPUs: requested {total_gpus_required} "
                             f"(num_workers={num_workers} × tensor_parallel={tensor_parallel}), "
                             f"but only {available_gpus} available")
    else:
        total_gpus_required = 0

    print(f"Using {num_workers} worker processes with tensor parallelism={tensor_parallel} "
          f"(total {total_gpus_required} GPUs required)")
    
    mp.set_start_method("spawn", force=True)
    ctx = get_context('spawn')
    barrier = ctx.Barrier(num_workers)
    
    if num_workers > 1:
        mp.spawn(
            run_worker,
            args=(num_workers, merged_args, cfg, run_root, barrier),
            nprocs=num_workers,
            join=True
        )
    else:
        run_worker(0, 1, merged_args, cfg, run_root, barrier)

if __name__ == "__main__":
    main()