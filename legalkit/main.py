import os
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

from legalkit.models import build_model
from legalkit.storage import StorageManager

def parse_args():
    parser = argparse.ArgumentParser(
        description="LegalKit Strategy Runner",
        argument_default=argparse.SUPPRESS
    )
    parser.add_argument("--config", type=str, help="YAML config file path")
    parser.add_argument("--models", nargs='+', help="Model specs: local paths, hf shortcuts, or api labels")
    parser.add_argument("--datasets", nargs='+', help="Dataset names")
    parser.add_argument("-a", "--accelerator", choices=["vllm", "lmdeploy"], help="Acceleration backend")
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
    
    for spec in models:
        if dist.is_initialized():
            dist.destroy_process_group()

        if os.path.isdir(spec):
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
            if mcfg["model_type"] == "local" and accelerator:
                mcfg['device'] = "meta"
                model = build_model(**mcfg)
                model = wrap_accelerator(model, accelerator, num_workers, tensor_parallel, gen_cfg, worker_id)

            elif mcfg["model_type"] == "local":
                mcfg['device'] = "cuda"
                mcfg['gen_cfg'] = gen_cfg
                mcfg['worker_id'] = worker_id
                model = build_model(**mcfg)

            elif mcfg["model_type"] == "api":
                mcfg['gen_cfg'] = gen_cfg
                model = build_model(**mcfg)

            for ds in datasets:
                ds_mod = import_module(f"legalkit.datasets.{ds}")
                tasks = ds_mod.load_tasks(sub_tasks=sub_tasks)
                
                # If "all", do per-subtask inference + eval
                if task_phase == "all":
                    results = {} if worker_id == 0 else None
                    evaluator = ds_mod.Evaluator() if worker_id == 0 else None
                    for task in tasks:
                        # Inference
                        storage = StorageManager(run_root, spec, task.id, worker_id)
                        if worker_id == 0:
                            storage.init(check_existing=is_resuming)
                        else:
                            storage.wait_until_initialized()
                            if is_resuming:
                                storage.existing_preds = storage.load_existing_predictions(run_root, spec, task.id)
                                
                        # Check which records still need to be processed
                        generator = ds_mod.Generator(model)
                        
                        if is_resuming:
                            # Filter out records that have already been processed
                            records = [rec for rec in task.records if rec['id'] not in storage.existing_preds]
                            if worker_id == 0:
                                print(f"Resuming task {task.id}: {len(records)}/{len(task.records)} records left to process")
                        else:
                            records = task.records
                            
                        assigned_recs = [
                            rec for idx, rec in enumerate(records)
                            if idx % num_workers == worker_id
                        ]

                        # Process remaining records
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
                            preds = StorageManager.load_predictions(run_root, spec, task.id)
                            score = evaluator.evaluate(task.id, task.records, preds)
                            results[task.id] = score
                            print(f"Result {task.id}: {score['score']:.4f}")

                    barrier.wait()

                    # Write final results once
                    if worker_id == 0:
                        import json
                        result_path = os.path.join(run_root, spec.replace("/","_"), 'result.json')
                        with open(result_path, 'w', encoding='utf-8') as f:
                            json.dump(results, f, ensure_ascii=False, indent=2)
                        print(f"Results saved to {result_path}")
                    
                    barrier.wait()

                else:
                    # Inference
                    if task_phase in ("infer",):
                        for task in tasks:
                            storage = StorageManager(run_root, task.id, worker_id)
                            if worker_id == 0:
                                storage.init(check_existing=is_resuming)
                            else:
                                storage.wait_until_initialized()
                                if is_resuming:
                                    storage.existing_preds = storage.load_existing_predictions(run_root, spec, task.id)
                            
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
                        results = {}
                        for task in tqdm(tasks, desc="Eval", leave=False):
                            preds = StorageManager.load_predictions(run_root, spec, task.id)
                            results[task.id] = evaluator.evaluate(task.id, task.records, preds)
                        import json
                        result_path = os.path.join(run_root, 'result.json')
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

    if dist.is_initialized():
        dist.destroy_process_group()

def main():
    args = parse_args()
    cfg = {}
    if hasattr(args, 'config'):
        cfg = load_config(args.config)
    merged_args = merge_config_with_args(cfg.get('args', {}), args)

    if not merged_args.get('models') or not merged_args.get('datasets'):
        raise ValueError("--models and --datasets must be provided (CLI or config).")

    if merged_args.get('resume'):
        run_root = merged_args['resume']
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_root = os.path.join(merged_args.get('output_dir', 'run_output'), timestamp)
        os.makedirs(run_root, exist_ok=True)
        with open(os.path.join(run_root, 'config.yaml'), 'w', encoding='utf-8') as cf:
            yaml.safe_dump({'args': merged_args, 'cfg': cfg}, cf, allow_unicode=True)

    num_workers = int(merged_args.get('num_workers', 1))
    tensor_parallel = int(merged_args.get('tensor_parallel', 1))
    
    # Validate that we have enough GPUs for the requested parallelism
    total_gpus_required = num_workers * tensor_parallel
    available_gpus = torch.cuda.device_count()
    
    if total_gpus_required > available_gpus:
        raise ValueError(f"Not enough GPUs: requested {total_gpus_required} "
                         f"(num_workers={num_workers} × tensor_parallel={tensor_parallel}), "
                         f"but only {available_gpus} available")
    
    print(f"Using {num_workers} worker processes with tensor parallelism={tensor_parallel} "
          f"(total {total_gpus_required} GPUs)")
    
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