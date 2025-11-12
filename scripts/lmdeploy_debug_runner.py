#!/usr/bin/env python3
"""Standalone LMDeploy inference harness for debugging CUDA OOMs and configuration issues.

This utility intentionally mirrors the logic in `legalkit.accelerator.lmdeploy_backend`
but runs outside the main LegalKit multiprocessing stack so you can isolate
initialisation or memory problems that only appear with the LMDeploy backend.

Example usage (single GPU):
    python scripts/lmdeploy_debug_runner.py \
        --model-path /path/to/model \
        --gpu-ids 0 \
        --prompts "简要介绍法律援助的基本流程" "What are the key elements of a contract?"

Example usage (two GPUs with tensor parallelism):
    python scripts/lmdeploy_debug_runner.py \
        --model-path /path/to/model \
        --gpu-ids 0,1 \
        --tp 2 \
        --batch-size 4 \
        --max-new-tokens 512 \
        --gpu-utilization 0.5

The script prints GPU memory snapshots before and after model load and generation,
and records timing for each batch so you can compare behaviour across LMDeploy releases.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Iterable, List

import torch
import torch.multiprocessing as mp
from transformers import AutoTokenizer, AutoConfig
from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone LMDeploy inference debugger")
    parser.add_argument("--model-path", required=True, help="Path or HF repo id of the base model")
    parser.add_argument(
        "--gpu-ids",
        default="0",
        help="Comma-separated GPU indices to expose (e.g., '0,1'). Overrides CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel degree to configure for Turbomind")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per generation call")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of data-parallel worker processes to simulate (mirrors LegalKit num_workers)",
    )
    parser.add_argument(
        "--worker-id",
        type=int,
        help="Run only the specified worker id (0-indexed). If omitted and num-workers > 1, all workers are spawned.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Maximum tokens to sample per prompt")
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=None,
        help="Override maximum context tokens (reduces KV cache if set)",
    )
    parser.add_argument(
        "--gpu-utilization",
        type=float,
        default=None,
        help="Scale context window by this factor (0-1) before applying explicit max context",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p nucleus sampling parameter")
    parser.add_argument(
        "--rep-penalty",
        type=float,
        default=1.0,
        help="Repetition penalty",
    )
    parser.add_argument(
        "--preproc-workers",
        type=int,
        default=None,
        help="Number of CPU workers to use for prompt templating/tokenization (default: half CPU cores)",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        help="Optional path to a JSON/JSONL/TXT file providing prompts (one per line or JSON array)",
    )
    parser.add_argument(
        "--prompts",
        nargs="*",
        default=None,
        help="Direct prompts provided via CLI (overrides prompt file if both supplied)",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        help="Optional path to dump inputs/outputs and timing information for later inspection",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Initialise the engine but skip actual generation (useful for catching init failures)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-prompt outputs immediately instead of only summary stats",
    )
    return parser.parse_args()


def _parse_gpu_ids_spec(spec: str) -> List[int]:
    """Parse a CUDA device specification string into a list of integer GPU ids."""
    if spec is None:
        return list(range(torch.cuda.device_count() or 1))
    if isinstance(spec, (list, tuple)):
        tokens = [str(item) for item in spec]
    else:
        tokens = [part.strip() for part in str(spec).split(",") if part.strip()]
    gpu_ids: List[int] = []
    for token in tokens:
        if token.lower().startswith("cuda:"):
            token = token.split(":", 1)[1]
        try:
            gpu_ids.append(int(token))
        except ValueError as exc:
            raise ValueError(f"Invalid GPU id '{token}' in spec '{spec}'") from exc
    if not gpu_ids:
        raise ValueError(f"No GPU ids parsed from specification '{spec}'")
    return gpu_ids


def _select_worker_devices(gpu_ids: List[int], worker_id: int, num_workers: int, tp: int) -> List[int]:
    """Return the physical GPU ids assigned to a worker given tp-style sharding."""
    if num_workers <= 0:
        raise ValueError("num_workers must be >= 1")
    if not (0 <= worker_id < num_workers):
        raise ValueError(f"worker_id {worker_id} out of range for num_workers={num_workers}")
    if tp <= 0:
        raise ValueError("tensor parallel degree must be >= 1")
    total_required = num_workers * tp
    if len(gpu_ids) < total_required:
        raise ValueError(
            f"Not enough GPUs for num_workers={num_workers}, tp={tp}. "
            f"Need {total_required}, have {len(gpu_ids)} ({gpu_ids})."
        )
    start = worker_id * tp
    end = start + tp
    if end > len(gpu_ids):
        raise ValueError(
            f"GPU slice [{start}:{end}] exceeds available devices {gpu_ids} for worker {worker_id}."
        )
    return gpu_ids[start:end]


def _worker_save_path(base_path: str | None, worker_id: int, num_workers: int) -> str | None:
    if not base_path:
        return None
    if num_workers <= 1:
        return base_path
    root, ext = os.path.splitext(base_path)
    if not ext:
        ext = ".json"
    return f"{root}_worker{worker_id}{ext}"


def _load_prompts(args: argparse.Namespace) -> List[str]:
    if args.prompts:
        return list(args.prompts)
    if not args.prompt_file:
        return [
            "简要介绍法律援助的基本流程。",
            "What legal rights does a tenant have when a landlord fails to repair critical utilities?",
        ]
    path = args.prompt_file
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    # Try JSON (array or dict with 'prompt' entries)
    try:
        data = json.loads(text)
        prompts: List[str] = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    prompts.append(item)
                elif isinstance(item, dict):
                    val = item.get("prompt") or item.get("input") or item.get("question")
                    if val:
                        prompts.append(str(val))
        elif isinstance(data, dict):
            for val in data.values():
                if isinstance(val, str):
                    prompts.append(val)
        if prompts:
            return prompts
    except json.JSONDecodeError:
        pass
    # Fall back to newline separated text
    prompts = [line.strip() for line in text.splitlines() if line.strip()]
    if not prompts:
        raise ValueError(f"No prompts parsed from file: {path}")
    return prompts


def _print_gpu_snapshot(label: str) -> None:
    if not torch.cuda.is_available():
        print(f"[{label}] CUDA not available")
        return
    device_count = torch.cuda.device_count()
    print(f"[{label}] CUDA devices visible: {device_count}")
    for idx in range(device_count):
        name = torch.cuda.get_device_name(idx)
        mem_alloc = torch.cuda.memory_allocated(idx) / 1024**3
        mem_reserved = torch.cuda.memory_reserved(idx) / 1024**3
        try:
            free_mem, total_mem = torch.cuda.mem_get_info(idx)
            free_mem /= 1024**3
            total_mem /= 1024**3
        except Exception:
            free_mem = total_mem = float("nan")
        print(
            f"  GPU {idx}: {name} | allocated={mem_alloc:.2f} GiB | reserved={mem_reserved:.2f} GiB | free={free_mem:.2f} / {total_mem:.2f} GiB"
        )


def _worker_entry(worker_id: int, args_dict: dict, prompts: List[str]) -> None:
    args = argparse.Namespace(**args_dict)
    _run_worker(worker_id, args, prompts)


def _run_worker(worker_id: int, args: argparse.Namespace, all_prompts: List[str]) -> None:
    num_workers = max(1, int(getattr(args, "num_workers", 1)))
    tp = max(1, int(getattr(args, "tp", 1)))
    gpu_ids = _parse_gpu_ids_spec(getattr(args, "gpu_ids", "0"))
    worker_devices = _select_worker_devices(gpu_ids, worker_id, num_workers, tp)

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, worker_devices))
    os.environ["LOCAL_RANK"] = "0"
    os.environ["RANK"] = str(worker_id)

    if num_workers == 1:
        subset_prompts = list(all_prompts)
    else:
        subset_prompts = [p for idx, p in enumerate(all_prompts) if idx % num_workers == worker_id]

    print(
        f"[worker {worker_id}] total_prompts={len(all_prompts)}, assigned={len(subset_prompts)}, "
        f"tp={tp}, devices={worker_devices}"
    )

    _print_gpu_snapshot(f"worker{worker_id}-before-init")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    config = AutoConfig.from_pretrained(args.model_path)

    base_max_len = getattr(config, "max_position_embeddings", 32768)
    if args.gpu_utilization is not None and 0.0 < args.gpu_utilization < 1.0:
        base_max_len = max(1024, int(base_max_len * args.gpu_utilization))
    max_context_tokens = args.max_context_tokens or base_max_len

    if max_context_tokens < args.max_new_tokens:
        raise ValueError(
            f"max_context_tokens ({max_context_tokens}) must be >= max_new_tokens ({args.max_new_tokens})"
        )

    preproc_workers = args.preproc_workers or max(2, os.cpu_count() // 2)
    print(
        f"[worker {worker_id}] Config summary:\n"
        f"  model_path: {args.model_path}\n"
        f"  CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}\n"
        f"  tensor_parallel: {tp}\n"
        f"  batch_size: {args.batch_size}\n"
        f"  max_new_tokens: {args.max_new_tokens}\n"
        f"  max_context_tokens: {max_context_tokens}\n"
        f"  gpu_utilization: {args.gpu_utilization}\n"
        f"  preproc_workers: {preproc_workers}\n"
    )

    engine_cfg = TurbomindEngineConfig(
        tp=tp,
        trust_remote_code=True,
        max_context_token_num=max_context_tokens,
    )

    def format_prompt(raw_prompt: str, limit: int) -> str:
        messages = [{"role": "user", "content": raw_prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        input_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(input_ids) > limit:
            input_ids = input_ids[-limit:]
            text = tokenizer.decode(input_ids, clean_up_tokenization_spaces=True)
        return text

    max_input_tokens = max(32, max_context_tokens - args.max_new_tokens)
    formatted_prompts = [format_prompt(prompt, max_input_tokens) for prompt in subset_prompts]

    print(f"[worker {worker_id}] Initialising LMDeploy pipeline…")
    init_start = time.time()
    try:
        pipe = pipeline(args.model_path, backend_config=engine_cfg)
    except Exception as exc:
        _print_gpu_snapshot(f"worker{worker_id}-init-failed")
        raise RuntimeError(f"LMDeploy pipeline initialisation failed: {exc}") from exc
    init_dur = time.time() - init_start
    print(f"[worker {worker_id}] Pipeline ready in {init_dur:.2f} s")
    _print_gpu_snapshot(f"worker{worker_id}-after-init")

    if args.dry_run:
        print(f"[worker {worker_id}] Dry run complete (no generation requested). Closing pipeline…")
        pipe.close()
        return

    gen_cfg = GenerationConfig(
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        max_new_tokens=args.max_new_tokens,
        do_sample=float(args.temperature) > 0,
        repetition_penalty=float(args.rep_penalty),
    )

    outputs: List[str] = []
    timings = []
    total = len(formatted_prompts)
    bs = max(1, args.batch_size)

    for start in range(0, total, bs):
        end = min(start + bs, total)
        batch_texts = formatted_prompts[start:end]
        torch.cuda.empty_cache()
        _print_gpu_snapshot(f"worker{worker_id}-before-batch-{start // bs}")
        batch_start = time.time()
        try:
            responses = pipe(batch_texts, gen_config=gen_cfg)
        except RuntimeError as exc:
            _print_gpu_snapshot(f"worker{worker_id}-oom-batch-{start // bs}")
            raise
        batch_dur = time.time() - batch_start
        timings.append({"batch": start // bs, "size": len(batch_texts), "duration_s": batch_dur})
        _print_gpu_snapshot(f"worker{worker_id}-after-batch-{start // bs}")
        for resp in responses:
            text = getattr(resp, "text", None)
            outputs.append(text if text is not None else str(resp))
        if args.verbose:
            for prompt, output in zip(subset_prompts[start:end], outputs[start:end]):
                print("-" * 40)
                print(f"[worker {worker_id}] Prompt: {prompt}\nResponse:\n{output}\n")

    print(f"[worker {worker_id}] Finished {total} prompt(s) across {len(timings)} batch(es)")
    if timings:
        avg = sum(t["duration_s"] for t in timings) / len(timings)
        print(f"[worker {worker_id}] Average batch time: {avg:.2f} s")

    save_path = _worker_save_path(args.save_json, worker_id, num_workers)
    if save_path:
        payload = {
            "config": {
                "model_path": args.model_path,
                "tp": tp,
                "gpu_ids": worker_devices,
                "batch_size": args.batch_size,
                "max_new_tokens": args.max_new_tokens,
                "max_context_tokens": max_context_tokens,
                "gpu_utilization": args.gpu_utilization,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "rep_penalty": args.rep_penalty,
                "preproc_workers": preproc_workers,
                "worker_id": worker_id,
                "num_workers": num_workers,
            },
            "prompts": subset_prompts,
            "outputs": outputs,
            "timings": timings,
        }
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[worker {worker_id}] Saved debug payload to {save_path}")

    print(f"[worker {worker_id}] Closing pipeline…")
    pipe.close()
    _print_gpu_snapshot(f"worker{worker_id}-after-close")


def main() -> None:
    args = _parse_args()
    prompts = _load_prompts(args)
    total_prompts = len(prompts)
    print(f"Loaded {total_prompts} prompt(s) across requested workers={args.num_workers}, tp={args.tp}")

    args_dict = dict(vars(args))

    if args.worker_id is not None:
        if not (0 <= args.worker_id < max(1, args.num_workers)):
            raise ValueError(
                f"worker-id {args.worker_id} is invalid for num-workers={args.num_workers}"
            )
        _worker_entry(args.worker_id, args_dict, prompts)
        return

    if args.num_workers > 1:
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            # start method may already be set elsewhere in the process lifecycle
            pass
        mp.spawn(_worker_entry, args=(args_dict, prompts), nprocs=args.num_workers, join=True)
    else:
        _worker_entry(0, args_dict, prompts)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user", file=sys.stderr)
        sys.exit(130)