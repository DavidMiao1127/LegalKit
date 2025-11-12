# LMDeploy Debugging Helper

This directory contains `lmdeploy_debug_runner.py`, a standalone script you can run to reproduce
LMDeploy initialisation and inference behaviour outside the full LegalKit orchestrator. The goal
is to isolate CUDA out-of-memory problems or regressions after upgrading LMDeploy.

## Quick start

```bash
python scripts/lmdeploy_debug_runner.py \
    --model-path /path/to/model \
    --gpu-ids 0 \
    --prompts "What are the essential elements of a contract?"
```

### Simulate multi-worker sharding

```bash
python scripts/lmdeploy_debug_runner.py \
    --model-path /path/to/model \
    --gpu-ids 0,1,2,3 \
    --num-workers 2 \
    --tp 2 \
    --batch-size 2
```

The command above spawns two worker processes. Each worker receives a slice of the provided GPU ids
(`0,1` for worker 0, `2,3` for worker 1) and processes its own shard of the prompt list, just like the
main LegalKit runner. Use `--worker-id <n>` to run a specific worker in isolation without spawning all of them.

## Useful flags

- `--gpu-ids`: set the exact GPUs to expose (e.g. `0,1`), matching how LegalKit workers pin devices.
- `--tp`: tensor parallel degree; must match the number of GPUs passed in `--gpu-ids` for Turbomind.
- `--num-workers`: spawn multiple data-parallel workers; each one automatically slices the GPU list and prompt shard.
- `--worker-id`: run only a single worker process (useful when you want to debug a specific shard).
- `--max-context-tokens` and `--gpu-utilization`: shrink the KV cache footprint to mitigate OOMs.
- `--batch-size`: test different micro-batch settings without restarting LegalKit.
- `--prompt-file`: supply a JSON/JSONL/TXT file with prompts to replay a workload.
- `--save-json`: dump configuration, prompts, outputs, and per-batch timings for later comparison.
- `--dry-run`: only initialise LMDeploy to diagnose setup errors.
- `--verbose`: print each prompt/response pair as they are generated.

## Interpreting output

The script prints GPU memory snapshots (`allocated`, `reserved`, and `free`) before and after key
stages (initialisation, each batch, and shutdown). If a CUDA OOM occurs during generation, the script
captures the memory snapshot right before the exception so you can compare utilisation with VLLM or
previous LMDeploy versions.

Use this tool to validate new LMDeploy releases, tune `max_context_tokens`, or reproduce worker-level
failures without the complexity of the full multi-process runner.
