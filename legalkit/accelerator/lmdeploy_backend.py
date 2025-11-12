import os
import math
import torch
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoTokenizer, AutoConfig
from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig

class LMDEPLOYAccelerator:
    # Keep class minimal; GPU assignment mirrors vLLM approach

    def __init__(self, model, num_workers: int, tensor_parallel: int, gen_cfg: dict, worker_id: int):
        self.model = model
        self.gen_cfg = gen_cfg
        self.worker_id = worker_id
        self.tensor_parallel = tensor_parallel
        # Concurrency controls (CPU-side pre-processing parallelism, GPU-side batching)
        self.batch_size = int(self.gen_cfg.get('batch_size', 4))  # reasonable default for local inference
        self.preproc_workers = int(self.gen_cfg.get('preproc_workers', max(2, os.cpu_count() // 2)))
        
        # GPU 选择：如果主进程已提前设置了 CUDA_VISIBLE_DEVICES（每个子进程一个掩码），这里直接使用该掩码，避免再次用物理索引计算导致越界
        explicit_ids = self.gen_cfg.get('gpu_ids')
        env_mask = os.environ.get('CUDA_VISIBLE_DEVICES')
        if env_mask and not explicit_ids:
            masked = [int(x.strip()) for x in env_mask.split(',') if x.strip()]
            if len(masked) < tensor_parallel:
                raise ValueError(f"当前可见 GPU 数 {len(masked)} 不满足 tensor_parallel={tensor_parallel}")
            tp_gpu_ids = masked[:tensor_parallel]
        else:
            # 根据显式 gpu_ids 或物理索引切片（仅在未预先掩码时使用）
            if explicit_ids:
                if isinstance(explicit_ids, str):
                    parsed = [int(x.strip().split(":")[-1]) for x in explicit_ids.split(',') if x.strip()]
                elif isinstance(explicit_ids, (list, tuple)):
                    parsed = [int(x) for x in explicit_ids]
                else:
                    raise ValueError("gpu_ids must be str or list")
                base = worker_id * tensor_parallel
                tp_gpu_ids = parsed[base: base + tensor_parallel]
            else:
                base_gpu_id = worker_id * tensor_parallel
                tp_gpu_ids = list(range(base_gpu_id, base_gpu_id + tensor_parallel))
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, tp_gpu_ids))
        self.worker_devices = tp_gpu_ids
        # Fix mmengine LOCAL_RANK issue: ensure index valid inside masked list
        try:
            lr = int(os.environ.get("LOCAL_RANK", "0"))
        except Exception:
            lr = 0
        if lr >= len(self.worker_devices) or lr < 0:
            os.environ["LOCAL_RANK"] = "0"
        else:
            # If tensor_parallel==1 just normalize to 0 to be safe
            if len(self.worker_devices) == 1 and lr != 0:
                os.environ["LOCAL_RANK"] = "0"
        # Remove global ranks that can confuse single-process TP context
        os.environ.pop("RANK", None)
        os.environ.pop("WORLD_SIZE", None)

        print(f"Worker {self.worker_id} using {self.tensor_parallel} GPUs: {self.worker_devices}")

        model_path = model.model_name if hasattr(model, 'model_name') else model.model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        config = AutoConfig.from_pretrained(model_path)
        # Override max context tokens
        cfg_mct = None
        try:
            cfg_mct = int(self.gen_cfg.get('max_context_tokens')) if self.gen_cfg.get('max_context_tokens') is not None else None
        except Exception:
            cfg_mct = None
        base_max_len = getattr(config, "max_position_embeddings", 32768)
        # Utilization hint
        gpu_util = None
        try:
            gpu_util = float(self.gen_cfg.get('gpu_utilization')) if self.gen_cfg.get('gpu_utilization') is not None else None
        except Exception:
            gpu_util = None
        if gpu_util is not None and 0.0 < gpu_util < 1.0:
            base_max_len = max(1024, int(base_max_len * gpu_util))
        self.max_model_len = cfg_mct or base_max_len

        self.max_new_tokens = int(self.gen_cfg.get("max_tokens", 512))
        if self.max_model_len < self.max_new_tokens:
            # Fallback if misconfigured
            safe_ctx = max(32, int(self.max_new_tokens * 1.05))
            print(
                f"[LMDeploy] Warning: max_context_tokens ({self.max_model_len}) < max_tokens ({self.max_new_tokens}). "
                f"Using fallback {safe_ctx} to avoid invalid configuration."
            )
            self.max_model_len = safe_ctx

        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        backend_cfg = TurbomindEngineConfig(
            tp=self.tensor_parallel,
            trust_remote_code=True,
            session_len=self.max_model_len
        )

        print(f"Initializing pipeline with tp={self.tensor_parallel}, gpu_ids={self.worker_devices}, session_len={self.max_model_len}")

        try:
            self.pipe = pipeline(
                model_path,
                backend_config=backend_cfg
            )
            print("Pipeline successfully initialized")
        except Exception as e:
            print(f"Error initializing LMDeploy pipeline: {e}")
            print(f"Visible devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}, "
                      f"Memory: {torch.cuda.memory_allocated(i)/1024**3:.2f}GB used")
            raise
        
    def _preprocess_single(self, p: str, max_input_tokens: int) -> str:
        """Format a single user prompt to raw text with chat template and truncate to fit context."""
        messages = [{"role": "user", "content": p}]
        raw = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        input_ids = self.tokenizer.encode(raw, add_special_tokens=False)
        if len(input_ids) > max_input_tokens:
            input_ids = input_ids[-max_input_tokens:]
            raw = self.tokenizer.decode(input_ids, clean_up_tokenization_spaces=True)
        return raw

    def _preprocess_batch(self, prompts: List[str], max_input_tokens: int) -> List[str]:
        """Preprocess a list of prompts concurrently on CPU to prepare inputs for the GPU pipeline."""
        if not prompts:
            return []
        # Use threads to parallelize tokenization/templating which is CPU-bound
        texts = [None] * len(prompts)
        with ThreadPoolExecutor(max_workers=self.preproc_workers) as ex:
            futures = {ex.submit(self._preprocess_single, p, max_input_tokens): i for i, p in enumerate(prompts)}
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    texts[i] = fut.result()
                except Exception:
                    # Fallback: minimal formatting on failure
                    texts[i] = prompts[i]
        return texts

    def generate(self, prompts: List[str]) -> List[str]:
        """High-throughput batched generation.

        Optimizations (module-local, main.py agnostic):
        - CPU-side prompt templating and truncation in a thread pool
        - GPU-side batched inference via LMDeploy pipeline
        - Preserve input order across batches
        """
        if not prompts:
            return []

        max_input_tokens = max(32, self.max_model_len - self.max_new_tokens)

        cfg = GenerationConfig(
            temperature=float(self.gen_cfg.get("temperature", 1.0)),
            top_p=float(self.gen_cfg.get("top_p", 1.0)),
            max_new_tokens=self.max_new_tokens,
            do_sample=float(self.gen_cfg.get("temperature", 1.0)) > 0,
            repetition_penalty=float(self.gen_cfg.get("rep_penalty", 1.2)),
        )

        outputs: List[str] = [None] * len(prompts)
        bs = max(1, int(self.batch_size))
        total = len(prompts)
        # Iterate in batches, preprocess each batch in parallel, and run a single GPU call per batch
        for start in range(0, total, bs):
            end = min(start + bs, total)
            batch_prompts = prompts[start:end]
            batch_texts = self._preprocess_batch(batch_prompts, max_input_tokens)
            # GPU batch inference
            # Attempt generation; if OOM occurs, try a one-time downscale of context and batch size
            try:
                responses = self.pipe(batch_texts, gen_config=cfg)
            except RuntimeError as e:
                msg = str(e)
                if 'CUDA out of memory' in msg or 'out of memory' in msg:
                    # Reduce batch by half (>=1) and shrink context window then retry once
                    new_bs = max(1, bs // 2)
                    if new_bs < bs:
                        print(f"[LMDeploy] OOM caught. Retrying with smaller batch: {new_bs}")
                        # Re-slice current window with smaller step
                        for inner_start in range(start, end, new_bs):
                            inner_end = min(inner_start + new_bs, end)
                            inner_prompts = prompts[inner_start:inner_end]
                            inner_texts = self._preprocess_batch(inner_prompts, max_input_tokens)
                            inner_responses = self.pipe(inner_texts, gen_config=cfg)
                            for i, r in enumerate(inner_responses):
                                outputs[inner_start + i] = getattr(r, 'text', str(r))
                        continue
                    # If batch already 1, reduce context tokens by 25% and retry once
                    shrink_tokens = int(max_input_tokens * 0.75)
                    if shrink_tokens >= 32:
                        print(f"[LMDeploy] OOM with batch=1. Retrying with smaller context tokens: {shrink_tokens}")
                        inner_texts = self._preprocess_batch(batch_prompts, shrink_tokens)
                        inner_responses = self.pipe(inner_texts, gen_config=cfg)
                        for i, r in enumerate(inner_responses):
                            outputs[start + i] = getattr(r, 'text', str(r))
                        continue
                raise
            for i, r in enumerate(responses):
                outputs[start + i] = getattr(r, 'text', str(r))

        # Ensure no None remains (robustness)
        return [o if o is not None else '' for o in outputs]
    
    def close(self):
        """Release GPU and engine resources used by LMDeploy pipeline."""
        try:
            if hasattr(self, "pipe") and hasattr(self.pipe, "close"):
                self.pipe.close()
                print(f"[LMDEPLOYAccelerator] pipeline closed successfully.")
            del self.pipe
            del self.tokenizer
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            print(f"[LMDEPLOYAccelerator] Memory cleaned.")
        except Exception as e:
            print(f"[LMDEPLOYAccelerator] Error during close(): {e}")