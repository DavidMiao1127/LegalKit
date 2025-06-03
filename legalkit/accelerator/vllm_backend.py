import os
import torch
from typing import List
from transformers import AutoTokenizer, AutoConfig
from vllm import LLM, SamplingParams

class VLLMAccelerator:
    def __init__(
        self,
        model,
        num_workers: int,
        tensor_parallel: int,
        gen_cfg: dict,
        worker_id: int,
    ):
        self.model = model
        self.gen_cfg = gen_cfg
        self.worker_id = worker_id
        self.tensor_parallel = tensor_parallel

        # Resolve model path
        model_path = (
            model.model_name
            if hasattr(model, "model_name")
            else model.model_path
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        config = AutoConfig.from_pretrained(model_path)
        self.max_model_len = getattr(config, "max_position_embeddings", 32768)

        # Assign GPUs to this worker
        base_gpu_id = worker_id * tensor_parallel
        tp_gpu_ids = list(range(base_gpu_id, base_gpu_id + tensor_parallel))
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, tp_gpu_ids))

        print(f"Worker {worker_id} using {tensor_parallel} GPUs: {tp_gpu_ids}")
        torch.cuda.empty_cache()

        # Initialize VLLM engine
        try:
            print("model_path:", model_path)
            self.llm = LLM(
                model=model_path,
                tensor_parallel_size=tensor_parallel,
                gpu_memory_utilization=0.8,
                dtype="auto",
            )
            print("VLLM model successfully initialized")
        except Exception as e:
            print(f"Error initializing VLLM: {e}")
            print(f"Visible devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                used = torch.cuda.memory_allocated(i) / 1024**3
                print(f"GPU {i}: {name}, Memory: {used:.2f}GB used")
            raise

    def generate(self, prompts: List[str]) -> List[str]:
        max_new_tokens = self.gen_cfg.get("max_tokens", 512)
        max_input_tokens = self.max_model_len - max_new_tokens
        texts = []
        for p in prompts:
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
            texts.append(raw)

        params = SamplingParams(
            temperature=self.gen_cfg.get("temperature", 1.0),
            top_p=self.gen_cfg.get("top_p", 1.0),
            max_tokens=self.gen_cfg.get("max_tokens", 512),
            repetition_penalty=self.gen_cfg.get("rep_penalty", 1.2)
        )

        batch_outputs = self.llm.generate(texts, params, use_tqdm=False)
        results = [
            out.outputs[0].text
            for out in batch_outputs
        ]
        return results
    
    def close(self):
        """Release GPU resources used by the VLLM engine."""
        try:
            del self.llm
            del self.tokenizer
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            print(f"[VLLMAccelerator] Memory cleaned.")
        except Exception as e:
            print(f"[VLLMAccelerator] Error during close(): {e}")