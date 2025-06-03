import os
import torch
from typing import List
from transformers import AutoTokenizer, AutoConfig
from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig
os.environ["LOCAL_RANK"] = "0"

class LMDEPLOYAccelerator:
    def __init__(self, model, num_workers: int, tensor_parallel: int, gen_cfg: dict, worker_id: int):
        self.model = model
        self.gen_cfg = gen_cfg
        self.worker_id = worker_id
        self.tensor_parallel = tensor_parallel
        
        model_path = model.model_name if hasattr(model, 'model_name') else model.model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        config = AutoConfig.from_pretrained(model_path)
        self.max_model_len = getattr(config, "max_position_embeddings", 32768)
        
        base_gpu_id = worker_id * tensor_parallel
        tp_gpu_ids = list(range(base_gpu_id, base_gpu_id + tensor_parallel))
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, tp_gpu_ids))
        
        print(f"Worker {worker_id} using {tensor_parallel} GPUs: {tp_gpu_ids}")
        torch.cuda.empty_cache()

        backend_cfg = TurbomindEngineConfig(
            tp=tensor_parallel,
            trust_remote_code=True,
            max_context_token_num=self.max_model_len
        )
        
        print(f"Initializing pipeline with tp={tensor_parallel}, gpu_ids={tp_gpu_ids}")
        
        try:
            self.pipe = pipeline(
                model_path,
                backend_config=backend_cfg
            )
            print(f"Pipeline successfully initialized")
        except Exception as e:
            print(f"Error initializing LMDeploy pipeline: {e}")
            print(f"Visible devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}, "
                      f"Memory: {torch.cuda.memory_allocated(i)/1024**3:.2f}GB used")
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

        cfg = GenerationConfig(
            temperature=self.gen_cfg.get("temperature", 1.0),
            top_p=self.gen_cfg.get("top_p", 1.0),
            max_new_tokens=max_new_tokens,
            do_sample=self.gen_cfg.get("temperature", 1.0) > 0,
            repetition_penalty=1.2
        )

        responses = self.pipe(texts, gen_config=cfg)
        return [r.text for r in responses]
    
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