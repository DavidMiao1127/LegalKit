import os
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
from typing import List
from .base import BaseModel

class LocalModel(BaseModel):
    """Model that loads from local checkpoint."""

    def __init__(self, model_path: str, device: str, gen_cfg, worker_id):
        super().__init__(model_path)
        self.device = device
        self.gen_cfg = gen_cfg
        self.worker_id = worker_id
        self._load_model(model_path)

    def _load_model(self, path):
        """Load tokenizer and model from a local directory."""
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Model directory not found: {path}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            self.model = AutoModelForCausalLM.from_pretrained(path).to(self.device)
            config = AutoConfig.from_pretrained(path)
            self.max_model_len = getattr(config, "max_position_embeddings", 32768)
        except Exception as e:
            raise RuntimeError(f"Failed to load local model: {e}")

    def generate(self, prompts: List[str]) -> List[str]:
        """Generate outputs from a list of prompts using the local model."""
        max_new_tokens = self.gen_cfg.get("max_tokens", 512)
        max_input_tokens = self.max_model_len - max_new_tokens
        results = []

        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            raw = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            ) if hasattr(self.tokenizer, "apply_chat_template") else prompt

            inputs = self.tokenizer(
                raw,
                return_tensors="pt",
                truncation=True,
                max_length=max_input_tokens,
                add_special_tokens=True
            ).to(self.device)

            try:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=self.gen_cfg.get("temperature", 1.0),
                    top_p=self.gen_cfg.get("top_p", 1.0),
                    repetition_penalty=self.gen_cfg.get("rep_penalty", 1.2)
                )
                text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                answer = text.split("assistant\n")[-1].strip()
                results.append(answer)
            except Exception as e:
                results.append(f"[Generation failed: {e}]")

        return results

    def close(self):
        """Release GPU and model resources used by LocalModel."""
        try:
            if hasattr(self, "model"):
                del self.model
            if hasattr(self, "tokenizer"):
                del self.tokenizer
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            print(f"[LocalModel] Worker {self.worker_id} memory cleaned.")
        except Exception as e:
            print(f"[LocalModel] Error during close(): {e}")