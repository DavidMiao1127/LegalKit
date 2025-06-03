from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from .base import BaseModel

class HuggingfaceModel(BaseModel):
    """Model loaded from Hugging Face Hub."""

    def __init__(self, model_name: str, device: str = "cuda", max_new_tokens: int = 256):
        super().__init__(model_name)
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._load_model(model_name)

    def _load_model(self, name):
        """
        Load tokenizer and model from Hugging Face Hub.
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(name)
            self.model = AutoModelForCausalLM.from_pretrained(name).to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load Huggingface model: {e}")

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate output from prompt using Huggingface model.
        """
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=kwargs.get("max_new_tokens", self.max_new_tokens),
                do_sample=kwargs.get("do_sample", False),
                temperature=kwargs.get("temperature", 1.0)
            )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            raise RuntimeError(f"Generation failed: {e}")