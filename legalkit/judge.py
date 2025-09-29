from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Dict, List, Optional

from legalkit.models import build_model


@dataclass
class JudgeConfig:
    model_spec: str
    accelerator: Optional[str] = None
    tensor_parallel: int = 1
    batch_size: int = 1
    gen_cfg: Dict[str, Any] = field(default_factory=dict)
    api_url: Optional[str] = None
    api_key: Optional[str] = None

    def __post_init__(self):
        if not self.gen_cfg:
            self.gen_cfg = {}
        if not self.tensor_parallel or self.tensor_parallel < 1:
            self.tensor_parallel = 1
        if not self.batch_size or self.batch_size < 1:
            self.batch_size = 1


class LLMJudgeRunner:
    """Utility wrapper for running an LLM-as-Judge model."""

    def __init__(self, config: JudgeConfig, worker_id: int = 0):
        self.config = config
        self.worker_id = worker_id
        self._model = None

    def _resolve_model_kwargs(self) -> Dict[str, Any]:
        spec = self.config.model_spec
        if not spec:
            raise ValueError("Judge model spec is empty")

        if spec.startswith("hf:"):
            return {"model_type": "hf", "model_name": spec.split("hf:", 1)[1]}
        if spec.startswith("api:"):
            name = spec.split("api:", 1)[1]
            return {
                "model_type": "api",
                "model_name": name,
                "api_url": self.config.api_url,
                "api_key": self.config.api_key,
            }

        # Default to local checkpoint
        return {"model_type": "local", "model_path": spec}

    def _wrap_accelerator(self, model):
        if not self.config.accelerator:
            return model
        mod = import_module(f"legalkit.accelerator.{self.config.accelerator}_backend")
        accel_cls = getattr(mod, f"{self.config.accelerator.upper()}Accelerator")
        return accel_cls(
            model,
            num_workers=1,
            tensor_parallel=self.config.tensor_parallel,
            gen_cfg=self.config.gen_cfg,
            worker_id=self.worker_id,
        )

    def _ensure_model(self):
        if self._model is not None:
            return self._model

        kwargs = self._resolve_model_kwargs()
        kwargs.setdefault("gen_cfg", self.config.gen_cfg)
        kwargs.setdefault("worker_id", self.worker_id)

        if kwargs["model_type"] == "local":
            kwargs.setdefault("device", "meta" if self.config.accelerator else "cuda")
        elif kwargs["model_type"] == "hf":
            kwargs.setdefault("device", "cuda")

        model = build_model(**kwargs)
        model = self._wrap_accelerator(model)
        self._model = model
        return self._model

    @property
    def batch_size(self) -> int:
        return max(1, int(self.config.batch_size or 1))

    def generate(self, prompts: List[str]) -> List[str]:
        model = self._ensure_model()
        outputs: List[str] = []
        bs = self.batch_size
        for idx in range(0, len(prompts), bs):
            batch = prompts[idx: idx + bs]
            outputs.extend(model.generate(batch))
        return outputs

    def close(self):
        if self._model is None:
            return
        try:
            if hasattr(self._model, "close"):
                self._model.close()
        finally:
            self._model = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
