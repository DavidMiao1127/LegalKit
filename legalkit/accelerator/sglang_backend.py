import os
import torch
from typing import List
from transformers import AutoTokenizer, AutoConfig


class SGLANGAccelerator:
	def __init__(
		self,
		model,
		num_workers: int,
		tensor_parallel: int,
		gen_cfg: dict,
		worker_id: int,
	):
		"""
		SGLANG accelerator wrapper to match the interface used by other backends.
		"""
		self.model = model
		self.gen_cfg = gen_cfg
		self.worker_id = worker_id
		self.tensor_parallel = tensor_parallel

		# Resolve model path
		self.model_path = (
			model.model_name if hasattr(model, "model_name") else getattr(model, "model_path", None)
		)
		if not self.model_path:
			raise ValueError("SGLANGAccelerator: could not resolve model path")

		# Tokenizer for chat templating and truncation
		self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
		config = AutoConfig.from_pretrained(self.model_path)
		self.max_model_len = getattr(config, "max_position_embeddings", 32768)

		# Assign GPUs to this worker
		base_gpu_id = worker_id * tensor_parallel
		tp_gpu_ids = list(range(base_gpu_id, base_gpu_id + tensor_parallel))
		os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, tp_gpu_ids))

		print(f"Worker {worker_id} using {tensor_parallel} GPUs: {tp_gpu_ids}")
		torch.cuda.empty_cache()

		# Initialize SGLang engine (support different versions/APIs)
		self._init_engine()

	def _init_engine(self):
		# Try a few known SGLang runtimes to maximize compatibility
		self.engine = None
		self._engine_type = None

		# Attempt: sglang.srt runtime engine (newer versions)
		try:
			from sglang.srt import Runtime  # type: ignore

			self.engine = Runtime(
				model_path=self.model_path,
				tp=self.tensor_parallel,
				trust_remote_code=True,
				max_seq_len=self.max_model_len,
			)
			self._engine_type = "srt"
			print("SGLANG (srt.Runtime) initialized")
			return
		except Exception as e:
			print(f"SGLANG srt.Runtime not available or failed to init: {e}")

		# Attempt: legacy sglang Runtime
		try:
			from sglang import Runtime  # type: ignore

			self.engine = Runtime(
				model_path=self.model_path,
				tp=self.tensor_parallel,
				trust_remote_code=True,
				max_seq_len=self.max_model_len,
			)
			self._engine_type = "runtime"
			print("SGLANG (Runtime) initialized")
			return
		except Exception as e:
			print(f"SGLANG Runtime not available or failed to init: {e}")

		# Attempt: fallback LLM style
		try:
			from sglang import LLM  # type: ignore

			self.engine = LLM(
				model=self.model_path,
				tensor_parallel_size=self.tensor_parallel,
				dtype="auto",
			)
			self._engine_type = "llm"
			print("SGLANG (LLM) initialized")
			return
		except Exception as e:
			print(f"SGLANG LLM not available or failed to init: {e}")

		raise RuntimeError(
			"Failed to initialize SGLANG backend. Please ensure sglang is installed and compatible."
		)

	def generate(self, prompts: List[str]) -> List[str]:
		max_new_tokens = self.gen_cfg.get("max_tokens", 512)
		max_input_tokens = self.max_model_len - max_new_tokens

		# Prepare chat-formatted inputs with truncation
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

		temperature = self.gen_cfg.get("temperature", 1.0)
		top_p = self.gen_cfg.get("top_p", 1.0)
		rep_penalty = self.gen_cfg.get("rep_penalty", 1.2)

		# Try sampling params across versions
		params = None
		params_kwargs = dict(
			temperature=temperature,
			top_p=top_p,
			max_new_tokens=max_new_tokens,
			repetition_penalty=rep_penalty,
		)

		# Build sampling params if available
		SamplingParams = None
		for modpath in ("sglang.srt", "sglang"):
			try:
				module = __import__(modpath, fromlist=["SamplingParams"])  # type: ignore
				SamplingParams = getattr(module, "SamplingParams", None)
				if SamplingParams is not None:
					break
			except Exception:
				continue

		if SamplingParams is not None:
			try:
				params = SamplingParams(**params_kwargs)
			except Exception:
				params = None

		# Execute generation with best-effort API compatibility
		try:
			if self._engine_type in ("srt", "runtime"):
				if params is not None:
					outputs = self.engine.generate(texts, params)  # type: ignore
				else:
					outputs = self.engine.generate(texts, **params_kwargs)  # type: ignore
				# Normalize outputs: could be list of objects with .text or plain strings
				results = []
				for r in outputs:
					if isinstance(r, str):
						results.append(r)
					else:
						txt = getattr(r, "text", None)
						if txt is None and hasattr(r, "outputs"):
							# Some engines return objects with outputs[0].text
							try:
								txt = r.outputs[0].text  # type: ignore
							except Exception:
								txt = ""
						results.append(txt or "")
				return results
			elif self._engine_type == "llm":
				# LLM-like engine often mirrors vLLM generate
				if params is not None:
					batch = self.engine.generate(texts, params, use_tqdm=False)  # type: ignore
				else:
					batch = self.engine.generate(texts, use_tqdm=False, **params_kwargs)  # type: ignore
				results = []
				for out in batch:
					# vLLM style: out.outputs[0].text
					try:
						results.append(out.outputs[0].text)
					except Exception:
						# Fallback if plain string
						results.append(out if isinstance(out, str) else "")
				return results
		except Exception as e:
			print(f"[SGLANGAccelerator] Generation error: {e}")
			return [f"[SGLANG generate failed: {e}]"] * len(prompts)

		# If we ever reach here, return empty strings to keep length
		return [""] * len(prompts)

	def close(self):
		"""Release resources used by SGLang engine."""
		try:
			if hasattr(self, "engine"):
				# Some engines may have close()
				try:
					if hasattr(self.engine, "close"):
						self.engine.close()
						print("[SGLANGAccelerator] engine closed successfully.")
				finally:
					del self.engine
			if hasattr(self, "tokenizer"):
				del self.tokenizer
			torch.cuda.empty_cache()
			import gc
			gc.collect()
			print("[SGLANGAccelerator] Memory cleaned.")
		except Exception as e:
			print(f"[SGLANGAccelerator] Error during close(): {e}")

