import requests
from .base import BaseModel
import os

class APIModel(BaseModel):
    """Model that calls a remote LLM service via HTTP API."""

    def __init__(self, model_name: str, api_url: str, api_key: str, gen_cfg, timeout: int = 30):
        super().__init__(model_name)
        self.api_url = api_url
        self.api_key = api_key
        self.gen_cfg = gen_cfg
        self.timeout = timeout

        http_proxy = os.getenv("http_proxy") or os.getenv("HTTP_PROXY")
        https_proxy = os.getenv("https_proxy") or os.getenv("HTTPS_PROXY")
        if http_proxy or https_proxy:
            self.proxies = {}
            if http_proxy:
                self.proxies["http"] = http_proxy
            if https_proxy:
                self.proxies["https"] = https_proxy
        else:
            self.proxies = None

    def generate(self, prompts: list[str]) -> list[str]:
        """
        Generate outputs from a list of prompts via the remote API.
        """
        max_tokens = self.gen_cfg.get("max_tokens", 512)
        temperature = self.gen_cfg.get("temperature", 1.0)
        top_p = self.gen_cfg.get("top_p", 1.0)

        headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        results: list[str] = []

        for prompt in prompts:
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "stream": False,
                "max_tokens": max_tokens,
                "enable_thinking": False,
                "temperature": temperature,
                "top_p": top_p,
            }

            try:
                response = requests.request(
                    "POST",
                    self.api_url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                    proxies=self.proxies
                )
                result = response.json()
                choice = result.get("choices", [{}])[0]
                message = choice.get("message", {})
                text = message.get("content", "").strip()

                # # If the API returns the full prompt + answer, strip off everything up to "assistant\n"
                # answer = text.split("assistant\n")[-1].strip()
                results.append(text)
            except requests.exceptions.RequestException as e:
                results.append(f"[API request failed: {e}]")
            except Exception as e:
                results.append(f"[Unexpected error during API generation: {e}]")

        return results