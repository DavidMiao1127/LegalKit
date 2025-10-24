import os
import requests
from .base import BaseModel
from typing import List


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

    def generate(self, prompts: List[str]) -> List[str]:
        """
        Generate outputs from a list of prompts via the remote API.
        """
        import time

        max_tokens = self.gen_cfg.get("max_tokens", 512)
        temperature = self.gen_cfg.get("temperature", 1.0)
        top_p = self.gen_cfg.get("top_p", 1.0)

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        results: List[str] = []

        for prompt in prompts:
            attempt = 0
            while attempt < 3:
                attempt += 1
                payload = {
                    "model": self.model_name,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "stream": False,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                }

                try:
                    response = requests.post(
                        self.api_url,
                        json=payload,
                        headers=headers,
                        timeout=self.timeout,
                        proxies=self.proxies,
                    )
                except requests.exceptions.RequestException as e:
                    if attempt < 3:
                        time.sleep(0.8 * attempt)
                        continue
                    results.append(f"[API request failed: {e}]")
                    break

                if response.status_code != 200:
                    # Try to read JSON error first
                    try:
                        err_json = response.json()
                        err_msg = err_json.get("error") or err_json
                    except Exception:
                        err_msg = response.text
                    # Retry on transient status
                    if response.status_code in (429, 500, 502, 503, 504) and attempt < 3:
                        time.sleep(0.8 * attempt)
                        continue
                    results.append(f"[API HTTP {response.status_code}: {err_msg}]")
                    break

                # Parse JSON body
                try:
                    result = response.json()
                except Exception as e:
                    if attempt < 3:
                        time.sleep(0.6 * attempt)
                        continue
                    results.append(f"[API invalid JSON: {e}]")
                    break

                # Error field in body
                if isinstance(result, dict) and result.get("error"):
                    err = result.get("error")
                    if isinstance(err, dict):
                        msg = err.get("message") or err
                    else:
                        msg = err
                    # Retry only if looks like a rate limiting error
                    if "rate" in str(msg).lower() and attempt < 3:
                        time.sleep(0.8 * attempt)
                        continue
                    results.append(f"[API error: {msg}]")
                    break

                # OpenAI-compatible schema
                choice = None
                if isinstance(result, dict):
                    choices = result.get("choices")
                    if isinstance(choices, list) and choices:
                        choice = choices[0]
                if not choice:
                    if attempt < 3:
                        time.sleep(0.5 * attempt)
                        continue
                    results.append("[API error: missing choices in response]")
                    break

                message = choice.get("message") or {}
                text = (message.get("content") or "").strip()
                if not text:
                    text = (choice.get("text") or "").strip()

                if not text:
                    if attempt < 3:
                        time.sleep(0.4 * attempt)
                        continue
                    results.append("[API empty content]")
                    break

                results.append(text)
                break

        return results