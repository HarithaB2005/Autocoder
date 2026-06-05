"""
GPU Model Registry
==================
Endpoints and model names are configured via environment variables (.env file).
Each model has its own URL env var since you have separate endpoints per model.

Set these in your .env:
    GPU_URL_QWEN=https://...
    GPU_URL_LLAMA=https://...
    GPU_URL_GEMMA=https://...
    GPU_DEPARTMENT=your-department-value
    GPU_ENV=prod
"""

import os
import time
import logging
import requests
from typing import Optional
from app.core.config import settings

logger = logging.getLogger(__name__)


MODELS: dict = {
    "qwen-coder-32b": {
        "url":         os.environ.get("GPU_URL_QWEN", ""),
        "model_name":  "qwen-text",
        "max_tokens":  4096,
        "temperature": 0.15,
    },
    "llama-3-70b": {
        "url":         os.environ.get("GPU_URL_LLAMA", ""),
        "model_name":  "llama-text",
        "max_tokens":  3072,
        "temperature": 0.2,
    },
    "gemma-2-9b": {
        "url":         os.environ.get("GPU_URL_GEMMA", ""),
        "model_name":  "gemma-text",
        "max_tokens":  2048,
        "temperature": 0.3,
    },
}


class GPUModelRegistry:

    def __init__(self):
        self._models = MODELS
        self._department = os.environ.get("GPU_DEPARTMENT", "YOUR_DEPARTMENT_HERE")

    def generate(
        self,
        model_name:  str,
        prompt:      str,
        max_tokens:  Optional[int]   = None,
        temperature: Optional[float] = None,
    ) -> str:
        cfg = self._get_config(model_name)

        final_max_tokens  = max_tokens  if max_tokens  is not None else cfg["max_tokens"]
        final_temperature = temperature if temperature is not None else cfg["temperature"]

        if not cfg["url"]:
            logger.warning(
                "[DEMO] Model '%s' has no URL configured — set GPU_URL_%s in your .env file.",
                model_name, model_name.upper().replace("-", "_"),
            )
            return self._demo_response(model_name, final_max_tokens, final_temperature)

        return self._call_with_retry(cfg, model_name, prompt, final_max_tokens, final_temperature)

    def list_models(self) -> list[str]:
        return list(self._models.keys())

    def _get_config(self, model_name: str) -> dict:
        if model_name not in self._models:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Available: {list(self._models.keys())}"
            )
        return self._models[model_name]

    def _call_with_retry(
        self,
        cfg:          dict,
        model_name:   str,
        prompt:       str,
        max_tokens:   int,
        temperature:  float,
        max_attempts: int   = 3,
        backoff:      float = 2.0,
    ) -> str:
        payload = {
            "prompt":        prompt,
            "model":         cfg["model_name"],
            "max_new_token": max_tokens,
            "temperature":   temperature,
            "department":    self._department,
            "env":           os.environ.get("GPU_ENV", "prod"),
        }

        last_error: Exception = Exception("Unknown")
        for attempt in range(1, max_attempts + 1):
            try:
                resp = requests.post(
                    cfg["url"],
                    data=payload,
                    timeout=settings.DEFAULT_TIMEOUT_SECONDS,
                )
                resp.raise_for_status()

                result = resp.json()
                return (
                    result.get("generated_text")
                    or result.get("response")
                    or result.get("text")
                    or str(result)
                ).strip()

            except requests.exceptions.Timeout as e:
                last_error = e
                logger.warning("[%s] Attempt %d/%d timed out.", model_name, attempt, max_attempts)
            except requests.exceptions.HTTPError as e:
                last_error = e
                logger.warning("[%s] Attempt %d/%d HTTP error: %s", model_name, attempt, max_attempts, e)
            except Exception as e:
                last_error = e
                logger.error("[%s] Attempt %d/%d unexpected error: %s", model_name, attempt, max_attempts, e)

            if attempt < max_attempts:
                sleep_for = backoff ** attempt
                logger.info("[%s] Retrying in %.1fs ...", model_name, sleep_for)
                time.sleep(sleep_for)

        raise RuntimeError(
            f"Model '{model_name}' failed after {max_attempts} attempts. "
            f"Last error: {last_error}"
        )

    @staticmethod
    def _demo_response(model_name: str, max_tokens: int, temperature: float) -> str:
        time.sleep(0.3)
        return (
            f"# [PLACEHOLDER — {model_name}]\n"
            f"# Set GPU_URL_QWEN / GPU_URL_LLAMA / GPU_URL_GEMMA in your .env file.\n\n"
            f"def placeholder_function():\n"
            f"    pass\n"
        )


# Singleton — imported everywhere as `from app.core.llm_registry import registry`
registry = GPUModelRegistry()
