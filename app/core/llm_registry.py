"""
GPU Model Registry
==================
Swap the placeholder URLs/keys below with your real vLLM / TGI / Ollama endpoints.
Everything else (retries, timeouts, per-model defaults) is handled automatically.

To add a new model:
    1. Add an entry to MODELS dict below.
    2. Reference the model name in agents.py.
    Done.
"""

import time
import logging
import requests
from typing import Optional
from app.core.config import settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 🔧  CONFIGURE YOUR GPU ENDPOINTS HERE
# ─────────────────────────────────────────────────────────────────────────────
MODELS: dict = {
    "qwen-coder-32b": {
        "url":         "http://YOUR_GPU_NODE_1_IP:8000/v1/completions",
        "api_key":     "YOUR_API_KEY_HERE",
        # Best for code generation: precise, high-token output
        "max_tokens":  4096,
        "temperature": 0.15,
    },
    "llama-3-70b": {
        "url":         "http://YOUR_GPU_NODE_2_IP:8000/v1/completions",
        "api_key":     "YOUR_API_KEY_HERE",
        # Best for debugging / refactoring: focused, low hallucination
        "max_tokens":  3072,
        "temperature": 0.2,
    },
    "gemma-2-9b": {
        "url":         "http://YOUR_GPU_NODE_3_IP:8000/v1/completions",
        "api_key":     "YOUR_API_KEY_HERE",
        # Best for explanation / validation / metadata: balanced
        "max_tokens":  2048,
        "temperature": 0.3,
    },
}
# ─────────────────────────────────────────────────────────────────────────────


class GPUModelRegistry:
    """
    Thread-safe, retry-aware client for your GPU inference cluster.

    Call signature:
        registry.generate(
            model_name  = "qwen-coder-32b",
            prompt      = "...",
            max_tokens  = None,   # None → uses model default
            temperature = None,   # None → uses model default
        )
    """

    def __init__(self):
        self._models = MODELS

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(
        self,
        model_name:  str,
        prompt:      str,
        max_tokens:  Optional[int]   = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Call the GPU endpoint and return the generated text.
        Falls back to a clearly labelled placeholder when the URL is still
        the default placeholder string — so the app runs end-to-end in demo mode.
        """
        cfg = self._get_config(model_name)

        final_max_tokens  = max_tokens  if max_tokens  is not None else cfg["max_tokens"]
        final_temperature = temperature if temperature is not None else cfg["temperature"]

        # ── Demo / placeholder mode ───────────────────────────────────────────
        if "YOUR_GPU_NODE" in cfg["url"]:
            logger.warning(
                "[DEMO] Model '%s' has no real endpoint configured — "
                "returning placeholder response.",
                model_name,
            )
            return self._demo_response(model_name, final_max_tokens, final_temperature)

        # ── Real GPU call ─────────────────────────────────────────────────────
        return self._call_with_retry(cfg, model_name, prompt, final_max_tokens, final_temperature)

    def list_models(self) -> list[str]:
        return list(self._models.keys())

    # ── Private helpers ───────────────────────────────────────────────────────

    def _get_config(self, model_name: str) -> dict:
        if model_name not in self._models:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Available: {list(self._models.keys())}"
            )
        return self._models[model_name]

    def _call_with_retry(
        self,
        cfg:         dict,
        model_name:  str,
        prompt:      str,
        max_tokens:  int,
        temperature: float,
        max_attempts: int = 3,
        backoff:     float = 2.0,
    ) -> str:
        """
        HTTP POST with exponential backoff.
        Raises RuntimeError after all attempts are exhausted.
        """
        headers = {
            "Authorization": f"Bearer {cfg['api_key']}",
            "Content-Type":  "application/json",
        }
        payload = {
            "model":       model_name,
            "prompt":      prompt,
            "max_tokens":  max_tokens,
            "temperature": temperature,
        }

        last_error: Exception = Exception("Unknown")
        for attempt in range(1, max_attempts + 1):
            try:
                resp = requests.post(
                    cfg["url"],
                    headers=headers,
                    json=payload,
                    timeout=settings.DEFAULT_TIMEOUT_SECONDS,
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["text"].strip()

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
        time.sleep(0.3)  # Tiny artificial delay so UI feels realistic
        return (
            f"# [PLACEHOLDER — {model_name}]\n"
            f"# max_tokens={max_tokens}  temperature={temperature}\n"
            f"# Replace the URL in app/core/llm_registry.py with your GPU endpoint.\n\n"
            f"def placeholder_function():\n"
            f"    \"\"\"Generated by {model_name} (demo mode).\"\"\"\n"
            f"    pass\n"
        )


# Singleton — imported everywhere as `from app.core.llm_registry import registry`
registry = GPUModelRegistry()
