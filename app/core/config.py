"""
Central configuration — all tunables in one place.
Override any value via environment variables or a .env file.

HARD LIMITS (never overridable by clients):
  - MAX_TOKENS_CEILING      : absolute token cap across all models
  - VALIDATION_MAX_RETRIES  : max self-correction loops per request (ceiling = 7)

AGENT TEMPERATURE BOUNDS:
  Temperature is not a free parameter — each agent type has a scientifically
  justified range. Clients may request a temperature, but the orchestrator
  clamps it to the agent's allowed range before it reaches the model.

  code_gen / debug / refactor  → [0.0 – 0.3]   near-deterministic, reduces hallucination
  validation                   → [0.0 – 0.1]   must be consistent; PASS/FAIL cannot vary
  explanation                  → [0.0 – 0.8]   needs some creativity for good prose
  metadata_extraction          → [0.0 – 0.2]   structured JSON output, must be stable
"""

from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    # ── Server ────────────────────────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    LOG_LEVEL: str = "info"
    ALLOWED_ORIGINS: List[str] = ["*"]

    # ── Auth ──────────────────────────────────────────────────────────────────
    API_SECRET_KEY: str = "change-me-in-production"
    AUTH_ENABLED: bool = True

    # ── Validation / Self-Correction ──────────────────────────────────────────
    VALIDATION_MAX_RETRIES: int = 3          # default; clients can request up to CEILING
    VALIDATION_MAX_RETRIES_CEILING: int = 7  # HARD LIMIT — never exceeded regardless of request

    # ── Token limits ──────────────────────────────────────────────────────────
    MAX_TOKENS_CEILING: int = 32_768         # HARD LIMIT — Pydantic + orchestrator both enforce
    DEFAULT_MAX_TOKENS: int = 4_096          # sensible default when request omits it
    DEFAULT_TEMPERATURE: float = 0.2         # global fallback when model config omits it
    DEFAULT_TIMEOUT_SECONDS: int = 120

    # ── Per-agent temperature bounds (min, max) ───────────────────────────────
    # Orchestrator clamps any user-supplied temperature into these ranges.
    # Format: (min, max)
    TEMP_BOUNDS_CODE_GEN:    tuple = (0.0, 0.3)   # generation / refactor / debug
    TEMP_BOUNDS_VALIDATION:  tuple = (0.0, 0.1)   # validator must be near-deterministic
    TEMP_BOUNDS_EXPLANATION: tuple = (0.0, 0.8)   # explainer benefits from fluency
    TEMP_BOUNDS_METADATA:    tuple = (0.0, 0.2)   # structured JSON output

    # ── Memory ───────────────────────────────────────────────────────────────
    MEMORY_WINDOW_SIZE: int = 10

    # ── Output ───────────────────────────────────────────────────────────────
    AUTO_SAVE_DIR: str = "generated_output"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
