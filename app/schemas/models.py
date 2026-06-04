"""
All Pydantic models for request validation and response serialization.

Temperature is NOT a client-facing parameter — each agent enforces its own
range internally. Exposing a global temperature knob would allow clients to
break code agents with high values, or make the validator non-deterministic.

What clients CAN control:
  - max_tokens              (64 – 32,768)
  - validation_max_retries  (0 – 7, hard ceiling)
"""

from __future__ import annotations
from typing import Optional, List, Literal
from pydantic import BaseModel, Field


# ─── Request ──────────────────────────────────────────────────────────────────

class CodeRequest(BaseModel):
    prompt: str = Field(
        ...,
        min_length=3,
        max_length=32_000,
        description="The user's natural-language task or question.",
        examples=["Write a FastAPI endpoint that returns paginated results from PostgreSQL."],
    )
    file_content: Optional[str] = Field(
        None,
        max_length=500_000,   # ~500 KB of source — beyond this, chunk it
        description="Optional source file content for the agents to operate on.",
    )
    file_name: Optional[str] = Field(
        None,
        max_length=255,
        description="Original filename (used for output naming and language hints).",
        examples=["main.py"],
    )
    session_id: Optional[str] = Field(
        None,
        max_length=128,
        description="Session ID for multi-turn memory. Omit to start a new session.",
    )

    # ── Token override ────────────────────────────────────────────────────────
    # Clients may request fewer tokens than the model default (e.g. for speed).
    # They may NOT exceed MAX_TOKENS_CEILING (32,768) — enforced here AND in the orchestrator.
    max_tokens: Optional[int] = Field(
        None,
        ge=64,
        le=32_768,
        description=(
            "Override max output tokens for this request. "
            "Range: 64–32,768. Defaults to the model's own tuned default."
        ),
    )

    # ── Retry override ────────────────────────────────────────────────────────
    # Hard ceiling is 7 — beyond that the cost/benefit is negative.
    validation_max_retries: Optional[int] = Field(
        None,
        ge=0,
        le=7,   # HARD CEILING — matches VALIDATION_MAX_RETRIES_CEILING in config
        description=(
            "How many self-correction loops to allow for this request. "
            "Range: 0 (skip validation) – 7. Defaults to server setting (3)."
        ),
    )


# ─── Response ─────────────────────────────────────────────────────────────────

class PipelineStep(BaseModel):
    type:    Literal["info", "warning", "success", "error"]
    message: str

class CodeResponse(BaseModel):
    session_id:    str
    intent:        str
    agent_used:    str
    model_used:    str

    # Exactly what was sent to the GPU — transparent to the caller
    max_tokens_used:  int
    temperature_used: float
    temperature_bounds: str   # e.g. "[0.0 – 0.3]" — tells client why their value was clamped

    validation_attempts: int = 0
    result:        str
    steps:         List[PipelineStep]
    saved_to:      Optional[str] = None


# ─── Session ──────────────────────────────────────────────────────────────────

class SessionInfo(BaseModel):
    session_id:    str
    message_count: int
    last_intent:   Optional[str] = None

class SessionList(BaseModel):
    sessions: List[SessionInfo]

