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
from typing import Any, Optional, List, Literal
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
        max_length=500_000,
        description="Optional source file content for the agents to operate on.",
    )
    file_name: Optional[str] = Field(
        None,
        max_length=255,
        description="Original filename (used for output naming and language hints).",
        examples=["main.py"],
    )

    # ── Template fields (for field_extraction intent) ─────────────────────────
    template_content: Optional[str] = Field(
        None,
        max_length=500_000,
        description=(
            "Template text containing placeholders like {field_name}, [FIELD], "
            "or __field__. Used with file_content for field extraction tasks. "
            "Get this value from the /upload endpoint."
        ),
    )
    template_name: Optional[str] = Field(
        None,
        max_length=255,
        description="Original template filename.",
        examples=["form_template.txt"],
    )
    session_id: Optional[str] = Field(
        None,
        max_length=128,
        description="Session ID for multi-turn memory. Omit to start a new session.",
    )
    max_tokens: Optional[int] = Field(
        None,
        ge=64,
        le=32_768,
        description=(
            "Override max output tokens for this request. "
            "Range: 64–32,768. Defaults to the model's own tuned default."
        ),
    )
    validation_max_retries: Optional[int] = Field(
        None,
        ge=0,
        le=7,
        description=(
            "How many self-correction loops to allow for this request. "
            "Range: 0 (skip validation) – 7. Defaults to server setting (3)."
        ),
    )


# ─── Field Extraction ─────────────────────────────────────────────────────────

class FieldResult(BaseModel):
    field:         str
    raw_value:     Optional[str]   = None
    cleaned_value: Optional[Any]   = None
    type:          str             = "text"
    status:        str             = "not_found"   # filled | not_found | conversion_error
    grounded:      Optional[bool]  = None
    error:         Optional[str]   = None


# ─── Response ─────────────────────────────────────────────────────────────────

class PipelineStep(BaseModel):
    type:    Literal["info", "warning", "success", "error"]
    message: str

class CodeResponse(BaseModel):
    session_id:  str
    intent:      str
    agent_used:  str
    model_used:  str

    needs_clarification:       bool            = False
    clarification_message:     Optional[str]   = None
    classification_confidence: Optional[float] = None
    classification_gap:        Optional[float] = None
    classification_source:     Optional[str]   = None

    max_tokens_used:   int
    temperature_used:  float
    temperature_bounds: str

    validation_attempts: int            = 0
    validation_score:    Optional[float] = Field(
        None,
        description="Final weighted validation score (0.0–1.0). None if validation was skipped."
    )

    result:   str
    steps:    List[PipelineStep]
    saved_to: Optional[str] = None

    # ── Field extraction results (populated only for field_extraction intent) ─
    field_results:    Optional[List[FieldResult]] = None
    coverage_score:   Optional[float]             = None
    grounding_score:  Optional[float]             = None
    type_score:       Optional[float]             = None
    missing_fields:   Optional[List[str]]         = None
    warnings:         Optional[List[str]]         = None
    validation_passed: Optional[bool]             = None


# ─── Session ──────────────────────────────────────────────────────────────────

class SessionInfo(BaseModel):
    session_id:    str
    message_count: int
    last_intent:   Optional[str] = None

class SessionList(BaseModel):
    sessions: List[SessionInfo]
