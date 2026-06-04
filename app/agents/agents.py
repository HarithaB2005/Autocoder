"""
Agent Definitions
=================
Each agent wraps one GPU model with a role-specific system prompt
AND enforces its own temperature bounds.

Temperature philosophy:
  - Code agents (gen/refactor/debug): 0.0–0.3  — deterministic, fewer hallucinations
  - Validation agent:                 0.0–0.1  — PASS/FAIL must be consistent
  - Explanation agent:                0.0–0.8  — fluent prose needs some creativity
  - Metadata agent:                   0.0–0.2  — structured JSON must be stable

Any user-supplied temperature is CLAMPED (not rejected) into the agent's range.
This means clients can still tune within reason — they just can't break the model.
"""

import logging
from typing import Optional
from app.core.llm_registry import registry
from app.core.config import settings

logger = logging.getLogger(__name__)


def _clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a float to [min_val, max_val]."""
    return max(min_val, min(max_val, value))


class BaseAgent:
    """
    Base class for all agents.

    Each subclass declares:
      - model        : which GPU model to call
      - role_prompt  : the system/role instruction prepended to every prompt
      - temp_default : what temperature to use when the caller provides none
      - temp_min     : hard lower bound (clamped, never violated)
      - temp_max     : hard upper bound (clamped, never violated)

    execute() always clamps the final temperature into [temp_min, temp_max]
    before the value reaches the GPU registry.
    """

    name:         str   = "Base Agent"
    model:        str   = "llama-3-70b"
    role_prompt:  str   = "You are a helpful AI assistant."
    temp_default: float = 0.2
    temp_min:     float = 0.0
    temp_max:     float = 1.0   # subclasses tighten this

    def execute(
        self,
        task:        str,
        max_tokens:  Optional[int]   = None,
        temperature: Optional[float] = None,
    ) -> str:
        # Use caller value if provided, else agent default — then clamp to agent range
        raw_temp  = temperature if temperature is not None else self.temp_default
        final_temp = _clamp(raw_temp, self.temp_min, self.temp_max)

        if temperature is not None and final_temp != temperature:
            logger.warning(
                "[%s] Temperature %.2f clamped to %.2f (agent bounds: %.2f–%.2f)",
                self.name, temperature, final_temp, self.temp_min, self.temp_max,
            )

        # Clamp tokens to the global ceiling too
        final_tokens = min(max_tokens, settings.MAX_TOKENS_CEILING) if max_tokens else None

        full_prompt = f"{self.role_prompt}\n\nTask:\n{task}"
        logger.info(
            "[%s] → model=%s  max_tokens=%s  temperature=%.2f",
            self.name, self.model, final_tokens or "model-default", final_temp,
        )

        result = registry.generate(
            model_name=self.model,
            prompt=full_prompt,
            max_tokens=final_tokens,
            temperature=final_temp,
        )
        logger.info("[%s] ← %d chars received", self.name, len(result))
        return result


# ── Specialised Agents ────────────────────────────────────────────────────────

class GeneralAssistantAgent(BaseAgent):
    """
    Handles greetings, casual conversation, and other non-code requests.
    Uses the general-purpose base model instead of a code-specialist model.
    """
    name         = "Assistant"
    model        = "llama-3-70b"
    temp_default = 0.3
    temp_min     = 0.0
    temp_max     = 0.8
    role_prompt  = (
        "You are a helpful general-purpose assistant. "
        "Answer the user's request naturally and directly. "
        "If the user is just greeting you, reply briefly and politely. "
        "If they ask a question, answer it clearly."
    )

class CodeGenerationAgent(BaseAgent):
    """
    Writes new code from a natural-language spec.
    Low temperature: deterministic output reduces hallucinated APIs.
    """
    name         = "Code Generator"
    model        = "qwen-coder-32b"
    temp_default = 0.15
    temp_min     = 0.0
    temp_max     = 0.3   # anything higher produces unreliable code
    role_prompt  = (
        "You are a senior software engineer. "
        "Produce syntactically correct, well-commented, enterprise-ready code. "
        "Include type hints, docstrings, and handle edge cases. "
        "Return ONLY the code — no prose, no markdown fences."
    )


class RefactorAgent(BaseAgent):
    """
    Restructures existing code without changing behaviour.
    Low temperature: must preserve semantics exactly.
    """
    name         = "Refactorer"
    model        = "llama-3-70b"
    temp_default = 0.15
    temp_min     = 0.0
    temp_max     = 0.3
    role_prompt  = (
        "You are an expert code refactorer. "
        "Apply SOLID principles, clean-code practices, and relevant design patterns. "
        "Preserve existing behaviour while improving readability and maintainability. "
        "Return ONLY the refactored code — no prose, no markdown fences."
    )


class DebuggingAgent(BaseAgent):
    """
    Identifies and fixes bugs in provided code or error traces.
    Low temperature: reproducible fixes, no creative detours.
    """
    name         = "Debugger"
    model        = "llama-3-70b"
    temp_default = 0.1
    temp_min     = 0.0
    temp_max     = 0.3
    role_prompt  = (
        "You are an expert debugger. "
        "Analyse the provided code or error trace, identify the root cause, "
        "and return a corrected version with a brief inline comment explaining each fix. "
        "Return ONLY the fixed code — no prose, no markdown fences."
    )


class ExplanationAgent(BaseAgent):
    """
    Explains code, architecture, or concepts in plain language.
    Higher temperature ceiling: natural, fluent prose.
    """
    name         = "Explainer"
    model        = "gemma-2-9b"
    temp_default = 0.5
    temp_min     = 0.0
    temp_max     = 0.8   # wider range — good explanations benefit from fluency
    role_prompt  = (
        "You are a senior technical educator. "
        "Read the user's request carefully: if they want a quick summary be concise; "
        "if they ask for a deep dive, provide a thorough breakdown. "
        "Use plain language, concrete examples, and step-by-step reasoning where helpful."
    )


class MetadataExtractionAgent(BaseAgent):
    """
    Extracts structured metadata (dependencies, symbols, etc.) as JSON.
    Low temperature: JSON output must be stable and parseable.
    """
    name         = "Metadata Extractor"
    model        = "gemma-2-9b"
    temp_default = 0.1
    temp_min     = 0.0
    temp_max     = 0.2   # structured JSON needs to be deterministic
    role_prompt  = (
        "You are a structured-data extraction bot. "
        "Parse the input and return a JSON object with keys: "
        "language, dependencies, entry_points, exported_symbols, file_purpose. "
        "Return ONLY valid JSON — no prose, no markdown fences."
    )


class ValidationAgent(BaseAgent):
    """
    Two-phase validator:
      Phase 1 — LLM logical critique  (does code satisfy original task?)
      Phase 2 — AST structural check  (is Python syntactically valid?)

    Temperature is intentionally pinned to near-zero: PASS/FAIL verdicts
    must be consistent across retries. Any variance here causes phantom
    failures or phantom passes — both are bugs.
    """
    name         = "Validator"
    model        = "gemma-2-9b"
    temp_default = 0.05
    temp_min     = 0.0
    temp_max     = 0.1   # HARD: non-deterministic validation is worse than no validation
    role_prompt  = (
        "You are a strict QA engineer. "
        "Review the code against the original task requirements. "
        "If it fully and correctly satisfies the task, respond with exactly: PASS\n"
        "If you find ANY logical flaw, hallucinated import, or missed requirement, "
        "describe the problem in one concise sentence so the generator can fix it."
    )

    def validate(
        self,
        code:          str,
        original_task: str,
    ) -> dict:
        """
        Returns:
            {"valid": True,  "feedback": "All checks passed."}
            {"valid": False, "feedback": "<specific flaw for the generator to fix>"}
        """
        import ast, traceback

        # ── Phase 1: LLM logical critique ────────────────────────────────────
        critique_prompt = (
            f"Original Task:\n{original_task}\n\n"
            f"Code to review:\n{code}\n\n"
            f"Does the code fully and correctly satisfy the task? "
            f"If yes: PASS. If no: explain the flaw concisely."
        )
        critique = registry.generate(
            model_name=self.model,
            prompt=f"{self.role_prompt}\n\n{critique_prompt}",
            max_tokens=512,
            temperature=self.temp_default,  # always uses agent's own pinned temperature
        )
        logger.debug("[Validator] LLM critique: %s", critique)

        if "PASS" not in critique.upper():
            return {"valid": False, "feedback": f"Logic QA failed: {critique.strip()}"}

        # ── Phase 2: AST syntax check ─────────────────────────────────────────
        try:
            ast.parse(code)
        except SyntaxError as e:
            detail = "".join(traceback.format_exception_only(type(e), e)).strip()
            return {"valid": False, "feedback": f"Syntax error: {detail}"}

        return {"valid": True, "feedback": "All checks passed (logic + syntax)."}


# ── Registry — lets the orchestrator look up agents by name ──────────────────
AGENT_REGISTRY: dict[str, BaseAgent] = {
    "general_chat":        GeneralAssistantAgent(),
    "generation":          CodeGenerationAgent(),
    "refactoring":         RefactorAgent(),
    "debugging":           DebuggingAgent(),
    "explanation":         ExplanationAgent(),
    "metadata_extraction": MetadataExtractionAgent(),
}

VALIDATION_AGENT = ValidationAgent()
