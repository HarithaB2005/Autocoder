"""
Orchestrator
=============
The central pipeline. For every request it:
  1. Classifies intent (CPU, ~10 ms)
  2. Routes to the right specialist agent
  3. For code intents: runs a validation + self-correction loop
     up to VALIDATION_MAX_RETRIES_CEILING (hard limit: 7)
  4. Auto-saves output
  5. Stores turn in session memory

Temperature is NOT a pipeline parameter — each agent owns its own
clamped range. This is intentional: a client cannot accidentally send
temperature=1.8 to the code generator.
"""

import os
import uuid
import logging
from typing import Optional

from app.core.config import settings
from app.core.llm_registry import registry
from app.agents.agents import AGENT_REGISTRY, VALIDATION_AGENT
from app.agents.intent_classifier import classifier
from app.agents.memory import memory_store
from app.schemas.models import PipelineStep, CodeResponse

logger = logging.getLogger(__name__)


class Orchestrator:

    def process(
        self,
        prompt:                 str,
        file_content:           Optional[str] = None,
        file_name:              Optional[str] = None,
        session_id:             Optional[str] = None,
        max_tokens:             Optional[int] = None,
        validation_max_retries: Optional[int] = None,
        auto_save:              bool = True,
    ) -> CodeResponse:
        """
        Run the full agentic pipeline and return a structured CodeResponse.

        Note: temperature is intentionally absent from the signature.
        Each agent enforces its own bounds — see agents.py.
        """
        steps:   list[PipelineStep] = []
        sid      = session_id or str(uuid.uuid4())
        session  = memory_store.get_or_create(sid)

        # Enforce retry ceiling — client can lower it, never raise above 7
        requested = (
            validation_max_retries
            if validation_max_retries is not None
            else settings.VALIDATION_MAX_RETRIES
        )
        max_retries = min(requested, settings.VALIDATION_MAX_RETRIES_CEILING)
        if requested > settings.VALIDATION_MAX_RETRIES_CEILING:
            log_fn = lambda t, m: (steps.append(PipelineStep(type=t, message=m)),
                                   logger.warning("[%s] %s: %s", sid, t.upper(), m))
            steps.append(PipelineStep(
                type="warning",
                message=(
                    f"Requested retries ({requested}) exceeds ceiling "
                    f"({settings.VALIDATION_MAX_RETRIES_CEILING}). Clamped."
                )
            ))

        def log(type_: str, msg: str) -> None:
            steps.append(PipelineStep(type=type_, message=msg))
            logger.info("[%s] %s: %s", sid, type_.upper(), msg)

        # ── Step 1: file context ──────────────────────────────────────────────
        if file_content and file_name:
            log("info", f"File loaded: {file_name} ({len(file_content):,} chars)")

        # ── Step 2: intent classification ─────────────────────────────────────
        log("info", "Classifying intent via CPU semantic model...")
        intent = classifier.classify(prompt)
        session.last_intent = intent
        log("info", f"Intent detected: {intent.upper()}")

        # ── Step 3: build enriched prompt ─────────────────────────────────────
        history  = session.get_context()
        enriched = self._build_prompt(prompt, file_content, file_name, history)

        # ── Step 4: route to agent ────────────────────────────────────────────
        agent = AGENT_REGISTRY.get(intent, AGENT_REGISTRY["generation"])
        log("info", f"Routing to {agent.name} (model: {agent.model}, "
                    f"temp bounds: [{agent.temp_min}–{agent.temp_max}])")

        # max_tokens is clamped inside agent.execute() against MAX_TOKENS_CEILING
        result = agent.execute(enriched, max_tokens=max_tokens)

        # Resolve actual values used (agents may have clamped max_tokens)
        model_cfg   = registry._models.get(agent.model, {})
        tokens_used = (
            min(max_tokens, settings.MAX_TOKENS_CEILING)
            if max_tokens
            else model_cfg.get("max_tokens", settings.DEFAULT_MAX_TOKENS)
        )
        temp_used   = agent.temp_default
        temp_bounds = f"[{agent.temp_min}–{agent.temp_max}]"

        validation_attempts = 0

        # ── Step 5: validation + self-correction ──────────────────────────────
        code_intents = ("generation", "refactoring", "debugging")

        if intent in code_intents:
            log("info", f"Starting validation (max {max_retries} retries, ceiling={settings.VALIDATION_MAX_RETRIES_CEILING})...")

            for attempt in range(1, max_retries + 1):
                validation_attempts = attempt
                val = VALIDATION_AGENT.validate(
                    code=result,
                    original_task=enriched,
                )

                if val["valid"]:
                    log("success", f"Validation passed on attempt {attempt}.")
                    break
                else:
                    log("warning", f"Attempt {attempt}/{max_retries}: {val['feedback']}")
                    if attempt < max_retries:
                        log("info", "Applying self-correction...")
                        fix_prompt = (
                            f"The QA validator found this issue:\n{val['feedback']}\n\n"
                            f"Fix the code below accordingly and return ONLY the corrected code:\n\n{result}"
                        )
                        result = agent.execute(fix_prompt, max_tokens=max_tokens)
                    else:
                        log("warning", f"All {max_retries} retries exhausted — returning best-effort output.")
        else:
            log("success", f"{intent.replace('_', ' ').title()} complete.")

        # ── Step 6: auto-save ─────────────────────────────────────────────────
        saved_to: Optional[str] = None
        if auto_save and intent in code_intents:
            saved_to = self._save(result, file_name)
            if saved_to:
                log("success", f"Saved to: {saved_to}")

        # ── Step 7: update session memory ─────────────────────────────────────
        session.add("user", prompt)
        session.add("assistant", result[:500])

        return CodeResponse(
            session_id=sid,
            intent=intent,
            agent_used=agent.name,
            model_used=agent.model,
            max_tokens_used=tokens_used,
            temperature_used=temp_used,
            temperature_bounds=temp_bounds,
            validation_attempts=validation_attempts,
            result=result,
            steps=steps,
            saved_to=saved_to,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _build_prompt(
        user_prompt:  str,
        file_content: Optional[str],
        file_name:    Optional[str],
        history:      str,
    ) -> str:
        parts = []
        if history:
            parts.append(f"[Conversation history]\n{history}\n")
        if file_content and file_name:
            parts.append(f"[File: {file_name}]\n```\n{file_content}\n```\n")
        parts.append(f"[User request]\n{user_prompt}")
        return "\n".join(parts)

    @staticmethod
    def _save(code: str, file_name: Optional[str]) -> Optional[str]:
        try:
            save_dir = settings.AUTO_SAVE_DIR
            os.makedirs(save_dir, exist_ok=True)
            if file_name:
                name, ext = os.path.splitext(file_name)
                out_name  = f"{name}_modified{ext}"
            else:
                out_name = "generated_code.py"
            out_path = os.path.join(save_dir, out_name)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(code)
            return out_path
        except Exception as e:
            logger.warning("Auto-save failed: %s", e)
            return None


# Singleton
orchestrator = Orchestrator()
