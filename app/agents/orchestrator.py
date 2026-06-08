"""
Orchestrator
=============
Central pipeline. For every request it:
  1. Gibberish check on uploaded file (CPU-only, instant — disable via agents.GIBBERISH_CHECK_ENABLED)
  2. Classifies intent (CPU, ~10 ms, all-MiniLM-L6-v2)
  3. Routes to the right specialist agent
  4. For code intents: runs 4-phase validation + self-correction loop
  5. Stores turn in session memory
  6. Returns full result in API response (auto_save=False by default)
"""

import os
import uuid
import logging
from pathlib import Path
from typing import Optional

from app.core.config import settings
from app.core.llm_registry import registry
from app.agents.agents import AGENT_REGISTRY, VALIDATION_AGENT, is_gibberish
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
        auto_save:              bool = False,
    ) -> CodeResponse:

        steps:  list[PipelineStep] = []
        sid     = session_id or str(uuid.uuid4())
        session = memory_store.get_or_create(sid)

        requested   = (
            validation_max_retries
            if validation_max_retries is not None
            else settings.VALIDATION_MAX_RETRIES
        )
        max_retries = min(requested, settings.VALIDATION_MAX_RETRIES_CEILING)
        if requested > settings.VALIDATION_MAX_RETRIES_CEILING:
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

        def error_response(reason: str) -> CodeResponse:
            return CodeResponse(
                session_id=sid,
                intent="error",
                agent_used="none",
                model_used="none",
                needs_clarification=False,
                clarification_message=None,
                classification_confidence=0.0,
                classification_gap=0.0,
                classification_source="pre_classification",
                max_tokens_used=0,
                temperature_used=0.0,
                temperature_bounds="[0.0 – 0.0]",
                validation_attempts=0,
                validation_score=None,
                result=reason,
                steps=steps,
                saved_to=None,
            )

        # ── Step 1: File validation ───────────────────────────────────────────
        if file_content and file_name:
            log("info", f"File received: {file_name} ({len(file_content):,} chars)")
            gibberish, reason = is_gibberish(file_content)
            if gibberish:
                log("error", f"Gibberish/binary file rejected: {reason}")
                return error_response(
                    f"The uploaded file '{file_name}' doesn't appear to contain valid code or "
                    f"readable text ({reason}). Please upload a source code file."
                )
            log("info", "File content validated — readable text confirmed.")

        # ── Step 2: Intent classification ─────────────────────────────────────
        log("info", "Classifying intent...")
        decision = classifier.classify(prompt, has_file=bool(file_content))
        session.last_intent = decision.intent

        if decision.needs_clarification:
            log("warning", "Intent unclear — returning clarification prompt.")
            session.add("user", prompt)
            session.add("assistant", decision.clarification_message or "Could you clarify?")
            return CodeResponse(
                session_id=sid,
                intent=decision.intent,
                agent_used="clarification",
                model_used="none",
                needs_clarification=True,
                clarification_message=decision.clarification_message,
                classification_confidence=decision.confidence,
                classification_gap=decision.confidence - decision.runner_up_confidence,
                classification_source=decision.source,
                max_tokens_used=0,
                temperature_used=0.0,
                temperature_bounds="[0.0 – 0.0]",
                validation_attempts=0,
                validation_score=None,
                result=decision.clarification_message or "Could you clarify?",
                steps=steps,
                saved_to=None,
            )

        intent = decision.intent
        log("info", f"Intent: {intent.upper()} (confidence={decision.confidence:.3f}, source={decision.source})")

        # ── Step 3: Build enriched prompt ─────────────────────────────────────
        history  = session.get_context()
        enriched = self._build_prompt(prompt, file_content, file_name, history, intent)

        # ── Step 4: Route to agent ────────────────────────────────────────────
        agent = AGENT_REGISTRY.get(intent, AGENT_REGISTRY["generation"])
        log("info", f"Agent: {agent.name} | Model: {agent.model} | Temp bounds: [{agent.temp_min}–{agent.temp_max}]")

        result = agent.execute(enriched, max_tokens=max_tokens)

        model_cfg   = registry._models.get(agent.model, {})
        tokens_used = (
            min(max_tokens, settings.MAX_TOKENS_CEILING)
            if max_tokens
            else model_cfg.get("max_tokens", settings.DEFAULT_MAX_TOKENS)
        )
        temp_used   = agent.temp_default
        temp_bounds = f"[{agent.temp_min}–{agent.temp_max}]"

        validation_attempts = 0
        validation_score    = None
        code_intents        = ("generation", "refactoring", "debugging")

        # ── Step 5: 4-phase validation + self-correction ──────────────────────
        if intent in code_intents:
            log("info", f"Starting 4-phase validation (max {max_retries} retries)...")

            for attempt in range(1, max_retries + 1):
                validation_attempts = attempt
                val = VALIDATION_AGENT.validate(code=result, original_task=enriched)
                validation_score = val["score"]

                # Log per-phase results
                phases = val.get("phases", {})
                if "sandbox" in phases:
                    sb = phases["sandbox"]
                    log("info", f"  Sandbox: ran={sb.get('ran')} score={sb.get('execution_score', 0):.2f} tests={len(sb.get('test_results', []))}")
                if "requirements" in phases:
                    log("info", f"  Requirements score: {phases['requirements'].get('score', 0):.2f}")
                if "quality" in phases:
                    log("info", f"  Quality score: {phases['quality'].get('score', 0):.2f}")

                log("info", f"  Overall validation score: {validation_score:.3f} (threshold={VALIDATION_AGENT.PASS_THRESHOLD})")

                if val["valid"]:
                    log("success", f"Validation PASSED on attempt {attempt} (score={validation_score:.3f}).")
                    break
                else:
                    log("warning", f"Attempt {attempt}/{max_retries} FAILED: {val['feedback']}")
                    if attempt < max_retries:
                        log("info", "Self-correcting...")
                        fix_prompt = (
                            f"The QA validator found these issues (score={validation_score:.2f}):\n"
                            f"{val['feedback']}\n\n"
                            f"Fix ONLY these issues in the code below. "
                            f"Return ONLY the corrected code:\n\n{result}"
                        )
                        result = agent.execute(fix_prompt, max_tokens=max_tokens)
                    else:
                        log("warning", f"All {max_retries} retries exhausted (final score={validation_score:.3f}). Returning best-effort output.")
        else:
            log("success", f"{intent.replace('_', ' ').title()} complete.")

        # ── Step 6: Auto-save (disabled by default) ───────────────────────────
        saved_to: Optional[str] = None
        if auto_save and intent in code_intents:
            saved_to = self._save(result, file_name)
            if saved_to:
                log("success", f"Saved to: {saved_to}")

        # ── Step 7: Update session memory ─────────────────────────────────────
        session.add("user", prompt)
        session.add("assistant", result[:500])

        return CodeResponse(
            session_id=sid,
            intent=intent,
            agent_used=agent.name,
            model_used=agent.model,
            needs_clarification=False,
            clarification_message=None,
            classification_confidence=decision.confidence,
            classification_gap=decision.confidence - decision.runner_up_confidence,
            classification_source=decision.source,
            max_tokens_used=tokens_used,
            temperature_used=temp_used,
            temperature_bounds=temp_bounds,
            validation_attempts=validation_attempts,
            validation_score=validation_score,
            result=result,
            steps=steps,
            saved_to=saved_to,
        )

    @staticmethod
    def _build_prompt(
        user_prompt:  str,
        file_content: Optional[str],
        file_name:    Optional[str],
        history:      str,
        intent:       str,
    ) -> str:
        """
        Build a rich, structured prompt that gives the agent maximum context.
        Different intents get slightly different framing.
        """
        parts = []

        if history:
            parts.append(f"[Conversation History]\n{history}\n")

        if file_content and file_name:
            ext = Path(file_name).suffix.lstrip(".").upper() or "CODE"
            parts.append(f"[Uploaded File: {file_name}]\n```{ext.lower()}\n{file_content}\n```\n")

        # Intent-specific framing
        task_frames = {
            "generation":          "Generate the following:",
            "refactoring":         "Refactor the code above as follows:",
            "debugging":           "Debug and fix the code above:",
            "explanation":         "Explain the following:",
            "metadata_extraction": "Extract structured metadata from the above code:",
            "general_chat":        "User message:",
        }
        frame = task_frames.get(intent, "Task:")
        parts.append(f"[{frame}]\n{user_prompt}")

        return "\n".join(parts)

    @staticmethod
    def _save(code: str, file_name: Optional[str]) -> Optional[str]:
        try:
            save_dir = Path(settings.AUTO_SAVE_DIR)
            if not save_dir.is_absolute():
                save_dir = Path(__file__).resolve().parents[2] / save_dir
            save_dir.mkdir(parents=True, exist_ok=True)
            if file_name:
                name, ext = os.path.splitext(Path(file_name).name)
                out_name = f"{name}_modified{ext}"
            else:
                out_name = "generated_code.py"
            out_path = save_dir / out_name
            out_path.write_text(code, encoding="utf-8")
            return str(out_path)
        except Exception as e:
            logger.warning("Auto-save failed: %s", e)
            return None


orchestrator = Orchestrator()
