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
from app.agents.agents import (
    AGENT_REGISTRY, VALIDATION_AGENT, is_gibberish,
    template_parser, field_extraction_agent, value_cleaner,
    cross_checker, field_extraction_validator,
)
from app.agents.intent_classifier import classifier
from app.agents.memory import memory_store
from app.schemas.models import PipelineStep, CodeResponse, FieldResult

logger = logging.getLogger(__name__)


class Orchestrator:

    def process(
        self,
        prompt:                 str,
        file_content:           Optional[str] = None,
        file_name:              Optional[str] = None,
        template_content:       Optional[str] = None,
        template_name:          Optional[str] = None,
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

        # Gibberish check on template too
        if template_content and template_name:
            log("info", f"Template received: {template_name} ({len(template_content):,} chars)")
            gibberish, reason = is_gibberish(template_content)
            if gibberish:
                log("error", f"Gibberish template rejected: {reason}")
                return error_response(
                    f"The template file '{template_name}' doesn't appear to contain readable text."
                )

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

        # ── FIELD EXTRACTION PIPELINE ─────────────────────────────────────────
        if intent == "field_extraction":
            return self._run_field_extraction(
                prompt=prompt,
                file_content=file_content or "",
                file_name=file_name or "document.txt",
                template_content=template_content or "",
                template_name=template_name or "template.txt",
                session=session,
                sid=sid,
                steps=steps,
                decision=decision,
            )
        # ─────────────────────────────────────────────────────────────────────

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
            "field_extraction":    "Extract fields and fill the template:",
            "general_chat":        "User message:",
        }
        frame = task_frames.get(intent, "Task:")
        parts.append(f"[{frame}]\n{user_prompt}")

        return "\n".join(parts)

    def _run_field_extraction(
        self,
        prompt:           str,
        file_content:     str,
        file_name:        str,
        template_content: str,
        template_name:    str,
        session,
        sid:              str,
        steps:            list,
        decision,
    ) -> CodeResponse:
        """
        Full field extraction pipeline:
        Phase 3 — Template Parsing
        Phase 4 — LLM Field Extraction
        Phase 5 — Value Cleaning
        Phase 6 — Cross-field Consistency
        Phase 7 — Template Filling
        Phase 8 — LangSmith Validation
        """
        def log(type_: str, msg: str):
            steps.append(PipelineStep(type=type_, message=msg))
            logger.info("[%s] %s: %s", sid, type_.upper(), msg)

        def fe_error(reason: str) -> CodeResponse:
            return CodeResponse(
                session_id=sid, intent="field_extraction",
                agent_used="Field Extractor", model_used=field_extraction_agent.model,
                needs_clarification=False, clarification_message=None,
                classification_confidence=decision.confidence,
                classification_gap=decision.confidence - decision.runner_up_confidence,
                classification_source=decision.source,
                max_tokens_used=0, temperature_used=0.0,
                temperature_bounds="[0.0–0.1]",
                validation_attempts=0, validation_score=None,
                result=reason, steps=steps, saved_to=None,
            )

        # ── Phase 3: Template Parsing ─────────────────────────────────────────
        if not template_content:
            return fe_error(
                "No template provided. Please upload a template file using "
                "/upload and pass its content as 'template_content'."
            )
        if not file_content:
            return fe_error(
                "No document provided. Please upload a document using "
                "/upload and pass its content as 'file_content'."
            )

        log("info", f"Parsing template: {template_name}")
        fields = template_parser.parse(template_content)
        if not fields:
            return fe_error(
                f"No placeholder fields found in template '{template_name}'. "
                "Supported formats: {{field_name}}, [FIELD_NAME], __field_name__, <field_name>"
            )
        log("info", f"Found {len(fields)} fields: {', '.join(f.name for f in fields)}")
        for f in fields:
            log("info", f"  {f.name} → type={f.type}" +
                (f" pattern={f.pattern}" if f.pattern else "") +
                (f" constraints={f.constraints}" if f.constraints else ""))

        # ── Phase 4: LLM Field Extraction ─────────────────────────────────────
        log("info", "Extracting field values from document (single LLM call)...")
        raw_extracted = field_extraction_agent.extract(file_content, fields)
        log("info", f"Raw extraction complete: {len([v for v in raw_extracted.values() if v])} values found")

        # ── Phase 5: Value Cleaning & Type Conversion ─────────────────────────
        log("info", "Cleaning and converting extracted values...")
        cleaned        = {}
        field_results  = []

        for f in fields:
            raw = raw_extracted.get(f.name)
            cleaned_val, error = value_cleaner.clean(raw, f)

            if raw is None or str(raw).strip().lower() in ('null', 'none', '', 'n/a', 'not found'):
                status = "not_found"
            elif error:
                status = "conversion_error"
            else:
                status = "filled"
                cleaned[f.name] = cleaned_val

            field_results.append({
                "field":         f.name,
                "raw_value":     str(raw) if raw is not None else None,
                "cleaned_value": cleaned_val,
                "type":          f.type,
                "status":        status,
                "grounded":      None,
                "error":         error,
            })
            log(
                "info" if status == "filled" else "warning",
                f"  {f.name}: {status}" +
                (f" → {cleaned_val}" if status == "filled" else "") +
                (f" (error: {error})" if error else "")
            )

        # ── Phase 6: Cross-field Consistency ──────────────────────────────────
        log("info", "Running cross-field consistency checks...")
        warnings = cross_checker.check(raw_extracted, cleaned, file_content)
        for w in warnings:
            log("warning", f"Cross-check: {w}")

        # ── Phase 7: Template Filling ──────────────────────────────────────────
        log("info", "Filling template with extracted values...")
        filled_template = template_content
        for f in fields:
            value_str   = str(cleaned.get(f.name, "")) if f.name in cleaned else "[NOT FOUND IN DOCUMENT]"
            # Replace all supported placeholder formats
            filled_template = filled_template.replace(f"{{{f.name}}}", value_str)
            filled_template = filled_template.replace(f"[{f.name.upper()}]", value_str)
            filled_template = filled_template.replace(f"__{f.name}__", value_str)
            filled_template = filled_template.replace(f"<{f.name}>", value_str)

        # ── Phase 8: LangSmith Validation ────────────────────────────────────
        log("info", "Validating extraction results (LangSmith)...")
        val = field_extraction_validator.validate(
            fields=fields,
            cleaned=cleaned,
            field_results=field_results,
            document=file_content,
            run_name=f"extraction_{sid[:8]}",
        )

        log(
            "success" if val["valid"] else "warning",
            f"Validation {'PASSED' if val['valid'] else 'FAILED'} "
            f"(score={val['score']:.3f} | "
            f"coverage={val['coverage_score']:.2f} "
            f"grounding={val['grounding_score']:.2f} "
            f"type={val['type_score']:.2f})"
        )
        for fb in val["feedback"]:
            log("warning", fb)

        # ── Update session ────────────────────────────────────────────────────
        session.add("user", prompt)
        session.add("assistant", f"Filled template with {len(cleaned)}/{len(fields)} fields.")

        return CodeResponse(
            session_id=sid,
            intent="field_extraction",
            agent_used=field_extraction_agent.name,
            model_used=field_extraction_agent.model,
            needs_clarification=False,
            clarification_message=None,
            classification_confidence=decision.confidence,
            classification_gap=decision.confidence - decision.runner_up_confidence,
            classification_source=decision.source,
            max_tokens_used=1024,
            temperature_used=field_extraction_agent.temp_default,
            temperature_bounds=f"[{field_extraction_agent.temp_min}–{field_extraction_agent.temp_max}]",
            validation_attempts=1,
            validation_score=val["score"],
            result=filled_template,
            steps=steps,
            saved_to=None,
            # Field extraction specific
            field_results=[FieldResult(**r) for r in field_results],
            coverage_score=val["coverage_score"],
            grounding_score=val["grounding_score"],
            type_score=val["type_score"],
            missing_fields=val["missing_fields"],
            warnings=warnings + val["feedback"],
            validation_passed=val["valid"],
        )

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
