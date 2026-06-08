"""
CPU-based Semantic Intent Classifier
=====================================
Uses all-MiniLM-L6-v2 (~80 MB) to classify user intent via cosine similarity.
Runs in ~5–15 ms on CPU — no GPU needed, no LLM call wasted on routing.

Recognised intents: general_chat | refactoring | debugging | explanation | metadata_extraction | generation
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional

# ── Fix 1: Force CPU — avoids CUDA driver version mismatch warning/crash ──────
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

# ── Fix 2: Redirect HuggingFace cache to a writable directory ─────────────────
# The default cache path (/apps/tmp or similar) may not be writable.
# Use ~/.cache/huggingface which is always writable, or a project-local dir.
_HF_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
os.environ.setdefault("HF_HOME", _HF_CACHE)
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(_HF_CACHE, "hub"))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", os.path.join(_HF_CACHE, "sentence_transformers"))
os.makedirs(_HF_CACHE, exist_ok=True)

from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClassificationDecision:
    intent: str
    confidence: float
    runner_up_intent: Optional[str] = None
    runner_up_confidence: float = 0.0
    needs_clarification: bool = False
    clarification_message: Optional[str] = None
    source: str = "semantic"

# ── Intent definitions ────────────────────────────────────────────────────────
# More example phrases per intent = better accuracy.
# Add domain-specific phrases here to tune for your codebase.
INTENT_DEFINITIONS: dict[str, list[str]] = {
    "general_chat": [
        "hi",
        "hello",
        "hey there",
        "how are you",
        "good morning",
        "good evening",
        "thanks",
        "can you help me",
    ],
    "refactoring": [
        "refactor this code to be cleaner",
        "apply SOLID principles to this code",
        "clean up this code and make it more maintainable",
        "restructure this code using design patterns",
        "reduce code duplication and improve readability",
        "apply clean code principles",
        "make this code follow best practices",
        "improve the architecture of this module",
    ],
    "debugging": [
        "fix this bug in my code",
        "debug this error message",
        "my code is crashing with an exception",
        "find and fix the issue in this code",
        "this function throws an error when I run it",
        "troubleshoot why this code is not working",
        "the program crashes with a runtime error",
        "there is an unexpected exception in my application",
    ],
    "explanation": [
        "explain how this code works",
        "summarize this codebase for me",
        "give me an overview of this architecture",
        "what does this function do",
        "break down this code step by step",
        "describe the logic behind this implementation",
        "help me understand this algorithm",
        "walk me through this code",
    ],
    "metadata_extraction": [
        "extract metadata from this project",
        "parse the dependencies from this file",
        "list all the imports and packages used",
        "what libraries does this project depend on",
        "extract file paths and configurations",
        "analyze the project structure and dependencies",
    ],
    "field_extraction": [
        "fill this template using information from the document",
        "extract fields from document and populate the form",
        "find values in this document and fill the template",
        "populate this form with data extracted from the file",
        "read the document and fill in the template fields",
        "extract and fill all placeholder fields from the document",
        "use the document to complete this form template",
    ],
    "generation": [
        "write a Python function that does this",
        "generate code for a REST API",
        "create a new class to handle this",
        "build a script that automates this task",
        "implement this feature from scratch",
        "write a program that solves this problem",
        "code a utility function for this",
        "develop a module that handles this",
    ],
}


class IntentClassifier:
    def __init__(self):
        logger.info("Loading intent classifier model (all-MiniLM-L6-v2)...")
        # Fix 1: Explicitly pass device="cpu" — prevents torch from attempting
        # CUDA initialisation with an incompatible driver.
        self._model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

        # Pre-compute embeddings once at startup
        self._intent_embeddings = {
            intent: self._model.encode(phrases, convert_to_tensor=True)
            for intent, phrases in INTENT_DEFINITIONS.items()
        }
        logger.info("Intent classifier ready.")

    def classify(
        self,
        user_prompt:   str,
        has_file:      bool  = False,
        threshold:     float = 0.45,
        gap_threshold: float = 0.05,
    ) -> ClassificationDecision:
        """
        Classify user intent using semantic similarity.

        Args:
            user_prompt:   The user's raw message.
            has_file:      True if the request includes an uploaded file.
                           Prevents routing to general_chat when code is attached.
            threshold:     Minimum cosine similarity to accept a classification.
            gap_threshold: Minimum gap between top-2 scores to avoid ambiguity.
        """
        prompt_emb = self._model.encode(user_prompt, convert_to_tensor=True)

        scored: list[tuple[str, float]] = []
        for intent, embeddings in self._intent_embeddings.items():
            scored.append((intent, float(util.cos_sim(prompt_emb, embeddings).max())))

        scored.sort(key=lambda item: item[1], reverse=True)
        best_intent, best_score = scored[0]
        runner_up_intent, runner_up_score = scored[1] if len(scored) > 1 else (None, 0.0)
        score_gap = best_score - runner_up_score

        # ── File attachment re-routing ────────────────────────────────────────
        # If a file is attached and top intent is general_chat, it almost
        # certainly means the user wants to do something with the code.
        # Re-route to the highest-scoring code intent instead.
        if has_file and best_intent == "general_chat":
            code_intents = {"debugging", "refactoring", "explanation", "generation", "metadata_extraction", "field_extraction"}
            for intent, score in scored:
                if intent in code_intents:
                    best_intent   = intent
                    best_score    = score
                    score_gap     = best_score - runner_up_score
                    break

        # ── High confidence: return immediately ───────────────────────────────
        if best_score >= threshold and score_gap >= gap_threshold:
            logger.debug("Intent='%s' (score=%.3f, gap=%.3f)", best_intent, best_score, score_gap)
            return ClassificationDecision(
                intent=best_intent,
                confidence=best_score,
                runner_up_intent=runner_up_intent,
                runner_up_confidence=runner_up_score,
                source="semantic",
            )

        # ── Low confidence: ask intelligent clarification ─────────────────────
        clarification = self._build_clarification(user_prompt, scored, has_file)
        logger.debug("Low confidence (score=%.3f, gap=%.3f) — asking for clarification", best_score, score_gap)
        return ClassificationDecision(
            intent="clarification",
            confidence=best_score,
            runner_up_intent=runner_up_intent,
            runner_up_confidence=runner_up_score,
            needs_clarification=True,
            clarification_message=clarification,
            source="semantic_low_confidence",
        )

    def _build_clarification(
        self,
        user_prompt: str,
        scored:      list[tuple[str, float]],
        has_file:    bool = False,
    ) -> str:
        """Generate a context-aware clarification question from the top intent scores."""
        top_intents = [intent for intent, score in scored[:3] if score > 0.2]

        _labels = {
            "general_chat":        "just have a general chat",
            "refactoring":         "refactor or clean up existing code",
            "debugging":           "debug or fix an error",
            "explanation":         "get an explanation of how something works",
            "metadata_extraction": "extract metadata or dependencies from a project",
            "generation":          "generate new code from scratch",
            "field_extraction":    "extract fields from a document and fill a template",
        }

        file_hint    = "I can see you've attached a file. " if has_file else ""
        prompt_lower = user_prompt.lower()

        code_words = {"code","function","class","error","bug","script","file","write","create","build","make","fix","run","test"}
        if any(w in prompt_lower for w in code_words):
            options = [_labels[i] for i in top_intents if i != "general_chat" and i in _labels]
            if options:
                opts_str = " or ".join(f'"{o}"' for o in options[:2])
                return f"{file_hint}Did you want to {opts_str}? A bit more detail would help!"

        if top_intents:
            options = [_labels[i] for i in top_intents if i in _labels]
            opts_str = " or ".join(f'"{o}"' for o in options[:2])
            return f"{file_hint}I'm not sure what you need. Did you want to {opts_str}?"

        return (
            f"{file_hint}Could you describe what you'd like help with? "
            "For example: generating code, debugging an error, refactoring, or explaining how something works."
        )


# Singleton — loaded once at startup, shared across all requests
classifier = IntentClassifier()
