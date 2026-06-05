"""
CPU-based Semantic Intent Classifier
=====================================
Uses all-MiniLM-L6-v2 (~80 MB) to classify user intent via cosine similarity.
Runs in ~5–15 ms on CPU — no GPU needed, no LLM call wasted on routing.

Recognised intents: general_chat | refactoring | debugging | explanation | metadata_extraction | generation
"""

import logging
from dataclasses import dataclass
from typing import Optional
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
        self._model = SentenceTransformer("all-MiniLM-L6-v2")

        # Pre-compute embeddings once at startup
        self._intent_embeddings = {
            intent: self._model.encode(phrases, convert_to_tensor=True)
            for intent, phrases in INTENT_DEFINITIONS.items()
        }
        logger.info("Intent classifier ready.")

    def classify(self, user_prompt: str, threshold: float = 0.7, gap_threshold: float = 0.15) -> ClassificationDecision:
        """Return the best intent plus confidence metadata.

        If confidence is low or the top-two gap is too small, return a clarification
        decision instead of guessing.
        """
        prompt_emb = self._model.encode(user_prompt, convert_to_tensor=True)

        scored: list[tuple[str, float]] = []
        for intent, embeddings in self._intent_embeddings.items():
            scored.append((intent, float(util.cos_sim(prompt_emb, embeddings).max())))

        scored.sort(key=lambda item: item[1], reverse=True)
        best_intent, best_score = scored[0]
        runner_up_intent, runner_up_score = scored[1] if len(scored) > 1 else (None, 0.0)
        score_gap = best_score - runner_up_score

        if best_score >= threshold and score_gap >= gap_threshold:
            logger.debug("Intent='%s' (score=%.3f, gap=%.3f)", best_intent, best_score, score_gap)
            return ClassificationDecision(
                intent=best_intent,
                confidence=best_score,
                runner_up_intent=runner_up_intent,
                runner_up_confidence=runner_up_score,
                source="semantic",
            )

        clarification = (
            "I'm not quite sure I caught that. "
            "Did you want to look at your dashboard or adjust your settings?"
        )
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


# Singleton — loaded once at startup, shared across all requests
classifier = IntentClassifier()
