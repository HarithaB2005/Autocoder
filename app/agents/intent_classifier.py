"""
CPU-based Semantic Intent Classifier
=====================================
Uses all-MiniLM-L6-v2 (~80 MB) to classify user intent via cosine similarity.
Runs in ~5–15 ms on CPU — no GPU needed, no LLM call wasted on routing.

Recognised intents: general_chat | refactoring | debugging | explanation | metadata_extraction | generation
"""

import logging
from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger(__name__)

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

    def classify(self, user_prompt: str, threshold: float = 0.35) -> str:
        """
        Returns the best-matching intent label.
        Falls back to 'general_chat' when confidence is below threshold.
        """
        prompt_emb = self._model.encode(user_prompt, convert_to_tensor=True)

        best_intent = "general_chat"
        best_score  = -1.0

        for intent, embeddings in self._intent_embeddings.items():
            max_score = float(util.cos_sim(prompt_emb, embeddings).max())
            if max_score > best_score:
                best_score  = max_score
                best_intent = intent

        if best_score < threshold:
            logger.debug("Low confidence (%.3f) — defaulting to 'general_chat'", best_score)
            return "general_chat"

        logger.debug("Intent='%s' (score=%.3f)", best_intent, best_score)
        return best_intent


# Singleton — loaded once at startup, shared across all requests
classifier = IntentClassifier()
