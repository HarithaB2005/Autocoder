MODELS: dict = {
    "qwen-coder-32b": {
        "url":         os.environ.get("GPU_URL_QWEN") or os.environ.get("GPU_URL", ""),
        "model_name":  "qwen-text",
        "max_tokens":  4096,
        "temperature": 0.15,
    },
    "llama-3-70b": {
        "url":         os.environ.get("GPU_URL_LLAMA") or os.environ.get("GPU_URL", ""),
        "model_name":  "llama-text",
        "max_tokens":  3072,
        "temperature": 0.2,
    },
    "gemma-2-9b": {
        "url":         os.environ.get("GPU_URL_GEMMA") or os.environ.get("GPU_URL", ""),
        "model_name":  "gemma-text",
        "max_tokens":  2048,
        "temperature": 0.3,
    },
}

def classify(self, user_prompt: str, threshold: float = 0.7, gap_threshold: float = 0.15) -> ClassificationDecision:
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

        # ── Intelligent clarification based on what the user actually typed ──
        clarification = self._build_clarification(user_prompt, scored)
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

    def _build_clarification(self, user_prompt: str, scored: list[tuple[str, float]]) -> str:
        """Generate a context-aware clarification question based on the top scoring intents."""
        top_intents = [intent for intent, score in scored[:3] if score > 0.3]

        _labels = {
            "general_chat":        "just have a general chat",
            "refactoring":         "refactor or clean up existing code",
            "debugging":           "debug or fix an error",
            "explanation":         "get an explanation of how something works",
            "metadata_extraction": "extract metadata or dependencies from a project",
            "generation":          "generate new code from scratch",
        }

        prompt_lower = user_prompt.lower()

        # If the prompt mentions code-related keywords, narrow the options
        if any(w in prompt_lower for w in ["code", "function", "class", "error", "bug", "script", "file"]):
            options = [_labels[i] for i in top_intents if i != "general_chat" and i in _labels]
            if options:
                opts_str = ", ".join(f'"{o}"' for o in options)
                return f'I can see this is about code — did you want to {opts_str}? Could you clarify a bit more?'

        # Generic fallback with top options
        if top_intents:
            options = [_labels[i] for i in top_intents if i in _labels]
            opts_str = " or ".join(f'"{o}"' for o in options[:2])
            return f'I\'m not quite sure what you need. Did you want to {opts_str}? A bit more detail would help!'

        return 'Could you describe what you\'d like help with? For example: generating code, debugging an error, refactoring, or explaining how something works.'
