# 🧠 Enterprise Agentic AutoCoder — FastAPI

Middleware-driven, multi-agent code generation system.  
Converted from Streamlit → production-grade FastAPI REST API.
1. app/core/llm_registry.py — change the 3 URL lines to fall back to a single GPU_URL:
pythonMODELS: dict = {
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
---

## Architecture

```
POST /api/v1/process
        │
        ▼
  ┌─────────────────────────────────────────────────────────┐
  │                     Orchestrator                        │
  │                                                         │
  │  1. CPU Intent Classifier (all-MiniLM-L6-v2, ~10ms)    │
  │         │                                               │
  │         ▼                                               │
  │  2. Agent Router                                        │
  │     ├── generation        → CodeGenerationAgent         │
  │     ├── refactoring       → RefactorAgent               │
  │     ├── debugging         → DebuggingAgent              │
  │     ├── explanation       → ExplanationAgent            │
  │     └── metadata_extract  → MetadataExtractionAgent     │
  │         │                                               │
  │         ▼                                               │
  │  3. Validation Loop (code intents only)                 │
  │     ┌──────────────────────────────────┐               │
  │     │ ValidationAgent (LLM logic check │               │
  │     │  + AST syntax check)             │               │
  │     │                                  │               │
  │     │  if FAIL → self-correct          │               │
  │     │  retry up to MAX_RETRIES times   │               │
  │     └──────────────────────────────────┘               │
  │         │                                               │
  │         ▼                                               │
  │  4. Auto-save to disk                                   │
  │  5. Update session memory                               │
  └─────────────────────────────────────────────────────────┘
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env — set API_SECRET_KEY

# 3. Point GPU endpoints (app/core/llm_registry.py)
# Replace "YOUR_GPU_NODE_X_IP" with your vLLM/TGI/Ollama IPs

# 4. Run
uvicorn main:app --reload
# API docs → http://localhost:8000/docs
```

---

## API Endpoints

### `POST /api/v1/process`
Run the full agent pipeline. Returns structured result + pipeline telemetry.

**Headers:**
```
Authorization: Bearer <API_SECRET_KEY>
Content-Type: application/json
```

**Request body:**
```json
{
  "prompt": "Write a FastAPI endpoint that returns paginated DB results",
  "file_content": "# optional: paste source code here",
  "file_name": "main.py",
  "session_id": "optional-uuid-for-multi-turn",
  "max_tokens": 4096,
  "temperature": 0.15,
  "validation_max_retries": 3
}
```

**Response:**
```json
{
  "session_id": "abc-123",
  "intent": "generation",
  "agent_used": "Code Generator",
  "model_used": "qwen-coder-32b",
  "max_tokens_used": 4096,
  "temperature_used": 0.15,
  "validation_attempts": 1,
  "result": "def paginate(...): ...",
  "saved_to": "generated_output/generated_code.py",
  "steps": [
    {"type": "info",    "message": "Intent detected: GENERATION"},
    {"type": "info",    "message": "Routing to Code Generator"},
    {"type": "success", "message": "Validation passed on attempt 1."}
  ]
}
```

### `POST /api/v1/process/stream`
Same as above but streams pipeline steps as **Server-Sent Events** — great for live UIs.

Each event:
```
data: {"type": "info",    "message": "Intent detected: GENERATION"}
data: {"type": "success", "message": "Validation passed on attempt 1."}
data: {"type": "done",    "data": { ...full CodeResponse... }}
```

### `GET /api/v1/sessions`
List all active sessions.

### `GET /api/v1/sessions/{session_id}`
Get session info (message count, last intent).

### `DELETE /api/v1/sessions/{session_id}`
Clear session memory.

### `GET /health`
Returns server status, available models, current config.

---

## Configuration (`.env`)

| Variable | Default | Description |
|---|---|---|
| `API_SECRET_KEY` | `change-me` | Bearer token clients must send |
| `AUTH_ENABLED` | `true` | Set `false` for local dev |
| `VALIDATION_MAX_RETRIES` | `3` | Self-correction loop limit (0 = no validation) |
| `DEFAULT_MAX_TOKENS` | `4096` | Global fallback if model config omits it |
| `DEFAULT_TEMPERATURE` | `0.2` | Global fallback if model config omits it |
| `DEFAULT_TIMEOUT_SECONDS` | `120` | HTTP timeout per GPU call |
| `MEMORY_WINDOW_SIZE` | `10` | Conversation turns kept per session |
| `AUTO_SAVE_DIR` | `generated_output` | Where generated code is saved |

Per-request overrides (`max_tokens`, `temperature`, `validation_max_retries`) always win over `.env` values.

---

## Plugging In GPU Models

Open `app/core/llm_registry.py` and replace the placeholder values:

```python
MODELS = {
    "qwen-coder-32b": {
        "url":         "http://10.0.0.1:8000/v1/completions",  # ← your vLLM node
        "api_key":     "sk-...",
        "max_tokens":  4096,
        "temperature": 0.15,
    },
    ...
}
```

The app runs in **demo mode** (returns placeholder responses) when a URL still contains `YOUR_GPU_NODE`.

---

## Adding a New Agent

1. Subclass `BaseAgent` in `app/agents/agents.py`
2. Add it to `AGENT_REGISTRY` at the bottom of the same file
3. Add example phrases for the new intent in `app/agents/intent_classifier.py`
4. Done — no changes needed anywhere else.

---

## Project Structure

```
agentic_autocoder_fastapi/
├── main.py                          # FastAPI app + router registration
├── requirements.txt
├── .env.example
└── app/
    ├── core/
    │   ├── config.py                # All settings (pydantic-settings)
    │   ├── llm_registry.py          # GPU model client + retry logic
    │   ├── security.py              # API key auth dependency
    │   └── logging_setup.py         # Structured JSON logging
    ├── agents/
    │   ├── agents.py                # BaseAgent + all specialist agents
    │   ├── intent_classifier.py     # CPU semantic classifier
    │   ├── memory.py                # Per-session sliding-window memory
    │   └── orchestrator.py          # Full pipeline + validation loop
    ├── api/
    │   ├── code.py                  # POST /process  +  POST /process/stream
    │   ├── health.py                # GET  /health
    │   └── sessions.py              # GET/DELETE /sessions
    └── schemas/
        └── models.py                # Pydantic request/response models
```
