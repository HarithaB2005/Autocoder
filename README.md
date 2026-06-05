# 🧠 Enterprise Agentic AutoCoder — FastAPI

Middleware-driven, multi-agent code generation system.  
Converted from Streamlit → production-grade FastAPI REST API.

---
import os
_HF_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
os.environ["HF_HOME"]                   = _HF_CACHE
os.environ["TRANSFORMERS_CACHE"]        = os.path.join(_HF_CACHE, "hub")
os.environ["HUGGINGFACE_HUB_CACHE"]     = os.path.join(_HF_CACHE, "hub")
os.environ["SENTENCE_TRANSFORMERS_HOME"]= os.path.join(_HF_CACHE, "sentence_transformers")
os.environ["HF_DATASETS_CACHE"]         = os.path.join(_HF_CACHE, "datasets")
os.environ["CUDA_VISIBLE_DEVICES"]      = ""
os.makedirs(os.path.join(_HF_CACHE, "hub"), exist_ok=True)
os.makedirs(os.path.join(_HF_CACHE, "sentence_transformers"), exist_ok=True)
os.makedirs(os.path.join(_HF_CACHE, "datasets"), exist_ok=True)

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
