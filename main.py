"""
Enterprise Agentic AutoCoder — FastAPI
Run:   uvicorn main:app --reload
Docs:  http://localhost:8000/docs
"""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import app.core.logging_setup  # noqa: F401 — sets up JSON logging at import time
from app.api import code, health, sessions
from app.core.config import settings

app = FastAPI(
    title="Enterprise Agentic AutoCoder",
    description=(
        "Multi-agent code generation, refactoring, debugging, and validation.\n\n"
        "**Quick start**\n"
        "1. Set `API_SECRET_KEY` in `.env`\n"
        "2. Point GPU endpoints in `app/core/llm_registry.py`\n"
        "3. `POST /api/v1/process` with `Authorization: Bearer <key>`\n"
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router,   tags=["Health"])
app.include_router(code.router,     prefix="/api/v1", tags=["Code Agents"])
app.include_router(sessions.router, prefix="/api/v1", tags=["Sessions"])


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL,
    )
