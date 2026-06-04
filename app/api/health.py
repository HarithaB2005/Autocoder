from fastapi import APIRouter
from app.core.llm_registry import registry
from app.core.config import settings

router = APIRouter()


@router.get("/health", summary="Health check")
def health():
    return {
        "status":  "ok",
        "version": "2.0.0",
        "models":  registry.list_models(),
        "validation_max_retries": settings.VALIDATION_MAX_RETRIES,
        "auth_enabled": settings.AUTH_ENABLED,
    }
