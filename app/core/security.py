"""
API Key authentication dependency.
Add  Authorization: Bearer <API_SECRET_KEY>  to every protected request.
Disable by setting AUTH_ENABLED=false in .env (e.g. during local dev).
"""

from fastapi import Security, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.core.config import settings

bearer_scheme = HTTPBearer(auto_error=False)


def require_api_key(
    credentials: HTTPAuthorizationCredentials = Security(bearer_scheme),
) -> str:
    if not settings.AUTH_ENABLED:
        return "auth-disabled"

    if credentials is None or credentials.credentials != settings.API_SECRET_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials
