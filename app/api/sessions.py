"""
Session Management
==================
GET  /sessions          — list all active sessions
GET  /sessions/{id}     — get one session info
DELETE /sessions/{id}   — clear session memory
"""

from fastapi import APIRouter, Depends, HTTPException
from app.agents.memory import memory_store
from app.schemas.models import SessionInfo, SessionList
from app.core.security import require_api_key

router = APIRouter()


@router.get("/sessions", response_model=SessionList, summary="List active sessions")
def list_sessions(_: str = Depends(require_api_key)):
    return SessionList(
        sessions=[SessionInfo(**s) for s in memory_store.list_sessions()]
    )


@router.get("/sessions/{session_id}", response_model=SessionInfo, summary="Get session info")
def get_session(session_id: str, _: str = Depends(require_api_key)):
    session = memory_store.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    return SessionInfo(
        session_id=session.session_id,
        message_count=session.message_count,
        last_intent=session.last_intent,
    )


@router.delete("/sessions/{session_id}", summary="Delete session memory")
def delete_session(session_id: str, _: str = Depends(require_api_key)):
    deleted = memory_store.delete(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"deleted": session_id}
