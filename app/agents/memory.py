"""
Session-scoped Conversation Memory
====================================
Keeps a sliding window of the last N messages per session_id so agents
have multi-turn context. Thread-safe for FastAPI's async workers.

Swap `_store` for Redis or a DB-backed store for multi-process deployments.
"""

import hashlib
import logging
import threading
from collections import OrderedDict, deque
from typing import Any, Optional
from app.core.config import settings

logger = logging.getLogger(__name__)


class SessionMemory:
    """In-process memory for one conversation session."""

    def __init__(self, session_id: str):
        self.session_id   = session_id
        self.last_intent: Optional[str] = None
        self._messages: deque[dict] = deque(maxlen=settings.MEMORY_WINDOW_SIZE * 2)

    def add(self, role: str, content: str) -> None:
        self._messages.append({"role": role, "content": content})

    def get_context(self) -> str:
        return "\n".join(f"{m['role']}: {m['content']}" for m in self._messages)

    @property
    def message_count(self) -> int:
        return len(self._messages)


class MemoryStore:
    """Global registry of all active sessions."""

    def __init__(self):
        self._sessions: dict[str, SessionMemory] = {}
        self._lock = threading.Lock()

    def get_or_create(self, session_id: str) -> SessionMemory:
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = SessionMemory(session_id)
                logger.info("New session created: %s", session_id)
            return self._sessions[session_id]

    def get(self, session_id: str) -> Optional[SessionMemory]:
        return self._sessions.get(session_id)

    def delete(self, session_id: str) -> bool:
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                logger.info("Session deleted: %s", session_id)
                return True
            return False

    def list_sessions(self) -> list[dict]:
        return [
            {
                "session_id":    sid,
                "message_count": s.message_count,
                "last_intent":   s.last_intent,
            }
            for sid, s in self._sessions.items()
        ]


class PromptCache:
    """Tiny in-memory LRU cache for prompt classification results."""

    def __init__(self, max_size: int = 1024):
        self._max_size = max_size
        self._lock = threading.Lock()
        self._store: OrderedDict[str, Any] = OrderedDict()

    @staticmethod
    def _normalize(prompt: str) -> str:
        return " ".join(prompt.strip().lower().split())

    @classmethod
    def _key(cls, prompt: str) -> str:
        normalized = cls._normalize(prompt)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def get(self, prompt: str) -> Any:
        key = self._key(prompt)
        with self._lock:
            value = self._store.get(key)
            if value is not None:
                self._store.move_to_end(key)
            return value

    def set(self, prompt: str, value: Any) -> None:
        key = self._key(prompt)
        with self._lock:
            self._store[key] = value
            self._store.move_to_end(key)
            while len(self._store) > self._max_size:
                self._store.popitem(last=False)


# Singleton
memory_store = MemoryStore()
prompt_cache = PromptCache()
