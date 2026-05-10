from __future__ import annotations

from datetime import datetime, timezone
from pydantic import BaseModel, Field


class MemoryNote(BaseModel):
    id: str
    app_name: str
    user_id: str
    author: str
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # A-Mem core fields
    embedding: list[float] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    context: str = ""
    links: list[str] = Field(default_factory=list)

    # Safety / provenance fields
    memory_type: str = "observation"
    source_type: str = "session_event"
    source_id: str | None = None
    is_derived: bool = False
    confidence: float = 1.0
    evidence_memory_ids: list[str] = Field(default_factory=list)

    # Lifecycle fields
    last_accessed_at: datetime | None = None
    access_count: int = 0
    expires_at: datetime | None = None