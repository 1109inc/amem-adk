from __future__ import annotations

from datetime import datetime, timezone
from pydantic import BaseModel, Field


class MemoryRevision(BaseModel):
    id: str
    memory_id: str
    triggered_by_memory_id: str

    old_keywords: list[str]
    new_keywords: list[str]

    old_tags: list[str]
    new_tags: list[str]

    old_context: str
    new_context: str

    reason: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))