from __future__ import annotations

from datetime import datetime, timezone
from pydantic import BaseModel, Field


class MemoryLink(BaseModel):
    id: str
    source_memory_id: str
    target_memory_id: str
    similarity_score: float
    reason: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))