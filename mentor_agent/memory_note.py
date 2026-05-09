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
    embedding: list[float] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    context: str = ""
    links: list[str] = Field(default_factory=list)