from __future__ import annotations

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from mentor_agent.database import Base


class MemoryNoteRecord(Base):
    __tablename__ = "memory_notes"

    id: Mapped[str] = mapped_column(String, primary_key=True)

    app_name: Mapped[str] = mapped_column(String, index=True)
    user_id: Mapped[str] = mapped_column(String, index=True)
    author: Mapped[str] = mapped_column(String)

    content: Mapped[str] = mapped_column(Text)
    timestamp: Mapped[DateTime] = mapped_column(DateTime)

    embedding: Mapped[list[float]] = mapped_column(JSON)
    keywords: Mapped[list[str]] = mapped_column(JSON)
    tags: Mapped[list[str]] = mapped_column(JSON)
    context: Mapped[str] = mapped_column(Text)
    links: Mapped[list[str]] = mapped_column(JSON)

    memory_type: Mapped[str] = mapped_column(String)
    source_type: Mapped[str] = mapped_column(String)
    source_id: Mapped[str | None] = mapped_column(String, nullable=True)
    is_derived: Mapped[bool] = mapped_column(Boolean)
    confidence: Mapped[float] = mapped_column(Float)
    evidence_memory_ids: Mapped[list[str]] = mapped_column(JSON)

    last_accessed_at: Mapped[DateTime | None] = mapped_column(DateTime, nullable=True)
    access_count: Mapped[int] = mapped_column(Integer)
    expires_at: Mapped[DateTime | None] = mapped_column(DateTime, nullable=True)
    importance: Mapped[float] = mapped_column(Float)
    memory_strength: Mapped[float] = mapped_column(Float)
    retention_score: Mapped[float] = mapped_column(Float)

class MemoryLinkRecord(Base):
    __tablename__ = "memory_links"

    id: Mapped[str] = mapped_column(String, primary_key=True)

    source_memory_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("memory_notes.id"),
        index=True,
    )
    target_memory_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("memory_notes.id"),
        index=True,
    )

    similarity_score: Mapped[float] = mapped_column(Float)
    reason: Mapped[str] = mapped_column(Text)
    timestamp: Mapped[DateTime] = mapped_column(DateTime)


class MemoryRevisionRecord(Base):
    __tablename__ = "memory_revisions"

    id: Mapped[str] = mapped_column(String, primary_key=True)

    memory_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("memory_notes.id"),
        index=True,
    )
    triggered_by_memory_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("memory_notes.id"),
        index=True,
    )

    old_keywords: Mapped[list[str]] = mapped_column(JSON)
    new_keywords: Mapped[list[str]] = mapped_column(JSON)

    old_tags: Mapped[list[str]] = mapped_column(JSON)
    new_tags: Mapped[list[str]] = mapped_column(JSON)

    old_context: Mapped[str] = mapped_column(Text)
    new_context: Mapped[str] = mapped_column(Text)

    reason: Mapped[str] = mapped_column(Text)
    timestamp: Mapped[DateTime] = mapped_column(DateTime)