from __future__ import annotations

from sqlalchemy import select

from mentor_agent.database import AsyncSessionLocal
from mentor_agent.db_models import (
    MemoryLinkRecord,
    MemoryNoteRecord,
    MemoryRevisionRecord,
)
from mentor_agent.memory_link import MemoryLink
from mentor_agent.memory_note import MemoryNote
from mentor_agent.memory_revision import MemoryRevision


class MemoryRepository:
    async def save_note(self, note: MemoryNote) -> None:
        async with AsyncSessionLocal() as session:
            record = MemoryNoteRecord(
                id=note.id,
                app_name=note.app_name,
                user_id=note.user_id,
                author=note.author,
                content=note.content,
                timestamp=note.timestamp,
                embedding=note.embedding,
                keywords=note.keywords,
                tags=note.tags,
                context=note.context,
                links=note.links,
                memory_type=note.memory_type,
                source_type=note.source_type,
                source_id=note.source_id,
                is_derived=note.is_derived,
                confidence=note.confidence,
                evidence_memory_ids=note.evidence_memory_ids,
                last_accessed_at=note.last_accessed_at,
                access_count=note.access_count,
                expires_at=note.expires_at,
            )

            await session.merge(record)
            await session.commit()

    async def load_notes(self, app_name: str, user_id: str) -> list[MemoryNote]:
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(MemoryNoteRecord).where(
                    MemoryNoteRecord.app_name == app_name,
                    MemoryNoteRecord.user_id == user_id,
                )
            )

            records = result.scalars().all()

            return [
                MemoryNote(
                    id=record.id,
                    app_name=record.app_name,
                    user_id=record.user_id,
                    author=record.author,
                    content=record.content,
                    timestamp=record.timestamp,
                    embedding=record.embedding or [],
                    keywords=record.keywords or [],
                    tags=record.tags or [],
                    context=record.context or "",
                    links=record.links or [],
                    memory_type=record.memory_type,
                    source_type=record.source_type,
                    source_id=record.source_id,
                    is_derived=record.is_derived,
                    confidence=record.confidence,
                    evidence_memory_ids=record.evidence_memory_ids or [],
                    last_accessed_at=record.last_accessed_at,
                    access_count=record.access_count,
                    expires_at=record.expires_at,
                )
                for record in records
            ]

    async def save_link(self, link: MemoryLink) -> None:
        async with AsyncSessionLocal() as session:
            record = MemoryLinkRecord(
                id=link.id,
                source_memory_id=link.source_memory_id,
                target_memory_id=link.target_memory_id,
                similarity_score=link.similarity_score,
                reason=link.reason,
                timestamp=link.timestamp,
            )

            await session.merge(record)
            await session.commit()

    async def load_links(self, memory_id: str) -> list[MemoryLink]:
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(MemoryLinkRecord).where(
                    MemoryLinkRecord.source_memory_id == memory_id
                )
            )

            records = result.scalars().all()

            return [
                MemoryLink(
                    id=record.id,
                    source_memory_id=record.source_memory_id,
                    target_memory_id=record.target_memory_id,
                    similarity_score=record.similarity_score,
                    reason=record.reason,
                    timestamp=record.timestamp,
                )
                for record in records
            ]

    async def save_revision(self, revision: MemoryRevision) -> None:
        async with AsyncSessionLocal() as session:
            record = MemoryRevisionRecord(
                id=revision.id,
                memory_id=revision.memory_id,
                triggered_by_memory_id=revision.triggered_by_memory_id,
                old_keywords=revision.old_keywords,
                new_keywords=revision.new_keywords,
                old_tags=revision.old_tags,
                new_tags=revision.new_tags,
                old_context=revision.old_context,
                new_context=revision.new_context,
                reason=revision.reason,
                timestamp=revision.timestamp,
            )

            await session.merge(record)
            await session.commit()

    async def load_revisions(self, memory_id: str) -> list[MemoryRevision]:
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(MemoryRevisionRecord).where(
                    MemoryRevisionRecord.memory_id == memory_id
                )
            )

            records = result.scalars().all()

            return [
                MemoryRevision(
                    id=record.id,
                    memory_id=record.memory_id,
                    triggered_by_memory_id=record.triggered_by_memory_id,
                    old_keywords=record.old_keywords or [],
                    new_keywords=record.new_keywords or [],
                    old_tags=record.old_tags or [],
                    new_tags=record.new_tags or [],
                    old_context=record.old_context,
                    new_context=record.new_context,
                    reason=record.reason,
                    timestamp=record.timestamp,
                )
                for record in records
            ]