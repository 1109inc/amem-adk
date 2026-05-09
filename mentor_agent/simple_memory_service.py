from __future__ import annotations
from mentor_agent.note_extractor import SimpleNoteExtractor
from typing import Dict, List
from uuid import uuid4

from google.adk.memory.base_memory_service import (
    BaseMemoryService,
    SearchMemoryResponse,
    MemoryEntry,
)
from google.adk.sessions import Session
from google.genai import types

from mentor_agent.memory_note import MemoryNote
from mentor_agent.embedding_service import EmbeddingService

class SimpleMemoryService(BaseMemoryService):
    """
    A tiny structured memory service.

    It stores A-Mem-style MemoryNote objects in memory:
      (app_name, user_id) -> list[MemoryNote]

    Still no vector search, links, or evolution yet.
    """

    def __init__(self):
        self._memories: Dict[tuple[str, str], List[MemoryNote]] = {}
        self._extractor = SimpleNoteExtractor()
        self._embedder = EmbeddingService()

    async def add_session_to_memory(self, session: Session) -> None:
        key = (session.app_name, session.user_id)

        if key not in self._memories:
            self._memories[key] = []

        for event in session.events:
            if not event.content or not event.content.parts:
                continue

            text_parts = []
            for part in event.content.parts:
                if getattr(part, "text", None):
                    text_parts.append(part.text)

            if not text_parts:
                continue

            content = " ".join(text_parts)

            keywords = self._extractor.extract_keywords(content)
            tags = self._extractor.extract_tags(content)
            context = self._extractor.create_context(content)

            embedding_text = " ".join(
                [
                    content,
                    " ".join(keywords),
                    " ".join(tags),
                    context,
                ]
            )
            note = MemoryNote(
                id=str(uuid4()),
                app_name=session.app_name,
                user_id=session.user_id,
                author=event.author,
                content=content,
                keywords=keywords,
                tags=tags,
                context=context,
                embedding=self._embedder.embed_text(embedding_text),
            )

            self._memories[key].append(note)

    async def search_memory(
        self,
        *,
        app_name: str,
        user_id: str,
        query: str,
    ) -> SearchMemoryResponse:
        key = (app_name, user_id)
        memories = self._memories.get(key, [])

        query_lower = query.lower()
        matches = [
            note
            for note in memories
            if query_lower in self._searchable_text(note).lower()
        ]
        entries = [
            MemoryEntry(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            text=self._format_note_for_agent(note)
                        )
                    ],
                ),
                author="simple_memory",
            )
            for note in matches[:5]
        ]

        return SearchMemoryResponse(memories=entries)

    def _format_note_for_agent(self, note: MemoryNote) -> str:
        return (
            f"Memory ID: {note.id}\n"
            f"Author: {note.author}\n"
            f"Time: {note.timestamp.isoformat()}\n"
            f"Content: {note.content}\n"
            f"Keywords: {note.keywords}\n"
            f"Tags: {note.tags}\n"
            f"Context: {note.context}\n"
            f"Embedding dimensions: {len(note.embedding)}\n"
            f"Links: {note.links}"
        )
    
    def _searchable_text(self, note: MemoryNote) -> str:
        return " ".join(
            [
                note.content,
                " ".join(note.keywords),
                " ".join(note.tags),
                note.context,
            ]
        )