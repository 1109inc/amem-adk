from __future__ import annotations
from mentor_agent.note_extractor import SimpleNoteExtractor
from typing import Dict, List
from uuid import uuid4
from mentor_agent.similarity import cosine_similarity
from google.adk.memory.base_memory_service import (
    BaseMemoryService,
    SearchMemoryResponse,
    MemoryEntry,
)
from google.adk.sessions import Session
from google.genai import types
from mentor_agent.similarity import cosine_similarity
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
        self._link_threshold = 0.65

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

            self._link_related_memories(
                new_note=note,
                existing_notes=self._memories[key],
            )

            self._evolve_related_memories(
                new_note=note,
                existing_notes=self._memories[key],
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

        if not memories:
            return SearchMemoryResponse(memories=[])

        query_embedding = self._embedder.embed_text(query)

        scored_notes = []
        for note in memories:
            score = cosine_similarity(query_embedding, note.embedding)
            scored_notes.append((score, note))

        scored_notes.sort(key=lambda item: item[0], reverse=True)

        top_notes = [
            (score, note)
            for score, note in scored_notes[:5]
            if score > 0.2
        ]
        expanded_notes = self._expand_with_linked_memories(
            top_notes=top_notes,
            all_notes=memories,
        )
        entries = [
            MemoryEntry(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            text=f"Similarity Score: {score:.4f}\n{self._format_note_for_agent(note)}"
                        )
                    ]
                ),
                author="simple_memory",
            )
            for score, note in expanded_notes[:5]
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
    def _link_related_memories(
        self,
        new_note: MemoryNote,
        existing_notes: List[MemoryNote],
    ) -> None:
        for old_note in existing_notes:
            score = cosine_similarity(new_note.embedding, old_note.embedding)

            if score >= self._link_threshold:
                if old_note.id not in new_note.links:
                    new_note.links.append(old_note.id)

                if new_note.id not in old_note.links:
                    old_note.links.append(new_note.id)
    def _get_note_by_id(
        self,
        notes: List[MemoryNote],
        note_id: str,
    ) -> MemoryNote | None:
        for note in notes:
            if note.id == note_id:
                return note
        return None
    def _expand_with_linked_memories(
        self,
        top_notes: list[tuple[float, MemoryNote]],
        all_notes: List[MemoryNote],
    ) -> list[tuple[float, MemoryNote]]:
        expanded: list[tuple[float, MemoryNote]] = []
        seen_ids: set[str] = set()

        for score, note in top_notes:
            if note.id not in seen_ids:
                expanded.append((score, note))
                seen_ids.add(note.id)

            for linked_id in note.links:
                linked_note = self._get_note_by_id(all_notes, linked_id)

                if linked_note and linked_note.id not in seen_ids:
                    # Linked memories get a slightly lower score because they were
                    # not directly retrieved by the query.
                    expanded.append((score * 0.9, linked_note))
                    seen_ids.add(linked_note.id)

        expanded.sort(key=lambda item: item[0], reverse=True)
        return expanded
    def _evolve_related_memories(
        self,
        new_note: MemoryNote,
        existing_notes: List[MemoryNote],
    ) -> None:
        for linked_id in new_note.links:
            old_note = self._get_note_by_id(existing_notes, linked_id)

            if old_note is None:
                continue

            merged_keywords = sorted(set(old_note.keywords + new_note.keywords))
            merged_tags = sorted(set(old_note.tags + new_note.tags))

            old_note.keywords = merged_keywords
            old_note.tags = merged_tags

            old_note.context = (
                "This memory is connected to related memories about "
                f"keywords={merged_keywords} and tags={merged_tags}."
            )