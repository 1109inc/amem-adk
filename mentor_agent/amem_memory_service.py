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
from datetime import datetime, timezone
import math
from mentor_agent.llm_link_judge import LLMLinkJudge
from google.adk.sessions import Session
from google.genai import types
from mentor_agent.memory_note import MemoryNote
from mentor_agent.embedding_service import EmbeddingService
from mentor_agent.memory_revision import MemoryRevision
from mentor_agent.memory_link import MemoryLink
from mentor_agent.memory_repository import MemoryRepository
from mentor_agent.database import init_db
from mentor_agent.llm_note_extractor import LLMNoteExtractor
from mentor_agent.llm_memory_evolver import LLMMemoryEvolver

class AMemMemoryService(BaseMemoryService):
    """
    A custom ADK memory service inspired by A-Mem.

    Current features:
    - structured MemoryNote storage
    - rule-based keyword/tag/context extraction
    - Gemini embeddings
    - semantic retrieval with cosine similarity
    - similarity-based link generation
    - graph-expanded retrieval
    - simple rule-based memory evolution
    """

    def __init__(self):
        self._memories: Dict[tuple[str, str], List[MemoryNote]] = {}
        self._extractor = SimpleNoteExtractor()
        self._embedder = EmbeddingService()
        self._link_threshold = 0.65
        self._revisions: Dict[str, List[MemoryRevision]] = {}
        self._links: Dict[str, List[MemoryLink]] = {}
        self._repo = MemoryRepository()
        self._llm_extractor = LLMNoteExtractor()
        self._link_judge = LLMLinkJudge()
        self._memory_evolver = LLMMemoryEvolver()
        self._duplicate_threshold = 0.92

    async def add_session_to_memory(self, session: Session) -> None:
        key = (session.app_name, session.user_id)

        if key not in self._memories:
            self._memories[key] = []
        existing_notes = await self._repo.load_notes(
            app_name=session.app_name,
            user_id=session.user_id,
        )
        if existing_notes:
            self._memories[key] = existing_notes
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

            try:
                metadata = self._llm_extractor.extract(content)
                keywords = metadata.keywords
                tags = metadata.tags
                context = metadata.context
            except Exception:
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
            source_type = "user_message" if event.author == "user" else "agent_message"
            confidence = 1.0 if event.author == "user" else 0.75
            source_id = getattr(event, "id", None)
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
                source_type=source_type,
                source_id=source_id,
                confidence=confidence,

            )
            duplicate_note, duplicate_score = self._find_most_similar_memory(
                new_note=note,
                existing_notes=self._memories[key],
            )
            if duplicate_note is not None and duplicate_score >= self._duplicate_threshold:
                now = datetime.now(timezone.utc)
                duplicate_note.access_count += 1
                duplicate_note.last_accessed_at = now
                duplicate_note.memory_strength += 1.0
                duplicate_note.retention_score = self._calculate_retention_score(duplicate_note, now)
                await self._repo.save_note(duplicate_note)
                continue

            await self._link_related_memories(
                new_note=note,
                existing_notes=self._memories[key],
            )

            await self._evolve_related_memories(
                new_note=note,
                existing_notes=self._memories[key],
            )

            self._memories[key].append(note)
            await self._repo.save_note(note)
    async def search_memory(
        self,
        *,
        app_name: str,
        user_id: str,
        query: str,
    ) -> SearchMemoryResponse:
        key = (app_name, user_id)
        memories = await self._repo.load_notes(app_name=app_name, user_id=user_id)

        if not memories:
            memories = self._memories.get(key, [])

        if not memories:
            return SearchMemoryResponse(memories=[])

        query_embedding = self._embedder.embed_text(query)
        now = datetime.now(timezone.utc)

        for note in memories:
            note.retention_score = self._calculate_retention_score(note, now)
            await self._repo.save_note(note)
        scored_notes = []
        for note in memories:
            semantic_score = cosine_similarity(query_embedding, note.embedding)
            final_score = self._calculate_final_score(
                semantic_score=semantic_score,
                note=note,
            )
            scored_notes.append((final_score, note))

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
        returned_notes = expanded_notes[:5]
        now = datetime.now(timezone.utc)
        for _, note in returned_notes:
            note.access_count += 1
            note.last_accessed_at = now
            note.memory_strength += 1.0
            note.retention_score = self._calculate_retention_score(note, now)
            await self._repo.save_note(note)
        entries = [
            MemoryEntry(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            text=f"Final Score: {score:.4f}\n{self._format_note_for_agent(note)}"
                        )
                    ]
                ),
                author="simple_memory",
            )
            for score, note in returned_notes
        ]

        return SearchMemoryResponse(memories=entries)

    def _format_note_for_agent(self, note: MemoryNote) -> str:
        revision_count = len(self._revisions.get(note.id, []))

        return (
            f"Memory ID: {note.id}\n"
            f"Author: {note.author}\n"
            f"Time: {note.timestamp.isoformat()}\n"
            f"Content: {note.content}\n"
            f"Keywords: {note.keywords}\n"
            f"Tags: {note.tags}\n"
            f"Context: {note.context}\n"
            f"Embedding dimensions: {len(note.embedding)}\n"
            f"Links: {note.links}\n"
            f"Revision count: {revision_count}\n"
            f"Memory type: {note.memory_type}\n"
            f"Source type: {note.source_type}\n"
            f"Source ID: {note.source_id}\n"
            f"Is derived: {note.is_derived}\n"
            f"Confidence: {note.confidence}\n"
            f"Evidence memory IDs: {note.evidence_memory_ids}\n"
            f"Access count: {note.access_count}\n"
            f"Last accessed at: {note.last_accessed_at}\n"
            f"Expires at: {note.expires_at}\n"
            f"Importance: {note.importance}\n"
            f"Memory strength: {note.memory_strength}\n"
            f"Retention score: {note.retention_score}\n"
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
    async def _link_related_memories(
        self,
        new_note: MemoryNote,
        existing_notes: List[MemoryNote],
    ) -> None:
        for old_note in existing_notes:
            score = cosine_similarity(new_note.embedding, old_note.embedding)

            if score >= self._link_threshold:
                judgement = self._link_judge.judge(
                    new_content=new_note.content,
                    old_content=old_note.content,
                    similarity_score=score,
                )
                if not judgement.should_link:
                    continue

                reason = (
                    f"Judgement reason: {judgement.reason} "

                    f"{score:.4f} exceeded threshold {self._link_threshold}."
                )

                if old_note.id not in new_note.links:
                    new_note.links.append(old_note.id)

                if new_note.id not in old_note.links:
                    old_note.links.append(new_note.id)

                await self._add_link(
                    source_memory_id=new_note.id,
                    target_memory_id=old_note.id,
                    similarity_score=score,
                    reason=reason,
                )

                await self._add_link(
                    source_memory_id=old_note.id,
                    target_memory_id=new_note.id,
                    similarity_score=score,
                    reason=reason,
                )
                await self._repo.save_note(old_note)
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
    async def _evolve_related_memories(
        self,
        new_note: MemoryNote,
        existing_notes: List[MemoryNote],
    ) -> None:
        for linked_id in new_note.links:
            old_note = self._get_note_by_id(existing_notes, linked_id)

            if old_note is None:
                continue

            old_keywords = list(old_note.keywords)
            old_tags = list(old_note.tags)
            old_context = old_note.context

            try:
                evolved = self._memory_evolver.evolve(
                old_content=old_note.content,
                old_keywords=old_note.keywords,
                old_tags=old_note.tags,
                old_context=old_note.context,
                new_content=new_note.content,
                new_keywords=new_note.keywords,
                new_tags=new_note.tags,
                new_context=new_note.context,
                )

                merged_keywords = evolved.keywords
                merged_tags = evolved.tags
                new_context = evolved.context

            except Exception:
                merged_keywords = sorted(set(old_note.keywords + new_note.keywords))
                merged_tags = sorted(set(old_note.tags + new_note.tags))
                new_context = (
                    "This memory is connected to related memories about "
                    f"keywords={merged_keywords} and tags={merged_tags}."
                )

            # If nothing changed, do not create a revision.
            if (
                old_keywords == merged_keywords
                and old_tags == merged_tags
                and old_context == new_context
            ):
                continue

            revision = MemoryRevision(
                id=str(uuid4()),
                memory_id=old_note.id,
                triggered_by_memory_id=new_note.id,
                old_keywords=old_keywords,
                new_keywords=merged_keywords,
                old_tags=old_tags,
                new_tags=merged_tags,
                old_context=old_context,
                new_context=new_context,
                reason="Memory evolved because a newly linked memory introduced related metadata.",
            )

            if old_note.id not in self._revisions:
                self._revisions[old_note.id] = []

            self._revisions[old_note.id].append(revision)
            await self._repo.save_revision(revision)
            old_note.keywords = merged_keywords
            old_note.tags = merged_tags
            old_note.context = new_context
            await self._repo.save_note(old_note)
    async def get_revision_history(self, memory_id: str) -> list[MemoryRevision]:
        revisions = await self._repo.load_revisions(memory_id)

        if revisions:
            return revisions

        return self._revisions.get(memory_id, [])
    async def _add_link(
        self,
        source_memory_id: str,
        target_memory_id: str,
        similarity_score: float,
        reason: str,
    ) -> None:
        link = MemoryLink(
            id=str(uuid4()),
            source_memory_id=source_memory_id,
            target_memory_id=target_memory_id,
            similarity_score=similarity_score,
            reason=reason,
        )

        if source_memory_id not in self._links:
            self._links[source_memory_id] = []

        already_exists = any(
            existing.target_memory_id == target_memory_id
            for existing in self._links[source_memory_id]
        )

        if not already_exists:
            self._links[source_memory_id].append(link)
            await self._repo.save_link(link)
    async def get_links(self, memory_id: str) -> list[MemoryLink]:
        links = await self._repo.load_links(memory_id)

        if links:
            return links

        return self._links.get(memory_id, [])
    async def initialize(self) -> None:
        await init_db()
    def _calculate_retention_score(
        self,
        note: MemoryNote,
        now: datetime,
    ) -> float:
        reference_time = note.last_accessed_at or note.timestamp

        if reference_time.tzinfo is None:
            reference_time = reference_time.replace(tzinfo=timezone.utc)

        elapsed_seconds = max((now - reference_time).total_seconds(), 0)
        elapsed_days = elapsed_seconds / 86400

        memory_strength = max(note.memory_strength, 1.0)

        return math.exp(-elapsed_days / memory_strength)
    def _calculate_final_score(
        self,
        semantic_score: float,
        note: MemoryNote,
    ) -> float:
        return (
            semantic_score * 0.75
            + note.retention_score * 0.10
            + note.confidence * 0.10
            + note.importance * 0.05
        )
    def _find_most_similar_memory(
        self,
        new_note: MemoryNote,
        existing_notes: List[MemoryNote],
    ) -> tuple[MemoryNote | None, float]:
        best_note = None
        best_score = 0.0

        for old_note in existing_notes:
            if not old_note.embedding or len(old_note.embedding) != len(new_note.embedding):
                continue

            score = cosine_similarity(new_note.embedding, old_note.embedding)

            if score > best_score:
                best_score = score
                best_note = old_note

        return best_note, best_score
    async def search_memory_vector_only(
        self,
        *,
        app_name: str,
        user_id: str,
        query: str,
        top_k: int = 5,
    ) -> SearchMemoryResponse:
        """
        Baseline retrieval method.

        This uses only embedding cosine similarity.
        It does NOT use:
        - graph expansion
        - links
        - memory evolution
        - retention/confidence/importance scoring

        This is useful for comparing plain vector memory vs A-Mem retrieval.
        """
        memories = await self._repo.load_notes(app_name=app_name, user_id=user_id)

        if not memories:
            return SearchMemoryResponse(memories=[])

        query_embedding = self._embedder.embed_text(query)

        scored_notes = []

        for note in memories:
            if not note.embedding or len(note.embedding) != len(query_embedding):
                continue

            score = cosine_similarity(query_embedding, note.embedding)
            scored_notes.append((score, note))

        scored_notes.sort(key=lambda item: item[0], reverse=True)

        top_notes = [
            (score, note)
            for score, note in scored_notes[:top_k]
            if score > 0.2
        ]

        entries = [
            MemoryEntry(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            text=f"Vector Score: {score:.4f}\n{self._format_note_for_agent(note)}"
                        )
                    ],
                ),
                author="vector_only_memory",
            )
            for score, note in top_notes
        ]

        return SearchMemoryResponse(memories=entries)