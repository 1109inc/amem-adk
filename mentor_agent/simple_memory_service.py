from __future__ import annotations

from typing import Dict, List

from google.adk.memory.base_memory_service import (
    BaseMemoryService,
    SearchMemoryResponse,
    MemoryEntry,
)
from google.adk.sessions import Session
from google.genai import types


class SimpleMemoryService(BaseMemoryService):
    """
    A tiny memory service for learning.

    It stores plain text memories in a Python dict:
      (app_name, user_id) -> list[str]

    Later, we will replace this with A-Mem notes.
    """

    def __init__(self):
        self._memories: Dict[tuple[str, str], List[str]] = {}

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

            if text_parts:
                memory_text = f"{event.author}: {' '.join(text_parts)}"
                self._memories[key].append(memory_text)

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
            memory
            for memory in memories
            if query_lower in memory.lower()
        ]

        entries = [
            MemoryEntry(
                content=types.Content(
                    role="model",
                    parts=[types.Part(text=memory)],
                ),
                author="simple_memory",
            )
            for memory in matches[:5]
        ]

        return SearchMemoryResponse(memories=entries)