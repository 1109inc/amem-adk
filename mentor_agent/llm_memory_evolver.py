from __future__ import annotations

import json
import os

from google import genai
from pydantic import BaseModel, Field


class EvolvedMemoryMetadata(BaseModel):
    keywords: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    context: str = ""


class LLMMemoryEvolver:
    """
    LLM-based memory evolution.

    It only evolves metadata:
    - keywords
    - tags
    - context

    It must never rewrite MemoryNote.content.
    """

    def __init__(self):
        self._client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self._model = "gemini-flash-lite-latest"

    def evolve(
        self,
        old_content: str,
        old_keywords: list[str],
        old_tags: list[str],
        old_context: str,
        new_content: str,
        new_keywords: list[str],
        new_tags: list[str],
        new_context: str,
    ) -> EvolvedMemoryMetadata:
        prompt = f"""
You are evolving metadata for an existing agent memory note.

The original memory content is source truth and must NOT be rewritten.
Only update metadata: keywords, tags, and context.

Rules:
- Do not invent facts.
- Preserve important existing keywords and tags.
- Add new keywords/tags only if supported by the new linked memory.
- Maximum 8 keywords.
- Maximum 5 tags.
- Tags must be snake_case.
- Context must be one concise sentence.
- Return ONLY valid JSON.

Existing memory content:
{old_content}

Existing metadata:
keywords={old_keywords}
tags={old_tags}
context={old_context}

New linked memory content:
{new_content}

New linked memory metadata:
keywords={new_keywords}
tags={new_tags}
context={new_context}

JSON format:
{{
  "keywords": ["..."],
  "tags": ["..."],
  "context": "..."
}}
"""

        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
        )

        raw_text = response.text.strip()

        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError:
            cleaned = (
                raw_text
                .replace("```json", "")
                .replace("```", "")
                .strip()
            )
            data = json.loads(cleaned)

        return EvolvedMemoryMetadata(
            keywords=data.get("keywords", [])[:8],
            tags=data.get("tags", [])[:5],
            context=data.get("context", ""),
        )