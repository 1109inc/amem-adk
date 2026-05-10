from __future__ import annotations

import json
import os

from google import genai
from pydantic import BaseModel, Field


class ExtractedNoteMetadata(BaseModel):
    keywords: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    context: str = ""


class LLMNoteExtractor:
    """
    LLM-based metadata extractor for A-Mem-style note construction.

    Important:
    - It does NOT rewrite the original memory content.
    - It only generates metadata around the source content.
    """

    def __init__(self):
        self._client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self._model = "gemini-flash-lite-latest"

    def extract(self, text: str) -> ExtractedNoteMetadata:
        prompt = f"""
You are constructing metadata for an agent memory note.

Given the source memory text, extract:
1. keywords: important entities, concepts, technologies, papers, tools
2. tags: broad categories like implementation, comparison, project_goal, bug, decision, retrieval, storage, evaluation
3. context: one concise sentence explaining what this memory is about

Rules:
- Do not invent facts.
- Do not rewrite the source memory as truth beyond what it says.
- Keep keywords short.
- Keep tags snake_case.
- Return ONLY valid JSON.

Source memory text:
{text}

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
            # Simple fallback if model wraps JSON in markdown fences.
            cleaned = (
                raw_text
                .replace("```json", "")
                .replace("```", "")
                .strip()
            )
            data = json.loads(cleaned)

        return ExtractedNoteMetadata(
            keywords=data.get("keywords", []),
            tags=data.get("tags", []),
            context=data.get("context", ""),
        )