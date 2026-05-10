from __future__ import annotations

import json
import os

from google import genai
from pydantic import BaseModel


class LinkJudgement(BaseModel):
    should_link: bool
    reason: str


class LLMLinkJudge:
    def __init__(self):
        self._client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self._model = "gemini-flash-lite-latest"

    def judge(
        self,
        new_content: str,
        old_content: str,
        similarity_score: float,
    ) -> LinkJudgement:
        prompt = f"""
You are deciding whether two agent memory notes should be linked.

A link means the two memories are meaningfully related and retrieving one could help understand the other.

Rules:
- Link if they discuss the same project, decision, comparison, implementation detail, bug, or research topic.
- Do not link only because they share generic words.
- Do not invent hidden meaning.
- Return ONLY valid JSON.
- Reason must be one short sentence.

New memory:
{new_content}

Existing memory:
{old_content}

Embedding similarity:
{similarity_score:.4f}

JSON format:
{{
  "should_link": true,
  "reason": "..."
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

        return LinkJudgement(
            should_link=bool(data.get("should_link", False)),
            reason=data.get("reason", "LLM judged memories as related."),
        )