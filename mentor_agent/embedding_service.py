from __future__ import annotations

import os

from google import genai
from dotenv import load_dotenv
load_dotenv()

class EmbeddingService:
    def __init__(self):
        self._client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self._model = "gemini-embedding-001"

    def embed_text(self, text: str) -> list[float]:
        response = self._client.models.embed_content(
            model=self._model,
            contents=text,
        )

        return response.embeddings[0].values