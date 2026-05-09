from __future__ import annotations


class SimpleNoteExtractor:
    """
    Temporary rule-based extractor.

    Later this will become an LLM-based A-Mem note constructor.
    """

    PROJECT_KEYWORDS = {
        "a-mem": "A-Mem",
        "amem": "A-Mem",
        "memorybank": "MemoryBank",
        "vertex": "Vertex AI",
        "adk": "ADK",
        "basememoryservice": "BaseMemoryService",
        "rag": "RAG",
        "embedding": "embeddings",
        "vector": "vector search",
    }

    def extract_keywords(self, text: str) -> list[str]:
        text_lower = text.lower()
        keywords = []

        for trigger, keyword in self.PROJECT_KEYWORDS.items():
            if trigger in text_lower:
                keywords.append(keyword)

        return sorted(set(keywords))

    def extract_tags(self, text: str) -> list[str]:
        text_lower = text.lower()
        tags = []

        if any(word in text_lower for word in ["compare", "versus", "vs", "better"]):
            tags.append("comparison")

        if any(word in text_lower for word in ["build", "implement", "create", "make"]):
            tags.append("implementation")

        if any(word in text_lower for word in ["resume", "interview", "project"]):
            tags.append("career_project")

        if any(word in text_lower for word in ["rag", "retrieval", "search"]):
            tags.append("retrieval")

        return sorted(set(tags))

    def create_context(self, text: str) -> str:
        keywords = self.extract_keywords(text)
        tags = self.extract_tags(text)

        if not keywords and not tags:
            return "General project discussion."

        return (
            "This memory relates to "
            f"keywords={keywords} and tags={tags}."
        )