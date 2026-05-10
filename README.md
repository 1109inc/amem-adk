# AMem-ADK

A research project mentor agent with a custom long-term memory backend inspired by A-Mem.

## Goal

Build an ADK agent that can remember, connect, and evolve project knowledge across sessions.

## Current Features

- Google ADK agent integration
- Custom `AMemMemoryService` using ADK `BaseMemoryService`
- Structured `MemoryNote` objects inspired by A-Mem
- Gemini embeddings for semantic retrieval
- Cosine-similarity memory search
- Similarity-based memory linking
- Graph-expanded retrieval
- Rule-based memory evolution
- Revision history for evolved memories
- Detailed memory link records with similarity scores
- SQLite persistence with async SQLAlchemy
- Automatic database initialization
- Separate persisted-memory search demo
- Provenance and lifecycle fields for safer memory design

## Next Milestones

- Replace rule-based metadata extraction with LLM-based note construction
- Add LLM-based link reasoning
- Add richer memory evolution logic
- Add confidence scoring and source-aware ranking
- Add evaluation against simple vector memory and MemoryBank-style baselines