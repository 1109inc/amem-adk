# AMem-ADK

A research project mentor agent with a custom long-term memory backend inspired by A-Mem.

## Goal

Build an ADK agent that can remember, connect, retrieve, and evolve project knowledge across sessions.

## Current Features

- Google ADK agent integration
- Custom `AMemMemoryService` using ADK `BaseMemoryService`
- Structured `MemoryNote` objects inspired by A-Mem
- LLM-based note construction for keywords, tags, and context
- Gemini embeddings for semantic retrieval
- Cosine-similarity memory search
- LLM-judged memory link generation
- Detailed `MemoryLink` records with similarity scores and reasons
- Graph-expanded retrieval over linked memories
- LLM-based memory evolution
- `MemoryRevision` history for evolved memories
- SQLite persistence with async SQLAlchemy
- Automatic database initialization
- Separate persisted-memory search demo
- Provenance and lifecycle fields for safer memory design
- ADK `load_memory` tool integration
- Agent can retrieve long-term memory during conversation through the custom `AMemMemoryService`
## Memory Pipeline

1. ADK session events are passed into `AMemMemoryService`.
2. Each event becomes a structured `MemoryNote`.
3. An LLM extracts note metadata: keywords, tags, and context.
4. Gemini embeddings are generated for semantic retrieval.
5. New memories are compared against existing memories.
6. Candidate links are judged by an LLM.
7. Approved links are stored as `MemoryLink` records with similarity scores and reasons.
8. Linked memories are evolved by an LLM without rewriting original source content.
9. Every evolution creates a `MemoryRevision` record.
10. Search uses semantic similarity plus graph-expanded retrieval.
11. Notes, links, and revisions are persisted in SQLite.
12. The ADK agent can call `load_memory`, which routes memory search through the custom `AMemMemoryService`.

## Current Safety Design

- Original memory `content` is treated as source truth.
- Memory evolution updates metadata only, not original content.
- Revisions track before/after metadata changes.
- Link records include similarity scores and reasons.
- Memory notes include provenance and lifecycle fields such as `source_type`, `source_id`, `confidence`, `is_derived`, `evidence_memory_ids`, `access_count`, and `expires_at`.

## Next Milestones

- Add source-aware confidence scoring
- Distinguish user messages, agent messages, summaries, and inferred memories
- Add recency and access-count based ranking
- Add MemoryBank-style forgetting / decay
- Add evaluation against simple vector memory and MemoryBank-style baselines
- Add a dashboard to inspect notes, links, revisions, and retrieval results

## Evaluation

The project includes a local retrieval evaluation script:

```bash
python eval_memory_retrieval.py
```
### Current evaluation compares:

Vector-only retrieval baseline
A-Mem graph-expanded retrieval

Initial results:

- Evaluation Type	Vector-only	A-Mem
- Multi-hop retrieval	2/3	3/3
- Overall retrieval	6/7	7/7

The direct retrieval queries are easy for both systems, but the multi-hop queries show the benefit of A-Mem graph expansion: linked memories provide supporting context that plain vector retrieval can miss.



