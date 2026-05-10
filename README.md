# AMem-ADK

A research project mentor agent with a custom long-term memory backend inspired by A-Mem.

## Goal

Build an ADK agent that can remember, connect, retrieve, and evolve project knowledge across sessions.

## Current Features

* Google ADK agent integration
* Custom `AMemMemoryService` using ADK `BaseMemoryService`
* Structured `MemoryNote` objects inspired by A-Mem
* LLM-based note construction for keywords, tags, and context
* Gemini embeddings for semantic retrieval
* Cosine-similarity memory search
* LLM-judged memory link generation
* Detailed `MemoryLink` records with similarity scores and reasons
* Graph-expanded retrieval over linked memories
* LLM-based memory evolution
* `MemoryRevision` history for evolved memories
* SQLite persistence with async SQLAlchemy
* Automatic database initialization
* Separate persisted-memory search demo
* Provenance and lifecycle fields for safer memory design
* Source-aware confidence scoring for user and agent messages
* Access tracking with `access_count` and `last_accessed_at`
* Memory reinforcement with `memory_strength`
* Retention-aware ranking using semantic similarity, retention, confidence, and importance
* Duplicate memory reinforcement instead of repeatedly storing near-identical memories
* ADK `load_memory` tool integration
* Agent can retrieve long-term memory during conversation through the custom `AMemMemoryService`

## Memory Pipeline

1. ADK session events are passed into `AMemMemoryService`.
2. Each event becomes a structured `MemoryNote`.
3. An LLM extracts note metadata: keywords, tags, and context.
4. Gemini embeddings are generated for semantic retrieval.
5. New memories are compared against existing memories.
6. Near-duplicate memories reinforce existing notes instead of creating duplicate records.
7. Candidate links are judged by an LLM.
8. Approved links are stored as `MemoryLink` records with similarity scores and reasons.
9. Linked memories are evolved by an LLM without rewriting original source content.
10. Every evolution creates a `MemoryRevision` record.
11. Search uses semantic similarity, retention, confidence, importance, and graph-expanded retrieval.
12. Notes, links, and revisions are persisted in SQLite.
13. The ADK agent can call `load_memory`, which routes memory search through the custom `AMemMemoryService`.

## Current Safety Design

* Original memory `content` is treated as source truth.
* Memory evolution updates metadata only, not original content.
* Revisions track before/after metadata changes.
* Link records include similarity scores and reasons.
* User messages and agent messages have different `source_type` and confidence values.
* Memory notes include provenance and lifecycle fields such as `source_type`, `source_id`, `confidence`, `is_derived`, `evidence_memory_ids`, `access_count`, `last_accessed_at`, `expires_at`, `importance`, `memory_strength`, and `retention_score`.
* Repeated or near-duplicate memories reinforce existing memories instead of creating duplicate memory records.

## Evaluation

The project includes a local retrieval evaluation script:

```bash
python eval_memory_retrieval.py
```

The evaluation compares:

* Vector-only retrieval baseline
* A-Mem graph-expanded retrieval

### Initial Results

| Evaluation Type     | Vector-only | A-Mem |
| ------------------- | ----------: | ----: |
| Direct retrieval    |         4/4 |   4/4 |
| Multi-hop retrieval |         2/3 |   3/3 |
| Overall retrieval   |         6/7 |   7/7 |

Direct retrieval queries are easy for both systems. The multi-hop queries show the benefit of A-Mem graph expansion: linked memories provide supporting context that plain vector retrieval can miss.

This is an initial synthetic evaluation, not a full benchmark.

### Why A-Mem Beats Vector-Only on Multi-Hop Queries

Vector-only retrieval works well when the answer is contained in one directly similar memory.

A-Mem helps when the answer is distributed across connected memories.

Example query:

```text
Why is Vertex relevant to my interview-ready A-Mem project?
```

Vector-only retrieved general project and comparison memories, but missed the specific Vertex detail in its top results.

A-Mem retrieved the direct project memories and expanded through memory links to also retrieve:

```text
Vertex AI Memory Bank is a managed production memory service, but it is more black-box than a custom implementation.
```

This shows the benefit of graph-expanded retrieval: the system can recover supporting context that is related to the main memory, even when it is not one of the top vector-only matches.

## Next Milestones

* Add richer source-aware confidence scoring
* Distinguish user messages, agent messages, summaries, and inferred memories more deeply
* Add stronger recency and access-count based ranking experiments
* Add MemoryBank-style forgetting / decay policies
* Add hard expiration filtering using `expires_at`
* Add evaluation against MemoryBank-style baselines
* Add Vertex AI Memory Bank comparison
* Add a dashboard to inspect notes, links, revisions, retrieval scores, and evaluation results
