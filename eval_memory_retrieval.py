import asyncio
import os

from google.genai import types

from mentor_agent.amem_memory_service import AMemMemoryService


APP_NAME = "amem_eval"
USER_ID = "eval_user"


class FakeEvent:
    def __init__(self, author: str, text: str):
        self.author = author
        self.id = None
        self.content = types.Content(
            role="user" if author == "user" else "model",
            parts=[types.Part(text=text)],
        )


class FakeSession:
    def __init__(self, app_name: str, user_id: str, events: list[FakeEvent]):
        self.app_name = app_name
        self.user_id = user_id
        self.events = events


def extract_memory_ids(response) -> list[str]:
    memory_ids = []

    for memory in response.memories:
        text = memory.content.parts[0].text

        for line in text.splitlines():
            if line.startswith("Memory ID:"):
                memory_ids.append(line.replace("Memory ID:", "").strip())
                break

    return memory_ids
def response_contains_hint(response, hint: str) -> bool:
    hint_lower = hint.lower()

    for memory in response.memories:
        text = memory.content.parts[0].text.lower()
        if hint_lower in text:
            return True

    return False
def response_contains_terms(response, terms: list[str]) -> bool:
    combined_text = " ".join(
        memory.content.parts[0].text.lower()
        for memory in response.memories
    )

    return all(term.lower() in combined_text for term in terms)
def print_response(title: str, response) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    if not response.memories:
        print("No memories returned.")
        return

    for memory in response.memories:
        print("-" * 80)
        print(memory.content.parts[0].text)


async def main():
    # Optional: remove old eval DB for clean results.
    # We use the same SQLite DB, but separate app/user scope.
    memory_service = AMemMemoryService()
    await memory_service.initialize()

    events = [
        FakeEvent(
            "user",
            "I am building an A-Mem style memory service for a resume project.",
        ),
        FakeEvent(
            "user",
            "The memory service should compare A-Mem with MemoryBank and Vertex AI Memory Bank.",
        ),
        FakeEvent(
            "user",
            "A-Mem should use structured memory notes, semantic links, graph-expanded retrieval, and memory evolution.",
        ),
        FakeEvent(
            "user",
            "MemoryBank is useful for forgetting because it uses an Ebbinghaus-style retention curve.",
        ),
        FakeEvent(
            "user",
            "Vertex AI Memory Bank is a managed production memory service, but it is more black-box than a custom implementation.",
        ),
        FakeEvent(
            "user",
            "The project should be interview-friendly by showing architecture tradeoffs, evaluation, and explainable memory revisions.",
        ),
    ]

    session = FakeSession(
        app_name=APP_NAME,
        user_id=USER_ID,
        events=events,
    )

    await memory_service.add_session_to_memory(session)

    test_queries = [
        {
            "query": "What memory project am I building?",
            "expected_terms": ["A-Mem", "memory service"],
        },
        {
            "query": "What did I want to compare A-Mem with?",
            "expected_terms": ["MemoryBank", "Vertex AI Memory Bank"],
        },
        {
            "query": "Why is MemoryBank relevant to forgetting?",
            "expected_terms": ["Ebbinghaus", "retention curve"],
        },
        {
            "query": "Why is this project good for interviews?",
            "expected_terms": ["architecture tradeoffs", "evaluation", "explainable", "revisions"],
        },
    ]
    vector_hits = 0
    amem_hits = 0
    for item in test_queries:
        query = item["query"]

        vector_response = await memory_service.search_memory_vector_only(
            app_name=APP_NAME,
            user_id=USER_ID,
            query=query,
            top_k=3,
        )

        amem_response = await memory_service.search_memory(
            app_name=APP_NAME,
            user_id=USER_ID,
            query=query,
        )
        vector_hit = response_contains_terms(vector_response, item["expected_terms"])
        amem_hit = response_contains_terms(amem_response, item["expected_terms"])
        if vector_hit:
            vector_hits += 1

        if amem_hit:
            amem_hits += 1
        print("\n\n" + "#" * 100)
        print(f"QUERY: {query}")
        print(f"EXPECTED TERMS: {item['expected_terms']}")
        print(f"\nVECTOR HIT: {vector_hit}")
        print(f"A-MEM HIT: {amem_hit}")
        print("#" * 100)

        print_response("VECTOR-ONLY BASELINE", vector_response)
        print_response("A-MEM GRAPH RETRIEVAL", amem_response)
    total = len(test_queries)

    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Vector-only hits: {vector_hits}/{total}")
    print(f"A-Mem hits: {amem_hits}/{total}")

if __name__ == "__main__":
    asyncio.run(main())