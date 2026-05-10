import asyncio
from types import SimpleNamespace

from google.genai import types
from mentor_agent.amem_memory_service import AMemMemoryService


async def main():
    memory_service = AMemMemoryService()

    fake_session = SimpleNamespace(
        app_name="amem_demo",
        user_id="pragat",
        events=[
            SimpleNamespace(
                author="user",
                content=types.Content(
                    role="user",
                    parts=[types.Part(text="I am building an A-Mem memory project.")]
                ),
            ),
            SimpleNamespace(
                author="mentor_agent",
                content=types.Content(
                    role="model",
                    parts=[types.Part(text="Great. Start with a simple memory service.")]
                ),
            ),
            SimpleNamespace(
                author="user",
                content=types.Content(
                    role="user",
                    parts=[types.Part(text="I want to compare A-Mem with MemoryBank.")]
                ),
            ),
        ],
    )

    await memory_service.add_session_to_memory(fake_session)

    result = await memory_service.search_memory(
        app_name="amem_demo",
        user_id="pragat",
        query="compare long term memory approaches",
    )

    print("Search results:")
    for memory in result.memories:
        text = memory.content.parts[0].text
        print("-", text)
    print("\nRevision histories:")

    for memory in result.memories:
        text = memory.content.parts[0].text
        lines = text.splitlines()

        memory_id_line = next(
            line for line in lines if line.startswith("Memory ID:")
        )
        memory_id = memory_id_line.replace("Memory ID:", "").strip()

        revisions = memory_service.get_revision_history(memory_id)

        if not revisions:
            continue

        print(f"\nMemory {memory_id} revisions:")

        for revision in revisions:
            print(f"- Revision ID: {revision.id}")
            print(f"  Triggered by: {revision.triggered_by_memory_id}")
            print(f"  Old keywords: {revision.old_keywords}")
            print(f"  New keywords: {revision.new_keywords}")
            print(f"  Old tags: {revision.old_tags}")
            print(f"  New tags: {revision.new_tags}")
            print(f"  Old context: {revision.old_context}")
            print(f"  New context: {revision.new_context}")
            print(f"  Reason: {revision.reason}")

        print("\nDetailed links:")

        for memory in result.memories:
            text = memory.content.parts[0].text
            lines = text.splitlines()

            memory_id_line = next(
                line for line in lines if line.startswith("Memory ID:")
            )
            memory_id = memory_id_line.replace("Memory ID:", "").strip()

            links = memory_service.get_links(memory_id)

            if not links:
                continue

            print(f"\nMemory {memory_id} links:")

            for link in links:
                print(f"- Target: {link.target_memory_id}")
                print(f"  Similarity: {link.similarity_score:.4f}")
                print(f"  Reason: {link.reason}")


if __name__ == "__main__":
    asyncio.run(main())