import asyncio
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from mentor_agent.amem_memory_service import AMemMemoryService
from mentor_agent.memory_note import MemoryNote


APP_NAME = "retention_test"
USER_ID = "test_user"


async def main():
    memory_service = AMemMemoryService()
    await memory_service.initialize()

    old_time = datetime.now(timezone.utc) - timedelta(days=10)

    old_note = MemoryNote(
        id=str(uuid4()),
        app_name=APP_NAME,
        user_id=USER_ID,
        author="user",
        content="I am testing whether old memories decay over time.",
        timestamp=old_time,
        keywords=["retention", "decay"],
        tags=["test"],
        context="Testing Ebbinghaus-style memory retention decay.",
        embedding=memory_service._embedder.embed_text(
            "I am testing whether old memories decay over time."
        ),
        source_type="user_message",
        confidence=1.0,
        memory_strength=1.0,
        retention_score=1.0,
    )
    now = datetime.now(timezone.utc)
    pre_access_retention = memory_service._calculate_retention_score(old_note, now)

    print("Expected pre-access retention:")
    print(f"{pre_access_retention:.6f}")
    await memory_service._repo.save_note(old_note)

    print("Before search:")
    print(f"Retention score: {old_note.retention_score}")
    print(f"Memory strength: {old_note.memory_strength}")
    print(f"Timestamp: {old_note.timestamp}")

    result = await memory_service.search_memory(
        app_name=APP_NAME,
        user_id=USER_ID,
        query="old memory decay test",
    )

    print("\nSearch results:")
    for memory in result.memories:
        print("-" * 80)
        print(memory.content.parts[0].text)


if __name__ == "__main__":
    asyncio.run(main())