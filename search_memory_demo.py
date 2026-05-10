import asyncio

from mentor_agent.amem_memory_service import AMemMemoryService


APP_NAME = "amem_demo"
USER_ID = "pragat"


async def main():
    memory_service = AMemMemoryService()
    await memory_service.initialize()
    query = "compare long term memory systems"

    result = await memory_service.search_memory(
        app_name=APP_NAME,
        user_id=USER_ID,
        query=query,
    )

    print(f"Search query: {query}")
    print("\nPersisted memory search results:")

    for memory in result.memories:
        print("-" * 80)
        print(memory.content.parts[0].text)


if __name__ == "__main__":
    asyncio.run(main())