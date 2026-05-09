import asyncio
from types import SimpleNamespace

from google.genai import types
from mentor_agent.simple_memory_service import SimpleMemoryService


async def main():
    memory_service = SimpleMemoryService()

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
        query="career_project",
    )

    print("Search results:")
    for memory in result.memories:
        text = memory.content.parts[0].text
        print("-", text)


if __name__ == "__main__":
    asyncio.run(main())