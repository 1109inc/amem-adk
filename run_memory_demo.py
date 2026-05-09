import asyncio

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from mentor_agent.agent import root_agent
from mentor_agent.amem_memory_service import AMemMemoryService
from dotenv import load_dotenv
load_dotenv()

APP_NAME = "amem_demo"
USER_ID = "pragat"
SESSION_ID = "demo_session"


async def send_message(runner: Runner, text: str):
    content = types.Content(
        role="user",
        parts=[types.Part(text=text)],
    )

    final_response = ""

    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=content,
    ):
        if event.is_final_response():
            if event.content and event.content.parts:
                final_response = event.content.parts[0].text

    print(f"\nUser: {text}")
    print(f"Agent: {final_response}")


async def main():
    session_service = InMemorySessionService()
    memory_service = AMemMemoryService()

    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
    )

    runner = Runner(
        app_name=APP_NAME,
        agent=root_agent,
        session_service=session_service,
        memory_service=memory_service,
    )

    await send_message(
        runner,
        "I am building an A-Mem style memory service for a resume project.",
    )

    await send_message(
        runner,
        "I want to compare it with MemoryBank and Vertex AI Memory Bank.",
    )

    session = await session_service.get_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
    )

    await memory_service.add_session_to_memory(session)

    result = await memory_service.search_memory(
        app_name=APP_NAME,
        user_id=USER_ID,
        query="A-Mem",
    )

    print("\nMemory search results:")
    for memory in result.memories:
        print("-", memory.content.parts[0].text)


if __name__ == "__main__":
    asyncio.run(main())