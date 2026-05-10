import asyncio

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from mentor_agent.agent import root_agent
from mentor_agent.amem_memory_service import AMemMemoryService


APP_NAME = "amem_demo"
USER_ID = "pragat"
SESSION_ID = "interactive_memory_demo"


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
        # Print tool calls so you can confirm load_memory is being used.
        if event.content and event.content.parts:
            for part in event.content.parts:
                if getattr(part, "function_call", None):
                    print(f"\n[Tool call] {part.function_call.name}: {part.function_call.args}")

        if event.is_final_response():
            if event.content and event.content.parts:
                final_response = event.content.parts[0].text

    print(f"\nUser: {text}")
    print(f"Agent: {final_response}")


async def main():
    session_service = InMemorySessionService()

    memory_service = AMemMemoryService()
    await memory_service.initialize()

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
        "What do you remember about my A-Mem project and what I wanted to compare it with?",
    )


if __name__ == "__main__":
    asyncio.run(main())