import asyncio
from uuid import uuid4

from mentor_agent.database import init_db
from mentor_agent.memory_note import MemoryNote
from mentor_agent.memory_repository import MemoryRepository


async def main():
    await init_db()

    repo = MemoryRepository()

    note = MemoryNote(
        id=str(uuid4()),
        app_name="amem_demo",
        user_id="pragat",
        author="user",
        content="I am testing SQLite persistence for A-Mem.",
        keywords=["A-Mem", "SQLite"],
        tags=["persistence"],
        context="Testing whether memory notes can be saved and loaded.",
        embedding=[0.1, 0.2, 0.3],
    )

    await repo.save_note(note)

    notes = await repo.load_notes(
        app_name="amem_demo",
        user_id="pragat",
    )

    print(f"Loaded {len(notes)} notes")

    for loaded_note in notes:
        print("- ID:", loaded_note.id)
        print("  Content:", loaded_note.content)
        print("  Keywords:", loaded_note.keywords)
        print("  Tags:", loaded_note.tags)


if __name__ == "__main__":
    asyncio.run(main())