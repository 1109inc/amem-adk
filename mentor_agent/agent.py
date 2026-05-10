from google.adk.agents.llm_agent import Agent
from google.adk.tools.load_memory_tool import LoadMemoryTool
from google.adk.models.lite_llm import LiteLlm


model=LiteLlm(model="openai/gpt-4o-mini")

root_agent = Agent(
    model=model,
    name='mentor_agent',
    description="A research project mentor that helps with architecture, tradeoffs, and implementation planning.",
    instruction="""
    You are a research project mentor.

    Help the user think deeply about long-term technical projects.
    Focus on architecture, tradeoffs, implementation plans, experiments,
    and explaining ideas clearly.

    For now, you do not have long-term memory.
    Later, you will be connected to a custom A-Mem memory service.
    """,
    tools=[
        LoadMemoryTool(),
    ],

)
