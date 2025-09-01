from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
llm = ChatGroq(model="deepseek-r1-distill-llama-70b")

# ✅ Return dict — because we need to pass state downstream
async def orchestrator_agent(state: dict) -> dict:
    user_input = state.get("user_input", "")

    routing_prompt = f"""
You're an AI agent that decides which analysis tools to run based on a user's request.

Based on the following input, choose one or more of: "narrative", "table", or "diagram".

Respond ONLY with a comma-separated list of tools to run.

User input:
{user_input}

Examples:
- "Compare Python and Java" → "narrative,table"
- "Draw architecture of a chatbot" → "diagram"
- "Explain AI in education" → "narrative,diagram"
- "Summarize the differences between GPT-4 and Claude" → "narrative,table,diagram"
"""

    response = await llm.ainvoke(routing_prompt)

    tools = response.content.strip().lower().replace(" ", "").split(",")

    # ✅ Save tool choices for conditional edge
    state["selected_tools"] = tools

    return state  