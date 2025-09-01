from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
llm = ChatGroq(model="deepseek-r1-distill-llama-70b")

# Mermaid prompt template
diagram_prompt_template = PromptTemplate.from_template(
    """
You are an expert in creating technical visualizations.

Generate a valid and properly formatted **Mermaid.js** diagram to explain the following topic:

**Topic**: {topic}

✅ Requirements:
- Choose the best Mermaid diagram type (flowchart, sequenceDiagram, classDiagram, etc.) based on the topic
- Show meaningful relationships, flow, or hierarchy
- Avoid too many nodes (4–8 is good)
- Output only **valid Mermaid code** between triple backticks (```mermaid)
- Do **not** include explanation outside the diagram
- Prefer left-to-right or top-down layout (LR or TD)

🎯 Goal: To help a beginner understand this concept visually.

Start directly with the Mermaid code block.
"""
)

# ✅ Robust & LangGraph-compatible agent function
async def diagram_agent(state: dict) -> dict:
    user_input = state.get("user_input")
    if not user_input:
        raise ValueError("Missing 'user_input' in state")

    formatted_prompt = diagram_prompt_template.format(topic=user_input)
    response = await llm.ainvoke(formatted_prompt)

    # ✅ Ensure the 'results' key exists
    state["results"] = state.get("results", {})
    state["results"]["diagram"] = response.content

    return state
