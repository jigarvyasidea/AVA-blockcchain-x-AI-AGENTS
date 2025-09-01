from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Load Groq LLM
llm = ChatGroq(model="deepseek-r1-distill-llama-70b")

# Prompt template
table_prompt_template = PromptTemplate.from_template(
    """
You are a helpful technical assistant.

Generate a clear and well-formatted **Markdown comparison table** for the following items or technologies:

**Topic**: {topic}

✅ Instructions:
- Compare at least 4–6 key features (e.g., performance, use case, popularity, ease of use, etc.)
- Use a proper markdown table format with `|` and `---` to separate columns and rows
- Include a short 1-line summary or insight **after** the table
- Do not include unnecessary explanation or formatting outside the table

🎯 Make sure the table is easy to read and visually balanced.

Start directly with the markdown table.
"""
)

# ✅ Robust LangGraph-compatible agent function
async def table_agent(state: dict) -> dict:
    user_input = state.get("user_input")
    if not user_input:
        raise ValueError("Missing 'user_input' in state")

    formatted_prompt = table_prompt_template.format(topic=user_input)
    response = await llm.ainvoke(formatted_prompt)

    # ✅ Ensure 'results' key exists
    state["results"] = state.get("results", {})
    state["results"]["table"] = response.content

    return state
