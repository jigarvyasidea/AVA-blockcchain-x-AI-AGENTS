from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="deepseek-r1-distill-llama-70b")

prompt_template = PromptTemplate.from_template(
    """
You are an expert educator and researcher.

Explain the following topic in a clear, well-structured, and informative way for someone with no prior knowledge.

Topic: {topic}

Your explanation must include:
1. A brief introduction.
2. Key concepts and their definitions.
3. Real-life examples or applications.
4. Why this topic is important or useful.
5. A short summary at the end.

Keep the tone friendly, use simple language, and avoid jargon unless explained.

Start your explanation below:
"""
)

# ✅ Safe & clean function
async def narrative_agent(state: dict) -> dict:
    user_input = state.get("user_input")
    if not user_input:
        raise ValueError("Missing 'user_input' in state")

    formatted_prompt = prompt_template.format(topic=user_input)
    response = await llm.ainvoke(formatted_prompt)

    # ✅ Safely update or create the 'results' key
    state["results"] = state.get("results", {})
    state["results"]["narrative"] = response.content

    return state
