from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
llm = ChatGroq(model="deepseek-r1-distill-llama-70b")

async def synthesizer(state: dict) -> dict:
    # ✅ Ensure 'results' key exists
    results = state.get("results", {})

    narrative = results.get("narrative", "")
    table = results.get("table", "")
    diagram = results.get("diagram", "")

    # 📝 Combined Markdown prompt
    prompt = f"""
Format this as a clean and presentable **Markdown** document.

---

### 🧠 Narrative
{narrative}

---

### 📊 Comparison Table
{table}

---

### 📈 Diagram
{diagram}

---

Wrap the content cleanly. Make it easy to read and beginner-friendly.
"""

    # 🌐 Call LLM
    response = await llm.ainvoke(prompt)

    # ✅ Safely store final result
    state["results"]["final_doc"] = response.content
    return state
