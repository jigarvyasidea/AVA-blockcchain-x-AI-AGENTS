from langgraph.graph import StateGraph, END
from orchestrator import orchestrator
from agents.narrative import narrative_agent
from agents.table import table_agent
from agents.diagram import diagram_agent
from agents.synthesizer import synthesizer

class DocState(dict): pass

def build_graph():
    graph = StateGraph(DocState)

    graph.add_node("orchestrator", orchestrator)
    graph.add_node("narrative_agent", narrative_agent)
    graph.add_node("table_agent", table_agent)
    graph.add_node("diagram_agent", diagram_agent)
    graph.add_node("synthesizer", synthesizer)

    graph.set_entry_point("orchestrator")

    graph.add_conditional_edges(
        "orchestrator",
        lambda state: "narrative" if "narrative" in state["tasks"] else (
            "table" if "table" in state["tasks"] else (
                "diagram" if "diagram" in state["tasks"] else END
            )
        ),
        {
            "narrative": "narrative_agent",
            "table": "table_agent",
            "diagram": "diagram_agent",
        }
    )

    for node in ["narrative_agent", "table_agent", "diagram_agent"]:
        graph.add_edge(node, "synthesizer")

    graph.add_edge("synthesizer", END)

    # ✅ This line must be inside the `build_graph()` function
    return graph.compile()
