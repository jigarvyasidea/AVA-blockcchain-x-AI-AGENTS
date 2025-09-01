import asyncio
from graph_builder import build_graph

async def main():
    graph = build_graph()
    query = "Compare GPT-4 and Claude with a diagram"
    result = await graph.ainvoke({"user_input": query})
    print(result["results"]["final_doc"])

asyncio.run(main())
