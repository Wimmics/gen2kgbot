import asyncio
from langgraph.graph import StateGraph, START, END
from app.core.scenarios.scenario_3.prompt import system_prompt_template
from app.core.utils.graph_nodes import (
    interpret_csv_query_results,
    select_similar_classes,
    create_query_generation_prompt,
    generate_query,
    run_query,
)
from app.core.utils.graph_routers import generate_query_router, run_query_router
from app.core.utils.graph_state import InputState, OverallState
import app.core.utils.config_manager as config
from app.core.utils.logger_manager import setup_logger

logger = setup_logger(__package__, __file__)

SCENARIO = "scenario_3"
config.set_scenario(SCENARIO)


def create_prompt(state: OverallState) -> OverallState:
    return create_query_generation_prompt(system_prompt_template, state)


builder = StateGraph(state_schema=OverallState, input=InputState, output=OverallState)

builder.add_node("select_similar_classes", select_similar_classes)
builder.add_node("create_prompt", create_prompt)
builder.add_node("generate_query", generate_query)
builder.add_node("run_query", run_query)
builder.add_node("interpret_results", interpret_csv_query_results)

builder.add_edge(START, "select_similar_classes")
builder.add_edge("select_similar_classes", "create_prompt")
builder.add_edge("create_prompt", "generate_query")
builder.add_conditional_edges("generate_query", generate_query_router)
builder.add_conditional_edges("run_query", run_query_router)
builder.add_edge("interpret_results", END)

graph = builder.compile()


def run_scenario(question: str):
    return graph.ainvoke(input=InputState({"initial_question": question}))


if __name__ == "__main__":
    asyncio.run(config.main(graph))
