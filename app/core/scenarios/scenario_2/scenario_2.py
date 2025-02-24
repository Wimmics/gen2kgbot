import asyncio
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from app.core.scenarios.scenario_2.prompt import system_prompt_template
from app.core.utils.graph_nodes import (
    interpret_csv_query_results,
    run_query,
)
from app.core.utils.graph_routers import generate_query_router, run_query_router
from app.core.utils.graph_state import InputState, OverallState
import app.core.utils.config_manager as config
from app.core.utils.logger_manager import setup_logger

logger = setup_logger(__package__, __file__)

SCENARIO = "scenario_2"


def init(state: OverallState) -> OverallState:
    logger.info(f"Running scenario: {SCENARIO}")
    return OverallState({"scenario_id": SCENARIO})


async def generate_query(state: OverallState) -> OverallState:
    logger.info(f"Question: {state["initial_question"]}")

    template = system_prompt_template

    if "kg_full_name" in template.input_variables:
        template = template.partial(kg_full_name=config.get_kg_full_name())

    if "kg_description" in template.input_variables:
        template = template.partial(kg_description=config.get_kg_description())

    if "initial_question" in state.keys():
        template = template.partial(initial_question=state["initial_question"])

    prompt = template.format()
    logger.debug(f"Prompt created:\n{prompt}")

    result = await config.get_llm(state["scenario_id"]).ainvoke(template.format())
    return OverallState({"messages": [HumanMessage(state["initial_question"]), result]})


builder = StateGraph(state_schema=OverallState, input=InputState, output=OverallState)

builder.add_node("init", init)
builder.add_node("generate_query", generate_query)
builder.add_node("run_query", run_query)
builder.add_node("interpret_results", interpret_csv_query_results)

builder.add_edge(START, "init")
builder.add_edge("init", "generate_query")
builder.add_conditional_edges("generate_query", generate_query_router)
builder.add_conditional_edges("run_query", run_query_router)
builder.add_edge("interpret_results", END)

graph = builder.compile()


if __name__ == "__main__":
    asyncio.run(config.main(graph))
