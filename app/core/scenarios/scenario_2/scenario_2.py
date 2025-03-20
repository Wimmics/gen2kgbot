import asyncio
from typing import Literal
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from app.core.scenarios.scenario_2.prompt import system_prompt_template
from app.core.utils.graph_nodes import (
    interpret_results,
    run_query,
    validate_question,
)
from app.core.utils.graph_routers import generate_query_router, run_query_router
from app.core.utils.graph_state import InputState, OverallState
import app.core.utils.config_manager as config
from app.core.utils.logger_manager import setup_logger

logger = setup_logger(__package__, __file__)

SCENARIO = "scenario_2"


def validate_question_router(
    state: OverallState,
) -> Literal["generate_query", "__end__"]:
    """
    Check the question validation results and decide whether to continue the process or stop.

    Args:
        state (OverallState): current state of the conversation

    Returns:
        Literal["generate_query", END]: next step in the conversation
    """
    results = state["question_validation_results"]
    if results.find("true") != -1 and results.find("false") == -1:
        logger.info("Question validation passed.")
        return "generate_query"
    else:
        logger.warning("Question validation failed.")
        return END


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

    result = await config.get_seq2seq_model(
        scenario_id=state["scenario_id"], node_name="generate_query"
    ).ainvoke(template.format())
    return OverallState({"messages": [HumanMessage(state["initial_question"]), result]})


builder = StateGraph(state_schema=OverallState, input=InputState, output=OverallState)

builder.add_node("init", init)
builder.add_node("validate_question", validate_question)
builder.add_node("generate_query", generate_query)
builder.add_node("run_query", run_query)
builder.add_node("interpret_results", interpret_results)

builder.add_edge(START, "init")
builder.add_edge("init", "validate_question")
builder.add_conditional_edges("validate_question", validate_question_router)
builder.add_conditional_edges("generate_query", generate_query_router)
builder.add_conditional_edges("run_query", run_query_router)
builder.add_edge("interpret_results", END)

graph = builder.compile()


if __name__ == "__main__":
    asyncio.run(config.main(graph))
