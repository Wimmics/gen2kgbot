import asyncio
from langgraph.graph import StateGraph, START, END
from app.scenarios.scenario_4.prompt import system_prompt_template
from app.utils.graph_nodes import (
    preprocess_question,
    select_similar_classes,
    get_class_context_from_cache,
    get_class_context_from_kg,
    create_query_generation_prompt,
    generate_query,
    run_query,
    interpret_results,
    validate_question,
)
from app.utils.graph_routers import (
    get_class_context_router,
    generate_query_router,
    run_query_router,
    validate_question_router,
)
from app.utils.graph_state import InputState, OverallState
import app.utils.config_manager as config
from app.utils.logger_manager import setup_logger

logger = setup_logger(__package__, __file__)

SCENARIO = "scenario_4"


def init(state: OverallState) -> OverallState:
    logger.info(f"Running scenario: {SCENARIO}")
    return OverallState({"scenario_id": SCENARIO})


def create_prompt(state: OverallState) -> OverallState:
    return create_query_generation_prompt(system_prompt_template, state)


builder = StateGraph(state_schema=OverallState, input=InputState, output=OverallState)

builder.add_node("init", init)
builder.add_node("validate_question", validate_question)
builder.add_node("preprocess_question", preprocess_question)
builder.add_node("select_similar_classes", select_similar_classes)
builder.add_node("get_context_class_from_cache", get_class_context_from_cache)
builder.add_node("get_context_class_from_kg", get_class_context_from_kg)

builder.add_node("create_prompt", create_prompt)
builder.add_node("generate_query", generate_query)
builder.add_node("run_query", run_query)
builder.add_node("interpret_results", interpret_results)

builder.add_edge(START, "init")
builder.add_edge("init", "validate_question")
builder.add_conditional_edges("validate_question", validate_question_router)
builder.add_edge("preprocess_question", "select_similar_classes")
builder.add_conditional_edges("select_similar_classes", get_class_context_router)
builder.add_edge("get_context_class_from_cache", "create_prompt")
builder.add_edge("get_context_class_from_kg", "create_prompt")
builder.add_edge("create_prompt", "generate_query")
builder.add_conditional_edges("generate_query", generate_query_router)
builder.add_conditional_edges("run_query", run_query_router)
builder.add_edge("interpret_results", END)

graph = builder.compile()


if __name__ == "__main__":
    asyncio.run(config.main(graph))
