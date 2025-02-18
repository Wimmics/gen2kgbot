import asyncio
from typing import Literal
from langgraph.graph import StateGraph, START, END
from app.core.scenarios.scenario_6.prompt import (
    system_prompt_template,
    retry_prompt,
)
from app.core.utils.graph_nodes import (
    preprocess_question,
    select_similar_classes,
    get_class_context_from_cache,
    get_class_context_from_kg,
    select_similar_query_examples,
    create_query_generation_prompt,
    generate_query,
    verify_query,
    run_query,
    SPARQL_QUERY_EXEC_ERROR,
    interpret_csv_query_results,
)
from app.core.utils.graph_routers import get_class_context_router, verify_query_router
from app.core.utils.graph_state import InputState, OverallState
import app.core.utils.config_manager as config
from app.core.utils.logger_manager import setup_logger

logger = setup_logger(__package__, __file__)

SCENARIO = "scenario_6"
config.set_scenario(SCENARIO)


# Router


def run_query_router(state: OverallState) -> Literal["interpret_results", "__end__"]:
    if state["last_query_results"].find(SPARQL_QUERY_EXEC_ERROR) == -1:
        logger.info("Query execution yielded some results")
        return "interpret_results"
    else:
        logger.info("Processing completed.")
        return END


# Node


def create_prompt(state: OverallState) -> OverallState:
    return create_query_generation_prompt(system_prompt_template, state)


def create_retry_prompt(state: OverallState) -> OverallState:
    logger.info("retry_prompt created successfuly.")

    query_regeneration_prompt = (
        f"{retry_prompt.content}\n\n"
        + f"The properties and their type when using the classes: \n {state["merged_classes_context"]}\n\n"
        + f"{state['selected_queries']}\n\n"
        + f"The user question:\n{state['initial_question']}\n\n"
        + "The last answer you provided that either don't contain or have a unparsable SPARQL query:\n"
        + f"-------------------------------------\n{state['messages'][-2].content}\n--------------------------------------------------\n\n"
        + f"The verification didn't pass because:\n-------------------------\n{state["messages"][-1].content}\n--------------------------------\n"
    )

    return {
        "query_generation_prompt": query_regeneration_prompt,
    }


builder = StateGraph(state_schema=OverallState, input=InputState, output=OverallState)
prepro_builder = StateGraph(
    state_schema=OverallState, input=OverallState, output=OverallState
)

prepro_builder.add_node("preprocess_question", preprocess_question)
prepro_builder.add_node("select_similar_classes", select_similar_classes)
prepro_builder.add_node("get_context_class_from_cache", get_class_context_from_cache)
prepro_builder.add_node("get_context_class_from_kg", get_class_context_from_kg)
prepro_builder.add_node("select_similar_query_examples", select_similar_query_examples)

prepro_builder.add_edge(START, "preprocess_question")
prepro_builder.add_edge("preprocess_question", "select_similar_query_examples")
prepro_builder.add_edge("preprocess_question", "select_similar_classes")
prepro_builder.add_edge("select_similar_query_examples", END)
prepro_builder.add_conditional_edges("select_similar_classes", get_class_context_router)
prepro_builder.add_edge("get_context_class_from_cache", END)
prepro_builder.add_edge("get_context_class_from_kg", END)


builder.add_node("preprocessing_subgraph", prepro_builder.compile())
builder.add_node("create_prompt", create_prompt)
builder.add_node("generate_query", generate_query)

builder.add_node("run_query", run_query)
builder.add_node("verify_query", verify_query)
builder.add_node("create_retry_prompt", create_retry_prompt)

builder.add_node("interpret_results", interpret_csv_query_results)

builder.add_edge(START, "preprocessing_subgraph")
builder.add_edge("preprocessing_subgraph", "create_prompt")
builder.add_edge("create_prompt", "generate_query")
builder.add_edge("generate_query", "verify_query")
builder.add_conditional_edges("verify_query", verify_query_router)
builder.add_edge("create_retry_prompt", "generate_query")
builder.add_conditional_edges("run_query", run_query_router)
builder.add_edge("interpret_results", END)

graph = builder.compile()


def run_scenario(question: str):
    return graph.ainvoke(input=InputState({"initial_question": question}))


if __name__ == "__main__":
    asyncio.run(config.main(graph))
