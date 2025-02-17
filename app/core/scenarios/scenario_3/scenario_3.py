import asyncio
from typing import Literal
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from app.core.scenarios.scenario_3.prompt import system_prompt_template
from app.core.utils.sparql_toolkit import find_sparql_queries
from app.core.utils.graph_nodes import (
    interpret_csv_query_results,
    select_similar_classes,
    create_prompt_from_template,
    run_query,
    SPARQL_QUERY_EXEC_ERROR,
)
from app.core.utils.graph_state import InputState, OverallState
import app.core.utils.config_manager as config
from app.core.utils.logger_manager import setup_logger

logger = setup_logger(__package__, __file__)

SCENARIO = "scenario_3"

llm = config.get_llm(SCENARIO)


def run_query_router(state: OverallState) -> Literal["interpret_results", "__end__"]:
    """
    Check if the query was successful and route to the next step accordingly.

    Args:
        state (OverallState): current state of the conversation

    Returns:
        Literal["interpret_results", END]: next step in the conversation
    """
    if state["last_query_results"].find(SPARQL_QUERY_EXEC_ERROR) == -1:
        logger.info("Query execution yielded some results")
        return "interpret_results"
    else:
        logger.info("Processing completed.")
        return END


def create_prompt(state: OverallState) -> OverallState:
    return create_prompt_from_template(system_prompt_template, state)


async def generate_query(state: OverallState):
    result = await llm.ainvoke(state["query_generation_prompt"])
    return {"messages": result}


def generate_query_router(state: OverallState) -> Literal["run_query", "__end__"]:
    """
    Check if the query generation task produced 0, 1 or more SPARQL query,
    and routes to the next step accordingly.
    If more than one query was produced, just send a warning and process the first one.
    """

    no_queries = len(find_sparql_queries(state["messages"][-1].content))

    if no_queries > 1:
        logger.warning(
            f"Query generation task produced {no_queries} SPARQL queries. Will process the first one."
        )
        return "run_query"
    elif no_queries == 1:
        logger.info("Query generation task produced one SPARQL query")
        return "run_query"
    else:
        logger.warning("Query generation task did not produce a proper SPARQL query")
        logger.info("Processing completed.")
        return END


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
