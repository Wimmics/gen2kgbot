import asyncio
from typing import Literal
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from app.core.scenarios.scenario_3.utils.prompt import system_prompt_template
from app.core.utils.sparql_toolkit import find_sparql_queries
from app.core.utils.graph_nodes import (
    interpret_csv_query_results,
    select_similar_classes,
    run_query,
    SPARQL_QUERY_EXEC_ERROR,
)
from app.core.utils.graph_state import InputState, OverallState
from app.core.utils.config_manager import (
    get_llm_from_config,
    main,
    setup_logger,
)
from rdflib.exceptions import ParserError
from app.core.utils.sparql_toolkit import run_sparql_query

logger = setup_logger(__package__, __file__)

SCENARIO = "scenario_3"

llm = get_llm_from_config(SCENARIO)


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


async def generate_query(state: OverallState):

    selected_classes = ""
    for item in state["selected_classes"]:
        selected_classes = f"{selected_classes}\n{item}"

    prompt = system_prompt_template.format(
        question=state["initial_question"], context=selected_classes
    )
    logger.info(f"Prompt created: {prompt}")
    result = await llm.ainvoke(prompt)
    return {"messages": [HumanMessage(state["initial_question"]), result]}


builder = StateGraph(
    state_schema=OverallState, input=InputState, output=OverallState
)


builder.add_node("select_similar_classes", select_similar_classes)
builder.add_node("generate_query", generate_query)
builder.add_node("run_query", run_query)
builder.add_node("interpret_results", interpret_csv_query_results)

builder.add_edge(START, "select_similar_classes")
builder.add_edge("select_similar_classes", "generate_query")
builder.add_conditional_edges("generate_query", generate_query_router)
builder.add_conditional_edges("run_query", run_query_router)
builder.add_edge("interpret_results", END)

graph = builder.compile()


def run_scenario(question: str):
    return graph.ainvoke(input=InputState({"initial_question": question}))


if __name__ == "__main__":
    asyncio.run(main(graph))
