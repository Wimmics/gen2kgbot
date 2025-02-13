import asyncio
from datetime import datetime, timezone
from typing import Literal
from langgraph.graph import StateGraph, START, END
from rdflib import Graph
from app.core.scenarios.scenario_4.utils.prompt import system_prompt
from app.core.utils.sparql_toolkit import find_sparql_queries
from app.core.utils.construct_util import (
    get_empty_graph_with_prefixes,
)
from app.core.utils.graph_nodes import (
    preprocess_question,
    select_similar_classes,
    get_class_context_from_cache,
    get_class_context_from_kg,
    run_query,
    SPARQL_QUERY_EXEC_ERROR,
    interpret_csv_query_results,
)
from app.core.utils.graph_routers import get_class_context_router
from app.core.utils.graph_state import InputState, OverallState
from app.core.utils.config_manager import (
    get_llm_from_config,
    main,
    setup_logger,
    get_temp_directory,
)

logger = setup_logger(__package__, __file__)

SCENARIO = "scenario_4"

llm = get_llm_from_config(SCENARIO)

# Routers


def run_query_router(state: OverallState) -> Literal["interpret_results", "__end__"]:
    if state["last_query_results"].find(SPARQL_QUERY_EXEC_ERROR) == -1:
        logger.info("Query execution yielded some results")
        return "interpret_results"
    else:
        logger.info("Processing completed.")
        return END


def generate_query_router(state: OverallState) -> Literal["run_query", "__end__"]:
    generated_queries = find_sparql_queries(state["messages"][-1].content)

    if len(generated_queries) > 0:
        logger.info("Query generation task produced one SPARQL query")
        return "run_query"
    else:
        logger.warning("Query generation task did not produce a proper SPARQL query")
        logger.info("Processing completed.")
        return END


# Nodes


def create_prompt(state: OverallState) -> OverallState:

    # Load all the class contexts in a common graph
    merged_graph = get_empty_graph_with_prefixes()
    for cls_context in state["selected_classes_context"]:
        g = Graph()
        merged_graph = merged_graph + g.parse(data=cls_context)

    # Save the graph
    timestr = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S.%f")[:-3]
    merged_graph.serialize(
        destination=f"{get_temp_directory()}/context-{timestr}.ttl",
        format="turtle",
    )

    merged_graph_ttl = merged_graph.serialize(format="turtle")

    logger.info(f"Context graph saved locally in {get_temp_directory()}/context-{timestr}.ttl")
    logger.info("prompt created successfuly.")

    query_generation_prompt = (
        f"{system_prompt.content}\n"
        + f"{state['messages'][-1].content}\n"
        + f"The properties and their type when using the classes: \n {merged_graph_ttl}\n\n"
        + f"The user question is: \n\n{state['initial_question']}\n"
    )

    return {
        "merged_classes_context": merged_graph_ttl,
        "query_generation_prompt": query_generation_prompt,
    }


async def generate_query(state: OverallState):
    result = await llm.ainvoke(state["query_generation_prompt"])
    return {"messages": result}


builder = StateGraph(state_schema=OverallState, input=InputState, output=OverallState)

builder.add_node("preprocess_question", preprocess_question)
builder.add_node("select_similar_classes", select_similar_classes)
builder.add_node("get_context_class_from_cache", get_class_context_from_cache)
builder.add_node("get_context_class_from_kg", get_class_context_from_kg)

builder.add_node("create_prompt", create_prompt)
builder.add_node("generate_query", generate_query)
builder.add_node("run_query", run_query)
builder.add_node("interpret_results", interpret_csv_query_results)

builder.add_edge(START, "preprocess_question")
builder.add_edge("preprocess_question", "select_similar_classes")
builder.add_conditional_edges("select_similar_classes", get_class_context_router)
builder.add_edge("get_context_class_from_cache", "create_prompt")
builder.add_edge("get_context_class_from_kg", "create_prompt")
builder.add_edge("create_prompt", "generate_query")
builder.add_conditional_edges("generate_query", generate_query_router)
builder.add_conditional_edges("run_query", run_query_router)
builder.add_edge("interpret_results", END)

graph = builder.compile()


def run_scenario(question: str):
    return graph.ainvoke(input=InputState({"initial_question": question}))


if __name__ == "__main__":
    asyncio.run(main(graph))
