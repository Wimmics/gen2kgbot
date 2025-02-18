import asyncio
from typing import Literal
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from app.core.scenarios.scenario_5.prompt import (
    system_prompt_template,
    retry_prompt,
)
from app.core.utils.sparql_toolkit import find_sparql_queries
from app.core.utils.graph_nodes import (
    preprocess_question,
    select_similar_classes,
    get_class_context_from_cache,
    get_class_context_from_kg,
    create_query_generation_prompt,
    generate_query,
    run_query,
    SPARQL_QUERY_EXEC_ERROR,
    interpret_csv_query_results,
)
from app.core.utils.graph_routers import get_class_context_router
from app.core.utils.graph_state import InputState, OverallState
import app.core.utils.config_manager as config
from app.core.utils.logger_manager import setup_logger
from rdflib.plugins.sparql.algebra import translateQuery
from rdflib.plugins.sparql.parser import parseQuery

logger = setup_logger(__package__, __file__)

SCENARIO = "scenario_5"
config.set_scenario(SCENARIO)

MAX_NUMBER_OF_TRIES: int = 3


# Routers


def run_query_router(state: OverallState) -> Literal["interpret_results", "__end__"]:
    if state["last_query_results"].find(SPARQL_QUERY_EXEC_ERROR) == -1:
        logger.info("Query execution yielded some results")
        return "interpret_results"
    else:
        logger.info("Processing completed.")
        return END


def verify_query_router(
    state: OverallState,
) -> Literal["run_query", "create_retry_prompt", "__end__"]:
    if "last_generated_query" in state:
        logger.info("Query generation task produced one SPARQL query")
        return "run_query"
    else:
        logger.warning("Query generation task did not produce a proper SPARQL query")
        if state["number_of_tries"] < MAX_NUMBER_OF_TRIES:
            logger.info(f"Tries left {MAX_NUMBER_OF_TRIES - state['number_of_tries']}")
            return "create_retry_prompt"
        else:
            logger.info("Max retries reached. Processing stopped.")
        return END


# Nodes


def create_prompt(state: OverallState) -> OverallState:
    return create_query_generation_prompt(system_prompt_template, state)


def verify_query(state: OverallState) -> OverallState:
    queries = find_sparql_queries(state["messages"][-1].content)

    if len(queries) == 0:
        return {
            "number_of_tries": state["number_of_tries"] + 1,
            "messages": [
                HumanMessage("No properly formatted SPARQL query was generated.")
            ],
        }

    try:
        translateQuery(parseQuery(queries[0]))
    except Exception as e:
        return {
            "number_of_tries": state["number_of_tries"] + 1,
            "messages": [AIMessage(f"{e}")],
        }

    return {"last_generated_query": queries[0]}


def create_retry_prompt(state: OverallState) -> OverallState:
    logger.info("retry_prompt created successfuly.")

    query_regeneration_prompt = (
        f"{retry_prompt.content}\n\n"
        + f"The properties and their type when using the classes: \n {state["merged_classes_context"]}\n\n"
        + f"The user question:\n{state['initial_question']}\n\n"
        + "The last answer you provided that either don't contain or have a unparsable SPARQL query:\n"
        + f"-------------------------------------\n{state['messages'][-2].content}\n--------------------------------------------------\n\n"
        + f"The verification didn't pass because:\n-------------------------\n{state["messages"][-1].content}\n--------------------------------\n"
    )

    return {
        "query_generation_prompt": query_regeneration_prompt,
    }


builder = StateGraph(state_schema=OverallState, input=InputState, output=OverallState)

builder.add_node("preprocess_question", preprocess_question)
builder.add_node("select_similar_classes", select_similar_classes)
builder.add_node("get_context_class_from_cache", get_class_context_from_cache)
builder.add_node("get_context_class_from_kg", get_class_context_from_kg)

builder.add_node("create_prompt", create_prompt)
builder.add_node("generate_query", generate_query)
builder.add_node("run_query", run_query)

builder.add_node("verify_query", verify_query)
builder.add_node("create_retry_prompt", create_retry_prompt)

builder.add_node("interpret_results", interpret_csv_query_results)

builder.add_edge(START, "preprocess_question")
builder.add_edge("preprocess_question", "select_similar_classes")
builder.add_conditional_edges("select_similar_classes", get_class_context_router)
builder.add_edge("get_context_class_from_cache", "create_prompt")
builder.add_edge("get_context_class_from_kg", "create_prompt")
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
