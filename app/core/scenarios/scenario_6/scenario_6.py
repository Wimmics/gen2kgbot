import ast
import asyncio
import os
import time
from typing import Literal
from rdflib import Graph
from rdflib.plugins.sparql.algebra import translateQuery
from rdflib.plugins.sparql.parser import parseQuery
from rdflib.exceptions import ParserError
from langgraph.constants import Send
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage, HumanMessage
from app.core.scenarios.scenario_6.utils.prompt import (
    system_prompt,
    retry_prompt,
)
from app.core.utils.sparql_toolkit import find_sparql_queries
from app.core.utils.graph_nodes import (
    interpret_csv_query_results,
    preprocess_question,
    select_similar_classes,
)
from app.core.utils.graph_state import InputState, OverallState
from app.core.utils.utils import (
    get_llm_from_config,
    get_query_vector_db_from_config,
    main,
    setup_logger,
)
from app.core.utils.sparql_toolkit import run_sparql_query
from app.core.utils.construct_util import (
    add_known_prefixes_to_query,
    format_class_graph_file,
    get_context_class,
    get_empty_graph_with_prefixes,
    tmp_directory,
)

SCENARIO = "scenario_6"
MAX_NUMBER_OF_TRIES: int = 3

logger = setup_logger(__package__, __file__)
llm = get_llm_from_config(SCENARIO)


# Router


def run_query_router(state: OverallState) -> Literal["interpret_results", "__end__"]:
    if state["messages"][-1].content.find("Error when running the query") == -1:
        logger.info("Query execution yielded results")
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


def get_context_class_router(
    state: OverallState,
) -> Literal["get_context_class_from_cache", "get_context_class_from_kg"]:

    next_nodes = []

    for doc in state["selected_classes"]:
        cls = ast.literal_eval(doc.page_content)
        cls_path = format_class_graph_file(cls[0])

        if os.path.exists(cls_path):
            logger.info(f"Classe context file path at {cls_path} found.")
            next_nodes.append(Send("get_context_class_from_cache", cls_path))
        else:
            logger.info(f"Classe context file path at {cls_path} not found.")
            next_nodes.append(Send("get_context_class_from_kg", cls))

    return next_nodes


# Node


def get_context_class_from_cache(cls_path: str) -> OverallState:
    with open(cls_path) as f:
        return {"selected_classes_context": ["\n".join(f.readlines())]}


def get_context_class_from_kg(cls: str) -> OverallState:
    graph_ttl = get_context_class(cls)
    return {"selected_classes_context": [graph_ttl]}


def select_similar_query_examples(state: OverallState) -> OverallState:

    db = get_query_vector_db_from_config(scenario=SCENARIO)

    query = state["messages"][-1].content

    # Retrieve the most similar text
    retrieved_documents = db.similarity_search(query, k=3)

    result = "These are some relevant queries for the query generation:\n"
    # show the retrieved document's content
    for doc in retrieved_documents:
        result = f"{result}\n```sparql\n{doc.page_content}\n```\n"

    logger.info("Done with selecting some similar queries to help query generation")

    return {"messages": AIMessage(result), "selected_queries": result}


def create_prompt(state: OverallState) -> OverallState:

    # if "selected_queries" in state and "selected_classes" in state:
    merged_graph = get_empty_graph_with_prefixes()

    for cls_context in state["selected_classes_context"]:
        g = Graph()
        merged_graph = merged_graph + g.parse(data=cls_context)

    # Save the graph
    timestr = time.strftime("%Y%m%d-%H%M%S")
    merged_graph.serialize(
        destination=f"{tmp_directory}/context-{timestr}.ttl",
        format="turtle",
    )

    merged_graph_ttl = merged_graph.serialize(format="turtle")

    logger.info(f"Context graph saved locally in {tmp_directory}/context-{timestr}.ttl")
    logger.info("prompt created successfuly.")

    query_generation_prompt = (
        f"{system_prompt.content}\n"
        + f"{state['messages'][-1].content}\n"
        + f"{state['selected_queries']}\n"
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
        translateQuery(parseQuery(add_known_prefixes_to_query(queries[0])))
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
        + f"{state['selected_queries']}\n\n"
        + f"The user question:\n{state['initial_question']}\n\n"
        + "The last answer you provided that either don't contain or have a unparsable SPARQL query:\n"
        + f"-------------------------------------\n{state['messages'][-2].content}\n--------------------------------------------------\n\n"
        + f"The verification didn't pass because:\n-------------------------\n{state["messages"][-1].content}\n--------------------------------\n"
    )

    return {
        "query_generation_prompt": query_regeneration_prompt,
    }


def run_query(state: OverallState):

    query = state["last_generated_query"]

    try:
        csv_result = run_sparql_query(query=query)
        return {"messages": csv_result}
    except ParserError as e:
        logger.warning(f"A parsing error occurred when running the query: {e}")
        return {"messages": AIMessage("Error when running the query")}
    except Exception as e:
        logger.warning(f"An error occurred when running the query: {e}")
        return {"messages": AIMessage("Error when running the query")}


s6_builder = StateGraph(
    state_schema=OverallState, input=InputState, output=OverallState
)
s6_preprocessing_builder = StateGraph(
    state_schema=OverallState, input=OverallState, output=OverallState
)

s6_preprocessing_builder.add_node("preprocess_question", preprocess_question)
s6_preprocessing_builder.add_node("select_similar_classes", select_similar_classes)
s6_preprocessing_builder.add_node(
    "get_context_class_from_cache", get_context_class_from_cache
)
s6_preprocessing_builder.add_node(
    "get_context_class_from_kg", get_context_class_from_kg
)
s6_preprocessing_builder.add_node(
    "select_similar_query_examples", select_similar_query_examples
)

s6_preprocessing_builder.add_edge(START, "preprocess_question")
s6_preprocessing_builder.add_edge(
    "preprocess_question", "select_similar_query_examples"
)
s6_preprocessing_builder.add_edge("preprocess_question", "select_similar_classes")
s6_preprocessing_builder.add_edge("select_similar_query_examples", END)
s6_preprocessing_builder.add_conditional_edges(
    "select_similar_classes", get_context_class_router
)
s6_preprocessing_builder.add_edge("get_context_class_from_cache", END)
s6_preprocessing_builder.add_edge("get_context_class_from_kg", END)


s6_builder.add_node("preprocessing_subgraph", s6_preprocessing_builder.compile())
s6_builder.add_node("create_prompt", create_prompt)
s6_builder.add_node("generate_query", generate_query)
s6_builder.add_node("run_query", run_query)
s6_builder.add_node("verify_query", verify_query)
s6_builder.add_node("create_retry_prompt", create_retry_prompt)
s6_builder.add_node("interpret_results", interpret_csv_query_results)

s6_builder.add_edge(START, "preprocessing_subgraph")
s6_builder.add_edge("preprocessing_subgraph", "create_prompt")
s6_builder.add_edge("create_prompt", "generate_query")
s6_builder.add_edge("generate_query", "verify_query")
s6_builder.add_conditional_edges("verify_query", verify_query_router)
s6_builder.add_edge("create_retry_prompt", "generate_query")
s6_builder.add_conditional_edges("run_query", run_query_router)
s6_builder.add_edge("interpret_results", END)

graph = s6_builder.compile()


def run_scenario(question: str):
    return graph.ainvoke(input=InputState({"initial_question": question}))


if __name__ == "__main__":
    asyncio.run(main(graph))
