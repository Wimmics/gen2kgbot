import ast
import asyncio
import os
from typing import Literal
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, START, END
from rdflib import Graph
from app.core.scenarios.scenario_4.utils.prompt import system_prompt
from app.core.utils.sparql_toolkit import find_sparql_queries
from app.core.utils.construct_util import (
    format_class_graph_file,
    get_context_class,
    get_empty_graph_with_prefixes,
    tmp_directory,
)
from app.core.utils.graph_nodes import (
    interpret_csv_query_results,
    preprocess_question,
    select_similar_classes,
)
from app.core.utils.graph_state import InputState, OverallState
from app.core.utils.utils import (
    get_llm_from_config,
    main,
    setup_logger,
)
from rdflib.exceptions import ParserError
from app.core.utils.sparql_toolkit import run_sparql_query
from langgraph.constants import Send
import time


logger = setup_logger(__package__, __file__)

SCENARIO = "scenario_4"

llm = get_llm_from_config(SCENARIO)

# Router


def run_query_router(state: OverallState) -> Literal["interpret_results", "__end__"]:
    if state["messages"][-1].content.find("Error when running the query") == -1:
        logger.info("Query execution yielded results")
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
        logger.warning(
            "Query generation task did not produce a proper SPARQL query"
        )
        logger.info("Processing completed.")
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


def create_prompt(state: OverallState) -> OverallState:

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


def run_query(state: OverallState):

    query = find_sparql_queries(state["messages"][-1].content)[0]

    try:
        csv_result = run_sparql_query(query=query)
        return {"messages": csv_result, "last_generated_query": query}
    except ParserError as e:
        logger.warning(f"A parsing error occurred when running the query: {e}")
        return {"messages": AIMessage("Error when running the query")}
    except Exception as e:
        logger.warning(f"An error occurred when running the query: {e}")
        return {"messages": AIMessage("Error when running the query")}


s4_builder = StateGraph(
    state_schema=OverallState, input=InputState, output=OverallState
)

s4_builder.add_node("preprocess_question", preprocess_question)
s4_builder.add_node("select_similar_classes", select_similar_classes)
s4_builder.add_node("get_context_class_from_cache", get_context_class_from_cache)
s4_builder.add_node("get_context_class_from_kg", get_context_class_from_kg)

s4_builder.add_node("create_prompt", create_prompt)
s4_builder.add_node("generate_query", generate_query)
s4_builder.add_node("run_query", run_query)
s4_builder.add_node("interpret_results", interpret_csv_query_results)

s4_builder.add_edge(START, "preprocess_question")
s4_builder.add_edge("preprocess_question", "select_similar_classes")
s4_builder.add_conditional_edges("select_similar_classes", get_context_class_router)
s4_builder.add_edge("get_context_class_from_cache", "create_prompt")
s4_builder.add_edge("get_context_class_from_kg", "create_prompt")
s4_builder.add_edge("create_prompt", "generate_query")
s4_builder.add_conditional_edges("generate_query", generate_query_router)
s4_builder.add_conditional_edges("run_query", run_query_router)
s4_builder.add_edge("interpret_results", END)

graph = s4_builder.compile()


def run_scenario(question: str):
    return graph.ainvoke(input=InputState({"initial_question": question}))


if __name__ == "__main__":
    asyncio.run(main(graph))
