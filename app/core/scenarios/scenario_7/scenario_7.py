import asyncio
from typing import Literal
from langgraph.graph import StateGraph, START, END
from app.core.scenarios.scenario_7.prompt import system_prompt_template
from app.core.utils.construct_util import add_known_prefixes_to_query
from app.core.utils.graph_nodes import (
    preprocess_question,
    select_similar_classes,
    get_class_context_from_cache,
    get_class_context_from_kg,
    select_similar_query_examples,
    create_query_generation_prompt,
    generate_query,
    validate_question,
    run_query,
    interpret_csv_query_results,
)
from app.core.utils.graph_routers import (
    get_class_context_router,
    preprocessing_subgraph_router,
    validate_question_router,
    run_query_router,
    MAX_NUMBER_OF_TRIES,
)
from app.core.utils.graph_state import InputState, OverallState
import app.core.utils.config_manager as config
from app.core.utils.logger_manager import setup_logger
from rdflib.plugins.sparql.parser import parseQuery
from rdflib.plugins.sparql.algebra import translateQuery
from langchain_core.messages import AIMessage, HumanMessage
from app.core.utils.sparql_toolkit import find_sparql_queries


logger = setup_logger(__package__, __file__)

SCENARIO = "scenario_7"


# Routers


def verify_query_router(
    state: OverallState,
) -> Literal["extract_query_qnames", "judge_regeneration_prompt", "__end__"]:
    """
    Decide whether to continue judging the query, retry or stop if max number of attemps is reached.

    Args:
        state (OverallState): current state of the conversation

    Returns:
        Literal["extract_query_qnames", "judge_regeneration_prompt", END]: next step in the conversation
    """
    if "last_generated_query" in state:
        return "extract_query_qnames"
    else:
        if state["number_of_tries"] < MAX_NUMBER_OF_TRIES:
            logger.info(
                f"Retries left: {MAX_NUMBER_OF_TRIES - state['number_of_tries']}"
            )
            return "judge_regeneration_prompt"
        else:
            logger.info("Max retries reached. Processing stopped.")
        return END


def judging_subgraph_router(
    state: OverallState,
) -> Literal["run_query", "__end__"]:
    """
    Decide whether to run the query, or stop if max number of attemps is reached.

    Args:
        state (OverallState): current state of the conversation

    Returns:
        Literal["run_query", END]: next step in the conversation
    """
    if "last_generated_query" in state:
        return "run_query"
    else:
        logger.info("Max retries reached. Processing stopped.")
        return END


def judge_query_router(
    state: OverallState,
) -> Literal["judge_regeneration_prompt", "__end__"]:
    """
    Decide whether to regeneration a new query, or stop if it judged sufficient.

    Args:
        state (OverallState): current state of the conversation

    Returns:
        Literal["judge_regeneration_prompt", END]: next step in the conversation
    """
    if "last_generated_query" in state:  # TODO: check if the query is invalid
        logger.info("The generated SPARQL query needs improvement.")
        return "judge_regeneration_prompt"
    else:
        logger.info("The generated SPARQL query is valid.")
        return END


# Nodes


def init(state: OverallState) -> OverallState:
    logger.info(f"Running scenario: {SCENARIO}")
    return OverallState({"scenario_id": SCENARIO})


def create_prompt(state: OverallState) -> OverallState:
    return create_query_generation_prompt(system_prompt_template, state)


def extract_query_qnames(state: OverallState):
    """
    Extract qnames from the generated query.
    """

    qname_list = []
    formated_qname_list = f"{", ".join(qname_list)}"
    logger.debug(f"Extracted following named entities: {formated_qname_list}")

    return OverallState(
        {
            "messages": AIMessage(formated_qname_list),
            "question_validation_results": qname_list,
        }
    )


def verify_query(state: OverallState) -> OverallState:
    """
    Check if a query was generated and if it is syntactically correct.
    If more than one query was produced, just log a warning and process the first one.

    Used in scenarios 5 and 6.

    Args:
        state (dict): current state of the conversation

    Return:
        dict: state updated with the query if it was correct (last_generated_query),
            otherwise increment number of tries (number_of_tries)
    """

    queries = find_sparql_queries(state["messages"][-1].content)
    no_queries = len(queries)

    if no_queries == 0:
        logger.info("Query generation task did not produce any SPARQL query.")
        return {
            "number_of_tries": state["number_of_tries"] + 1,
            "messages": [HumanMessage("No SPARQL query was generated.")],
        }
    if no_queries > 1:
        logger.warning(
            f"Query generation task produced {no_queries} SPARQL queries. Will process the first one."
        )

    try:
        query = queries[0]
        logger.info("Query generation task produced a SPARQL query.")
        logger.debug(f"Generated SPARQL query:\n{query}")
        translateQuery(parseQuery(add_known_prefixes_to_query(queries[0])))
    except Exception as e:
        logger.warning(f"The generated SPARQL query is invalid: {e}")
        return {
            "number_of_tries": state["number_of_tries"] + 1,
            "messages": [HumanMessage(f"{e}")],
        }

    logger.info("The generated SPARQL query is syntactically correct.")
    return {"last_generated_query": queries[0]}


def find_qnames_info(state: OverallState):
    """ """

    qname_list = []
    formated_qname_list = f"{", ".join(qname_list)}"
    logger.debug(f"Extracted following named entities: {formated_qname_list}")

    return OverallState(
        {
            "messages": AIMessage(formated_qname_list),
            "question_validation_results": qname_list,
        }
    )


def judge_query(state: OverallState):
    """ """

    qname_list = []
    formated_qname_list = f"{", ".join(qname_list)}"
    logger.debug(f"Extracted following named entities: {formated_qname_list}")

    return OverallState(
        {
            "messages": AIMessage(formated_qname_list),
            "question_validation_results": qname_list,
        }
    )


def judge_regeneration_prompt(state: OverallState):
    """ """

    qname_list = []
    formated_qname_list = f"{", ".join(qname_list)}"
    logger.debug(f"Extracted following named entities: {formated_qname_list}")

    return OverallState(
        {
            "messages": AIMessage(formated_qname_list),
            "question_validation_results": qname_list,
        }
    )


def judge_regenerate_query(state: OverallState):
    """ """

    qname_list = []
    formated_qname_list = f"{", ".join(qname_list)}"
    logger.debug(f"Extracted following named entities: {formated_qname_list}")

    return OverallState(
        {
            "messages": AIMessage(formated_qname_list),
            "question_validation_results": qname_list,
        }
    )


# Subgraph for preprocessing the question: generate context with classes and examples queries
prepro_builder = StateGraph(
    state_schema=OverallState, input=OverallState, output=OverallState
)

# Preprocessing graph for generating context with classes and examples queries
prepro_builder.add_node("init", init)
prepro_builder.add_node("validate_question", validate_question)
prepro_builder.add_node("preprocess_question", preprocess_question)
prepro_builder.add_node("select_similar_classes", select_similar_classes)
prepro_builder.add_node("get_context_class_from_cache", get_class_context_from_cache)
prepro_builder.add_node("get_context_class_from_kg", get_class_context_from_kg)
prepro_builder.add_node("select_similar_query_examples", select_similar_query_examples)

prepro_builder.add_edge(START, "init")
prepro_builder.add_edge("init", "validate_question")
prepro_builder.add_conditional_edges("validate_question", validate_question_router)
prepro_builder.add_edge("preprocess_question", "select_similar_query_examples")
prepro_builder.add_edge("preprocess_question", "select_similar_classes")
prepro_builder.add_edge("select_similar_query_examples", END)
prepro_builder.add_conditional_edges("select_similar_classes", get_class_context_router)
prepro_builder.add_edge("get_context_class_from_cache", END)
prepro_builder.add_edge("get_context_class_from_kg", END)


# Subgraph for judging the generated query
judge_builder = StateGraph(
    state_schema=OverallState, input=OverallState, output=OverallState
)

# Judging graph for verifying the generated query

judge_builder.add_node("verify_query", verify_query)
judge_builder.add_node("extract_query_qnames", extract_query_qnames)
judge_builder.add_node("find_qnames_info", find_qnames_info)
judge_builder.add_node("judge_query", judge_query)
judge_builder.add_node("judge_regeneration_prompt", judge_regeneration_prompt)
judge_builder.add_node("judge_regenerate_query", judge_regenerate_query)


judge_builder.add_edge(START, "verify_query")
judge_builder.add_conditional_edges("verify_query", verify_query_router)
judge_builder.add_edge("extract_query_qnames", "find_qnames_info")
judge_builder.add_edge("find_qnames_info", "judge_query")
judge_builder.add_conditional_edges("judge_query", judge_query_router)
judge_builder.add_edge("judge_regeneration_prompt", "judge_regenerate_query")
judge_builder.add_edge("judge_regenerate_query", "verify_query")


# Main graph for generating and executing the query
builder = StateGraph(state_schema=OverallState, input=InputState, output=OverallState)

builder.add_node("preprocessing_subgraph", prepro_builder.compile())
builder.add_node("create_prompt", create_prompt)
builder.add_node("generate_query", generate_query)
builder.add_node("run_query", run_query)
builder.add_node("interpret_results", interpret_csv_query_results)
builder.add_node("judging_subgraph", judge_builder.compile())

builder.add_edge(START, "preprocessing_subgraph")
builder.add_conditional_edges("preprocessing_subgraph", preprocessing_subgraph_router)
builder.add_edge("create_prompt", "generate_query")
builder.add_edge("generate_query", "judging_subgraph")
builder.add_conditional_edges("judging_subgraph", judging_subgraph_router)
builder.add_conditional_edges("run_query", run_query_router)
builder.add_edge("interpret_results", END)

graph = builder.compile()


if __name__ == "__main__":
    asyncio.run(config.main(graph))
