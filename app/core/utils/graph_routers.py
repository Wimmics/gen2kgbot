"""
This module implements the Langgraph contitional edges, aka. routers,
that are common to multiple scenarios
"""

import ast
from typing import Literal
import os
from app.core.utils.graph_state import OverallState
import app.core.utils.config_manager as config
from app.core.utils.sparql_toolkit import find_sparql_queries
from app.core.utils.construct_util import generate_class_context_filename
from app.core.utils.graph_nodes import SPARQL_QUERY_EXEC_ERROR
from langgraph.graph import END
from langgraph.constants import Send


logger = config.setup_logger(__package__, __file__)

MAX_NUMBER_OF_TRIES: int = 3


def get_class_context_router(
    state: OverallState,
) -> Literal['get_context_class_from_cache', 'get_context_class_from_kg']:
    """
    Looks in the cache folder whether the context of the selected similar classes are
    already present. If not, route to the node that will fetch the context from the knowledge graph.

    This function must be invoked after classes similar to the user question were set in OverallState.selected_classes.

    Args:
        state (dict): current state of the conversation

    Returns:
        List[Send]: node to be executed next with class context file path.
            Next node should be one of "get_context_class_from_cache" or "get_context_class_from_kg".
            If "get_context_class_from_cache", the additional arg is the file path.
            If "get_context_class_from_kg", the additional arg is a tuple (uri, label, description).
    """
    next_nodes = []

    logger.info("Looking for class contexts (in ttl) from the cache folder...")

    for item in state["selected_classes"]:
        cls = ast.literal_eval(item)
        cls_path = generate_class_context_filename(cls[0])

        if os.path.exists(cls_path):
            logger.debug(f"Class context found in cache: {cls_path}.")
            next_nodes.append(Send("get_context_class_from_cache", cls_path))
        else:
            logger.debug(f"Class context not found in cache for class {cls}.")
            next_nodes.append(Send("get_context_class_from_kg", cls))

    return next_nodes


def generate_query_router(state: OverallState) -> Literal["run_query", "__end__"]:
    """
    Check if the query generation task produced 0, 1 or more SPARQL query, and route to the next step.
    If more than one query was produced, just send a warning and process the first one.

    Only used in scenarios 2, 3, 4.

    Args:
        state (OverallState): current state of the conversation

    Returns:
        Literal["run_query", END]: next step in the conversation
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


def verify_query_router(
    state: OverallState,
) -> Literal["run_query", "create_retry_prompt", "__end__"]:
    """
    Decide whether to run the query, retry or stop if max number of attemps is reached.

    Used in scenarios 5 and 6 after the verify_query node.

    Args:
        state (OverallState): current state of the conversation

    Returns:
        Literal["run_query", "create_retry_prompt", END]: next step in the conversation
    """
    if "last_generated_query" in state:
        return "run_query"
    else:
        if state["number_of_tries"] < MAX_NUMBER_OF_TRIES:
            logger.info(
                f"Retries left: {MAX_NUMBER_OF_TRIES - state['number_of_tries']}"
            )
            return "create_retry_prompt"
        else:
            logger.info("Max retries reached. Processing stopped.")
        return END


def run_query_router(state: OverallState) -> Literal["interpret_results", "__end__"]:
    """
    Check the SPARQL query results and decide whether to go to intepretation or stop.

    Args:
        state (OverallState): current state of the conversation

    Returns:
        Literal["interpret_results", END]: next step in the conversation
    """
    results = state["last_query_results"]
    if results.find(SPARQL_QUERY_EXEC_ERROR) == -1:
        nolines = len(results.splitlines())
        if nolines == 0:
            logger.warning("SPARQL query returned invalid csv output")
            return END
        elif nolines == 1:
            logger.info("SPARQL query executed succcessully but returned empty results")
            return END
        else:
            logger.info(
                "SPARQL query executed succcessully and returned non-empty results"
            )
            return "interpret_results"
    else:
        logger.info("SPARQL query execution failed.")
        return END
