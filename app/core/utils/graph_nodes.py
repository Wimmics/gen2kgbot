"""
This module implements the Langgraph nodes that are common to multiple scenarios
"""

from app.core.utils.graph_state import OverallState
from app.core.utils.question_preprocessing import extract_relevant_entities_spacy
from app.core.utils.sparql_toolkit import find_sparql_queries, run_sparql_query
from app.core.utils.config_manager import (
    get_class_vector_db_from_config,
    get_current_llm,
    setup_logger,
)
from app.core.utils.construct_util import (
    get_class_context,
)
from langchain_core.messages import AIMessage
from app.core.utils.prompts import interpret_csv_query_results_prompt

logger = setup_logger(__package__, __file__)

SPARQL_QUERY_EXEC_ERROR = "Error when running the SPARQL query"


def preprocess_question(input: OverallState) -> OverallState:
    logger.debug("Preprocessing the question...")

    extracted_classes = extract_relevant_entities_spacy(input["initial_question"])
    relevant_entities = AIMessage(f"{",".join(extracted_classes)}")
    logger.debug(f"Extracted following entities: {extracted_classes}")

    return {
        "messages": relevant_entities,
        "initial_question": input["initial_question"],
        "number_of_tries": 0,
    }


def select_similar_classes(state: OverallState) -> OverallState:
    """
    Retrieve, from the vector db, the descritption of ontology classes
    related to the question

    Args:
        state (dict): current state of the conversation

    Returns:
        dict: state updated with selected_classes
    """

    db = get_class_vector_db_from_config()

    question = state["initial_question"]
    logger.info(f"Users' question: {question}")
    logger.info("Looking for ontology classes related to the question...")

    # Retrieve the most similar text
    retrieved_documents = db.similarity_search(question, k=10)
    retrieved_classes = [item.page_content for item in retrieved_documents]

    result = "These are some relevant classes for the query generation:"
    for item in retrieved_documents:
        result = f"{result}\n{item.page_content}"
    result = f"{result}\n\n"

    logger.info(f"Found {len(retrieved_documents)} classes related to the question.")
    return {"messages": AIMessage(result), "selected_classes": retrieved_classes}


def get_class_context_from_cache(cls_path: str) -> OverallState:
    """
    Retrieve a class context from the cache

    Args:
        cls_path (str): path to the class context file

    Returns:
        dict: state where selected_classes_context.
            This will be added to selected_classes_context in the current context
    """
    with open(cls_path) as f:
        return {"selected_classes_context": ["\n".join(f.readlines())]}


def get_class_context_from_kg(cls: tuple) -> OverallState:
    """
    Retrieve a class context from the knowledge graph

    Args:
        cls (tuple): (class URI, label, description)

    Returns:
        dict: state where selected_classes_context.
            This will be added to selected_classes_context in the current context
    """
    graph_ttl = get_class_context(cls)
    return {"selected_classes_context": [graph_ttl]}


async def interpret_csv_query_results(state: OverallState) -> OverallState:
    csv_results_message = state["last_query_results"]
    llm = get_current_llm()
    result = await llm.ainvoke(
        interpret_csv_query_results_prompt.format(
            question=state["initial_question"], results=csv_results_message
        )
    )

    logger.debug(f"Interpretation of the query results:\n{result.content}")
    return OverallState({"messages": result, "results_interpretation": result})


def run_query(state: OverallState) -> OverallState:
    """
    Submit the generated SPARQL query to the endpoint and return the results

    Args:
        state (dict): current state of the conversation

    Returns:
        dict: state updated with last query (last_generated_query) and query results (last_query_results)
    """

    # Depending on the scenario, the last generated query may already be in the state or not
    if "last_generated_query" in state:
        query = state["last_generated_query"]
    else:
        query = find_sparql_queries(state["messages"][-1].content)[0]

    try:
        csv_result = run_sparql_query(query=query)
        logger.debug(f"Query execution results:\n{csv_result}")
        return {"last_generated_query": query, "last_query_results": csv_result}
    except Exception as e:
        logger.warning(f"An error occurred when running the query: {e}")
        return {
            "last_generated_query": query,
            "last_query_results": SPARQL_QUERY_EXEC_ERROR,
        }
