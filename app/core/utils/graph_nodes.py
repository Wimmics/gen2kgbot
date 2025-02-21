"""
This module implements the Langgraph nodes that are common to multiple scenarios
"""

from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from app.core.utils.graph_state import OverallState
from app.core.utils.question_preprocessing import extract_relevant_entities_spacy
from app.core.utils.sparql_toolkit import find_sparql_queries, run_sparql_query
import app.core.utils.config_manager as config
from app.core.utils.construct_util import (
    get_class_context,
    get_empty_graph_with_prefixes,
    add_known_prefixes_to_query,
)
from app.core.utils.prompts import interpret_csv_query_results_prompt
from rdflib import Graph
from rdflib.plugins.sparql.parser import parseQuery
from rdflib.plugins.sparql.algebra import translateQuery

logger = config.setup_logger(__package__, __file__)

SPARQL_QUERY_EXEC_ERROR = "Error when running the SPARQL query"


def preprocess_question(state: OverallState) -> OverallState:
    """
    Extract named entities from the user question, to help selected similar SPARQL queries.
    Only applies in scenario 6.
    """

    if state["scenario_id"] != "scenario_6":
        return {
            "initial_question": state["initial_question"],
            "number_of_tries": 0,
        }

    logger.debug("Preprocessing the question...")

    extracted_classes = extract_relevant_entities_spacy(state["initial_question"])
    logger.debug(f"Extracted following named entities: {extracted_classes}")
    relevant_entities = f"{",".join(extracted_classes)}"

    return {
        "messages": AIMessage(relevant_entities),
        "initial_question": state["initial_question"],
        "question_relevant_entities": relevant_entities,
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

    db = config.get_class_context_vector_db(state["scenario_id"])

    question = state["initial_question"]
    logger.info("Looking for classes related to the question in the vector db...")

    # Retrieve the most similar text
    retrieved_documents = db.similarity_search(question, k=10)
    retrieved_classes = [item.page_content for item in retrieved_documents]

    logger.info(f"Found {len(retrieved_classes)} classes related to the question.")
    return {"selected_classes": retrieved_classes}


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


def select_similar_query_examples(state: OverallState) -> OverallState:
    """
    Retrieve the SPARQL queries most similar to the question

    Args:
        state (dict): current state of the conversation

    Returns:
        dict: state updated messages and queries (selected_queries)
    """
    question = state["question_relevant_entities"]
    retrieved_documents = config.get_query_vector_db(
        state["scenario_id"]
    ).similarity_search(question, k=3)

    # Show the retrieved document's content
    result = ""
    for item in retrieved_documents:
        result = f"{result}\n```sparql\n{item.page_content}\n```\n"
    logger.info(
        f"Selected {len(retrieved_documents)} SPARQL queries similar to the named entities of the quesion."
    )

    return {"messages": AIMessage(result), "selected_queries": result}


def create_query_generation_prompt(
    template: PromptTemplate, state: OverallState
) -> OverallState:
    """
    Generate a prompt from a template using the inputs available in the current state.
    Depending on the scenario, the inputs variables may not be the same.

    Args:
        template (PromptTemplate): template to use
        state (dict): current state of the conversation

    Returns:
        dict: state updated with the prompt generated (query_generation_prompt)
            and optionally the class contexts all merged in a single graph (merged_classes_context)
    """
    # logger.debug(f"Query generation prompt template: {template}")

    if "kg_full_name" in template.input_variables:
        template = template.partial(kg_full_name=config.get_kg_full_name())

    if "kg_description" in template.input_variables:
        template = template.partial(kg_description=config.get_kg_description())

    if "initial_question" in template.input_variables:
        template = template.partial(initial_question=state["initial_question"])

    if (
        "selected_classes" in template.input_variables
        and "selected_classes" in state.keys()
    ):
        selected_classes_str = ""
        for item in state["selected_classes"]:
            selected_classes_str = f"{selected_classes_str}\n{item}"
        template = template.partial(selected_classes=selected_classes_str)

    has_merged_classes_context = "merged_classes_context" in template.input_variables

    if has_merged_classes_context:
        # Load all the class contexts in a common graph
        merged_graph = get_empty_graph_with_prefixes()
        for cls_context in state["selected_classes_context"]:
            g = Graph()
            merged_graph = merged_graph + g.parse(data=cls_context)

        # Save the graph
        # timestr = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S.%f")[:-3]
        # merged_graph_file = f"{config.get_temp_directory()}/context-{timestr}.ttl"
        # merged_graph.serialize(
        #     destination=merged_graph_file,
        #     format="turtle",
        # )
        # logger.info(f"Graph of selected classes context saved to {merged_graph_file}")
        merged_graph_ttl = merged_graph.serialize(format="turtle")
        template = template.partial(merged_classes_context=merged_graph_ttl)

    if (
        "selected_queries" in template.input_variables
        and "selected_queries" in state.keys()
    ):
        template = template.partial(selected_queries=state["selected_queries"])

    # Make sure there are no more unset input variables
    if template.input_variables:
        raise Exception(
            f"Template has unused input variables: {template.input_variables}"
        )

    prompt = template.format()
    logger.info(f"Query generation prompt created:\n{prompt}.")

    if has_merged_classes_context:
        return {
            "messages": SystemMessage(prompt),
            "merged_classes_context": merged_graph_ttl,
            "query_generation_prompt": prompt,
        }
    else:
        return {
            "messages": SystemMessage(prompt),
            "query_generation_prompt": prompt,
        }


def create_retry_query_generation_prompt(
    template: PromptTemplate, state: OverallState
) -> OverallState:
    """
    Generate a prompt from a template using the inputs available in the current state,
    to retry the query generation task.

    Args:
        template (PromptTemplate): template to use
        state (dict): current state of the conversation

    Returns:
        dict: state updated with the retry prompt (query_generation_prompt)
    """
    # logger.debug(f"Retry query generation prompt template: {template}")

    if "kg_full_name" in template.input_variables:
        template = template.partial(kg_full_name=config.get_kg_full_name())

    if "kg_description" in template.input_variables:
        template = template.partial(kg_description=config.get_kg_description())

    if "initial_question" in template.input_variables:
        template = template.partial(initial_question=state["initial_question"])

    if (
        "merged_classes_context" in template.input_variables
        and "merged_classes_context" in state.keys()
    ):
        template = template.partial(
            merged_classes_context=state["merged_classes_context"]
        )

    if (
        "selected_queries" in template.input_variables
        and "selected_queries" in state.keys()
    ):
        template = template.partial(selected_queries=state["selected_queries"])

    if "last_answer" in template.input_variables:
        template = template.partial(last_answer=state["messages"][-2].content)

    if "last_answer_error_cause" in template.input_variables:
        template = template.partial(
            last_answer_error_cause=state["messages"][-1].content
        )

    # Make sure there are no more unset input variables
    if template.input_variables:
        raise Exception(
            f"Template has unused input variables: {template.input_variables}"
        )

    prompt = template.format()
    logger.info(f"Retry query generation prompt created:\n{prompt}.")
    return {
        "query_generation_prompt": prompt,
    }


def generate_query(state: OverallState):
    """
    Invoke the LLM with the prompt asking to create a SPARQL query
    """
    result = config.get_llm(state["scenario_id"]).invoke(
        state["query_generation_prompt"]
    )
    logger.debug(f"Query generation response:\n{result.content}")
    return {"messages": result}


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


def run_query(state: OverallState) -> OverallState:
    """
    Submit the generated SPARQL query to the endpoint.
    Return the SPARQL CSV results or SPARQL_QUERY_EXEC_ERROR error string in the current state (last_query_results).

    Args:
        state (dict): current state of the conversation

    Returns:
        dict: state updated with last query (last_generated_query) and query results (last_query_results)
    """

    # The last generated query may already be in the state (secnarios 4-6)
    # or in the conversation (scenarios 1-3)
    if "last_generated_query" in state:
        query = state["last_generated_query"]
    else:
        query = find_sparql_queries(state["messages"][-1].content)[0]

    logger.info("Submitting the generated SPARQL query to the endpoint...")
    try:
        csv_result = run_sparql_query(query=query)
        logger.info("SPARQL execution completed.")
        logger.debug(f"Query execution results:\n{csv_result}")
        return {"last_generated_query": query, "last_query_results": csv_result}
    except Exception as e:
        logger.warning(f"SPARQL query executon failed: {e}")
        return {
            "last_generated_query": query,
            "last_query_results": SPARQL_QUERY_EXEC_ERROR,
        }


def interpret_csv_query_results(state: OverallState) -> OverallState:
    """
    Generate a prompt asking the interpret the SPARQL CSV results and invoke the LLM.

    Args:
        state (dict): current state of the conversation

    Returns:
        dict: state updated with the response from the LLM in results_interpretation
    """

    template = interpret_csv_query_results_prompt
    # logger.debug(f"Results interpretation prompt template: {template}")

    if "kg_full_name" in template.input_variables:
        template = template.partial(kg_full_name=config.get_kg_full_name())

    if "kg_description" in template.input_variables:
        template = template.partial(kg_description=config.get_kg_description())

    if "initial_question" in state.keys():
        template = template.partial(initial_question=state["initial_question"])

    sparql_csv_results = state["last_query_results"]
    if "last_query_results" in state.keys():
        template = template.partial(last_query_results=sparql_csv_results)

    # Make sure there are no more unset input variables
    if template.input_variables:
        raise Exception(
            f"Template has unused input variables: {template.input_variables}"
        )

    prompt = template.format()
    logger.info(f"Results interpretation prompt created:\n{prompt}.")
    result = config.get_llm(state["scenario_id"]).invoke(prompt)

    logger.debug(f"Interpretation of the query results:\n{result.content}")
    return OverallState({"messages": result, "results_interpretation": result})
