"""
This module implements the Langgraph nodes that are common to multiple scenarios
"""

from datetime import timezone, datetime
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from app.utils.graph_state import JudgeStatus, OverallState
from app.utils.question_preprocessing import extract_relevant_entities_spacy
from app.utils.sparql_toolkit import find_sparql_queries, run_sparql_query
import app.utils.config_manager as config
from app.utils.construct_util import (
    get_class_context,
    get_connected_classes,
    get_empty_graph_with_prefixes,
    add_known_prefixes_to_query,
    fulliri_to_prefixed,
)
from app.utils.prompts import (
    interpret_results_prompt,
    validate_question_prompt,
)
from rdflib import Graph
from rdflib.plugins.sparql.parser import parseQuery
from rdflib.plugins.sparql.algebra import translateQuery

logger = config.setup_logger(__package__, __file__)

SPARQL_QUERY_EXEC_ERROR = "Error when running the SPARQL query"


def preprocess_question(state: OverallState) -> OverallState:
    """
    Extract named entities from the user question.

    Args:
        state (dict): current state of the conversation

    Returns:
        dict: state updated with initial_question, number_of_tries,
            and question_relevant_entities that contains the comma-separated list of named entities
    """

    logger.debug("Preprocessing the question...")
    relevant_entities_list = extract_relevant_entities_spacy(state["initial_question"])
    relevant_entities = f"{", ".join(relevant_entities_list)}"
    logger.info(f"Extracted following named entities: {relevant_entities}")

    return {
        "messages": AIMessage(relevant_entities),
        "question_relevant_entities": relevant_entities_list,
        "number_of_tries": 0,
    }


def select_similar_classes(state: OverallState) -> OverallState:
    """
    Retrieve, from the vector db, the descritption of ontology classes
    related to the named entities extracted from the question

    Args:
        state (dict): current state of the conversation

    Returns:
        dict: state updated with selected_classes
    """

    db = config.get_class_context_vector_db(state["scenario_id"])

    relevant_entities_list = state["question_relevant_entities"]
    logger.info(
        f"Looking for classes related to the question's named entities: {", ".join(relevant_entities_list)}"
    )

    # 1st, retrieve from the vector DB the classes whose label is exactly the same as one named entity
    classes_str = []
    for entity in relevant_entities_list:
        matches = db.similarity_search(entity, k=1)
        if len(matches) > 0:
            if eval(matches[0].page_content)[1].lower() == entity.lower():
                classes_str.append(matches[0].page_content)

    # Then, retrieve from the vector DB the classes similar to all the entites together
    for doc in db.similarity_search(
        ", ".join(relevant_entities_list), k=config.get_max_similar_classes()
    ):
        classes_str.append(fulliri_to_prefixed(doc.page_content))


    classes_str = list(set(classes_str))
    logger.info(f"Found {len(classes_str)} classes related to the question.")
    logger.debug(
        f"Classes found: {", ".join([f"{eval(cls)[0]} ({eval(cls)[1]})" for cls in classes_str])}"
    )

    # Extend the initial list of classes by retrieving, from the KG, additional classes connected to the initial ones
    if config.expand_similar_classes():
        classes_uris = [eval(cls)[0] for cls in classes_str]
        for cls, label, description in get_connected_classes(classes_uris):
            if cls not in classes_uris:
                cls_tuple = (cls, label, description)
                classes_str.append(f"{fulliri_to_prefixed(str(cls_tuple))}")
        logger.info(f"Expanded to {len(classes_str)} classes related to the question.")

    classes_str = list(set(classes_str))

    # Filter out classes marked as to be excluded
    classes_filtered_str = []
    for cls in classes_str:
        keep_cls = True
        for excluded_class in config.get_excluded_classes_namespaces():
            if cls.find(excluded_class) != -1:
                keep_cls = False
                break
        if keep_cls:
            classes_filtered_str.append(cls)

    logger.info(f"Keeping {len(classes_filtered_str)} classes after filtering.")
    return {"selected_classes": classes_filtered_str}


def get_class_context_from_cache(cls_path: str) -> OverallState:
    """
    Retrieve a class context from the cache

    Args:
        cls_path (str): path to the class context file

    Returns:
        dict: state with selected_classes_context .
            This will be added to the current context.
    """
    cls_f = open(cls_path, "r", encoding="utf8")
    return {"selected_classes_context": ["".join(cls_f.readlines())]}


def get_class_context_from_kg(cls: tuple) -> OverallState:
    """
    Retrieve a class context from the knowledge graph,
    i.e., a description of the properties that instances of a class have.
    This includes triples/tuples (class uri, property uri, type), and
    tuples (property uri, label, description).

    Args:
        cls (tuple): (class URI, label, description)

    Returns:
        dict: state with selected_classes_context.
            These will be added to the current context.
    """
    return {"selected_classes_context": [get_class_context(cls)]}


def select_similar_query_examples(state: OverallState) -> OverallState:
    """
    Retrieve the SPARQL queries most similar to the question

    Args:
        state (dict): current state of the conversation

    Returns:
        dict: state updated messages and queries (selected_queries)
    """
    question = ", ".join(state["question_relevant_entities"])
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


def class_description_tuple_to_nl(description: str) -> str:
    """
    Convert a class description formatted as a tuple into natural language

    Args:
        description (str): class description as a tuplr "(class uri, label, description)"
    """
    uri, label, descr = eval(description)
    serialization = f"Class '{uri}':\n"
    if label is not None:
        serialization += f"  - Label: '{label}'\n"
    if descr is not None:
        serialization += f"  - Description: '{descr}'\n"
    return serialization


def class_context_tuple_to_nl(context: str) -> str:
    """
    Convert a class context formatted as a tuple into natural language

    Args:
        context (str): class context as a tuple "(class uri, property uri, property label, value type)"
    """
    serialization = ""
    for c in context.splitlines():
        uri, prop_uri, prop_label, value_type = eval(c)
        serialization += f"Instances of class '{uri}' have property '{prop_uri}' ({prop_label}) with value type '{value_type}'.\n"
    return serialization


def create_query_generation_prompt(
    template: PromptTemplate, state: OverallState
) -> OverallState:
    """
    Generate a prompt from a template using the inputs available in the current state.
    This function does not assume the presence of input variable in the template,
    and simply replaces them if they are present.
    Depending on the scenario, and wether this is a 1st time generation or a retry,
    the inputs variables may not be the same.

    Args:
        template (PromptTemplate): template to use
        state (dict): current state of the conversation

    Returns:
        dict: state updated with the prompt generated (query_generation_prompt)
            and optionally the class contexts all merged (merged_classes_context)
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
            selected_classes_str += (
                f"\n{class_description_tuple_to_nl(fulliri_to_prefixed(item))}"
            )
        template = template.partial(selected_classes=selected_classes_str)

    if (
        "selected_queries" in template.input_variables
        and "selected_queries" in state.keys()
    ):
        template = template.partial(selected_queries=state["selected_queries"])

    # Manage the detailed context of selected classes
    has_merged_classes_context = "merged_classes_context" in template.input_variables
    if has_merged_classes_context:

        if "merged_classes_context" in state.keys():
            # This is a retry, state["merged_classes_context"] has already been set during the previous attempt
            merged_cls_context = state["merged_classes_context"]
        else:
            # This is the first attempt, merge all class contexts together
            if config.get_class_context_format() == "turtle":
                # Load all the class contexts in a common graph
                merged_graph = get_empty_graph_with_prefixes()
                for cls_context in state["selected_classes_context"]:
                    merged_graph = merged_graph + Graph().parse(data=cls_context)
                # save_full_context(merged_graph)
                merged_cls_context = (
                    "```turtle\n" + merged_graph.serialize(format="turtle") + "```"
                )

            elif config.get_class_context_format() == "tuple":
                merged_cls_context = ""
                merged_cls_context = "Format is: ('class uri', 'property uri', 'property label', 'value type')\n"
                for cls_context in state["selected_classes_context"]:
                    if cls_context not in ["", "\n"]:
                        merged_cls_context += f"\n{fulliri_to_prefixed(cls_context)}"

            elif config.get_class_context_format() == "nl":
                merged_cls_context = ""
                for cls_context in state["selected_classes_context"]:
                    if cls_context not in ["", "\n"]:
                        merged_cls_context += f"\n{class_context_tuple_to_nl(fulliri_to_prefixed(cls_context))}"

            else:
                raise ValueError(
                    f"Invalid requested format for class context: {format}"
                )

        template = template.partial(merged_classes_context=merged_cls_context)

    # Keep track of wether this is a retry or a first attempt
    is_retry = (
        "last_answer" in template.input_variables
        or "last_answer_error_cause" in template.input_variables
    )

    # If retry, add the answer previously given by the model, and that was incorrect
    if "last_answer" in template.input_variables:
        template = template.partial(last_answer=state["messages"][-2].content)

    # If retry, add the cause for the last error
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
    result_state = {
        "query_generation_prompt": prompt,
        "messages": SystemMessage(prompt),
    }
    if is_retry:
        logger.info(f"Retry query generation prompt created:\n{prompt}.")
    else:
        logger.info(f"First-time query generation prompt created:\n{prompt}.")
        if has_merged_classes_context:
            result_state["merged_classes_context"] = merged_cls_context

    return result_state


async def generate_query(state: OverallState):
    """
    Invoke the LLM with the prompt asking to create a SPARQL query
    """
    result = await config.get_seq2seq_model(
        scenario_id=state["scenario_id"], node_name="generate_query"
    ).ainvoke(state["query_generation_prompt"])
    # logger.debug(f"Query generation response:\n{result.content}")
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
    if (
        "query_judgements" in state
        and len(state["query_judgements"]) > 0
        and state["query_judgements"][-1]["judge_status"]
        in [
            JudgeStatus.JUDGE_HIGH_SCORE,
            JudgeStatus.JUDGE_LOW_SCORE_RUN_QUERY,
        ]
    ):
        query = state["query_judgements"][-1]["query"]
    elif "last_generated_query" in state:
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


async def interpret_results(state: OverallState) -> OverallState:
    """
    Generate a prompt asking the interpret the SPARQL CSV results and invoke the LLM.

    Args:
        state (dict): current state of the conversation

    Returns:
        dict: state updated with the response from the LLM in results_interpretation
    """

    template = interpret_results_prompt
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
    result = await config.get_seq2seq_model(
        scenario_id=state["scenario_id"], node_name="interpret_results"
    ).ainvoke(prompt)

    logger.debug(f"Interpretation of the query results:\n{result.content}")
    return OverallState({"messages": result, "results_interpretation": result.content})


def save_full_context(graph: Graph):
    timestr = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S.%f")[:-3]
    graph_file = f"{config.get_temp_directory()}/context-{timestr}.ttl"
    graph.serialize(
        destination=graph_file,
        format="turtle",
    )
    logger.info(f"Graph of selected classes context saved to {graph_file}")


async def validate_question(state: OverallState) -> OverallState:
    """
    Check if a question fits the context of the current Knowledge Graph. Used in all the scenarios.

    Args:
        state (dict): current state of the conversation

    Return:
        dict: state updated with the validation result (question_validation_results)
    """

    logger.info("Validating the question ...")

    template = validate_question_prompt

    if "kg_full_name" in template.input_variables:
        template = template.partial(kg_full_name=config.get_kg_full_name())

    if "kg_description" in template.input_variables:
        template = template.partial(kg_description=config.get_kg_description())

    if "initial_question" in state.keys():
        template = template.partial(initial_question=state["initial_question"])

    # Make sure there are no more unset input variables
    if template.input_variables:
        raise Exception(
            f"Template has unused input variables: {template.input_variables}"
        )

    prompt = template.format()

    logger.debug(f"Validation question prompt created:\n{prompt}.")
    result = await config.get_seq2seq_model(
        scenario_id=state["scenario_id"], node_name="validate_question"
    ).ainvoke(prompt)

    logger.debug(f"Validation of the question results: {result.content}")

    return OverallState(
        {"messages": result, "question_validation_results": result.content}
    )
