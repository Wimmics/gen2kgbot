import asyncio
import re
from typing import Literal
from langgraph.graph import StateGraph, START, END
from pydantic import ValidationError
from rdflib import Graph
from app.core.scenarios.scenario_7.prompt import (
    system_prompt_template,
    error_cause_no_query,
    retry_system_prompt_template,
    judge_query_prompt,
)
from app.core.utils.construct_util import (
    add_known_prefixes_to_query,
    fulliri_to_prefixed,
    get_empty_graph_with_prefixes,
    run_sparql_query,
)
from app.core.utils.graph_nodes import (
    class_context_tuple_to_nl,
    class_description_tuple_to_nl,
    preprocess_question,
    select_similar_classes,
    get_class_context_from_cache,
    get_class_context_from_kg,
    select_similar_query_examples,
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
from langchain_core.prompts import PromptTemplate
from app.core.utils.graph_state import (
    InputState,
    JudgeGrade,
    JudgeState,
    JudgeStatus,
    OverallState,
)
import app.core.utils.config_manager as config
from app.core.utils.logger_manager import setup_logger
from rdflib.plugins.sparql.parser import parseQuery
from rdflib.plugins.sparql.algebra import translateQuery
from langchain_core.messages import AIMessage, SystemMessage
from app.core.utils.sparql_toolkit import find_json, find_sparql_queries


logger = setup_logger(__package__, __file__)

SCENARIO = "scenario_7"


# Routers


def validate_sparql_syntax_router(
    state: OverallState,
) -> Literal["extract_query_qnames", "judge_regeneration_prompt", "__end__"]:
    """
    Decide whether to continue judging the query, retry or stop if max number of attemps is reached.

    Args:
        state (OverallState): current state of the conversation

    Returns:
        Literal["extract_query_qnames", "judge_regeneration_prompt", END]: next step in the conversation
    """
    if state["query_judgements"][-1]["judge_status"] == JudgeStatus.VALID_SYNTAX:
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
    if state["query_judgements"][-1]["judge_status"] in [
        JudgeStatus.JUDGE_HIGH_SCORE,
        JudgeStatus.JUDGE_LOW_SCORE_RUN_QUERY,
    ]:
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

    last_judge_status = state["query_judgements"][-1]["judge_status"]

    if last_judge_status == JudgeStatus.JUDGE_HIGH_SCORE:
        return END

    else:

        if state["number_of_tries"] < MAX_NUMBER_OF_TRIES:
            logger.info("The generated SPARQL query needs improvement.")
            logger.info(
                f"Retries left: {MAX_NUMBER_OF_TRIES - state['number_of_tries']}"
            )
            return "judge_regeneration_prompt"

        else:

            judging_grade_threshold_run = config.get_judging_grade_threshold_run()
            last_judge_grade = state["query_judgements"][-1]["judging_grade"]

            if last_judge_grade < judging_grade_threshold_run:
                logger.info("Max retries reached. The generated SPARQL query needs improvement. And it is not good enough to run.")
                state["query_judgements"][-1]["judge_status"] = JudgeStatus.JUDGE_LOW_SCORE_END
            else:
                logger.info("Max retries reached. The generated SPARQL query needs improvement. But it is good enough to run.")
                state["query_judgements"][-1]["judge_status"] = JudgeStatus.JUDGE_LOW_SCORE_RUN_QUERY

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

    Used in scenarios 7.

    Args:
        state (dict): current state of the conversation

    Return:
        dict: state updated with the extracted QNames
    """

    query_judgement = state["query_judgements"].pop()

    # Extract WHERE clause
    where_clause = extract_where_clause(query_judgement["query"])

    # Extract QNames and variables
    qnames = extract_qnames(where_clause)

    logger.info("Done extracting QNames from the generated query.")

    logger.debug(f"Extracted following QNames: {qnames}")

    query_judgement["query_qnames"] = qnames

    return {
        "query_judgements": [query_judgement],
        "messages": AIMessage("Done extracting QNames from the generated query"),
    }


def extract_where_clause(sparql_query):
    """
    Extract the WHERE clause from a SPARQL query.

    Args:
        sparql_query (str): a SPARQL query

    Return:
        str: the content of the WHERE clause
    """
    match = re.search(r"WHERE\s*\{(.*)\}", sparql_query, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else None


def extract_qnames(where_clause):
    """
    Extract QNames from a WHERE clause.

    Args:
        where_clause (str): the content of a WHERE clause

    Return:
        list: a list of QNames
    """
    qname_pattern = r"([a-zA-Z0-9_]+:[a-zA-Z0-9_]+)|(<[^>]+>)"  # Match QName, URIs
    qnames = re.findall(qname_pattern, where_clause)

    # Return the first matching group (QName or URI)
    return [item[0] or item[1] for item in qnames]


def validate_sparql_syntax(state: OverallState) -> OverallState:
    """
    Check if a query was generated and if it is syntactically correct.
    If more than one query was produced, just log a warning and process the first one.

    Used in scenarios 7.

    Args:
        state (dict): current state of the conversation

    Return:
        dict: state updated
    """

    generated_answer = state["messages"][-1].content
    queries = find_sparql_queries(state["messages"][-1].content)
    no_queries = len(queries)

    if no_queries == 0:
        logger.info(error_cause_no_query)
        query_judgement = JudgeState(
            judge_status=JudgeStatus.NO_QUERY,
            generated_answer=generated_answer,
            judgement=error_cause_no_query,
        )

        return {
            "number_of_tries": state["number_of_tries"] + 1,
            "query_judgements": [query_judgement],
            "messages": [AIMessage(error_cause_no_query)],
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
        query_judgement = JudgeState(
            judge_status=JudgeStatus.INVALID_SYNTAX,
            query=queries[0],
            judgement=e,
            generated_answer=generated_answer,
        )
        return {
            "number_of_tries": state["number_of_tries"] + 1,
            "query_judgements": [query_judgement],
            "messages": [AIMessage(f"{e}")],
        }

    logger.info("The generated SPARQL query is syntactically correct.")
    query_judgement = JudgeState(
        judge_status=JudgeStatus.VALID_SYNTAX,
        query=queries[0],
        generated_answer=generated_answer,
    )
    return {
        "query_judgements": [query_judgement],
        "messages": [AIMessage("The generated SPARQL query is syntactically correct.")],
    }


def find_qnames_info(state: OverallState):
    """
    Retrieve the properties and value types for the QNames.
    
    Used in scenarios 7.

    Args:
        state (dict): current state of the conversation

    Return:
        dict: state updated with additional info about the QNames

    """

    endpoint = config.get_ontologies_sparql_endpoint_url()
    properties = config.get_properties_qnames_info()
    query_judgement = state["query_judgements"].pop()

    qnames = query_judgement["query_qnames"]

    qnameInfo = []

    for item in qnames:
        qnameInfo.append(getPropertyDetails(item, endpoint, properties))

    logger.info("Done retrieving properties for the QNames.")

    for i in range(len(qnameInfo)):
        logger.debug(f"QName: {qnames[i]}\n{qnameInfo[i]}\n\n")

    query_judgement["qnames_info"] = qnameInfo

    return OverallState(
        {
            "messages": AIMessage(
                f"Done retrieving properties for the QNames.\n {qnameInfo}"
            ),
            "query_judgements": [query_judgement],
        }
    )


def getPropertyDetails(
    propertyUri: str, sparqlEndpoint: str, listOfProperties: list[str]
):
    """
    Retrieve some properties for a given QName.

    Args:
        propertyUri (str): the QName to retrieve properties for
        sparqlEndpoint (str): the SPARQL endpoint to query
        listOfProperties (list): the list of properties to retrieve

    Return:
        str: the properties for the provided QName
    """
    sparqlQuery: str = """PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX dc: <http://purl.org/dc/elements/1.1/description>
    PREFIX dcterms: <http://purl.org/dc/terms/>
    PREFIX obo: <http://purl.obolibrary.org/obo/>
    SELECT """

    for i in range(len(listOfProperties)):
        sparqlQuery += f"?p{i} "

    sparqlQuery += "WHERE {\n"

    for i in range(len(listOfProperties)):
        sparqlQuery += f"OPTIONAL {{ <{propertyUri}> {listOfProperties[i]} ?p{i} .}}\n"

    sparqlQuery += "}"

    formated_results = []
    _sparql_results = run_sparql_query(sparqlQuery, sparqlEndpoint)
    if _sparql_results is None:
        logger.warning(
            f"Failed to retrieve the properties and value types for property {propertyUri}"
        )
    else:
        for result in _sparql_results:
            for i in range(len(listOfProperties)):
                if f"p{i}" in result:
                    formated_results.append(
                        f"{listOfProperties[i]}: {result[f"p{i}"]["value"]}"
                    )

        logger.debug(
            f"Retrieved {len(formated_results)} properties for QName {propertyUri}"
        )

    return "\n".join(set(formated_results))


async def judge_query(state: OverallState):
    """
    Judge the generated query. If no valid JSON (conforming to `JudgeGrade` schema) is found in the response,
    the judgement is set to `NO_VALID_JSON`.

    Otherwise, the judgement is set to `JUDGE_LOW_SCORE` if the grade is below the threshold, and `JUDGE_HIGH_SCORE` otherwise.

    Used in scenarios 7.

    Args:
        state (dict): current state of the conversation

    Return:
        dict: state updated with the judgement of the generated query
    """

    llm = config.get_seq2seq_model(scenario_id=SCENARIO, node_name="judge_query")
    judging_grade_threshold_retry = config.get_judging_grade_threshold_retry()
    query_judgement = state["query_judgements"].pop()

    try:
        # Invoke the LLM with the judging prompt
        result = await llm.ainvoke(
            judge_query_prompt.format(
                initial_question=state["initial_question"],
                sparql=query_judgement["query"],
                qname_info=query_judgement["qnames_info"],
            )
        )

        # Extract the grade and justification from the response
        grade = find_json(result.content)

        if len(grade) > 0:

            # Validate the extracted JSON
            data = JudgeGrade.model_validate_json(grade[0])

            if data.grade >= judging_grade_threshold_retry:
                logger.info("The query passed the judging process.")
                query_judgement["judge_status"] = JudgeStatus.JUDGE_HIGH_SCORE

            else:
                logger.info("The query did not pass the judging process.")
                query_judgement["judge_status"] = JudgeStatus.JUDGE_LOW_SCORE

            logger.debug(f"Grade: {data.grade} - Justification: {data.justification}")

        else:

            # No JSON found in the response
            logger.debug("No JSON found in the response.")
            query_judgement["judge_status"] = JudgeStatus.NO_VALID_JSON

    except ValidationError as e:
        logger.debug(f"Judge Schema not valid: {e}")
        query_judgement["judge_status"] = JudgeStatus.NO_VALID_JSON

    # Add the judgement to the query_judgement
    query_judgement["judgement"] = result.content

    return OverallState({"messages": [result], "query_judgements": [query_judgement]})


def judge_regeneration_prompt(state: OverallState):
    return create_query_generation_prompt(retry_system_prompt_template, state)


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
                # merged_cls_context = "Format is: ('class uri', 'property uri', 'property label', 'value type')\n"
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
        template = template.partial(
            last_answer=state["query_judgements"][-1]["generated_answer"]
        )

    # If retry, add the cause for the last error
    if "last_answer_error_cause" in template.input_variables:
        if state["query_judgements"][-1]["judge_status"] in [
            JudgeStatus.INVALID_SYNTAX,
            JudgeStatus.NO_QUERY,
            JudgeStatus.JUDGE_LOW_SCORE,
        ]:
            template = template.partial(
                last_answer_error_cause=state["query_judgements"][-1]["judgement"]
            )

    # Make sure there are no more unset input variables
    if template.input_variables:
        raise Exception(
            f"Template has unused input variables: {template.input_variables}"
        )

    prompt = template.format()

    if is_retry:
        query_judgement = state["query_judgements"].pop()
        query_judgement["query_regeneration_prompt"] = prompt
        result_state = {
            "query_judgements": [query_judgement],
            "messages": SystemMessage(prompt),
        }

        logger.info(f"Retry query generation prompt created:\n{prompt}.")
    else:
        result_state = {
            "query_generation_prompt": prompt,
            "messages": SystemMessage(prompt),
        }
        logger.info(f"First-time query generation prompt created:\n{prompt}.")
        if has_merged_classes_context:
            result_state["merged_classes_context"] = merged_cls_context

    return result_state


async def judge_regenerate_query(state: OverallState):
    """
    Invoke the LLM with the regeneration prompt asking to create a SPARQL query
    """
    result = await config.get_seq2seq_model(
        scenario_id=state["scenario_id"], node_name="judge_regenerate_query"
    ).ainvoke(state["query_judgements"][-1]["query_regeneration_prompt"])
    # logger.debug(f"Query generation response:\n{result.content}")
    return {"messages": result}


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

judge_builder.add_node("validate_sparql_syntax", validate_sparql_syntax)
judge_builder.add_node("extract_query_qnames", extract_query_qnames)
judge_builder.add_node("find_qnames_info", find_qnames_info)
judge_builder.add_node("judge_query", judge_query)
judge_builder.add_node("judge_regeneration_prompt", judge_regeneration_prompt)
judge_builder.add_node("judge_regenerate_query", judge_regenerate_query)


judge_builder.add_edge(START, "validate_sparql_syntax")
judge_builder.add_conditional_edges(
    "validate_sparql_syntax", validate_sparql_syntax_router
)
judge_builder.add_edge("extract_query_qnames", "find_qnames_info")
judge_builder.add_edge("find_qnames_info", "judge_query")
judge_builder.add_conditional_edges("judge_query", judge_query_router)
judge_builder.add_edge("judge_regeneration_prompt", "judge_regenerate_query")
judge_builder.add_edge("judge_regenerate_query", "validate_sparql_syntax")


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
