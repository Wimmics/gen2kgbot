import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import MessagesState


class InputState(TypedDict):
    """
    Attributes:
        initial_question (str): the question to be answered by the Gen²KGBot workflow.
    """
    initial_question: str


class OverallState(MessagesState, InputState):
    """
    Attributes:
        scenario_id (str): identifier of the current scenario

        question_relevant_entities (list[str]): named entities extracted from the question by spacy

        selected_classes (list[str]):
            classes that are similar to the question and can be relevant to generate a SPARQL query

        selected_classes_context (list[str]):
            context of the classes in in selected_classes, that is, for each class a serialization describing the properties and value types

        merged_classes_context (str): concatenation of the contexts in selected_classes_context

        selected_classes_properties (list[str]):
            label and description of the properties used with instances of the classes in selected_classes

        merged_classes_context (str): concatenation of the contexts in selected_classes_properties after eliminating duplicates

        selected_queries (str):
            example SPARQL queries selected using the named entities extracted from the question

        query_generation_prompt (str): first prompt with question and context to ask the generation of a SPAQL query

        number_of_tries (int): number of attemps if asking mutliples times to generate a SPARQL query

        last_generated_query (str): last generated SPARQL query

        last_query_results (str): results of executing the last generated SPARQL query

        results_interpretation (str): interpretation of the SPARQL results of the last executed SPARQL query
    """

    scenario_id: str

    question_relevant_entities: list[str]

    selected_classes: list[str]

    selected_classes_context: Annotated[list[str], operator.add]
    merged_classes_context: str

    selected_classes_properties: Annotated[list[str], operator.add]
    merged_classes_properties: str

    selected_queries: str

    query_generation_prompt: str

    number_of_tries: int

    last_generated_query: str
    last_query_results: str

    results_interpretation: str
