import operator
from typing import Annotated, TypedDict
from langgraph.graph import MessagesState


class InputState(TypedDict):
    initial_question: str


class OverallState(MessagesState, InputState):
    """
    Attributes:
        question_relevant_entities (list[str]): named entities extracted from the question by spacy
        selected_queries (str):
        selected_classes (list[str]): 
            classes retrieved from the vector DB, that are similar to the question and can be relevant to generate a SPARQL query
        selected_classes_context (list[str]): 
            context of the selected classes, that is, for each class a Turtle graph describing the properties and value types
        merged_classes_context (str)
        query_generation_prompt (str)
        number_of_tries (int)
        last_generated_query (str): last generated SPARQL query
        last_query_results (str): results of executing the last generated SPARQL query
        results_interpretation (str): interpretation of the SPARQL results of the last executed SPARQL query
    """

    question_relevant_entities: list[str]
    selected_queries: str
    selected_classes: list[str]
    selected_classes_context: Annotated[list[str], operator.add]
    merged_classes_context: str
    query_generation_prompt: str
    number_of_tries: int
    last_generated_query: str
    last_query_results: str
    results_interpretation: str
