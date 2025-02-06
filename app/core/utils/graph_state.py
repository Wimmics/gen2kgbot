import operator
from typing import Annotated, TypedDict
from langgraph.graph import MessagesState
from langchain_core.documents import Document


class InputState(TypedDict):
    initial_question: str


class OverallState(MessagesState, InputState):
    """
    Attributes:
        last_generated_query (str): last generated SPARQL query
        last_query_results (str): results of executing the last generated SPARQL query
    """

    question_relevant_entities: list[str]
    selected_queries: str
    selected_classes: list[Document]
    selected_classes_context: Annotated[list[str], operator.add]
    merged_classes_context: str
    query_generation_prompt: str
    number_of_tries: int
    last_generated_query: str
    last_query_results: str
    results_interpretation: str
