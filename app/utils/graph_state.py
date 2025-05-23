from enum import Enum
import json
import operator
from typing import Annotated
from openai import BaseModel
from typing_extensions import TypedDict
from langgraph.graph import MessagesState


class InputState(TypedDict):
    """
    Attributes:
        initial_question (str): the question to be answered by the Gen²KGBot workflow.
    """

    initial_question: str


class JudgeStatus(Enum):
    NO_QUERY = "No generated query to judge."
    INVALID_SYNTAX = "Invalid syntax."
    VALID_SYNTAX = "Valid syntax."
    NO_VALID_JSON = "No JSON found in the judging answer."
    JUDGE_LOW_SCORE = "Low score."
    JUDGE_LOW_SCORE_RUN_QUERY = "Low score, run the query."
    JUDGE_LOW_SCORE_END = "Low score, end of the conversation."
    JUDGE_HIGH_SCORE = "High score."


class JudgeState(TypedDict):
    """
    Attributes:

        generated_answer (str): the dump answer generated by the model

        query (str): the query extracted from the `generated_answer` to be judged

        judge_status (JudgeStatus): status of the query judging process

        query_qnames (list[str]): qnames extracted from the query to be judged

        qnames_info (list[str]): information about the qnames extracted from the query to be judged

        judgement (str): judgement of the generated answer

        query_regeneration_prompt (str): prompt to ask for the regeneration of the query
    """

    generated_answer: str

    query: str

    judge_status: JudgeStatus

    query_qnames: list[str]

    qnames_info: list[str]

    judging_grade: int = 0

    judgement: str

    query_regeneration_prompt: str


class OverallState(MessagesState, InputState):
    """
    Attributes:
        scenario_id (str): identifier of the current scenario

        question_validation_results (str): results of checking if the question is clear, answerable, and relevant to the knowledge graph

        question_relevant_entities (list[str]): named entities extracted from the question by spacy

        selected_classes (list[str]):
            classes that are similar to the question and can be relevant to generate a SPARQL query.
            Formatted as "(uri, label, comment)".

        selected_classes_context (list[str]):
            context of the classes in selected_classes, that is, for each class, a serialization describing the properties and value types

        merged_classes_context (str): concatenation of the contexts in selected_classes_context

        selected_queries (str):
            example SPARQL queries selected using the named entities extracted from the question

        query_generation_prompt (str): first prompt with question and context to ask the generation of a SPAQL query

        number_of_tries (int): number of attemps if asking mutliples times to generate a SPARQL query

        last_generated_query (str): last generated SPARQL query

        last_query_results (str): results of executing the last generated SPARQL query

        results_interpretation (str): interpretation of the SPARQL results of the last executed SPARQL query
    """

    scenario_id: str

    question_validation_results: str

    question_relevant_entities: list[str]

    selected_classes: list[str]

    selected_classes_context: Annotated[list[str], operator.add]

    merged_classes_context: str

    selected_queries: str

    query_generation_prompt: str

    number_of_tries: int

    last_generated_query: str

    last_query_results: str

    results_interpretation: str

    query_judgements: Annotated[list[JudgeState], operator.add]


class JudgeGrade(BaseModel):
    """
    Attributes:
        grade (int): grade of the query
        justification (str): justification of the grade
    """

    grade: int
    justification: str


class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value  # Convert Enum to its value (string)
        return super().default(obj)
