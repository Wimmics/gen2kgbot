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


class JudgeState(TypedDict):
    """
    Attributes:

        query (str): the query to be judged

        query_qnames (list[str]): qnames extracted from the query to be judged

        qnames_info (list[str]): information about the qnames extracted from the query to be judged

        query_judgement (str): judgement of the generated query

        query_regeneration_prompt (str): prompt to ask for the regeneration of the query
    """

    query: str

    query_qnames: list[str]

    qnames_info: list[str]

    query_judgement: str

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
        
        agents (list): list of agent instances for the multi-agent debate
        
        agent_proposals (list): current round proposals from all agents
        
        agent_summaries (list): current round summaries from all agents
        
        agent_proposals_history (list): history of proposals from all agents across rounds
        
        agent_summaries_history (list): history of summaries from all agents across rounds
        
        debate_round (int): current debate round number
        
        agent_types (list): list of agent types/roles in the debate
        
        final_proposals (list): final proposals from all agents after debate
        
        kg_description (str): description of the knowledge graph
        
        selected_query (str): the query selected by the moderator
        
        max_debate_rounds (int): maximum number of debate rounds to run
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
    
    agents: list
    
    agent_proposals: list
    
    agent_summaries: list
    
    agent_proposals_history: list
    
    agent_summaries_history: list
    
    debate_round: int
    
    agent_types: list
    
    final_proposals: list
    
    kg_description: str
    
    selected_query: str
    
    max_debate_rounds: int
