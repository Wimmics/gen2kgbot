import argparse
import ast
from math import log
import operator
import os
from pathlib import Path
import re
from typing import Annotated, List, Literal
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from rdflib import Graph
from app.core.scenarios.scenario_6.utils.prompt import (
    system_prompt,
    interpreter_prompt,
    retry_prompt,
)
from app.core.utils import construct_util
from app.core.utils.printing import new_log
from app.core.utils.utils import setup_logger
from rdflib.exceptions import ParserError
from app.core.utils.sparql_toolkit import run_sparql_query
from langchain_community.vectorstores import FAISS
from app.core.scenarios.scenario_6.utils.preprocessing import (
    extract_relevant_entities_spacy,
)
from langgraph.constants import Send
from langchain_core.documents import Document
import time

from rdflib.plugins.sparql.algebra import translateQuery
from rdflib.plugins.sparql.parser import parseQuery

logger = setup_logger(__package__, __file__)

# openai_api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOllama(model="llama3.2:1b")
# llm = ChatOpenAI(
#                 model="gpt-4o",
#                 openai_api_key=openai_api_key,
#             )
faiss_embedding_classes_directory = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "data"
    / "faiss_embeddings"
    / "idsm"
    / "v3_4_full_nomic_faiss_index"
)

faiss_embedding_query_directory = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "data"
    / "faiss_embeddings"
    / "idsm"
    / "query_v1_nomic_faiss_index"
)

MAX_NUMBER_OF_TRIES: int = 3


# State Class


class OverAllState(MessagesState):
    initial_question: str
    selected_classes: List[Document]
    selected_queries: str

    selected_classes_context: Annotated[list[str], operator.add]
    merged_classes_context: str
    query_generation_prompt: str

    number_of_tries: int
    last_generated_query: str


# Router


def run_query_router(state: OverAllState) -> Literal["interpret_results", END]:
    if state["messages"][-1].content.find("Error when running the query") == -1:
        logger.info(f"query run succesfully and it yielded")
        return "interpret_results"
    else:
        logger.info(f"Ending the process")
        return END


def verify_query_router(
    state: OverAllState,
) -> Literal["create_retry_prompt", "run_query", END]:
    if state["last_generated_query"] != None:
        logger.info(f"query generated task completed with a generated SPARQL query")
        return "run_query"
    else:
        logger.warning(
            f"query generated task completed without generating a proper SPARQL query"
        )
        if state["number_of_tries"] < MAX_NUMBER_OF_TRIES:
            logger.info(f"Tries left {MAX_NUMBER_OF_TRIES - state['number_of_tries']}")
            return "create_retry_prompt"
        else:
            logger.info(f"Max retries ... Ending the process")
        return END


def get_context_class_router(
    state: OverAllState,
) -> Literal["get_context_class_from_cache", "get_context_class_from_kg"]:

    next_nodes = []

    for doc in state["selected_classes"]:
        cls = ast.literal_eval(doc.page_content)
        cls_path = construct_util.format_class_graph_file(cls[0])

        if os.path.exists(cls_path):
            logger.info(f"Classe context file path at {cls_path} found.")
            next_nodes.append(Send("get_context_class_from_cache", cls_path))
        else:
            logger.info(f"Classe context file path at {cls_path} not found.")
            next_nodes.append(Send("get_context_class_from_kg", cls))

    return next_nodes


# Node


def preprocess_question(state: OverAllState) -> OverAllState:
    result = AIMessage(
        f"{extract_relevant_entities_spacy(state["messages"][-1].content)}"
    )
    logger.info(f"Preprocessing the question was done succesfully")
    return {
        "messages": result,
        "initial_question": state["messages"][-1].content,
        "number_of_tries": 0,
    }


def get_context_class_from_cache(cls_path: str) -> OverAllState:
    with open(cls_path) as f:
        return {"selected_classes_context": ["\n".join(f.readlines())]}


def get_context_class_from_kg(cls: str) -> OverAllState:
    graph_ttl = construct_util.get_context_class(cls)
    return {"selected_classes_context": [graph_ttl]}


def select_similar_classes(state: OverAllState) -> OverAllState:

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
    )

    db = FAISS.load_local(
        faiss_embedding_classes_directory,
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )

    query = state["messages"][-1].content

    logger.info(f"query: {query}")

    # Retrieve the most similar text
    retrieved_documents = db.similarity_search(query, k=10)

    result = "These are some relevant classes for the query generation:\n"
    # show the retrieved document's content
    for doc in retrieved_documents:
        result = f"{result}\n{doc.page_content}\n"

    logger.info(f"Done with selecting some similar classes to help query generation")

    return {"messages": AIMessage(result), "selected_classes": retrieved_documents}

def select_similar_query_examples(state: OverAllState) -> OverAllState: 
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
    )

    db = FAISS.load_local(
        faiss_embedding_query_directory,
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )

    query = state["messages"][-1].content

    # Retrieve the most similar text
    retrieved_documents = db.similarity_search(query, k=3)

    result = "These are some relevant queries for the query generation:\n"
    # show the retrieved document's content
    for doc in retrieved_documents:
        result = f"{result}\n```sparql\n{doc.page_content}\n```\n"

    logger.info(f"Done with selecting some similar queries to help query generation")

    return {"messages": AIMessage(result), "selected_queries": result}

def create_prompt(state: OverAllState) -> OverAllState:

    # if "selected_queries" in state and "selected_classes" in state:
        merged_graph = construct_util.get_graph_with_prefixes()

        for cls_context in state["selected_classes_context"]:
            g = Graph()
            merged_graph = merged_graph + g.parse(data=cls_context)

        # Save the graph
        timestr = time.strftime("%Y%m%d-%H%M%S")
        merged_graph.serialize(
            destination=f"{construct_util.tmp_directory}/context-{timestr}.ttl",
            format="turtle",
        )

        merged_graph_ttl = merged_graph.serialize(format="turtle")

        logger.info(
            f"Context graph saved locally in {construct_util.tmp_directory}/context-{timestr}.ttl"
        )
        logger.info(f"prompt created successfuly.")

        query_generation_prompt = (
            f"{system_prompt.content}\n"
            + f"{state['messages'][-1].content}\n"
            + f"{state['selected_queries']}\n"
            + f"The properties and their type when using the classes: \n {merged_graph_ttl}\n\n"
            + f"The user question is: \n\n{state['initial_question']}\n"
        )

        return {
            "merged_classes_context": merged_graph_ttl,
            "query_generation_prompt": query_generation_prompt,
        }


def generate_query(state: OverAllState):
    result = llm.invoke(state["query_generation_prompt"])
    return {"messages": result}


def verify_query(state: OverAllState) -> OverAllState:
    queries = re.findall(
        "```sparql\n(.*)\n```", state["messages"][-1].content, re.DOTALL
    )
    if len(queries) == 0:
        return {
            "number_of_tries": state["number_of_tries"] + 1,
            "messages": [HumanMessage("No properly formatted SPARQL query was generated.")],
            "last_generated_query": None
        }

    try:
        translateQuery(parseQuery(construct_util.add_known_prefixes_to_query(queries[0])))
    except Exception as e:
        return {
            "number_of_tries": state["number_of_tries"] + 1,
            "messages": [AIMessage(f"{e}")],
            "last_generated_query": None
        }

    return {"last_generated_query": queries[0]}


def create_retry_prompt(state: OverAllState) -> OverAllState:
    logger.info(f"retry_prompt created successfuly.")

    query_regeneration_prompt = (
        f"{retry_prompt.content}\n\n"
        + f"The properties and their type when using the classes: \n {state["merged_classes_context"]}\n\n"
        + f"{state['selected_queries']}\n\n"
        + f"The user question:\n{state['initial_question']}\n\n"
        + f"The last answer you provided that either don't contain or have a unparsable SPARQL query:\n"
        + f"-------------------------------------\n{state['messages'][-2].content}\n--------------------------------------------------\n\n"
        + f"The verification didn't pass because:\n-------------------------\n{state["messages"][-1].content}\n--------------------------------\n"
    )

    return {
        "query_generation_prompt": query_regeneration_prompt,
    }


def run_query(state: OverAllState):

    query = state["last_generated_query"]

    try:
        csv_result = run_sparql_query(query=query)
        return {"messages": csv_result}
    except ParserError as e:
        logger.warning(f"A parsing error occurred when running the query: {e}")
        return {"messages": AIMessage("Error when running the query")}
    except Exception as e:
        logger.warning(f"An error occurred when running the query: {e}")
        return {"messages": AIMessage("Error when running the query")}


def interpret_results(state: OverAllState):
    csv_results_message = state["messages"][-1]
    result = llm.invoke([interpreter_prompt] + [csv_results_message])
    logger.info(f"the interpretatin of the query result is done")
    return {"messages": result}


s6_builder = StateGraph(OverAllState)

s6_builder.add_node("preprocess_question", preprocess_question)
s6_builder.add_node("select_similar_classes", select_similar_classes)
s6_builder.add_node("get_context_class_from_cache", get_context_class_from_cache)
s6_builder.add_node("get_context_class_from_kg", get_context_class_from_kg)

s6_builder.add_node("select_similar_query_examples", select_similar_query_examples)

s6_builder.add_node("create_prompt", create_prompt)
s6_builder.add_node("generate_query", generate_query)
s6_builder.add_node("run_query", run_query)

s6_builder.add_node("verify_query", verify_query)
s6_builder.add_node("create_retry_prompt", create_retry_prompt)

s6_builder.add_node("interpret_results", interpret_results)

s6_builder.add_edge(START, "preprocess_question")
s6_builder.add_edge("preprocess_question", "select_similar_query_examples")
s6_builder.add_edge("preprocess_question", "select_similar_classes")
s6_builder.add_edge("select_similar_query_examples", "create_prompt")
s6_builder.add_conditional_edges("select_similar_classes", get_context_class_router)
s6_builder.add_edge("get_context_class_from_cache", "create_prompt")
s6_builder.add_edge("get_context_class_from_kg", "create_prompt")
# s6_builder.add_edge(["select_similar_query_examples","get_context_class_from_cache","get_context_class_from_kg"], "create_prompt")
s6_builder.add_edge("create_prompt", "generate_query")
s6_builder.add_edge("generate_query", "verify_query")
s6_builder.add_conditional_edges("verify_query", verify_query_router)
s6_builder.add_edge("create_retry_prompt", "generate_query")
s6_builder.add_conditional_edges("run_query", run_query_router)
s6_builder.add_edge("interpret_results", END)

graph = s6_builder.compile()


def main():

    parser = argparse.ArgumentParser(description="Process the scenario with the predifined or custom question.")
    
    parser.add_argument('-c', '--custom', type=str,
                        help="Provide a custom question.")
    
    args = parser.parse_args()
    
    if args.custom:
        question = args.custom
    else:
        question = "What protein targets does donepezil (CHEBI_53289) inhibit with an IC50 less than 10 µM?"
    
    state = graph.invoke({"messages":HumanMessage(question)})

    new_log()
    for m in state["messages"]:
        m.pretty_print()
    new_log()


if __name__ == "__main__":
    main()