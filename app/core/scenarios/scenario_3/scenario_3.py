import argparse
import os
from pathlib import Path
import re
from typing import Literal
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from app.core.scenarios.scenario_3.utils.prompt import system_prompt, interpreter_prompt
from app.core.utils.printing import new_log
from app.core.utils.utils import setup_logger
from rdflib.exceptions import ParserError
from app.core.utils.sparql_toolkit import run_sparql_query
from langchain_community.vectorstores import FAISS

logger = setup_logger(__package__, __file__)

# openai_api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOllama(model="llama3.2:1b")
# llm = ChatOpenAI(
#     model="gpt-4o",
#     openai_api_key=openai_api_key,
# )
faiss_embedding_directory = (
    Path(__file__).resolve().parent.parent.parent.parent.parent
    / "data"
    / "faiss_embeddings"
    / "idsm"
    / "v3_4_full_nomic_faiss_index"
)


def run_query_router(state: MessagesState) -> Literal["interpret_results", END]:
    """
    Check if the query was run successfully and routes to the next step accordingly.

    Args:
        state (MessagesState): The current state of the conversation

    Returns:
        Literal["interpret_results", END]: The next step in the conversation
    """
    if state["messages"][-1].content.find("Error when running the query") == -1:
        logger.info(f"query run succesfully and it yielded")
        return "interpret_results"
    else:
        logger.info(f"Ending the process")
        return END


def generate_query_router(state: MessagesState) -> Literal["run_query", END]:
    if state["messages"][-1].content.find("```sparql") != -1:
        logger.info(f"query generated task completed with a generated SPARQL query")
        return "run_query"
    else:
        logger.warning(
            f"query generated task completed without generating a proper SPARQL query"
        )
        logger.info(f"Ending the process")
        return END


# Node
def select_similar_classes(state: MessagesState) -> MessagesState:

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
    )

    db = FAISS.load_local(
        faiss_embedding_directory,
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )

    query = state["messages"][-1].content

    # Retrieve the most similar text
    retrieved_documents = db.similarity_search(query, k=10)

    result = "These are some relevant classes for the query generation:\n"
    # show the retrieved document's content
    for doc in retrieved_documents:
        result = f"{result}\n{doc.page_content}\n"

    logger.info(f"Done with selecting some similar classes to help query generation")

    return {"messages": AIMessage(result)}


def create_prompt(state: MessagesState):
    result = [system_prompt] + state["messages"]
    state["messages"].clear()
    logger.info(f"prompt created successfuly.")
    return {"messages": result}


def generate_query(state: MessagesState):
    result = llm.invoke(state["messages"])
    return {"messages": result}


def run_query(state: MessagesState):

    query = re.findall(
        "```sparql\n(.*)\n```", state["messages"][-1].content, re.DOTALL
    )[0]

    try:
        csv_result = run_sparql_query(query=query)
        return {"messages": csv_result}
    except ParserError as e:
        logger.warning(f"A parsing error occurred when running the query: {e}")
        return {"messages": AIMessage("Error when running the query")}
    except Exception as e:
        logger.warning(f"An error occurred when running the query: {e}")
        return {"messages": AIMessage("Error when running the query")}


def interpret_results(state: MessagesState):
    csv_results_message = state["messages"][-1]
    result = llm.invoke([interpreter_prompt] + [csv_results_message])
    logger.info(f"the interpretatin of the query result is done")
    return {"messages": result}


s3_builder = StateGraph(MessagesState)


s3_builder.add_node("select_similar_classes", select_similar_classes)
s3_builder.add_node("create_prompt", create_prompt)
s3_builder.add_node("generate_query", generate_query)
s3_builder.add_node("run_query", run_query)
s3_builder.add_node("interpret_results", interpret_results)

s3_builder.add_edge(START, "select_similar_classes")
s3_builder.add_edge("select_similar_classes", "create_prompt")
s3_builder.add_edge("create_prompt", "generate_query")
s3_builder.add_conditional_edges("generate_query", generate_query_router)
s3_builder.add_conditional_edges("run_query", run_query_router)
s3_builder.add_edge("interpret_results", END)

graph = s3_builder.compile()

def run_scenario(question: str):
    return graph.invoke({"messages": HumanMessage(question)})


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