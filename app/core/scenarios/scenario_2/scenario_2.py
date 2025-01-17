import argparse
import os
import re
from typing import Literal
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from app.core.scenarios.scenario_2.utils.prompt import system_prompt, interpreter_prompt
from app.core.utils.printing import new_log
from app.core.utils.utils import get_llm_from_config, main, setup_logger
from rdflib.exceptions import ParserError
from app.core.utils.sparql_toolkit import run_sparql_query

logger = setup_logger(__package__, __file__)


SPARQL_QUERY_EXEC_ERROR = "SPARQL query execution failed"
SCENARIO = "scenario_2"

llm = get_llm_from_config(SCENARIO)

# Router
def run_query_router(state: MessagesState) -> Literal["interpret_results", END]:
    if state["messages"][-1].content.find(SPARQL_QUERY_EXEC_ERROR) == -1:
        logger.info(f"query executed successfully")
        return "interpret_results"
    else:
        logger.info(f"Ending the process")
        return END


def generate_query_router(state: MessagesState) -> Literal["run_query", END]:
    if state["messages"][-1].content.find("```sparql") != -1:
        logger.info(f"query generation task completed successfully")
        return "run_query"
    else:
        logger.warning(
            f"query generation task completed without generating a proper SPARQL query"
        )
        logger.info(f"Ending the process")
        return END


# Node
def create_prompt(state: MessagesState):
    result = [system_prompt] + state["messages"]
    state["messages"].clear()
    return {"messages": result}


def generate_query(state: MessagesState):
    result = llm.invoke(state["messages"])
    return {"messages": result}


def run_query(state: MessagesState):
    query = re.findall(
        "```sparql\n(.*)\n```", state["messages"][-1].content, re.DOTALL
    )[0]
    logger.info(f"Executing SPARQL query extracted from llm's response: \n{query}")

    try:
        csv_result = run_sparql_query(query=query)
        return {"messages": csv_result}
    except ParserError as e:
        logger.warning(f"A parsing error occurred when running the query: {e}")
        return {"messages": AIMessage(SPARQL_QUERY_EXEC_ERROR)}
    except Exception as e:
        logger.warning(f"An error occurred when running the query: {e}")
        return {"messages": AIMessage(SPARQL_QUERY_EXEC_ERROR)}


def interpret_results(state: MessagesState):
    csv_results_message = state["messages"][-1]
    result = llm.invoke([interpreter_prompt] + [csv_results_message])
    logger.info(f"Interpretation of query results by llm: {result.content}")
    return {"messages": result}


s2_builder = StateGraph(MessagesState)

s2_builder.add_node("create_prompt", create_prompt)
s2_builder.add_node("generate_query", generate_query)
s2_builder.add_node("run_query", run_query)
s2_builder.add_node("interpret_results", interpret_results)

s2_builder.add_edge(START, "create_prompt")
s2_builder.add_edge("create_prompt", "generate_query")
s2_builder.add_conditional_edges("generate_query", generate_query_router)
s2_builder.add_conditional_edges("run_query", run_query_router)
s2_builder.add_edge("interpret_results", END)

graph = s2_builder.compile()

def run_scenario(question: str):
    return graph.invoke({"messages": HumanMessage(question)})

if __name__ == "__main__":
    main(graph)