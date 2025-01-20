import re
from typing import Literal
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from app.core.scenarios.scenario_3.utils.prompt import system_prompt, interpreter_prompt
from app.core.utils.graph_nodes import interpret_csv_query_results, select_similar_classes
from app.core.utils.graph_state import InputState, OverAllState
from app.core.utils.utils import (
    get_llm_from_config,
    get_class_vector_db_from_config,
    main,
    setup_logger,
)
from rdflib.exceptions import ParserError
from app.core.utils.sparql_toolkit import run_sparql_query

logger = setup_logger(__package__, __file__)

SCENARIO = "scenario_3"

llm = get_llm_from_config(SCENARIO)


def run_query_router(state: OverAllState) -> Literal["interpret_results", END]:
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


def generate_query_router(state: OverAllState) -> Literal["run_query", END]:
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
def create_prompt(state: OverAllState):
    result = [system_prompt] + state["messages"]
    state["messages"].clear()
    logger.info(f"prompt created successfuly.")
    return {"messages": result}


def generate_query(state: OverAllState):
    result = llm.invoke(state["messages"])
    return {"messages": result}


def run_query(state: OverAllState):

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
        return {
            "messages": AIMessage("Error when running the query"),
            "last_generated_query": query,
        }


s3_builder = StateGraph(
    state_schema=OverAllState, input=InputState, output=OverAllState
)


s3_builder.add_node("select_similar_classes", select_similar_classes)
s3_builder.add_node("create_prompt", create_prompt)
s3_builder.add_node("generate_query", generate_query)
s3_builder.add_node("run_query", run_query)
s3_builder.add_node("interpret_results", interpret_csv_query_results)

s3_builder.add_edge(START, "select_similar_classes")
s3_builder.add_edge("select_similar_classes", "create_prompt")
s3_builder.add_edge("create_prompt", "generate_query")
s3_builder.add_conditional_edges("generate_query", generate_query_router)
s3_builder.add_conditional_edges("run_query", run_query_router)
s3_builder.add_edge("interpret_results", END)

graph = s3_builder.compile()


def run_scenario(question: str):
    return graph.invoke({"messages": HumanMessage(question)})


if __name__ == "__main__":
    main(graph)
