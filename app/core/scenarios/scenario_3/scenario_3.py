import asyncio
from typing import Literal
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from app.core.scenarios.scenario_3.utils.prompt import system_prompt_template
from app.core.utils.sparql_toolkit import find_sparql_queries
from app.core.utils.graph_nodes import (
    interpret_csv_query_results,
    select_similar_classes,
)
from app.core.utils.graph_state import InputState, OverallState
from app.core.utils.utils import (
    get_llm_from_config,
    main,
    setup_logger,
)
from rdflib.exceptions import ParserError
from app.core.utils.sparql_toolkit import run_sparql_query

logger = setup_logger(__package__, __file__)

SCENARIO = "scenario_3"

llm = get_llm_from_config(SCENARIO)


def run_query_router(state: OverallState) -> Literal["interpret_results", "__end__"]:
    """
    Check if the query was run successfully and routes to the next step accordingly.

    Args:
        state (MessagesState): The current state of the conversation

    Returns:
        Literal["interpret_results", END]: The next step in the conversation
    """
    if state["messages"][-1].content.find("Error when running the query") == -1:
        logger.info("query run succesfully and it yielded")
        return "interpret_results"
    else:
        logger.info("Ending the process")
        return END


def generate_query_router(state: OverallState) -> Literal["run_query", "__end__"]:
    if len(find_sparql_queries(state["messages"][-1].content)) > 0:
        logger.info("query generated task completed with a generated SPARQL query")
        return "run_query"
    else:
        logger.warning(
            "query generated task completed without generating a proper SPARQL query"
        )
        logger.info("Ending the process")
        return END


async def generate_query(state: OverallState):
    result = await llm.ainvoke(
        system_prompt_template.format(question=state["initial_question"])
    )
    return {"messages": [HumanMessage(state["initial_question"]), result]}


def run_query(state: OverallState):

    query = find_sparql_queries(state["messages"][-1].content)[0]

    try:
        csv_result = run_sparql_query(query=query)
        return {"messages": csv_result, "last_generated_query": query}
    except ParserError as e:
        logger.warning(f"A parsing error occurred when running the query: {e}")
        return {"messages": AIMessage("Error when running the query")}
    except Exception as e:
        logger.warning(f"An error occurred when running the query: {e}")
        return {"messages": AIMessage("Error when running the query")}


s3_builder = StateGraph(
    state_schema=OverallState, input=InputState, output=OverallState
)


s3_builder.add_node("select_similar_classes", select_similar_classes)
s3_builder.add_node("generate_query", generate_query)
s3_builder.add_node("run_query", run_query)
s3_builder.add_node("interpret_results", interpret_csv_query_results)

s3_builder.add_edge(START, "select_similar_classes")
s3_builder.add_edge("select_similar_classes", "generate_query")
s3_builder.add_conditional_edges("generate_query", generate_query_router)
s3_builder.add_conditional_edges("run_query", run_query_router)
s3_builder.add_edge("interpret_results", END)

graph = s3_builder.compile()


def run_scenario(question: str):
    return graph.ainvoke(input={"initial_question": question})


if __name__ == "__main__":
    asyncio.run(main(graph))
