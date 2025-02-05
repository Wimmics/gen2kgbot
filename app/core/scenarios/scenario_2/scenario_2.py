import asyncio
from typing import Literal
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from app.core.scenarios.scenario_2.utils.prompt import system_prompt_template
from app.core.utils.sparql_toolkit import find_sparql_queries
from app.core.utils.graph_nodes import interpret_csv_query_results
from app.core.utils.graph_state import InputState, OverallState
from app.core.utils.utils import (
    get_llm_from_config,
    main,
    setup_logger,
)
from rdflib.exceptions import ParserError
from app.core.utils.sparql_toolkit import run_sparql_query

logger = setup_logger(__package__, __file__)

SPARQL_QUERY_EXEC_ERROR = "SPARQL query execution failed"
SCENARIO = "scenario_2"

llm = get_llm_from_config(SCENARIO)


# Router
def run_query_router(state: OverallState) -> Literal["interpret_results", "__end__"]:
    if state["messages"][-1].content.find(SPARQL_QUERY_EXEC_ERROR) == -1:
        logger.info("query executed successfully")
        return "interpret_results"
    else:
        logger.info("Ending the process")
        return END


def generate_query_router(state: OverallState) -> Literal["run_query", "__end__"]:
    if len(find_sparql_queries(state["messages"][-1].content)) > 0:
        logger.info("query generation task completed successfully")
        return "run_query"
    else:
        logger.warning(
            "query generation task completed without generating a proper SPARQL query"
        )
        logger.info("Ending the process")
        return END


# Node
async def generate_query(state: OverallState) -> OverallState:
    logger.info(f"Question: {state["initial_question"]}")
    result = await llm.ainvoke(
        system_prompt_template.format(question=state["initial_question"])
    )
    return OverallState({"messages": [HumanMessage(state["initial_question"]), result]})


def run_query(state: OverallState) -> OverallState:
    query = find_sparql_queries(state["messages"][-1].content)[0]
    logger.info("Executing SPARQL query extracted from llm's response:")
    logger.info(f"{query}")

    try:
        csv_result = run_sparql_query(query=query)
        return OverallState({"messages": csv_result, "last_generated_query": query})
    except ParserError as e:
        logger.warning(f"A parsing error occurred when running the query: {e}")
        return OverallState({"messages": BaseMessage(SPARQL_QUERY_EXEC_ERROR)})
    except Exception as e:
        logger.warning(f"An error occurred when running the query: {e}")
        return OverallState({"messages": AIMessage(SPARQL_QUERY_EXEC_ERROR)})


s2_builder = StateGraph(
    state_schema=OverallState, input=InputState, output=OverallState
)

s2_builder.add_node("generate_query", generate_query)
s2_builder.add_node("run_query", run_query)
s2_builder.add_node("interpret_results", interpret_csv_query_results)

s2_builder.add_edge(START, "generate_query")
s2_builder.add_conditional_edges("generate_query", generate_query_router)
s2_builder.add_conditional_edges("run_query", run_query_router)
s2_builder.add_edge("interpret_results", END)

graph = s2_builder.compile()


def run_scenario(question: str):
    return graph.ainvoke(input=InputState({"initial_question": question}))


if __name__ == "__main__":
    asyncio.run(main(graph))
