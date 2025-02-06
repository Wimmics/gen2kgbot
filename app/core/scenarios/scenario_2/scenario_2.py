import asyncio
from typing import Literal
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from app.core.scenarios.scenario_2.utils.prompt import system_prompt_template
from app.core.utils.sparql_toolkit import find_sparql_queries
from app.core.utils.graph_nodes import interpret_csv_query_results, run_query
from app.core.utils.graph_state import InputState, OverallState
from app.core.utils.utils import (
    get_llm_from_config,
    main,
    setup_logger,
)
from rdflib.exceptions import ParserError
from app.core.utils.sparql_toolkit import run_sparql_query

logger = setup_logger(__package__, __file__)

SPARQL_QUERY_EXEC_ERROR = "Error when running the SPARQL query"
SCENARIO = "scenario_2"

llm = get_llm_from_config(SCENARIO)


# Router
def run_query_router(state: OverallState) -> Literal["interpret_results", "__end__"]:
    if state["last_query_results"].find(SPARQL_QUERY_EXEC_ERROR) == -1:
        logger.info("Query execution yielded some results")
        return "interpret_results"
    else:
        logger.info("Processing completed.")
        return END


def generate_query_router(state: OverallState) -> Literal["run_query", "__end__"]:
    if len(find_sparql_queries(state["messages"][-1].content)) > 0:
        logger.info("Query generation task produced a SPARQL query")
        return "run_query"
    else:
        logger.warning("Query generation task did not produce a proper SPARQL query")
        logger.info("Processing completed.")
        return END


# Node
async def generate_query(state: OverallState) -> OverallState:
    logger.info(f"Question: {state["initial_question"]}")
    result = await llm.ainvoke(
        system_prompt_template.format(question=state["initial_question"])
    )
    return OverallState({"messages": [HumanMessage(state["initial_question"]), result]})


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
