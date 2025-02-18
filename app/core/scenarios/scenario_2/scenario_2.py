import asyncio
from typing import Literal
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from app.core.scenarios.scenario_2.prompt import system_prompt_template
from app.core.utils.sparql_toolkit import find_sparql_queries
from app.core.utils.graph_nodes import (
    interpret_csv_query_results,
    run_query,
)
from app.core.utils.graph_routers import run_query_router
from app.core.utils.graph_state import InputState, OverallState
import app.core.utils.config_manager as config
from app.core.utils.logger_manager import setup_logger

logger = setup_logger(__package__, __file__)

SCENARIO = "scenario_2"
config.set_scenario(SCENARIO)


def generate_query_router(state: OverallState) -> Literal["run_query", "__end__"]:
    if len(find_sparql_queries(state["messages"][-1].content)) > 0:
        logger.info("Query generation task produced a SPARQL query")
        return "run_query"
    else:
        logger.warning("Query generation task did not produce a proper SPARQL query")
        logger.info("Processing completed.")
        return END


async def generate_query(state: OverallState) -> OverallState:
    logger.info(f"Question: {state["initial_question"]}")

    template = system_prompt_template

    if "kg_full_name" in template.input_variables:
        template = template.partial(kg_full_name=config.get_kg_full_name())

    if "kg_description" in template.input_variables:
        template = template.partial(kg_description=config.get_kg_description())

    if "initial_question" in state.keys():
        template = template.partial(initial_question=state["initial_question"])

    prompt = template.format()
    logger.debug(f"Prompt created:\n{prompt}")

    result = await config.get_llm().ainvoke(template.format())
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
    asyncio.run(config.main(graph))
