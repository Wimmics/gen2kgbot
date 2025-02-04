import asyncio
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from app.core.utils.graph_state import InputState, OverallState
from app.core.utils.utils import main, setup_logger, get_llm_from_config
from app.core.scenarios.scenario_1.utils.prompt import system_prompt_template


logger = setup_logger(__package__, __file__)

SCENARIO = "scenario_1"

llm = get_llm_from_config(SCENARIO)


async def interpret_results(state: OverallState):
    logger.info(f"Question: {state["initial_question"]}")
    result = await llm.ainvoke(
        system_prompt_template.format(question=state["initial_question"])
    )
    return {"messages": [HumanMessage(state["initial_question"]), result]}


s1_builder = StateGraph(
    state_schema=OverallState, input=InputState, output=OverallState
)
s1_builder.add_node("Interpret_results", interpret_results)
s1_builder.add_edge(START, "Interpret_results")
s1_builder.add_edge("Interpret_results", END)

graph = s1_builder.compile()


def run_scenario(question: str):
    return graph.ainvoke(input={"initial_question": question})


if __name__ == "__main__":
    asyncio.run(main(graph))
