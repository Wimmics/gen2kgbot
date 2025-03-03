import asyncio
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from app.core.utils.graph_state import InputState, OverallState
import app.core.utils.config_manager as config
from app.core.utils.logger_manager import setup_logger
from app.core.scenarios.scenario_1.prompt import system_prompt_template


logger = setup_logger(__package__, __file__)

SCENARIO = "scenario_1"


def init(state: OverallState) -> OverallState:
    logger.info(f"Running scenario: {SCENARIO}")
    return OverallState({"scenario_id": SCENARIO})


async def ask_question(state: OverallState):
    logger.info(f"Question: {state["initial_question"]}")
    result = await config.get_seq2seq_model(state["scenario_id"]).ainvoke(
        system_prompt_template.format(initial_question=state["initial_question"])
    )
    logger.info(f"Model's response:\n{result.content}")
    return {"messages": [HumanMessage(state["initial_question"]), result]}


builder = StateGraph(state_schema=OverallState, input=InputState, output=OverallState)

builder.add_node("init", init)
builder.add_node("ask_question", ask_question)

builder.add_edge(START, "init")
builder.add_edge("init", "ask_question")
builder.add_edge("ask_question", END)

graph = builder.compile()


if __name__ == "__main__":
    asyncio.run(config.main(graph))
