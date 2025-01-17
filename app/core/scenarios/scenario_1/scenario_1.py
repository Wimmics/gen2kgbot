from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from app.core.utils.utils import main, setup_logger, get_llm_from_config
from app.core.scenarios.scenario_1.utils.prompt import PROMPT


logger = setup_logger(__package__, __file__)

SCENARIO = "scenario_1"

llm = get_llm_from_config(SCENARIO)

def interpret_results(state: MessagesState):
    logger.info(f"Question: {state["messages"]}")
    result = llm.invoke([PROMPT] + state["messages"])
    return {"messages": result}

s1_builder = StateGraph(MessagesState)
s1_builder.add_node("Interpret_results", interpret_results)
s1_builder.add_edge(START, "Interpret_results")
s1_builder.add_edge("Interpret_results", END)

graph = s1_builder.compile()

def run_scenario(question: str):
    return graph.invoke({"messages": HumanMessage(question)})

if __name__ == "__main__":
    main(graph)