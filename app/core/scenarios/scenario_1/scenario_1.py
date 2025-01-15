import os
import argparse
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from app.core.utils.utils import setup_logger
from app.core.scenarios.scenario_1.utils.prompt import PROMPT


logger = setup_logger(__package__, __file__)

llm = ChatOllama(model="llama3.2:1b")
# openai_api_key = os.getenv("OPENAI_API_KEY")
# llm = ChatOpenAI(
#     model="gpt-4o",
#     openai_api_key=openai_api_key,
# )
logger.info(f"LLM initialized")


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


def main():

    parser = argparse.ArgumentParser(
        description="Process the scenario with the predifined or custom question."
    )
    parser.add_argument("-c", "--custom", type=str, help="Provide a custom question.")
    args = parser.parse_args()

    if args.custom:
        question = args.custom
    else:
        question = "What protein targets does donepezil (CHEBI_53289) inhibit with an IC50 less than 10 µM?"

    state = graph.invoke({"messages": HumanMessage(question)})

    logger.info("----------------------------------------------------------------")
    for m in state["messages"]:
        logger.info(m)
    logger.info("----------------------------------------------------------------")


if __name__ == "__main__":
    main()
