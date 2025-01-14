import argparse
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from app.core.utils.printing import new_log

from app.core.scenarios.scenario_1.utils.prompt import PROMPT


llm = ChatOllama(model="llama3.2:1b")

def interpret_results(state: MessagesState):
    result = llm.invoke([PROMPT] + state["messages"])
    return {"messages": result}


s1_builder = StateGraph(MessagesState)
s1_builder.add_node("Interpret_results", interpret_results)
s1_builder.add_edge(START, "Interpret_results")
s1_builder.add_edge("Interpret_results", END)


graph = s1_builder.compile()

def main():

    parser = argparse.ArgumentParser(description="Process the scenario with the predifined or custom question.")
    
    parser.add_argument('-c', '--custom', type=str,
                        help="Provide a custom question.")
    
    args = parser.parse_args()
    
    if args.custom:
        question = args.custom
    else:
        question = "What protein targets does donepezil (CHEBI_53289) inhibit with an IC50 less than 10 µM?"
    
    state = graph.invoke({"messages":HumanMessage(question)})

    new_log()
    for m in state["messages"]:
        m.pretty_print()
    new_log()


if __name__ == "__main__":
    main()