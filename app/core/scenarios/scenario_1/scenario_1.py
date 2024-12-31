from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from app.core.utils.printing import new_log

from app.core.scenarios.scenario_1.utils.prompt import PROMPT


llm = ChatOllama(model="llama3.2")

# Node
# def interpret_results(state: MessagesState):
#     result =  llm.invoke((PROMPT + state["messages"]).messages)
#     return { "messages": (result + state["messages"]).messages}


def interpret_results(state: MessagesState):
    result = llm.invoke([PROMPT] + state["messages"])
    return {"messages": result}


s1_builder = StateGraph(MessagesState)
s1_builder.add_node("Interpret_results", interpret_results)
s1_builder.add_edge(START, "Interpret_results")
s1_builder.add_edge("Interpret_results", END)

# config = {"configurable": {"thread_id": "1"}}

graph = s1_builder.compile(checkpointer=MemorySaver())

# question = "What protein targets does donepezil (CHEBI_53289) inhibit with an IC50 less than 10 µM?"
# state = graph.invoke({"messages":HumanMessage(question)},config)
# state = graph.invoke({"messages":HumanMessage("sure?")},config)

# new_log()
# for m in state["messages"]:
#    m.pretty_print()
# new_log()
