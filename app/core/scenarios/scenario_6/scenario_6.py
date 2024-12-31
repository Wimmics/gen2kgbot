from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState
from langchain_core.messages import AIMessage,HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from app.core.utils.printing import new_log

from app.core.scenarios.scenario_1.utils.prompt import PROMPT


llm = ChatOllama(model="llama3.2")

# Node
def preprocess_question(state: MessagesState):   
    result =  AIMessage("I am preprocessing the quesion")
    return { "messages": (result + state["messages"]).messages}

def get_context_class(state: MessagesState):   
    result =  AIMessage("I created")
    return { "messages": (result + state["messages"]).messages}

def get_context_class_from_cache(state: MessagesState):   
    result =  AIMessage("I created")
    return { "messages": (result + state["messages"]).messages}

def get_context_class_from_kg(state: MessagesState):   
    result =  AIMessage("I created")
    return { "messages": (result + state["messages"]).messages}

def create_prompt(state: MessagesState):   
    result =  AIMessage("I created")
    return { "messages": (result + state["messages"]).messages}

def select_similar_classes(state: MessagesState):   
    result =  AIMessage("I created")
    return { "messages": (result + state["messages"]).messages}

def select_similar_query_examples(state: MessagesState):   
    result =  AIMessage("I created")
    return { "messages": (result + state["messages"]).messages}

def generate_query(state: MessagesState):   
    result =  llm.invoke((PROMPT + state["messages"]).messages)
    return { "messages": (result + state["messages"]).messages}

def verify_query(state: MessagesState):   
    result =  llm.invoke((PROMPT + state["messages"]).messages)
    return { "messages": (result + state["messages"]).messages}

def create_retry_prompt(state: MessagesState):   
    result =  llm.invoke((PROMPT + state["messages"]).messages)
    return { "messages": (result + state["messages"]).messages}

def run_query(state: MessagesState):   
    result =  llm.invoke((PROMPT + state["messages"]).messages)
    return { "messages": (result + state["messages"]).messages}

def interpret_results(state: MessagesState):   
    result =  llm.invoke((PROMPT + state["messages"]).messages)
    return { "messages": (result + state["messages"]).messages}

s6_builder = StateGraph(MessagesState)

s6_builder.add_node("preprocess_question", preprocess_question)
s6_builder.add_node("select_similar_classes", select_similar_classes)
s6_builder.add_node("select_similar_query_examples", select_similar_query_examples)
s6_builder.add_node("get_context_class_from_cache", get_context_class_from_cache)
s6_builder.add_node("get_context_class_from_kg", get_context_class_from_kg)
s6_builder.add_node("create_prompt", create_prompt)
s6_builder.add_node("generate_query", generate_query)
s6_builder.add_node("verify_query", verify_query)
s6_builder.add_node("create_retry_prompt", create_retry_prompt)
s6_builder.add_node("run_query", run_query)
s6_builder.add_node("interpret_results", interpret_results)

s6_builder.add_edge(START, "preprocess_question")
s6_builder.add_edge("preprocess_question", "select_similar_classes")
s6_builder.add_edge("preprocess_question", "select_similar_query_examples")
s6_builder.add_edge("select_similar_query_examples", "create_prompt")
s6_builder.add_conditional_edges("select_similar_classes", get_context_class, ["get_context_class_from_cache","get_context_class_from_kg"])
s6_builder.add_edge("get_context_class_from_cache", "create_prompt")
s6_builder.add_edge("get_context_class_from_kg", "create_prompt")
s6_builder.add_edge("create_prompt", "generate_query")
s6_builder.add_edge("generate_query", "verify_query")
s6_builder.add_conditional_edges("verify_query", get_context_class, ["create_retry_prompt","run_query"])
s6_builder.add_edge("create_retry_prompt", "generate_query")
s6_builder.add_edge("run_query","interpret_results")
s6_builder.add_edge("interpret_results",END)

graph = s6_builder.compile()

# question = "What protein targets does donepezil (CHEBI_53289) inhibit with an IC50 less than 10 µM?"
# state = graph.invoke({"messages":[PROMPT, HumanMessage(question)]})

# new_log()
# for m in state["messages"]:
#    m.pretty_print()
# new_log()
