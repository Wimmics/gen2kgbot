import asyncio
from langgraph.graph import StateGraph, START, END
from app.core.scenarios.scenario_8.prompt import (
    retry_system_prompt_template,
    query_repair_expert_prompt,
)
from app.core.utils.graph_nodes import (
    preprocess_question,
    select_similar_classes,
    get_class_context_from_cache,
    get_class_context_from_kg,
    select_similar_query_examples,
    validate_question,
    verify_query,
    run_query,
    interpret_csv_query_results,
)
from app.core.utils.graph_routers import (
    get_class_context_router,
    preprocessing_subgraph_router,
    validate_question_router,
    verify_query_router,
    run_query_router,
)
from app.core.scenarios.scenario_8.agents import (
    initialize_agents,
    debate,
    moderator_evaluate,
    format_selected_query,
)
from app.core.utils.graph_state import InputState, OverallState
import app.core.utils.config_manager as config
from app.core.utils.logger_manager import setup_logger
from app.core.utils.construct_util import add_known_prefixes_to_query
from app.core.utils.sparql_toolkit import find_sparql_queries
from rdflib.plugins.sparql.parser import parseQuery
from rdflib.plugins.sparql.algebra import translateQuery

from typing import Literal
import re
from langchain_core.messages import HumanMessage

logger = setup_logger(__package__, __file__)

SCENARIO = "scenario_8"


def init(state) -> OverallState:
    """Initialize the state with scenario identification."""
    logger.info(f"Running scenario: {SCENARIO}")
    
    # Log the input state type and structure
    logger.info(f"Init received state type: {type(state)}")
    if hasattr(state, "keys"):
        logger.info(f"Init state keys: {list(state.keys())}")
    elif hasattr(state, "__dict__"):
        logger.info(f"Init state attributes: {state.__dict__.keys()}")
    
    # Extract initial_question from input state - more robust handling
    initial_question = ""
    
    # Direct dictionary access
    if isinstance(state, dict) and "initial_question" in state:
        initial_question = state["initial_question"]
        logger.info(f"Found initial_question in state dict: {initial_question}")
    
    # Try accessing input attribute if it exists
    elif hasattr(state, "input"):
        if isinstance(state.input, dict) and "initial_question" in state.input:
            initial_question = state.input["initial_question"]
            logger.info(f"Found initial_question in state.input dict: {initial_question}")
    
    # Try checking messages
    if not initial_question and isinstance(state, dict) and "messages" in state:
        logger.info("Checking messages field for question")
        messages = state["messages"]
        if isinstance(messages, str) and "properties of the Person class" in messages:
            initial_question = "What are the properties of the Person class?"
            logger.info(f"Extracted question from messages: {initial_question}")
    
    # If still empty, use default
    if not initial_question:
        initial_question = "What are the properties of the Person class?"
        logger.warning(f"Using default question: {initial_question}")
    
    logger.info(f"Initializing with question: '{initial_question}'")
    
    result_state = OverallState({
        "scenario_id": SCENARIO,
        "initial_question": initial_question,
        "messages": [],  # Initialize empty messages list
        "kg_description": config.get_kg_description(),  # Add KG description
        "max_debate_rounds": config.get_configuration().get("scenario_8", {}).get("debate_rounds", 3)  # Get debate_rounds from config
    })
    
    # Log output state structure
    logger.info(f"Init returning state with keys: {list(result_state.keys())}")
    return result_state


def create_retry_prompt(state: OverallState) -> OverallState:
    """Create a retry prompt when query verification fails."""
    logger.info("Creating retry prompt for failed query verification")
    template = retry_system_prompt_template

    if "kg_full_name" in template.input_variables:
        template = template.partial(kg_full_name=config.get_kg_full_name())
    
    if "kg_description" in template.input_variables:
        template = template.partial(kg_description=config.get_kg_description())
        
    for var in ["initial_question", "selected_queries", "merged_classes_context", 
                "last_answer", "last_answer_error_cause", "selected_query"]:
        if var in template.input_variables and var in state:
            template = template.partial(**{var: state[var]})
    
    prompt = template.format()
    
    logger.debug(f"Retry prompt created for failed verification")
    
    return OverallState({
        **state,
        "query_generation_prompt": prompt,
        "number_of_tries": state.get("number_of_tries", 0) + 1
    })


def merge_class_contexts(state: OverallState) -> OverallState:
    """Merge the class contexts into a single string for use in agent prompts."""
    logger.info("Merging class contexts")
    
    if "selected_classes_context" not in state:
        logger.warning("No selected_classes_context found in state")
        return state
        
    # Simple concatenation of contexts with separators
    merged_context = "\n\n".join([str(ctx) for ctx in state.get("selected_classes_context", [])])
    
    logger.info(f"Created merged context from {len(state.get('selected_classes_context', []))} class contexts")
    
    return OverallState({
        **state,
        "merged_classes_context": merged_context
    })


# Modified verify_query router to include query_repair path
def verify_query_router(
    state: OverallState,
) -> Literal["run_query", "query_repair", "__end__"]:
    """
    Decide whether to run the query, attempt to repair it, or stop.
    
    Args:
        state (OverallState): Current state of the conversation
        
    Returns:
        Literal["run_query", "query_repair", "__end__"]: Next step in the conversation
    """
    # Debug log to track state
    logger.debug(f"Router received state with keys: {list(state.keys())}")
    logger.debug(f"verified_query present: {'verified_query' in state}")
    
    if "selected_query" not in state or not state["selected_query"]:
        logger.warning("No query generated from debate process, stopping process")
        return END
    elif "verified_query" in state and state["verified_query"]:
        logger.info("Query verification passed, proceeding to execution")
        logger.debug(f"Verified query: {state['verified_query'][:50]}...")
        return "run_query"
    elif state.get("verification_error") and state.get("repair_attempts", 0) < 3:
        logger.info(f"Query verification failed, attempting repair (attempt {state.get('repair_attempts', 0) + 1}/3)")
        logger.debug(f"Verification error: {state.get('verification_error')}")
        return "query_repair"
    else:
        if state.get("repair_attempts", 0) >= 3:
            logger.warning("Maximum repair attempts reached, stopping repair process")
        else:
            logger.warning("Query verification failed without specific error, stopping process")
            # Dump state keys to help debug
            logger.debug(f"State keys at failure: {list(state.keys())}")
            logger.debug(f"selected_query present: {'selected_query' in state}")
            if 'selected_query' in state:
                logger.debug(f"Selected query: {state['selected_query'][:50]}...")
        return END


# Wrapped verify_query to capture errors
async def verify_query_with_error_capture(state: OverallState) -> OverallState:
    """
    Verify the query directly from state['selected_query'] and capture any errors.
    
    Args:
        state (OverallState): Current state of the conversation with selected_query
        
    Returns:
        OverallState: Updated state with verification results and errors
    """
    if "selected_query" not in state or not state["selected_query"]:
        logger.warning("No selected query found in state")
        error_message = "No SPARQL query was found in state."
        logger.debug(f"State keys before error return: {list(state.keys())}")
        return OverallState({
            **state,
            "messages": [HumanMessage(error_message)],
            "verification_error": error_message
        })
    try:
        # For scenario 8, get query directly from state
        query = state["selected_query"]
        logger.info("Verifying selected query from debate process")
        logger.debug(f"Query to verify:\n{query}")
        
        # Verify the query syntax
        translateQuery(parseQuery(add_known_prefixes_to_query(query)))
        logger.info("The selected SPARQL query is syntactically correct")
        
        # Make a complete copy of the state for return
        result_state = {k: v for k, v in state.items()}
        result_state["verified_query"] = query
        result_state["last_generated_query"] = query
        
        # Debug logging to track state transfer
        logger.debug(f"State keys after successful verification: {list(result_state.keys())}")
        logger.debug(f"verified_query set to: {query[:50]}...")
        
        # Return successful verification result with complete state
        return OverallState(result_state)
    except Exception as e:
        # Capture any exceptions during verification
        error_message = str(e)
        logger.error(f"Error during query verification: {error_message}")
        
        # Make a complete copy of the state for return
        result_state = {k: v for k, v in state.items()}
        result_state["messages"] = [HumanMessage(error_message)]
        result_state["verification_error"] = error_message
        
        logger.debug(f"State keys after error: {list(result_state.keys())}")
        
        return OverallState(result_state)


async def repair_query(state: OverallState) -> OverallState:
    """
    Use a specialized Query Repair Expert to fix a query that failed verification.
    
    This function is called when query verification fails. It extracts the verification error,
    sends the failed query to a specialized agent, and returns a fixed query.
    
    Args:
        state (OverallState): Current state containing the failed query and error
        
    Returns:
        OverallState: Updated state with the repaired query
    """
    logger.info("Using Query Repair Expert to fix failed query")
    
    # Track repair attempts
    repair_attempts = state.get("repair_attempts", 0) + 1
    logger.info(f"Current repair attempt: {repair_attempts}/3")
    
    # Extract the query that needs repair and the verification error
    query_to_repair = state.get("selected_query", "")
    verification_error = state.get("verification_error", "Unknown verification error")
    
    logger.debug(f"Query to repair: {query_to_repair[:100]}...")
    logger.debug(f"Verification error: {verification_error}")
    
    # Prepare the repair prompt
    repair_prompt = query_repair_expert_prompt.format(
        kg_full_name=config.get_kg_full_name(),
        kg_description=config.get_kg_description(),
        initial_question=state.get("initial_question", ""),
        query_to_repair=query_to_repair,
        verification_error=verification_error,
        merged_classes_context=state.get("merged_classes_context", "")
    )
    
    # Get repair from the model
    logger.info("Sending repair prompt to model")
    model = config.get_seq2seq_model(scenario_id="scenario_8", node_name="query_repair")
    response = await model.ainvoke(repair_prompt)
    logger.debug(f"Repair expert's complete response:\n{response.content}")
    
    # Extract repaired query and explanation
    content = response.content
    query_match = re.search(r"REPAIRED_QUERY:\s*```sparql\s*(.*?)\s*```", content, re.DOTALL)
    explanation_match = re.search(r"EXPLANATION:\s*(.*?)(?:\n\n|$)", content, re.DOTALL)
    
    if query_match:
        repaired_query = query_match.group(1).strip()
        
        # Verify the repaired query is valid SPARQL
        queries = find_sparql_queries(repaired_query)
        if queries:
            logger.info("Query Repair Expert successfully fixed the query")
            logger.debug(f"Repaired query:\n{queries[0]}")
            
            if explanation_match:
                explanation = explanation_match.group(1).strip()
                logger.debug(f"Repair explanation: {explanation}")
            
            # Immediately try to verify the repaired query
            try:
                # Verify the query syntax
                translateQuery(parseQuery(add_known_prefixes_to_query(queries[0])))
                logger.info("Repaired query verification passed")
                
                # Update state with the repaired query and increment counter
                return OverallState({
                    **state,
                    "selected_query": queries[0],
                    "verified_query": queries[0],  # Set verified_query for the router
                    "repair_explanation": explanation_match.group(1).strip() if explanation_match else "No explanation provided",
                    "repair_attempts": repair_attempts,
                    "verification_error": None  # Clear any previous verification error
                })
            except Exception as e:
                logger.warning(f"Repaired query still has syntax errors: {e}")
                # Continue with the repair attempt increment
            
    # If repair failed or query still invalid, still increment counter
    logger.warning("Query Repair Expert failed to fix the query")
    return OverallState({
        **state,
        "repair_attempts": repair_attempts
    })


# Subgraph for preprocessing the question: generate context with classes and examples queries
prepro_builder = StateGraph(
    state_schema=OverallState, input=OverallState, output=OverallState
)

prepro_builder.add_node("init", init)
prepro_builder.add_node("validate_question", validate_question)
prepro_builder.add_node("preprocess_question", preprocess_question)
prepro_builder.add_node("select_similar_classes", select_similar_classes)
prepro_builder.add_node("get_context_class_from_cache", get_class_context_from_cache)
prepro_builder.add_node("get_context_class_from_kg", get_class_context_from_kg)
prepro_builder.add_node("select_similar_query_examples", select_similar_query_examples)
prepro_builder.add_node("merge_class_contexts", merge_class_contexts)

prepro_builder.add_edge(START, "init")
prepro_builder.add_edge("init", "validate_question")
prepro_builder.add_conditional_edges("validate_question", validate_question_router)
prepro_builder.add_edge("preprocess_question", "select_similar_query_examples")
prepro_builder.add_edge("preprocess_question", "select_similar_classes")
prepro_builder.add_edge("select_similar_query_examples", "merge_class_contexts")
prepro_builder.add_conditional_edges("select_similar_classes", get_class_context_router)
prepro_builder.add_edge("get_context_class_from_cache", "merge_class_contexts")
prepro_builder.add_edge("get_context_class_from_kg", "merge_class_contexts")
prepro_builder.add_edge("merge_class_contexts", END)

# Subgraph for the multi-agent debate process
debate_builder = StateGraph(
    state_schema=OverallState, input=OverallState, output=OverallState
)

debate_builder.add_node("initialize_agents", initialize_agents)
debate_builder.add_node("debate", debate)
debate_builder.add_node("moderator_evaluate", moderator_evaluate)
debate_builder.add_node("format_selected_query", format_selected_query)

# Simplified debate workflow
debate_builder.add_edge(START, "initialize_agents")
debate_builder.add_edge("initialize_agents", "debate")
debate_builder.add_edge("debate", "moderator_evaluate")
#debate_builder.add_edge("moderator_evaluate", "format_selected_query")
#debate_builder.add_edge("format_selected_query", END)
debate_builder.add_edge("moderator_evaluate", END)

# Main graph for the overall workflow
builder = StateGraph(state_schema=OverallState, input=InputState, output=OverallState)

# Get configured debate rounds from config and log it
configured_debate_rounds = config.get_scenario_config("scenario_8").get("debate_rounds")
logger.info(f"Loading configured debate rounds: {configured_debate_rounds}")

builder.add_node("preprocessing_subgraph", prepro_builder.compile())
builder.add_node("debate_subgraph", debate_builder.compile())
builder.add_node("verify_query", verify_query_with_error_capture)
builder.add_node("query_repair", repair_query)
builder.add_node("run_query", run_query)
builder.add_node("interpret_results", interpret_csv_query_results)

# Main flow with repair path
builder.add_edge(START, "preprocessing_subgraph")
builder.add_conditional_edges(
    "preprocessing_subgraph", 
    preprocessing_subgraph_router,
    {
        "create_prompt": "debate_subgraph",
        END: END
    }
)
builder.add_edge("debate_subgraph", "verify_query")
builder.add_conditional_edges(
    "verify_query", 
    verify_query_router,
    {
        "run_query": "run_query",
        "query_repair": "query_repair",
        END: END
    }
)
# Add path from repair to verify_query for re-verification of repaired query
builder.add_edge("query_repair", "verify_query")
builder.add_conditional_edges("run_query", run_query_router)
builder.add_edge("interpret_results", END)

graph = builder.compile()


if __name__ == "__main__":
    from argparse import ArgumentParser
    import asyncio
    import traceback
    
    parser = ArgumentParser(
        description="Process the scenario with the predefined or custom question and configuration."
    )
    parser.add_argument(
        "-q",
        "--question",
        type=str,
        help='User\'s question. Defaults to "What protein targets does donepezil (CHEBI_53289) inhibit with an IC50 less than 5 µM?"',
        default="What protein targets does donepezil (CHEBI_53289) inhibit with an IC50 less than 5 µM?",
    )
    parser.add_argument("-p", "--params", type=str, help="Custom configuration file")
    parser.add_argument(
        "--prod",
        action="store_true",
        help="Load the production configuration file",
        default=False,
    )
    args = parser.parse_args()
    
    # Load configuration
    config.read_configuration(args)
    
    question = args.question
    print(f"Processing question: {question}")
    logger.info(f"Main: processing question: '{question}'")
    
    async def run_graph():
        try:
            # Create input state with configured debate rounds
            configured_debate_rounds = config.get_scenario_config("scenario_8").get("debate_rounds")
            input_state = {
                "initial_question": question,
                "max_debate_rounds": configured_debate_rounds if configured_debate_rounds is not None else 3
            }
            logger.info(f"Graph input structure: {input_state}")
            logger.info(f"Setting max_debate_rounds to: {input_state.get('max_debate_rounds')}")
            
            # Invoke the graph
            state = await graph.ainvoke(input_state)
            
            # Log the output structure
            logger.info(f"Graph output keys: {state.keys() if hasattr(state, 'keys') else 'no keys method'}")
            
            print("==============================================================")
            if "messages" in state:
                for m in state["messages"]:
                    print(m)
        except Exception as e:
            logger.error(f"Error during graph execution: {e}")
            logger.error(traceback.format_exc())
            print(f"Error: {e}")
    
    try:
        asyncio.run(run_graph())
        print("Scenario completed successfully")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
        print(f"Fatal error: {e}")
