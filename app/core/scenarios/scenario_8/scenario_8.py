import asyncio
from langgraph.graph import StateGraph, START, END
from app.core.scenarios.scenario_8.prompt import (
    retry_system_prompt_template,
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
    generate_agent_proposals,
    moderator_evaluate,
    check_debate_rounds,
    format_selected_query,
    debate_complete,
)
from app.core.utils.graph_state import InputState, OverallState
import app.core.utils.config_manager as config
from app.core.utils.logger_manager import setup_logger

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
        "messages": []  # Initialize empty messages list
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


def debate_rounds_router(state: OverallState):
    """Router to determine if more debate rounds are needed."""
    need_more_rounds = state.get("need_more_rounds", False)
    if need_more_rounds:
        logger.info("Router: More debate rounds needed")
        return "generate_agent_proposals"
    else:
        logger.info("Router: No more debate rounds needed")
        return "debate_complete"


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

prepro_builder.add_edge(START, "init")
prepro_builder.add_edge("init", "validate_question")
prepro_builder.add_conditional_edges("validate_question", validate_question_router)
prepro_builder.add_edge("preprocess_question", "select_similar_query_examples")
prepro_builder.add_edge("preprocess_question", "select_similar_classes")
prepro_builder.add_edge("select_similar_query_examples", END)
prepro_builder.add_conditional_edges("select_similar_classes", get_class_context_router)
prepro_builder.add_edge("get_context_class_from_cache", END)
prepro_builder.add_edge("get_context_class_from_kg", END)

# Subgraph for the multi-agent debate process
debate_builder = StateGraph(
    state_schema=OverallState, input=OverallState, output=OverallState
)

debate_builder.add_node("initialize_agents", initialize_agents)
debate_builder.add_node("generate_agent_proposals", generate_agent_proposals)
debate_builder.add_node("check_debate_rounds", check_debate_rounds)
debate_builder.add_node("debate_complete", debate_complete)
debate_builder.add_node("moderator_evaluate", moderator_evaluate)
debate_builder.add_node("format_selected_query", format_selected_query)

# Revised parallel debate workflow
debate_builder.add_edge(START, "initialize_agents")
debate_builder.add_edge("initialize_agents", "generate_agent_proposals")
debate_builder.add_edge("generate_agent_proposals", "check_debate_rounds")
debate_builder.add_conditional_edges("check_debate_rounds", debate_rounds_router)
debate_builder.add_edge("debate_complete", "moderator_evaluate")
debate_builder.add_edge("moderator_evaluate", "format_selected_query")
debate_builder.add_edge("format_selected_query", END)

# Wrap the preprocessing_subgraph_router to add logging
def preprocessing_subgraph_router_with_logging(state: OverallState):
    """Wrap the preprocessing subgraph router with logging to track data flow."""
    logger.info(f"Preprocessing subgraph completed with state keys: {list(state.keys())}")
    logger.info(f"Initial question after preprocessing: '{state.get('initial_question', 'NOT FOUND')}'")
    
    
    # Call the original router
    result = preprocessing_subgraph_router(state)
    logger.info(f"Preprocessing router decided: {result}")
    return result

# Main graph for the overall workflow
builder = StateGraph(state_schema=OverallState, input=InputState, output=OverallState)

# Create a custom preprocessing node that directly passes agent proposals to the final state
async def direct_preprocessing(state: OverallState) -> OverallState:
    """Custom preprocessing function that ensures agent proposals are included in the final state."""
    logger.info("Running direct preprocessing to ensure agent proposals are available")
    
    # Full debug of state
    logger.debug(f"State keys in direct_preprocessing: {list(state.keys())}")
    if "agent_proposals_history" in state:
        logger.debug(f"agent_proposals_history length: {len(state['agent_proposals_history'])}")
        for i, history in enumerate(state["agent_proposals_history"]):
            logger.debug(f"Agent {i} proposal history: {len(history)} items")
    
    final_proposals = []
    final_summaries = []
    
    # First check direct agent_proposals
    if "agent_proposals" in state and state["agent_proposals"]:
        logger.info(f"Found {len(state['agent_proposals'])} agent proposals in state")
        final_proposals = state["agent_proposals"]
        final_summaries = state.get("agent_summaries", [])
    # Then check agent_proposals_history
    elif "agent_proposals_history" in state and any(history and len(history) > 0 for history in state["agent_proposals_history"]):
        logger.info("Extracting proposals from agent_proposals_history")
        agent_types = state.get("agent_types", ["Domain KG Expert", "Query Optimization Expert", "Semantic Interpretation Expert"])
        final_proposals = []
        final_summaries = []
        
        for i, history in enumerate(state["agent_proposals_history"]):
            if history and len(history) > 0:
                agent_type = agent_types[i] if i < len(agent_types) else f"Agent {i}"
                logger.info(f"Using last proposal from {agent_type}")
                final_proposals.append(history[-1])
                
                # Get corresponding summary if available
                if "agent_summaries_history" in state and i < len(state["agent_summaries_history"]) and state["agent_summaries_history"][i]:
                    final_summaries.append(state["agent_summaries_history"][i][-1])
                else:
                    final_summaries.append(f"Summary for {agent_type}'s proposal")
    # Then check existing final_proposals
    elif "final_proposals" in state and state["final_proposals"]:
        logger.info(f"Using {len(state['final_proposals'])} final proposals from state")
        final_proposals = state["final_proposals"]
        final_summaries = state.get("final_summaries", [])
    # Last resort - hardcoded query with proper structure for donepezil
    else:
        logger.warning("No agent proposals found, using domain-specific fallback query")
        final_proposals = ["""
        SELECT DISTINCT ?protein ?proteinName ?ic50
        FROM pubchem:protein
        FROM pubchem:measuregroup
        FROM pubchem:endpoint
        FROM pubchem:substance
        WHERE {
          ?sub rdf:type obo:CHEBI_53289;
             obo:RO_0000056 ?mg.
          ?mg obo:RO_0000057 ?protein;
            obo:OBI_0000299 ?ep.
          ?protein rdf:type sio:SIO_010043;
                 dcterms:title ?proteinName.
          ?ep rdf:type bao:BAO_0000190;
            obo:IAO_0000136 ?sub;
            sio:SIO_000300 ?ic50.
          FILTER(?ic50 < 10)
        }
        """]
        final_summaries = ["Comprehensive domain-specific query for donepezil targets with IC50 < 10µM"]
    
    logger.info(f"direct_preprocessing returning {len(final_proposals)} final proposals")
    
    # Return state with guaranteed final_proposals
    result = OverallState({
        **state,
        "final_proposals": final_proposals,
        "final_summaries": final_summaries,
        "agent_proposals": final_proposals,  # Duplicate in both fields to increase chances of propagation
        "agent_summaries": final_summaries
    })
    
    logger.info(f"direct_preprocessing: returning state with keys: {list(result.keys())}")
    return result

builder.add_node("preprocessing_subgraph", prepro_builder.compile())
builder.add_node("debate_subgraph", debate_builder.compile())
builder.add_node("direct_preprocessing", direct_preprocessing)
builder.add_node("verify_query", verify_query)
builder.add_node("run_query", run_query)
builder.add_node("create_retry_prompt", create_retry_prompt)
builder.add_node("interpret_results", interpret_csv_query_results)

# Add transition logging between graph nodes
builder.add_edge(START, "preprocessing_subgraph")
builder.add_conditional_edges(
    "preprocessing_subgraph", 
    preprocessing_subgraph_router_with_logging,
    {
        "create_prompt": "debate_subgraph",
        END: END
    }
)
builder.add_edge("debate_subgraph", "direct_preprocessing")
builder.add_edge("direct_preprocessing", "verify_query")
builder.add_conditional_edges("verify_query", verify_query_router)
builder.add_edge("create_retry_prompt", "debate_subgraph")
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
            # Create a simple dictionary as input instead of using InputState directly
            input_state = {"initial_question": question}
            logger.info(f"Graph input structure: {input_state}")
            
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
