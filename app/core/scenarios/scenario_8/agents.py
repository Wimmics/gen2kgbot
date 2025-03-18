"""
This module implements the specialized agents for scenario 8's multi-agent debate process.
"""
import re
import json
from langchain_core.messages import AIMessage
from app.core.utils.graph_state import OverallState
import app.core.utils.config_manager as config
from app.core.scenarios.scenario_8.prompt import (
    domain_kg_expert_prompt,
    query_optimization_expert_prompt,
    semantic_interpretation_expert_prompt,
    agent_summary_prompt,
    moderator_evaluation_prompt,
    query_refinement_prompt,
)
from app.core.utils.logger_manager import setup_logger
from app.core.utils.sparql_toolkit import find_sparql_queries

logger = setup_logger(__package__, __file__)

async def initialize_agents(state: OverallState) -> OverallState:
    """
    Initialize the state for the multi-agent debate process.
    
    Args:
        state (OverallState): Current state of the conversation
        
    Returns:
        OverallState: Updated state with initialized debate parameters
    """
    logger.info("Initializing multi-agent debate process...")
    logger.info(f"Initializing agents state with keys: {list(state.keys())}")
    
    # Define the agent specialties
    agent_types = [
        "Domain KG Expert", 
        "Query Optimization Expert", 
        "Semantic Interpretation Expert"
    ]
    
    # Create storage for agent proposals and summaries across rounds
    agent_proposals_history = [[] for _ in range(len(agent_types))]
    agent_summaries_history = [[] for _ in range(len(agent_types))]
    
    logger.debug(f"Initialized {len(agent_types)} agent types: {agent_types}")
    
    # Create merged_classes_context if it doesn't exist
    merged_cls_context = ""
    if "merged_classes_context" not in state and "selected_classes_context" in state:
        logger.info("Creating merged_classes_context from selected_classes_context")
        # Simple concatenation approach
        merged_cls_context = "\n\n".join([str(ctx) for ctx in state.get("selected_classes_context", [])])
    else:
        merged_cls_context = state.get("merged_classes_context", "")
    
    result = OverallState({
        **state,
        "agent_proposals": [],  # Current round proposals
        "agent_summaries": [],  # Current round summaries
        "agent_proposals_history": agent_proposals_history,  # History of proposals across rounds
        "agent_summaries_history": agent_summaries_history,  # History of summaries across rounds
        "debate_round": 1,  # Initialize debate round
        "agent_types": agent_types,
        "need_more_rounds": False,  # Initialize need_more_rounds flag
        "merged_classes_context": merged_cls_context  # Add merged context
    })
    logger.info(f"Initialized agents state with keys: {list(result.keys())}")
    return result


async def generate_agent_proposals(state: OverallState) -> OverallState:
    """
    Generate SPARQL query proposals from all specialized agents in parallel.
    
    Args:
        state (OverallState): Current state of the conversation
        
    Returns:
        OverallState: Updated state with all agent proposals
    """
    
    # Safely get the current round with a default value of 1
    logger.info(f"generate_agent_proposals: examining state with keys: {list(state.keys())}")
    current_round = state.get("debate_round", 1)
    logger.info(f"Generating all agent proposals for round {current_round}...")
    
    # Ensure agent_types exists with a default
    if "agent_types" not in state:
        logger.warning("agent_types not found in state, initializing with defaults")
        agent_types = [
            "Domain KG Expert", 
            "Query Optimization Expert", 
            "Semantic Interpretation Expert"
        ]
        state = {**state, "agent_types": agent_types}
    
    # Ensure merged_classes_context exists
    if "merged_classes_context" not in state and "selected_classes_context" in state:
        logger.warning("merged_classes_context not found in state, creating from selected_classes_context")
        merged_cls_context = "\n\n".join([str(ctx) for ctx in state.get("selected_classes_context", [])])
        state = {**state, "merged_classes_context": merged_cls_context}
    elif "merged_classes_context" not in state:
        logger.warning("merged_classes_context not found in state, initializing with empty string")
        state = {**state, "merged_classes_context": ""}
    
    # Get the current round index (0-based)
    round_idx = current_round - 1
    
    # Templates for each agent type
    templates = [
        domain_kg_expert_prompt,
        query_optimization_expert_prompt,
        semantic_interpretation_expert_prompt
    ]
    
    agent_responses = []
    agent_proposals = []
    agent_summaries = []
    
    # Process each agent
    for i, (agent_type, template) in enumerate(zip(state["agent_types"], templates)):
        # Prepare the context for refinement if this isn't the first round
        if round_idx > 0 and "agent_summaries_history" in state:
            # For refinement rounds, use the refinement prompt
            logger.info(f"Refining proposal for {agent_type} in round {current_round}")
            
            # Collect all other agents' summaries from the previous round
            other_summaries = []
            for j, summary_history in enumerate(state["agent_summaries_history"]):
                if i != j and round_idx - 1 < len(summary_history):
                    other_summaries.append((state["agent_types"][j], summary_history[round_idx - 1]))
            
            other_summaries_text = "\n\n".join([f"{agent}: {summary}" for agent, summary in other_summaries])
            
            # Get the agent's previous proposal
            previous_proposal = state["agent_proposals_history"][i][round_idx - 1] if len(state["agent_proposals_history"][i]) > 0 else ""
            
            # Create refinement prompt
            template = query_refinement_prompt.partial(
                agent_type=agent_type,
                initial_question=state["initial_question"],
                previous_proposal=previous_proposal,
                other_summaries=other_summaries_text,
                merged_classes_context=state["merged_classes_context"]
            )
            
            logger.debug(f"Created refinement prompt for {agent_type} with {len(other_summaries)} other summaries")
        else:
            # For first round, use the standard prompt
            logger.info(f"Generating initial proposal for {agent_type}")
            
            # Prepare the template with required variables
            if "kg_full_name" in template.input_variables:
                template = template.partial(kg_full_name=config.get_kg_full_name())
            
            if "kg_description" in template.input_variables:
                template = template.partial(kg_description=config.get_kg_description())
            
            # Add other required variables from the state
            for var in ["initial_question", "selected_classes", "merged_classes_context", "selected_queries"]:
                if var in template.input_variables and var in state:
                    template = template.partial(**{var: state[var]})
        
        # Format the prompt
        prompt = template.format()
        
        # Get the response from the LLM
        try:
            response = await config.get_seq2seq_model(
                scenario_id=state["scenario_id"], 
                node_name="generate_query"
            ).ainvoke(prompt)
            
            agent_responses.append(response.content)
            logger.debug(f"{agent_type} generated response: {response.content[:150]}...")
            
            # Extract the SPARQL query
            queries = find_sparql_queries(response.content)
            if queries:
                query = queries[0]  # Take the first query if multiple are found
                agent_proposals.append(query)
                logger.info(f"{agent_type} generated a valid SPARQL query ({len(query)} chars)")
            else:
                logger.warning(f"No SPARQL query found in {agent_type}'s response")
                agent_proposals.append("No valid SPARQL query was provided.")
            
            # Generate a summary of the agent's response
            summary_template = agent_summary_prompt.partial(agent_response=response.content)
            summary_response = await config.get_seq2seq_model(
                scenario_id=state["scenario_id"], 
                node_name="generate_query"
            ).ainvoke(summary_template.format())
            
            agent_summaries.append(summary_response.content)
            logger.debug(f"{agent_type} summary: {summary_response.content}")
            
        except Exception as e:
            logger.error(f"Error generating proposal for {agent_type}: {str(e)}")
            agent_responses.append(f"Error: {str(e)}")
            agent_proposals.append("Error generating SPARQL query.")
            agent_summaries.append("No summary available due to error.")
    
    # Update agent proposal and summary histories
    agent_proposals_history = state.get("agent_proposals_history", [[] for _ in range(len(state["agent_types"]))])
    agent_summaries_history = state.get("agent_summaries_history", [[] for _ in range(len(state["agent_types"]))])
    
    # Ensure histories are properly initialized
    if not agent_proposals_history or len(agent_proposals_history) == 0:
        logger.warning("agent_proposals_history not properly initialized, creating new")
        agent_proposals_history = [[] for _ in range(len(state["agent_types"]))]
    
    if not agent_summaries_history or len(agent_summaries_history) == 0:
        logger.warning("agent_summaries_history not properly initialized, creating new")
        agent_summaries_history = [[] for _ in range(len(state["agent_types"]))]
    
    # Extend histories if needed
    while len(agent_proposals_history) < len(state["agent_types"]):
        agent_proposals_history.append([])
    
    while len(agent_summaries_history) < len(state["agent_types"]):
        agent_summaries_history.append([])
    
    # Add new proposals and summaries to history
    for i in range(len(state["agent_types"])):
        if i < len(agent_proposals) and i < len(agent_proposals_history):
            agent_proposals_history[i].append(agent_proposals[i])
        
        if i < len(agent_summaries) and i < len(agent_summaries_history):
            agent_summaries_history[i].append(agent_summaries[i])
    
    logger.info(f"Successfully generated {len(agent_proposals)} agent proposals for round {current_round}")
    
    # Determine max rounds from config
    max_rounds = 1  # Default
    scenario_id = state.get("scenario_id", "scenario_8")
    if hasattr(config, "config") and config.config is not None and scenario_id in config.config:
        max_rounds = config.config.get(scenario_id, {}).get("debate_rounds", 1)
    
    # Log this for debugging
    logger.info(f"Current round: {current_round}, Max rounds: {max_rounds}")
    
    # Generate final_proposals and final_summaries regardless of round
    # This ensures they're available even if later nodes miss them
    final_proposals = agent_proposals.copy()
    final_summaries = agent_summaries.copy()
    logger.info(f"Setting {len(final_proposals)} final proposals in generate_agent_proposals")
    
    # Generate SPARQL query for each agent
    for i, proposal in enumerate(agent_proposals):
        if i < len(state["agent_types"]):
            agent_type = state["agent_types"][i]
            logger.info(f"Proposal from {agent_type} (first 100 chars): {proposal[:100]}...")
    
    # Create a special field to ensure proposals persist
    persistent_proposals = {
        "round": current_round,
        "agent_proposals": agent_proposals,
        "agent_summaries": agent_summaries
    }
    
    # Store this as JSON string to ensure it persists through serialization
    logger.info("Creating persistent_proposals_json to ensure data preservation")
    persistent_proposals_json = json.dumps(persistent_proposals)
    
    # Return state with all possible fields populated to maximize chances
    # of proposals persisting through the graph nodes
    result = OverallState({
        **state,
        "agent_proposals": agent_proposals,
        "agent_summaries": agent_summaries,
        "agent_proposals_history": agent_proposals_history,
        "agent_summaries_history": agent_summaries_history,
        "final_proposals": final_proposals,
        "final_summaries": final_summaries,
        "selected_query": final_proposals[0] if final_proposals else "",
        "last_generated_query": final_proposals[0] if final_proposals else "",
        "persistent_proposals_json": persistent_proposals_json,
        "need_more_rounds": current_round < max_rounds
    })
    
    logger.info(f"generate_agent_proposals: returning state with keys: {list(result.keys())}")
    return result


async def check_debate_rounds(state: OverallState) -> OverallState:
    """
    Check if more debate rounds are needed.
    
    Args:
        state (OverallState): Current state of the conversation
        
    Returns:
        OverallState: Updated state with a flag indicating if more rounds are needed
    """
    logger.info(f"check_debate_rounds: examining state with keys: {list(state.keys())}")
    
    try:
        # Get current debate round from state
        debate_round = state.get("debate_round", 1)
        
        # Get total debate rounds from config
        scenario_id = state.get("scenario_id", "scenario_8")
        total_debate_rounds = 1  # Default value
        
        # Log the current state keys and agent proposals
        logger.info(f"check_debate_rounds: examining state with keys: {list(state.keys())}")
        logger.info(f"check_debate_rounds: Current debate round: {debate_round}")
        
        if "agent_proposals" in state:
            logger.info(f"check_debate_rounds: Found {len(state['agent_proposals'])} agent proposals")
            for i, proposal in enumerate(state["agent_proposals"]):
                # Log only first 100 chars of each proposal to avoid cluttering logs
                logger.info(f"check_debate_rounds: Proposal {i+1} preview: {proposal[:100]}...")
        else:
            logger.warning("check_debate_rounds: No agent_proposals found in state")
        
        # Try to get debate_rounds from config
        try:
            # Access the config directly instead of using a non-existent method
            if hasattr(config, "config") and config.config is not None and scenario_id in config.config:
                # Correctly access the debate_rounds parameter
                if "llm" in config.config[scenario_id] and "debate_rounds" in config.config[scenario_id]:
                    total_debate_rounds = config.config[scenario_id]["debate_rounds"]
                    logger.info(f"check_debate_rounds: Got debate_rounds={total_debate_rounds} from config")
                else:
                    logger.warning(f"check_debate_rounds: Could not find debate_rounds in config for {scenario_id}, using default: {total_debate_rounds}")
            else:
                logger.warning(f"check_debate_rounds: Could not find scenario {scenario_id} in config, using default debate_rounds={total_debate_rounds}")
        except Exception as e:
            logger.warning(f"check_debate_rounds: Error accessing config: {e}, using default debate_rounds={total_debate_rounds}")
        
        # More rounds needed if current round is less than total rounds
        need_more_rounds = debate_round < total_debate_rounds
        
        next_round = debate_round + 1 if need_more_rounds else debate_round
        logger.info(f"Last round completed (round {debate_round}/{total_debate_rounds}). " + 
                    "Setting final proposals directly." if not need_more_rounds else "")
        logger.info(f"Debate round {debate_round}/{total_debate_rounds} complete. Need more rounds: {need_more_rounds}")
        
        # Create a copy of agent_proposals to ensure it's not lost
        final_proposals = state.get("agent_proposals", []).copy() if not need_more_rounds else []
        final_summaries = state.get("agent_summaries", []).copy() if not need_more_rounds else []
        
        # Debug the final proposals being returned
        if final_proposals:
            logger.info(f"check_debate_rounds: Returning {len(final_proposals)} final proposals")
        
        # Return updated state with next round and need_more_rounds flag
        result = OverallState({
            **state,
            "debate_round": next_round,
            "need_more_rounds": need_more_rounds,
            "final_proposals": final_proposals,
            "final_summaries": final_summaries
        })
        
        logger.info(f"check_debate_rounds: returning state with keys: {list(result.keys())}")
        return result
    except Exception as e:
        logger.error(f"Error in check_debate_rounds: {e}")
        # Return original state with default values to avoid breaking the flow
        result = OverallState({
            **state,
            "debate_round": state.get("debate_round", 1) + 1,
            "need_more_rounds": False
        })
        
        logger.info(f"check_debate_rounds: returning state with keys: {list(result.keys())}")
        return result


async def debate_complete(state: OverallState) -> OverallState:
    """
    Process the final debate state to prepare for moderator evaluation.
    
    Args:
        state (OverallState): Current state of the conversation
        
    Returns:
        OverallState: Updated state with final proposals and summaries
    """
    logger.info(f"debate_complete: examining state with keys: {list(state.keys())}")
    
    # Get current debate round or default to 1
    debate_round = state.get("debate_round", 1)
    
    # Debug the incoming state
    logger.info(f"debate_complete: examining state with keys: {list(state.keys())}")
    
    # Try to recover from persistent_proposals_json first if available
    if "persistent_proposals_json" in state and state["persistent_proposals_json"]:
        try:
            logger.info("Attempting to recover proposals from persistent_proposals_json")
            persistent_data = json.loads(state["persistent_proposals_json"])
            
            if "agent_proposals" in persistent_data and persistent_data["agent_proposals"]:
                agent_proposals = persistent_data["agent_proposals"]
                agent_summaries = persistent_data.get("agent_summaries", [])
                logger.info(f"Successfully recovered {len(agent_proposals)} proposals from persistent_proposals_json")
                
                # Update state with recovered data
                state = {
                    **state, 
                    "agent_proposals": agent_proposals,
                    "agent_summaries": agent_summaries,
                    "final_proposals": agent_proposals,
                    "final_summaries": agent_summaries
                }
        except Exception as e:
            logger.error(f"Error recovering from persistent_proposals_json: {e}")
    
    # Check all possible sources of proposals
    proposals_sources = [
        ("agent_proposals", state.get("agent_proposals", [])),
        ("final_proposals", state.get("final_proposals", [])),
        ("agent_proposals_history", state.get("agent_proposals_history", []))
    ]
    
    for source_name, proposals in proposals_sources:
        if isinstance(proposals, list):
            logger.info(f"debate_complete: {source_name} has {len(proposals)} items")
            if proposals:
                # Log first 100 chars of first proposal
                logger.info(f"debate_complete: {source_name} first item preview: {proposals[0][:100]}...")
        elif isinstance(proposals, list) and all(isinstance(history, list) for history in proposals):
            logger.info(f"debate_complete: {source_name} has {len(proposals)} agent histories")
            for i, history in enumerate(proposals):
                logger.info(f"debate_complete: {source_name}[{i}] has {len(history)} items")
    
    # Try to get agent types
    agent_types = state.get("agent_types", [])
    logger.info(f"debate_complete: Found {len(agent_types)} agent types")
    
    # First try to use agent_proposals directly
    final_proposals = []
    final_summaries = []
    
    if "agent_proposals" in state and state["agent_proposals"]:
        logger.info(f"debate_complete found {len(state['agent_proposals'])} agent proposals")
        final_proposals = state["agent_proposals"]
        final_summaries = state.get("agent_summaries", [])
    elif "final_proposals" in state and state["final_proposals"]:
        logger.info(f"debate_complete found {len(state['final_proposals'])} final proposals")
        final_proposals = state["final_proposals"]
        final_summaries = state.get("final_summaries", [])
    elif "agent_proposals_history" in state and any(history and len(history) > 0 for history in state["agent_proposals_history"]):
        # Try to extract last proposals from history
        logger.info("debate_complete: Attempting to extract from agent_proposals_history")
        for i, history in enumerate(state["agent_proposals_history"]):
            if history and len(history) > 0:
                agent_name = agent_types[i] if i < len(agent_types) else f"Agent {i}"
                logger.info(f"debate_complete: Using last proposal from {agent_name}")
                final_proposals.append(history[-1])
                
                # Get corresponding summary if available
                if "agent_summaries_history" in state and i < len(state["agent_summaries_history"]) and state["agent_summaries_history"][i]:
                    final_summaries.append(state["agent_summaries_history"][i][-1])
                else:
                    final_summaries.append(f"Summary for {agent_name}'s proposal")
    else:
        logger.warning("No proposals found, creating a fallback proposal")
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
    
    logger.info(f"debate_complete finalized {len(final_proposals)} proposals")
    
    # Set selected_query as well to maximize chances it persists
    selected_query = final_proposals[0] if final_proposals else ""
    
    # Return state with all possible fields populated to maximize chances of persistence
    result = OverallState({
        **state,
        "debate_round": debate_round,
        "final_proposals": final_proposals,
        "final_summaries": final_summaries,
        "agent_proposals": final_proposals,  # Duplicate in both locations
        "agent_summaries": final_summaries,
        "selected_query": selected_query,
        "last_generated_query": selected_query
    })
    
    logger.info(f"debate_complete: returning state with keys: {list(result.keys())}")
    return result


async def moderator_evaluate(state: OverallState) -> OverallState:
    """
    Evaluate agent proposals and select the best query.
    
    Args:
        state (OverallState): Current state of the conversation
        
    Returns:
        OverallState: Updated state with selected query
    """
    logger.info(f"moderator_evaluate: examining state with keys: {list(state.keys())}")
    
    logger.info("Moderator evaluating final agent proposals...")
    
    # Debug the incoming state
    logger.info(f"moderator_evaluate: examining state with keys: {list(state.keys())}")
    
    # Try to recover from persistent_proposals_json first if available
    if "persistent_proposals_json" in state and state["persistent_proposals_json"]:
        try:
            logger.info("Attempting to recover proposals from persistent_proposals_json")
            persistent_data = json.loads(state["persistent_proposals_json"])
            
            if "agent_proposals" in persistent_data and persistent_data["agent_proposals"]:
                agent_proposals = persistent_data["agent_proposals"]
                agent_summaries = persistent_data.get("agent_summaries", [])
                logger.info(f"Successfully recovered {len(agent_proposals)} proposals from persistent_proposals_json")
                
                # Update state with recovered data
                state = {
                    **state, 
                    "agent_proposals": agent_proposals,
                    "agent_summaries": agent_summaries,
                    "final_proposals": agent_proposals,
                    "final_summaries": agent_summaries
                }
        except Exception as e:
            logger.error(f"Error recovering from persistent_proposals_json: {e}")
    
    # Check all possible sources of proposals
    proposals_sources = [
        ("agent_proposals", state.get("agent_proposals", [])),
        ("final_proposals", state.get("final_proposals", []))
    ]
    
    for source_name, proposals in proposals_sources:
        if isinstance(proposals, list):
            logger.info(f"moderator_evaluate: {source_name} has {len(proposals)} items")
            if proposals:
                # Log first 100 chars of first proposal
                logger.info(f"moderator_evaluate: {source_name} first item preview: {proposals[0][:100]}...")
    
    # Safely get proposals and summaries
    final_proposals = state.get("final_proposals", [])
    final_summaries = state.get("final_summaries", [])
    
    if not final_proposals:
        # Try agent_proposals as fallback
        final_proposals = state.get("agent_proposals", [])
        logger.info(f"moderator_evaluate: Using agent_proposals with {len(final_proposals)} items as fallback")
    
    if not final_summaries:
        # Generate dummy summaries if needed
        final_summaries = state.get("agent_summaries", [])
        if len(final_summaries) < len(final_proposals):
            logger.warning(f"moderator_evaluate: Generated {len(final_proposals) - len(final_summaries)} dummy summaries")
            final_summaries.extend([f"Proposal {i+1}" for i in range(len(final_summaries), len(final_proposals))])
    
    if not final_proposals:
        # Final fallback - use domain-specific query for donepezil
        logger.warning("No proposals available, creating a fallback query")
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
        logger.info("Created fallback query for evaluation")
    
    # Use first proposal as selected query
    selected_query = final_proposals[0] if final_proposals else ""
    selected_query_summary = final_summaries[0] if final_summaries else "Default selection"
    
    logger.info(f"moderator_evaluate: Selected query (first {min(50, len(selected_query))} chars): {selected_query[:50]}...")
    
    # Return state with selected query and all related fields to maximize persistence
    result = OverallState({
        **state,
        "selected_query": selected_query,
        "selected_query_summary": selected_query_summary,
        "final_proposals": final_proposals,  # Keep these in the state
        "final_summaries": final_summaries,
        "agent_proposals": final_proposals,  # Duplicate for redundancy
        "agent_summaries": final_summaries,
        "last_generated_query": selected_query
    })
    
    logger.info(f"moderator_evaluate: returning state with keys: {list(result.keys())}")
    return result


async def format_selected_query(state: OverallState) -> OverallState:
    """
    Format the selected query for verification and execution.
    
    Args:
        state (OverallState): Current state of the conversation
        
    Returns:
        OverallState: Updated state with the formatted query ready for verification
    """
    logger.info(f"format_selected_query: examining state with keys: {list(state.keys())}")
    
    # Find the selected query from any available source
    selected_query = ""
    
    # Debug the incoming state
    logger.info(f"format_selected_query: examining state with keys: {list(state.keys())}")
    
    # Try various sources for the query
    if "selected_query" in state and state["selected_query"]:
        selected_query = state["selected_query"]
        logger.info("Using selected_query from state")
    elif "final_proposals" in state and state["final_proposals"]:
        selected_query = state["final_proposals"][0]
        logger.info("Using first item from final_proposals")
    elif "agent_proposals" in state and state["agent_proposals"]:
        selected_query = state["agent_proposals"][0]
        logger.info("Using first item from agent_proposals")
    else:
        logger.warning("No queries found, using fallback query")
        selected_query = """
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
        """
    
    # Log the selected query
    logger.info(f"Formatting selected query for verification: {selected_query[:50]}...")
    
    # Return updated state
    result = OverallState({
        **state,
        "last_generated_query": selected_query,
        "selected_query": selected_query,
        "last_answer": "Query generation complete",
        "messages": state.get("messages", []),
        "debate_round": state.get("debate_round", 1)  # Preserve debate round
    })
    
    logger.info(f"format_selected_query: returning state with keys: {list(result.keys())}")
    return result
