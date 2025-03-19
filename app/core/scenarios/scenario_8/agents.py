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

class Agent:
    """Base class for specialized agents in the debate process."""
    def __init__(self, agent_type: str, prompt_template):
        self.agent_type = agent_type
        self.prompt_template = prompt_template
        
        # Map agent type to model name in config
        model_name_map = {
            "Domain KG Expert": "domain_kg_expert",
            "Query Optimization Expert": "query_optimization_expert",
            "Semantic Interpretation Expert": "semantic_interpretation_expert"
        }
        
        # Use agent-specific model if available, otherwise fallback to generate_query
        model_node_name = model_name_map.get(agent_type, "generate_query")
        logger.info(f"Initializing {agent_type} with model: {model_node_name}")
        self.model = config.get_seq2seq_model(scenario_id="scenario_8", node_name=model_node_name)
    
    async def generate_proposal(self, state: OverallState, round_number: int = 1) -> str:
        """Generate a SPARQL query proposal based on the current state and round number."""
        logger.info(f"{self.agent_type} generating proposal for round {round_number}")
        
        if round_number == 1:
            logger.info(f"{self.agent_type} using standard prompt for first round")
            # First round: use standard prompt
            prompt = self.prompt_template.format(
                kg_full_name=config.get_kg_full_name(),
                kg_description=state.get("kg_description", ""),
                initial_question=state.get("initial_question", ""),
                selected_classes=state.get("selected_classes", ""),
                merged_classes_context=state.get("merged_classes_context", ""),
                selected_queries=state.get("selected_queries", "")
            )
        else:
            logger.info(f"{self.agent_type} using refinement prompt for round {round_number}")
            # Subsequent rounds: use refinement prompt
            # Get previous proposal and other agents' summaries
            previous_proposal = state.get("agent_proposals_history", [[] for _ in range(3)])[state.get("agent_types", []).index(self.agent_type)][-1]
            logger.debug(f"{self.agent_type}'s previous proposal: {previous_proposal[:100]}...")
            
            # Collect other agents' summaries
            other_summaries = []
            for i, agent_type in enumerate(state.get("agent_types", [])):
                if agent_type != self.agent_type and i < len(state.get("agent_summaries_history", [])):
                    other_summaries.append(f"{agent_type}: {state['agent_summaries_history'][i][-1]}")
            
            other_summaries_text = "\n\n".join(other_summaries)
            logger.debug(f"Other agents' summaries: {other_summaries_text[:100]}...")
            
            # Use refinement prompt
            prompt = query_refinement_prompt.format(
                kg_full_name=config.get_kg_full_name(),
                kg_description=state.get("kg_description", ""),
                agent_type=self.agent_type,
                initial_question=state.get("initial_question", ""),
                previous_proposal=previous_proposal,
                other_summaries=other_summaries_text,
                merged_classes_context=state.get("merged_classes_context", "")
            )
        
        # Get response from the model
        logger.info(f"{self.agent_type} sending prompt to model")
        response = await self.model.ainvoke(prompt)
        logger.debug(f"{self.agent_type}'s model response: {response.content[:100]}...")
        
        # Extract SPARQL query from response
        queries = find_sparql_queries(response.content)
        if queries:
            logger.info(f"{self.agent_type} successfully generated a valid SPARQL query")
            return queries[0]  # Take the first query if multiple are found
        else:
            logger.warning(f"{self.agent_type} failed to generate a valid SPARQL query")
            return "No valid SPARQL query was provided."
    
    async def generate_summary(self, proposal: str) -> str:
        """Generate a summary of the proposal."""
        logger.info(f"{self.agent_type} generating summary for proposal")
        logger.debug(f"Proposal to summarize: {proposal[:100]}...")
        
        # Prepare the summary prompt
        prompt = agent_summary_prompt.format(agent_response=proposal)
        
        # Get summary from the model
        logger.info(f"{self.agent_type} sending summary prompt to model")
        response = await self.model.ainvoke(prompt)
        logger.debug(f"{self.agent_type}'s summary: {response.content[:100]}...")
        
        return response.content

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
    
    # Define the agent specialties and their prompts
    agent_types = [
        "Domain KG Expert", 
        "Query Optimization Expert", 
        "Semantic Interpretation Expert"
    ]
    agent_prompts = [
        domain_kg_expert_prompt,
        query_optimization_expert_prompt,
        semantic_interpretation_expert_prompt
    ]
    
    # Create agent instances
    agents = [
        Agent(agent_type, prompt)
        for agent_type, prompt in zip(agent_types, agent_prompts)
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
        "agents": agents,  # Store agent instances in state
        "agent_proposals": [],  # Current round proposals
        "agent_summaries": [],  # Current round summaries
        "agent_proposals_history": agent_proposals_history,  # History of proposals across rounds
        "agent_summaries_history": agent_summaries_history,  # History of summaries across rounds
        "debate_round": 1,  # Initialize debate round
        "agent_types": agent_types,
        "merged_classes_context": merged_cls_context  # Add merged context
    })
    logger.info(f"Initialized agents state with keys: {list(result.keys())}")
    return result


async def debate(state: OverallState) -> OverallState:
    """Generate proposals from all agents and check if debate is complete."""
    logger.info("Starting agent proposal generation and debate round check")
    logger.info(f"Debate state keys: {list(state.keys())}")
    
    # Get current debate round and max rounds
    current_round = state.get("debate_round", 1)
    max_rounds = state.get("max_debate_rounds", 3)
    
    # Initialize or get existing proposals and summaries
    agent_proposals = state.get("agent_proposals", [])
    agent_summaries = state.get("agent_summaries", [])
    agent_proposals_history = state.get("agent_proposals_history", [[] for _ in range(3)])
    agent_summaries_history = state.get("agent_summaries_history", [[] for _ in range(3)])
    
    # Loop over debate rounds
    while current_round <= max_rounds:
        logger.info(f"Starting debate round {current_round}/{max_rounds}")
        
        # Generate proposals for each agent
        for i, agent in enumerate(state.get("agents", [])):
            logger.info(f"Generating proposal for agent {i}")
            try:
                # Generate proposal with current round number
                proposal = await agent.generate_proposal(state, current_round)
                summary = await agent.generate_summary(proposal)
                
                # Store in history
                agent_proposals_history[i].append(proposal)
                agent_summaries_history[i].append(summary)
                
                # Update current proposals
                if len(agent_proposals) <= i:
                    agent_proposals.append(proposal)
                    agent_summaries.append(summary)
                else:
                    agent_proposals[i] = proposal
                    agent_summaries[i] = summary
                    
            except Exception as e:
                logger.error(f"Error generating proposal for agent {i}: {e}")
                continue
        
        logger.info(f"Completed round {current_round} with {len(agent_proposals)} proposals")
        
        # Check if we need more rounds
        if current_round < max_rounds:
            # Update state for next round
            state = OverallState({
                **state,
                "agent_proposals": agent_proposals,
                "agent_summaries": agent_summaries,
                "agent_proposals_history": agent_proposals_history,
                "agent_summaries_history": agent_summaries_history,
                "debate_round": current_round + 1
            })
            current_round += 1
        else:
            break
    
    # Final state update after all rounds - only pass final_proposals
    updated_state = OverallState({
        **state,
        "final_proposals": agent_proposals  # Only pass the final proposals
    })
    
    logger.info(f"Debate complete after {current_round} rounds")
    return updated_state


async def moderator_evaluate(state: OverallState) -> OverallState:
    """
    Evaluate agent proposals and select the best query.
    
    Args:
        state (OverallState): Current state of the conversation
        
    Returns:
        OverallState: Updated state with selected query
    """
    logger.info("Moderator evaluating final agent proposals...")
    
    # Get final proposals
    final_proposals = state.get("final_proposals", [])
    
    if not final_proposals:
        logger.warning("No proposals available for evaluation, checking agent_proposals")
        final_proposals = state.get("agent_proposals", [])
        
    if not final_proposals:
        logger.error("No proposals available for evaluation")
        return OverallState({
            **state,
            "selected_query": "",
            "error": "No proposals were generated during the debate process"
        })
    
    # Ensure we have 3 proposals (pad with empty ones if needed)
    while len(final_proposals) < 3:
        logger.warning(f"Only {len(final_proposals)} proposals available, padding with empty ones")
        final_proposals.append("# Empty proposal\nSELECT * WHERE { ?s ?p ?o } LIMIT 0")
    
    # Prepare moderator prompt
    prompt = moderator_evaluation_prompt.format(
        initial_question=state.get("initial_question", ""),
        kg_description=state.get("kg_description", ""),
        merged_classes_context=state.get("merged_classes_context", ""),
        agent_proposals=final_proposals
    )
    
    # Get evaluation from the model - use dedicated moderator_evaluate model
    try:
        logger.info("Using dedicated moderator_evaluate model")
        model = config.get_seq2seq_model(scenario_id="scenario_8", node_name="moderator_evaluate")
    except KeyError:
        logger.warning("No specific moderator_evaluate model configured, falling back to generate_query")
        model = config.get_seq2seq_model(scenario_id="scenario_8", node_name="generate_query")
        
    response = await model.ainvoke(prompt)
    
    # Extract the selected query from the response
    # The model should indicate which proposal it selected
    selected_proposal_index = 0  # Default to first proposal
    try:
        # Try to find a proposal number in the response
        match = re.search(r"proposal\s+(\d+)", response.content.lower())
        if match:
            selected_proposal_index = int(match.group(1)) - 1  # Convert to 0-based index
            if selected_proposal_index >= len(final_proposals):
                selected_proposal_index = 0  # Fallback to first proposal if index is invalid
    except Exception as e:
        logger.warning(f"Error parsing moderator response: {e}")
    
    # Select the query
    selected_query = final_proposals[selected_proposal_index]
    
    # Log the selection
    logger.info(f"Moderator selected proposal {selected_proposal_index + 1}")
    logger.info(f"Selected query (first {min(50, len(selected_query))} chars): {selected_query[:50]}...")
    
    # Return state with selected query and evaluation
    return OverallState({
        **state,
        "selected_query": selected_query,
        "moderator_evaluation": response.content  # Store the evaluation reasoning
    })


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
        logger.error("No queries found in state")
        # Return state with empty query to indicate failure
        return OverallState({
            **state,
            "last_generated_query": "",
            "selected_query": "",
            "messages": state.get("messages", []),
            "error": "No queries were found in the state"
        })
    
    # Log the selected query
    logger.info(f"Formatting selected query for verification: {selected_query[:50]}...")
    
    # Return updated state
    result = OverallState({
        **state,
        "last_generated_query": selected_query,
        "selected_query": selected_query,
        "messages": state.get("messages", [])
    })
    
    logger.info(f"format_selected_query: returning state with keys: {list(result.keys())}")
    return result
