"""
Utility functions for SPARQL query extraction and processing specific to scenario 8.
"""
import re

def extract_sparql_from_proposal_summary(message: str) -> list:
    """
    Extract SPARQL queries from a response that contains both PROPOSAL and SUMMARY sections.
    
    This function is designed to handle the specific format used in scenario 8, where:
    1. The proposal section begins with "PROPOSAL:" 
    2. The proposal section ends with "SUMMARY:"
    3. The SPARQL query is wrapped in markdown code blocks with ```sparql and ```
    
    Args:
        message (str): The full response from the LLM containing both proposal and summary
        
    Returns:
        list: A list of extracted SPARQL queries (typically just one)
    """
    # Extract the proposal section (everything between "PROPOSAL:" and "SUMMARY:")
    proposal_match = re.search(r"PROPOSAL:\s*(.*?)(?:SUMMARY:|$)", message, re.DOTALL)
    
    if not proposal_match:
        return []
    
    proposal = proposal_match.group(1).strip()
    
    # Extract SPARQL query from markdown code blocks
    # This regex allows for whitespace after 'sparql' and before the closing ```
    queries = re.findall(r"```sparql\s*(.*?)\s*```", proposal, re.DOTALL)
    
    return queries 