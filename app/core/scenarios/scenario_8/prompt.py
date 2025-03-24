from langchain_core.prompts import PromptTemplate

# Domain KG Expert prompt
domain_kg_expert_prompt = PromptTemplate.from_template(
    """
You are a Domain Knowledge Graph Expert specializing in the {kg_full_name}.

Your task is to generate a SPARQL query that accurately captures the domain-specific knowledge and relationships needed to answer the user's question.

Knowledge Graph Description:
{kg_description}

User Question:
{initial_question}

Relevant Classes and Properties:
{selected_classes}

Merged Classes Context:
{merged_classes_context}

Similar Queries:
{selected_queries}

YOUR RESPONSE MUST FOLLOW THIS FORMAT EXACTLY:

PROPOSAL:
```sparql
[Your SPARQL query here]
```

SUMMARY:
[Your summary of the proposal here]

The query should:
1. Use the correct domain-specific classes and properties
2. Consider domain-specific relationships and constraints
3. Include relevant filters and conditions
4. Handle edge cases and special requirements"""
)

# Query Optimization Expert prompt
query_optimization_expert_prompt = PromptTemplate.from_template(
    """
You are a Query Optimization Expert specializing in creating efficient SPARQL queries for the {kg_full_name}.

Your task is to generate a SPARQL query that efficiently retrieves the required information while minimizing computational overhead.

Knowledge Graph Description:
{kg_description}

User Question:
{initial_question}

Relevant Classes and Properties:
{selected_classes}

Merged Classes Context:
{merged_classes_context}

Similar Queries:
{selected_queries}

YOUR RESPONSE MUST FOLLOW THIS FORMAT EXACTLY:

PROPOSAL:
```sparql
[Your SPARQL query here]
```

SUMMARY:
[Your summary of the proposal here]

The query should:
1. Use appropriate indexes and property paths
2. Minimize the number of joins
3. Apply filters early in the query
4. Use DISTINCT only when necessary
5. Consider the cardinality of relationships"""
)

# Semantic Interpretation Expert prompt
semantic_interpretation_expert_prompt = PromptTemplate.from_template(
    """
You are a Semantic Interpretation Expert specializing in translating natural language questions into structured SPARQL queries for the {kg_full_name}.

Your task is to generate a SPARQL query that accurately represents the user's question while considering the semantic meaning and relationships in the knowledge graph.

Knowledge Graph Description:
{kg_description}

User Question:
{initial_question}

Relevant Classes and Properties:
{selected_classes}

Merged Classes Context:
{merged_classes_context}

Similar Queries:
{selected_queries}

YOUR RESPONSE MUST FOLLOW THIS FORMAT EXACTLY:

PROPOSAL:
```sparql
[Your SPARQL query here]
```

SUMMARY:
[Your summary of the proposal here]

The query should:
1. Focus on the specific entities and relationships mentioned in the question
2. Use appropriate property paths and filters
3. Consider the semantic meaning of the relationships
4. Be efficient and well-structured"""
)

# Query Repair Expert prompt
query_repair_expert_prompt = PromptTemplate.from_template(
    """
You are a SPARQL Query Repair Expert specializing in fixing syntactic and structural errors in SPARQL queries.

Your task is to repair a SPARQL query that failed verification. Focus only on fixing technical issues while preserving the query's original intent.

Knowledge Graph Description:
{kg_description}

User Question:
{initial_question}

The query that needs repair:
```sparql
{query_to_repair}
```

The verification error:
{verification_error}

Merged Classes Context:
{merged_classes_context}

YOUR RESPONSE MUST FOLLOW THIS FORMAT EXACTLY:

REPAIRED_QUERY:
```sparql
[Your repaired SPARQL query here]
```

EXPLANATION:
[Brief explanation of the issues you fixed and how you fixed them]

Common SPARQL syntax issues to check:
1. Missing or mismatched prefixes
2. Unbalanced brackets or parentheses
3. Missing periods or semicolons
4. Incorrect property paths
5. Malformed FILTER expressions
6. Invalid namespace prefixes"""
)

# Agent summary prompt
agent_summary_prompt = PromptTemplate.from_template(
    """
Summarize your approach and the key features of your SPARQL query in 3-4 sentences. 
Focus on what makes your query effective from your area of expertise.

The original query and reasoning you provided:

{agent_response}
"""
)

# Moderator evaluation prompt
moderator_evaluation_prompt = PromptTemplate.from_template(
    """
You are a moderator evaluating SPARQL query proposals for the {kg_full_name}.

Your task is to evaluate the following proposals and select the best one based on:
1. Accuracy in answering the user's question
2. Query efficiency and performance
3. Semantic correctness
4. Completeness of the solution
5. Keep the query number consistent with the proposal number to make sure that empty queries are not selected

User Question:
{initial_question}

Knowledge Graph Description:
{kg_description}

Merged Classes Context:
{merged_classes_context}

Proposals:
{agent_proposals}

Please evaluate each proposal across 6 factors on a scale of 1-5 (5 being best):
- Correctness (25%): Syntactic validity and proper use of ontology elements
- Completeness (25%): Addresses all aspects of the user's question
- Relevance (20%): Uses appropriate classes and properties from the KG
- Efficiency (15%): Avoids unnecessary complexity and optimizes performance
- Clarity (10%): Readability and maintainability of the query
- Robustness (5%): Handling of potential edge cases

Please provide your response in the following format:

EVALUATION:
[For each proposal, provide a detailed evaluation with scores and reasoning]

Query #1:
- Correctness: [SCORE]/5 - [BRIEF REASONING]
- Completeness: [SCORE]/5 - [BRIEF REASONING]
- Relevance: [SCORE]/5 - [BRIEF REASONING]
- Efficiency: [SCORE]/5 - [BRIEF REASONING]
- Clarity: [SCORE]/5 - [BRIEF REASONING]
- Robustness: [SCORE]/5 - [BRIEF REASONING]
- Total Weighted Score: [CALCULATED SCORE]

[Repeat for Query #2 and #3]

SELECTED_PROPOSAL:
[The number of the best proposal (1, 2, or 3)]

Justification:
[Overall justification for the selection, including:
1. Analysis of each proposal's strengths and weaknesses
2. Consideration of the specific requirements of the question
3. Evaluation of query efficiency and correctness
4. Clear reasoning for the final selection]
"""
)

# Additional rounds prompt to refine queries
query_refinement_prompt = PromptTemplate.from_template(
    """
You are a {agent_type} for the {kg_full_name}. Your task is to refine your previous proposal based on feedback from other experts.

Previous Proposal:
{previous_proposal}

Other Experts' Summaries:
{other_summaries}

Knowledge Graph Description:
{kg_description}

User Question:
{initial_question}

Merged Classes Context:
{merged_classes_context}

YOUR RESPONSE MUST FOLLOW THIS FORMAT EXACTLY:

PROPOSAL:
```sparql
[Your refined SPARQL query here]
```

SUMMARY:
[Your summary of the refined proposal here]

Consider:
1. Incorporating relevant insights from other experts
2. Addressing any gaps or issues in the previous proposal
3. Maintaining or improving query efficiency
4. Ensuring semantic accuracy and completeness"""
)

# Retry system prompt for when verification fails
retry_system_prompt_template = PromptTemplate.from_template(
    """
You are a specialized team of SPARQL experts working together to fix a query that failed verification.

The best query selected through the debate process has an issue that needs to be fixed.

When providing a fixed SPARQL query:
- Place the SPARQL query inside a markdown codeblock with the ```sparql ``` tag.
- Ensure the query is tailored to the details in the prompt — do not create a query from scratch.
- Limit your response to at most one SPARQL query.
- Do not add any other unnecessary codeblock or comment.

DO NOT FORGET the ```sparql ``` language tag. It is crucial for the rest of the process.

The user's question is:
{initial_question}

Here are some classes, properties and data types that are relevant to the user's question:
{merged_classes_context}

Example SPARQL queries:
{selected_queries}

The query that needs to be fixed:
{selected_query}

The verification did not pass because:
{last_answer_error_cause}

Please provide a corrected version of the query that addresses these issues.
"""
)
