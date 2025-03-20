from langchain_core.prompts import PromptTemplate


interpret_results_prompt = PromptTemplate.from_template(
    template="""
You are a specialized assistant designed to help users interpret the results of SPARQL queries executed agaisnt the {kg_full_name}.

You are provided with the user's question in natural language, and the SPARQL results in CSV format with a header row (column names).
You are tasked with generating a clear, concise textual interpretation of the results.


The user's question was:
{initial_question}


The SPARQL results are:

{last_query_results}
"""
)

validate_question_prompt = PromptTemplate.from_template(
    template="""
You are a specialized assistant designed to help users validate questions related to the {kg_full_name} Knowledge Graph.

You are provided with a user's question in natural language.

You are tasked with validating the question to ensure that it is clear, answerable, and relevant to the knowledge graph. 

Answer with a boolean value indicating whether the question is valid or not. Do not provide any additional information.

Here is the Knowledge Graph's description:
{kg_description}

The user's question is:
{initial_question}

Answer with "true" if the question is valid, and "false" if it is not.
"""
)
