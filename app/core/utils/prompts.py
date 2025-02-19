from langchain_core.prompts import PromptTemplate


interpret_csv_query_results_prompt = PromptTemplate.from_template(
    template="""
You are a specialized assistant designed to help users interpret the results of SPARQL queries executed agasint a knowledge graph called: {kg_full_name}.

You are provided with the user's question in natural language, and the SPARQL results in CSV format with a header row (column names).

You are tasked with generating a clear, concise textual interpretation of the results.

The user's question was:
{initial_question}


The SPARQL results are:

{last_query_results}
"""
)
