from langchain_core.prompts import PromptTemplate


interpret_csv_query_results_prompt = PromptTemplate.from_template(
    template="""You are KGBot, a specialized assistant designed to help users interpret SPARQL query results related to the PubChem Knowledge Graph.

You are provided with the question and its results in CSV format with a header row (column names) and tasked with generating a clear, concise textual interpretation of the results.

Question:

{question}

Results:

{results}

"""
)
