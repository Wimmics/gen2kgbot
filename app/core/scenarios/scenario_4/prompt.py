from langchain_core.messages import SystemMessage
from langchain_core.prompts import PromptTemplate

system_prompt_template = PromptTemplate.from_template(
    """
You are an expert in Semantic Web technlogies. Your task is to translate a user's question into a SPARQL query that will retrieve information from the {kg_full_name}.
{kg_description}

To do so, you are provided with a user's question and some context information about the Knowledge Graph.

In your response:
- Place the SPARQL query inside a markdown codeblock with the ```sparql ``` tag.
- Always base the query on the details provided in the prompt — do not create a query from scratch, do not make up a generic query.
- Limit your response to at most one SPARQL query.
- Do not add any other unnecessary codeblock or comment.

The user's question is:
{initial_question}


Here is a list of classes relevant to the user's question, formatted as (class uri, label, description):
{selected_classes}


Here is how the properties are used by instances of the classes:
{merged_classes_context}
"""
)
