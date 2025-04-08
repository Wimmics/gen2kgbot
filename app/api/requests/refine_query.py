from pydantic import BaseModel


class RefineQuery(BaseModel):
    """
    RefineQuery is a Pydantic model that represents the request body for the /api/dataset_forge/judge_query endpoint.

    Attributes:
        modelProvider: str: The provider of the language model.
        modelName: str: The name of the language model.
        base_uri: str: The base URI of the language model.
        question: str: The asked question to be judged in natural language.
        sparql_query: str: The sparql query to be judged.
        sparql_query_context: str: The list of QNames and Full QNames used in the sparql query with some additional context e.g. rdfs:label, rdfs:comment.
    """
    modelProvider: str
    modelName: str
    base_uri: str = ""
    question: str
    sparql_query: str
    sparql_query_context: str
