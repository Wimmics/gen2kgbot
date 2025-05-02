from pydantic import BaseModel


class RefineQuery(BaseModel):
    """
    Attributes:
        model_config_id: str: The ID of the selected model configuration.
        question: str: The asked question to be judged in natural language.
        sparql_query: str: The sparql query to be judged.
        sparql_query_context: str: The list of QNames and Full QNames used in the sparql query with some additional context e.g. rdfs:label, rdfs:comment.
    """
    model_config_id: str
    question: str
    sparql_query: str
    sparql_query_context: str
