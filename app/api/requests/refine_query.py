from pydantic import BaseModel, Field


class RefineQuery(BaseModel):
    """
    Attributes:
        model_config_id: str: The ID of the selected model configuration.
        question: str: The asked question to be judged in natural language.
        sparql_query: str: The sparql query to be judged.
        sparql_query_context: str: The list of QNames and Full QNames used in the sparql query with some additional context e.g. rdfs:label, rdfs:comment.
    """
    model_config_id: str = Field(..., description="The ID of the selected model configuration.")
    question: str = Field(..., description="The asked question to be judged in natural language.")
    sparql_query: str = Field(..., description="The sparql query to be judged.")
    sparql_query_context: str = Field(..., description="The list of QNames and Full QNames used in the sparql query with some additional context e.g. rdfs:label, rdfs:comment.")
