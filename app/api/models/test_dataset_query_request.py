from pydantic import BaseModel


class TestDatasetQueryRequest(BaseModel):
    modelProvider: str
    modelName: str
    base_uri: str = ""
    question: str
    sparql_query: str
    sparql_query_context: str
