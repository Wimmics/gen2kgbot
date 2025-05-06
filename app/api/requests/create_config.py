from pydantic import BaseModel, Field


class QueryExample(BaseModel):
    """
    Attributes:
        question (str): The competency question.
        query (str): The SPARQL query corresponding to the competency question.
    """
    question: str = Field(..., description="The competency question.")
    query: str = Field(..., description="The SPARQL query corresponding to the competency question.")


class CreateConfig(BaseModel):
    """
    Attributes:
        kg_full_name (str): The full name of the knowledge graph.
        kg_short_name (str): The short name of the knowledge graph.
        kg_description (str): A description of the knowledge graph.
        kg_sparql_endpoint_url (str): The URL of the SPARQL endpoint for the knowledge graph.
        ontologies_sparql_endpoint_url (str): The URL of the SPARQL endpoint for the ontologies.
        properties_qnames_info (list[str]): A list of property QNames information.
        prefixes (dict[str, str]): A dictionary of prefixes and their corresponding URIs.
        ontology_named_graphs (list[str]): A list of ontology named graphs.
        excluded_classes_namespaces (list[str]): A list of excluded class namespaces.
        queryExamples (list[QueryExample]): A list of competency questions and their corresponding SPARQL queries.
    """

    kg_full_name: str = Field(..., description="The full name of the knowledge graph.")
    kg_short_name: str = Field(..., description="The short name of the knowledge graph.")
    kg_description: str = Field(..., description="A description of the knowledge graph.")
    kg_sparql_endpoint_url: str = Field(..., description="The URL of the SPARQL endpoint for the knowledge graph.")
    ontologies_sparql_endpoint_url: str = Field(..., description="The URL of the SPARQL endpoint for the ontologies.")
    properties_qnames_info: list[str] = Field(..., description="A list of property QNames information.")
    prefixes: dict[str, str] = Field(..., description="A dictionary of prefixes and their corresponding URIs.")
    ontology_named_graphs: list[str] = Field(..., description="A list of ontology named graphs.")
    excluded_classes_namespaces: list[str] = Field(..., description="A list of excluded class namespaces.")
    queryExamples: list[QueryExample] = Field(..., description="A list of competency questions and their corresponding SPARQL queries.")
