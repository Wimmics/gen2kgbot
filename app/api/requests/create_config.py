from pydantic import BaseModel


class QueryExample(BaseModel):
    """
    Attributes:
        question (str): The competency question.
        query (str): The SPARQL query corresponding to the competency question.
    """
    question: str
    query: str


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
    """

    kg_full_name: str
    kg_short_name: str
    kg_description: str
    kg_sparql_endpoint_url: str
    ontologies_sparql_endpoint_url: str
    properties_qnames_info: list[str]
    prefixes: dict[str, str]
    ontology_named_graphs: list[str]
    excluded_classes_namespaces: list[str]
    queryExamples: list[QueryExample]
