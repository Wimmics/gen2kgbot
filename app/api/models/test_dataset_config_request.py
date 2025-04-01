from pydantic import BaseModel


class TestDatasetConfigRequest(BaseModel):
    """
    TestDatasetConfigRequest is a Pydantic model that defines the structure of the request body for the
    create_test_dataset_create_config endpoint of the TestDataset API.

    Attributes:
        kg_full_name (str): The full name of the knowledge graph.
        kg_short_name (str): The short name of the knowledge graph.
        kg_description (str): A description of the knowledge graph.
        kg_sparql_endpoint_url (str): The URL of the SPARQL endpoint for the knowledge graph.
        ontologies_sparql_endpoint_url (str): The URL of the SPARQL endpoint for the ontologies.
        properties_qnames_info (list[str]): A list of property QNames information.
        prefixes (dict[str, str]): A dictionary of prefixes and their corresponding URIs.
        ontology_named_graphs (list[str]): A list of ontology named graphs.
        max_similar_classes (int): The maximum number of similar classes to retrieve.
        expand_similar_classes (bool): Whether to expand similar classes or not.
        class_context_format (str): The format for class context.
        excluded_classes_namespaces (list[str]): A list of excluded class namespaces.
        data_directory (str): The directory where data is stored.
        class_embeddings_subdir (str): The subdirectory for class embeddings.
        property_embeddings_subdir (str): The subdirectory for property embeddings.
        queries_embeddings_subdir (str): The subdirectory for query embeddings.
        temp_directory (str): The temporary directory for storing files.
    """

    kg_full_name: str
    kg_short_name: str
    kg_description: str
    kg_sparql_endpoint_url: str
    ontologies_sparql_endpoint_url: str
    properties_qnames_info: list[str]
    prefixes: dict[str, str]
    ontology_named_graphs: list[str]
    max_similar_classes: int
    expand_similar_classes: bool
    class_context_format: str
    excluded_classes_namespaces: list[str]
    data_directory: str
    class_embeddings_subdir: str
    property_embeddings_subdir: str
    queries_embeddings_subdir: str
    temp_directory: str
    seq2seq_models: dict
    text_embedding_models: dict
    scenario_1: dict
    scenario_2: dict
    scenario_3: dict
    scenario_4: dict
    scenario_5: dict
    scenario_6: dict
    scenario_7: dict
