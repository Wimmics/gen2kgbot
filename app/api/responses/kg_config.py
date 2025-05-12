from pydantic import BaseModel, Field


class KGConfig(BaseModel):
    """
    This class represents the configuration of a Knowledge Graph (KG) used in the application.
    """

    kg_full_name: str = Field(..., description="The full name of the Knowledge Graph.")
    kg_short_name: str = Field(..., description="The short name of the Knowledge Graph.")
    kg_description: str = Field(..., description="A brief description of the Knowledge Graph.")
    kg_sparql_endpoint_url: str = Field(..., description="The SPARQL endpoint URL of the Knowledge Graph.")
    ontologies_sparql_endpoint_url: str = Field(..., description="The SPARQL endpoint URL for the ontologies.")
    ontology_named_graphs: list[str] = Field(default=None, description="Named graphs where to look for ontology definitions (optional)")
    properties_qnames_info: list[str] = Field(..., description="A list of property QNames used in the Knowledge Graph.")
    prefixes: dict = Field(..., description="A dictionary of prefixes used in the Knowledge Graph.")
    excluded_classes_namespaces: list[str] = Field(..., description="A list of excluded class namespaces.")
    seq2seq_models: dict = Field(default=None, description="A dictionary of seq2seq models used in the Knowledge Graph.")
    text_embedding_models: dict = Field(default=None, description="A dictionary of text embedding models used in the Knowledge Graph.")
    scenario_1: dict = Field(default=None, description="Configuration for scenario 1.")
    scenario_2: dict = Field(default=None, description="Configuration for scenario 2.")
    scenario_3: dict = Field(default=None, description="Configuration for scenario 3.")
    scenario_4: dict = Field(default=None, description="Configuration for scenario 4.")
    scenario_5: dict = Field(default=None, description="Configuration for scenario 5.")
    scenario_6: dict = Field(default=None, description="Configuration for scenario 6.")
    scenario_7: dict = Field(default=None, description="Configuration for scenario 7.")
