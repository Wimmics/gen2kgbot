from pydantic import BaseModel


class ActivateConfig(BaseModel):
    """
    ActivateConfig is a Pydantic model that defines the structure of the request body for the
    create_dataset_forge_activate_config endpoint of the DatasetForge API.

    Attributes:
        kg_short_name (str): The short name of the knowledge graph.
    """

    kg_short_name: str
