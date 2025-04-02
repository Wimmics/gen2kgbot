from pydantic import BaseModel


class TestDatasetActivateConfigRequest(BaseModel):
    """
    TestDatasetConfigRequest is a Pydantic model that defines the structure of the request body for the
    create_test_dataset_activate_config endpoint of the TestDataset API.

    Attributes:
        kg_short_name (str): The short name of the knowledge graph.
    """

    kg_short_name: str
