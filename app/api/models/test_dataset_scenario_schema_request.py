from pydantic import BaseModel


class TestDatasetScenarioSchemaRequest(BaseModel):
    """
    TestDatasetScenarioGraphRequest is a Pydantic model that defines the structure of the request body for the
    generate_questions endpoint of the TestDataset API.

    Attributes:
        scenario_id (str): The ID of the scenario.
    """
    scenario_id: str
