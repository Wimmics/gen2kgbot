from pydantic import BaseModel


class ScenarioSchema(BaseModel):
    """
    ScenarioSchema is a Pydantic model that defines the structure of the request body for the
    generate_questions endpoint of the DatasetForge API.

    Attributes:
        scenario_id (str): The ID of the scenario.
    """
    scenario_id: int
