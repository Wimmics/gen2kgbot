from pydantic import BaseModel, Field


class ScenarioSchema(BaseModel):
    """
    Attributes:
        scenario_id (str): The ID of the scenario.
    """
    scenario_id: int = Field(..., description="The ID of the scenario.")
