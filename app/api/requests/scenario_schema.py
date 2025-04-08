from pydantic import BaseModel


class ScenarioSchema(BaseModel):
    """
    Attributes:
        scenario_id (str): The ID of the scenario.
    """
    scenario_id: int
