from pydantic import BaseModel, Field


class ScenarioSchema(BaseModel):
    """
    A class representing the schema of a scenario.
    Attributes:
        scenario_id (int): The ID of the scenario.
        graph_schema (str): The schema of the scenario in mermaid format.
    """

    scenario_id: str = Field(..., description="The ID of the scenario.")
    graph_schema: str = Field(..., description="The schema of the scenario in mermaid format.")
