from pydantic import BaseModel, Field


class ActivateConfig(BaseModel):
    """
    Attributes:
        kg_short_name (str): The short name of the knowledge graph.
    """

    kg_short_name: str = Field(..., description="The short name of the knowledge graph to use.")
