from pydantic import BaseModel


class ActivateConfig(BaseModel):
    """
    Attributes:
        kg_short_name (str): The short name of the knowledge graph.
    """

    kg_short_name: str
