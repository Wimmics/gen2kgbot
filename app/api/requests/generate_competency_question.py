from typing import Optional
from pydantic import BaseModel


class GenerateCompetencyQuestion(BaseModel):
    """
    Attributes:
        model_config_id: str: The ID of the selected model configuration.
        kg_description: str: The description of the knowledge graph in natural language.
        kg_schema: str: The schema of the knowledge graph.
        additional_context: str: Some additional context  e.g. abstract of the paper presenting the KG.
        number_of_questions: int: The number of questions to generate.
        enforce_structured_output: bool: A flag to enforce structured output in JSON with a predified schema.
    """
    model_config_id: str
    kg_description: str
    kg_schema: Optional[str] = None
    additional_context: Optional[str] = None
    number_of_questions: int
    enforce_structured_output: bool
