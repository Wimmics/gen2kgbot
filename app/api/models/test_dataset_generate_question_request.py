from pydantic import BaseModel


class TestDatasetGenerateQuestionRequest(BaseModel):
    """
    TestDatasetGenerateQuestionRequest is a Pydantic model that defines the structure of the request body for the
    generate_questions endpoint of the TestDataset API.

    Attributes:
        model_provider: str: The name of the model provider e.g. Ollama, OpenAI.
        model_name: str: The name of the model e.g. llama3.2:1b, o3-mini.
        base_uri: str: The base URI of the model if needed.
        kg_description: str: The description of the knowledge graph in natural language.
        kg_schema: str: The schema of the knowledge graph.
        additional_context: str: Some additional context  e.g. abstract of the paper presenting the KG.
        number_of_questions: int: The number of questions to generate.
        enforce_structured_output: bool: A flag to enforce structured output in JSON with a predified schema.
    """
    model_provider: str
    model_name: str
    base_uri: str = ""
    kg_description: str
    kg_schema: str
    additional_context: str
    number_of_questions: int
    enforce_structured_output: bool
