from pydantic import BaseModel


class TestDatasetAnswerQuestionRequest(BaseModel):
    """
    TestDatasetGenerateQuestionRequest is a Pydantic model that defines the structure of the request body for the
    generate_questions endpoint of the TestDataset API.

    Attributes:
        model_provider: str: The name of the model provider e.g. Ollama, OpenAI.
        model_name: str: The name of the model e.g. llama3.2:1b, o3-mini.
        base_uri: str: The base URI of the model if needed.
        question: str: The question to be answered.
    """
    model_provider: str
    model_name: str
    base_uri: str = ""
    question: str
