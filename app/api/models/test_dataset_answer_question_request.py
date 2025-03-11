from typing import Optional
from pydantic import BaseModel


class TestDatasetAnswerQuestionRequest(BaseModel):
    """
    TestDatasetGenerateQuestionRequest is a Pydantic model that defines the structure of the request body for the
    generate_questions endpoint of the TestDataset API.

    Attributes:
        seq2seq_model: str: The name of the seq2seq model to be used for answering the question.
        text_embedding_model: Optional[str]: The name of the text embedding model to be used for answering the question.
        question: str: The question to be answered.
        scenario_id: int: The ID of the scenario to be used for the question generation.
    """

    validate_question_model: str
    ask_question_model: str
    generate_query_model: str
    interpret_csv_query_results_model: str
    text_embedding_model: Optional[str] = None
    question: str
    scenario_id: int = 6
