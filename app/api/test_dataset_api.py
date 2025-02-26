import os
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from app.api.models.test_dataset_answer_question_request import TestDatasetAnswerQuestionRequest
from app.api.models.test_dataset_generate_question_request import (
    TestDatasetGenerateQuestionRequest,
)
from app.api.models.test_dataset_query_request import TestDatasetQueryRequest
from app.api.services.answer_question_service import generate_stream_responses
from app.api.services.generate_question_dataset_service import generate_questions
from fastapi.middleware.cors import CORSMiddleware

from app.api.services.test_answer_dataset_service import judge_answer


def get_env_variable(var_name: str) -> str:
    value = os.getenv(var_name)
    if value is None:
        raise ValueError(f"Environement variable `{var_name}` not found.")

    return value


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/test_dataset/judge_query")
async def test_dataset_judge_query(test_request: TestDatasetQueryRequest):
    """
    This endpoint is used to judge the answer of a question based on the given SPARQL query.

    Args:
        test_request (TestDatasetQueryRequest): The request object containing the necessary information to judge the answer.

    Returns:
        dict: The result of the judgement.
    """
    answer = await judge_answer(
        base_uri=test_request.base_uri,
        model_provider=test_request.modelProvider,
        model_name=test_request.modelName,
        question=test_request.question,
        sparql_query=test_request.sparql_query,
        sparql_query_context=test_request.sparql_query_context,
    )
    return {"result": answer}


@app.post("/api/test_dataset/generate-question")
async def test_dataset_generate_question(
    test_request: TestDatasetGenerateQuestionRequest,
):
    """
    This endpoint is used to generate questions about a given Knowledge Graph using a given LLM.

    Args:
        test_request (TestDatasetGenerateQuestionRequest): The request object containing the necessary information to generate questions.

    Returns:
        dict: The generated questions.
    """
    answer = await generate_questions(
        base_uri=test_request.base_uri,
        model_provider=test_request.model_provider,
        model_name=test_request.model_name,
        number_of_questions=test_request.number_of_questions,
        additional_context=test_request.additional_context,
        kg_description=test_request.kg_description,
        kg_schema=test_request.kg_schema,
        enforce_structured_output=test_request.enforce_structured_output,
    )
    return {"result": answer}


@app.post("/api/test_dataset/answer_question")
def test_dataset_answer_question(
    test_request: TestDatasetAnswerQuestionRequest,
):
    """
    This endpoint is used to generate questions about a given Knowledge Graph using a given LLM.

    Args:
        test_request (TestDatasetGenerateQuestionRequest): The request object containing the necessary information to generate questions.

    Returns:
        dict: The generated questions.
    """

    # setLLM(
    #     model_provider=test_request.model_provider,
    #     model_name=test_request.model_name,
    #     base_uri=test_request.base_uri,
    # )

    return StreamingResponse(
        generate_stream_responses(question=test_request.question),
        # media_type="text/event-stream",
        media_type="application/json",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
