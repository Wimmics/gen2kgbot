import json
import os
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from app.api.models.test_dataset_answer_question_request import (
    TestDatasetAnswerQuestionRequest,
)
from app.api.models.test_dataset_generate_question_request import (
    TestDatasetGenerateQuestionRequest,
)
from app.api.models.test_dataset_query_request import TestDatasetQueryRequest

from app.api.services.answer_question_service import generate_stream_responses
from app.api.services.generate_question_dataset_service import generate_questions
from fastapi.middleware.cors import CORSMiddleware

from app.api.services.graph_mermaid_service import get_scenarios_schema
from app.api.services.test_answer_dataset_service import judge_answer
from app.core.utils.config_manager import (
    get_configuration,
    read_configuration,
    set_custom_scenario_configuration,
    setup_cli,
)


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
        StreamingResponse: The stream of the model judgement.
    """

    return StreamingResponse(
        judge_answer(
            base_uri=test_request.base_uri,
            model_provider=test_request.modelProvider,
            model_name=test_request.modelName,
            question=test_request.question,
            sparql_query=test_request.sparql_query,
            sparql_query_context=test_request.sparql_query_context,
        ),
        media_type="application/json",
    )


@app.post("/api/test_dataset/generate-question")
async def test_dataset_generate_question(
    test_request: TestDatasetGenerateQuestionRequest,
):
    """
    This endpoint is used to generate questions about a given Knowledge Graph using a given LLM.

    Args:
        test_request (TestDatasetGenerateQuestionRequest): The request object containing the necessary information to generate questions.

    Returns:
        StreamingResponse: The stream of the generated questions.
    """

    return StreamingResponse(
        generate_questions(
            base_uri=test_request.base_uri,
            model_provider=test_request.model_provider,
            model_name=test_request.model_name,
            number_of_questions=test_request.number_of_questions,
            additional_context=test_request.additional_context,
            kg_description=test_request.kg_description,
            kg_schema=test_request.kg_schema,
            enforce_structured_output=test_request.enforce_structured_output,
        ),
        media_type="application/json",
    )


@app.post("/api/test_dataset/answer_question")
def test_dataset_answer_question(
    test_request: TestDatasetAnswerQuestionRequest,
):
    """
    This endpoint is used to answer questions about a given Knowledge Graph

    Args:
        test_request (TestDatasetAnswerQuestionRequest): The request object containing the necessary information to answer a question.

    Returns:
        StreamingResponse: the stream of the answer to the question.
    """
    set_custom_scenario_configuration(
        scenario_id=test_request.scenario_id,
        validate_question_model=test_request.validate_question_model,
        ask_question_model=test_request.ask_question_model,
        generate_query_model=test_request.generate_query_model,
        judge_query_model=test_request.judge_query_model,
        judge_regenerate_query_model=test_request.judge_regenerate_query_model,
        interpret_results_model=test_request.interpret_results_model,
        text_embedding_model=test_request.text_embedding_model,
    )
    return StreamingResponse(
        generate_stream_responses(
            scenario_id=test_request.scenario_id, question=test_request.question
        ),
        # media_type="text/event-stream",
        media_type="application/json",
    )


@app.get("/api/test_dataset/scenarios_graph_schema")
def test_dataset_scenario_schema():
    """
    This endpoint is used to generate questions about a given Knowledge Graph using a given LLM.

    Returns:
        list:
            A list containing the different scenarios schemas following keys:
            - `scenario_id` (int): the id of the scenario
            - `schema` (str): the mermaid schema of the scenario.
    """

    return get_scenarios_schema()


@app.get("/api/test_dataset/default_config")
def test_dataset_default_config():
    """
    This endpoint is used to get the default configuration of the test dataset.

    Returns:
        dict:
            A dictionary containing the default configuration of the test dataset.
    """
    args = setup_cli()
    read_configuration(args=args)
    yaml_data = get_configuration()

    return json.dumps(yaml_data, indent=4)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
