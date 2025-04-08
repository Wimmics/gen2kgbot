import json
import os
from pathlib import Path
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
import yaml
from app.api.models.test_dataset_activate_config_request import (
    TestDatasetActivateConfigRequest,
)
from app.api.models.test_dataset_answer_question_request import (
    TestDatasetAnswerQuestionRequest,
)
from app.api.models.test_dataset_config_request import TestDatasetConfigRequest
from app.api.models.test_dataset_generate_question_request import (
    TestDatasetGenerateQuestionRequest,
)
from app.api.models.test_dataset_query_request import TestDatasetQueryRequest

from app.api.services.answer_question_service import generate_stream_responses
from app.api.services.config_manager_service import add_missing_config_params
from app.api.services.generate_question_dataset_service import generate_questions
from fastapi.middleware.cors import CORSMiddleware

from app.api.services.graph_mermaid_service import get_scenarios_schema
from app.api.services.test_answer_dataset_service import judge_answer
from app.utils.config_manager import (
    get_configuration,
    read_configuration,
    set_custom_scenario_configuration,
    setup_cli,
)
from app.utils.logger_manager import setup_logger
from app.preprocessing.compute_embeddings import start_compute_embeddings
from app.preprocessing.gen_descriptions import generate_descriptions
import app.utils.config_manager as config


logger = setup_logger(__package__, __file__)


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


@app.post("/api/test_dataset/config/create")
def test_dataset_create_config(config_request: TestDatasetConfigRequest):
    """
    This endpoint is used to create a new configuration Yaml file.

    Returns:
        dict:
            A dictionary containing the created configuration.
    """
    try:

        logger.debug(f"Received configuration request: {config_request}")

        config_path = (
            Path(__file__).resolve().parent.parent
            / "config"
            / f"params_{config_request.kg_short_name}.yml"
        )

        # Check if the file already exists
        if config_path.exists():
            logger.error(f"Configuration file already exists at {config_path}")
            return Response(
                status_code=400,
                content=json.dumps(
                    {
                        "error": f"Configuration file already exists: {config_request.kg_short_name}"
                    }
                ),
                media_type="application/json",
            )

        # Create the configuration dictionary from the request
        with open(config_path, "w", encoding="utf-8") as file:
            # Convert the request to a dictionary
            config_dict = config_request.model_dump()

            # Add missing parameters to the configuration
            updated_config = add_missing_config_params(config_dict)

            # Write the configuration to the file
            yaml.safe_dump(updated_config, file)

            logger.info(f"Configuration file created at {config_path}")

            return Response(
                status_code=200,
                content=config_request.model_dump_json(),
                media_type="application/json",
            )
    except Exception as e:
        logger.error(f"Error creating configuration file: {str(e)}")
        return Response(
            status_code=500,
            content={"error": f"Error creating configuration file: {str(e)}"},
            media_type="application/json",
        )


@app.post("/api/test_dataset/config/activate")
def test_dataset_activate_config(config_request: TestDatasetActivateConfigRequest):
    """
    This endpoint is used to create a new configuration Yaml file.

    Returns:
        dict:
            A dictionary containing the created configuration.
    """
    try:

        logger.debug(f"Configuration to activate: {config_request}")

        config_to_activate_path = (
            Path(__file__).resolve().parent.parent
            / "config"
            / f"params_{config_request.kg_short_name}.yml"
        )

        # Check if the file already exists
        if not config_to_activate_path.exists():
            logger.error(
                f"Configuration file does not exist at {config_to_activate_path}"
            )
            return Response(
                status_code=400,
                content=json.dumps(
                    {
                        "error": f"Configuration file does not exist {config_request.kg_short_name}"
                    }
                ),
                media_type="application/json",
            )

        active_config_path = (
            Path(__file__).resolve().parent.parent / "config" / "params.yml"
        )

        with open(config_to_activate_path, "r", encoding="utf-8") as file_to_activate:
            config_data = yaml.safe_load(file_to_activate)

            # Activate the configuration
            with open(active_config_path, "w", encoding="utf-8") as active_file:
                yaml.safe_dump(config_data, active_file)
                logger.info(f"Configuration file activated at {active_file}")
                return Response(
                    status_code=200,
                    content=config_request.model_dump_json(),
                    media_type="application/json",
                )
    except Exception as e:
        logger.error(f"Error activating configuration file: {str(e)}")
        return Response(
            status_code=500,
            content={"error": f"Error activating configuration file: {str(e)}"},
            media_type="application/json",
        )


@app.post("/api/test_dataset/config/kg_descriptions")
def test_dataset_generate_kg_descriptions(
    config_request: TestDatasetActivateConfigRequest,
):
    """
    This endpoint is used to generate KG description of a given Knowledge Graph.

    Returns:
        dict:
            A dictionary containing the created configuration.
    """
    try:

        generate_descriptions()

        directory = config.get_preprocessing_directory()
        generated_files = {}
        for file in directory.iterdir():
            if file.is_file():
                with file.open("r", encoding="utf-8") as f:
                    content = f.read()
                    generated_files[file.name] = content

        return Response(
            status_code=200,
            content=json.dumps(generated_files),
            media_type="application/json",
        )
    except Exception as e:
        logger.error(f"Error  configuration file: {str(e)}")
        return Response(
            status_code=500,
            content={"error": f"Error creating configuration file: {str(e)}"},
            media_type="application/json",
        )


@app.post("/api/test_dataset/config/kg_embeddings")
def test_dataset_generate_kg_embeddings(
    config_request: TestDatasetActivateConfigRequest,
):
    """
    This endpoint is used to generate KG embeddings of a given Knowledge Graph.

    Returns:
        dict:
            A dictionary containing the created configuration.
    """
    try:

        start_compute_embeddings()

        return Response(
            status_code=200,
            content=json.dumps({"message": "KG embeddings generated successfully"}),
            media_type="application/json",
        )
    except Exception as e:
        logger.error(f"Error  configuration file: {str(e)}")
        return Response(
            status_code=500,
            content={"error": f"Error generated embeddings: {str(e)}"},
            media_type="application/json",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
