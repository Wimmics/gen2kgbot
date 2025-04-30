import json
from pathlib import Path
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
import yaml
from app.api.requests.activate_config import ActivateConfig
from app.api.requests.answer_question import AnswerQuestion
from app.api.requests.create_config import CreateConfig
from app.api.requests.generate_competency_question import GenerateCompetencyQuestion
from app.api.requests.refine_query import RefineQuery
from app.api.services.answer_question import answer_question
from app.api.services.config_manager import add_missing_config_params, save_query_examples_to_file
from app.api.services.generate_competency_question import generate_competency_questions
from fastapi.middleware.cors import CORSMiddleware
from app.api.services.graph_mermaid import get_scenarios_schema
from app.api.services.refine_query import refine_query
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


# setup logger
logger = setup_logger(__package__, __file__)

# setup FastAPI
app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/q2forge/judge_query")
async def judge_query_endpoint(refine_query_request: RefineQuery):
    """
    This endpoint is used to judge the answer of a question based on the given SPARQL query.

    Args:
        refine_query_request (RefineQuery): The request object containing the necessary information to judge the answer.

    Returns:
        StreamingResponse: The stream of the model judgement.
    """

    return StreamingResponse(
        refine_query(
            base_uri=refine_query_request.base_uri,
            model_provider=refine_query_request.modelProvider,
            model_name=refine_query_request.modelName,
            question=refine_query_request.question,
            sparql_query=refine_query_request.sparql_query,
            sparql_query_context=refine_query_request.sparql_query_context,
        ),
        media_type="application/json",
    )


@app.post("/api/q2forge/generate-question")
async def generate_question_endpoint(
    generate_competency_question_request: GenerateCompetencyQuestion,
):
    """
    This endpoint is used to generate questions about a given Knowledge Graph using a given LLM.

    Args:
        generate_competency_question_request (GenerateCompetencyQuestion): The request object containing the necessary information to generate competency questions.

    Returns:
        StreamingResponse: The stream of the generated questions.
    """

    return StreamingResponse(
        generate_competency_questions(
            base_uri=generate_competency_question_request.base_uri,
            model_provider=generate_competency_question_request.model_provider,
            model_name=generate_competency_question_request.model_name,
            number_of_questions=generate_competency_question_request.number_of_questions,
            additional_context=generate_competency_question_request.additional_context,
            kg_description=generate_competency_question_request.kg_description,
            kg_schema=generate_competency_question_request.kg_schema,
            enforce_structured_output=generate_competency_question_request.enforce_structured_output,
        ),
        media_type="application/json",
    )


@app.post("/api/q2forge/answer_question")
def answer_question_endpoint(answer_question_request: AnswerQuestion):
    """
    This endpoint is used to answer questions about a given Knowledge Graph

    Args:
        answer_question_request (AnswerQuestion): The request object containing the necessary information to answer a question.

    Returns:
        StreamingResponse: the stream of the answer to the question.
    """
    set_custom_scenario_configuration(
        scenario_id=answer_question_request.scenario_id,
        validate_question_model=answer_question_request.validate_question_model,
        ask_question_model=answer_question_request.ask_question_model,
        generate_query_model=answer_question_request.generate_query_model,
        judge_query_model=answer_question_request.judge_query_model,
        judge_regenerate_query_model=answer_question_request.judge_regenerate_query_model,
        interpret_results_model=answer_question_request.interpret_results_model,
        text_embedding_model=answer_question_request.text_embedding_model,
    )
    return StreamingResponse(
        answer_question(
            scenario_id=answer_question_request.scenario_id,
            question=answer_question_request.question,
        ),
        # media_type="text/event-stream",
        media_type="application/json",
    )


@app.get("/api/q2forge/scenarios_graph_schema")
def get_scenario_schema_endpoint():
    """
    This endpoint is used to generate questions about a given Knowledge Graph using a given LLM.

    Returns:
        list:
            A list containing the different scenarios schemas following keys:
            - `scenario_id` (int): the id of the scenario
            - `schema` (str): the mermaid schema of the scenario.
    """

    return get_scenarios_schema()


@app.get("/api/q2forge/default_config")
def get_active_config_endpoint():
    """
    This endpoint is used to get the default configuration of the Q²Forge API.

    Returns:
        dict:
            A dictionary containing the default configuration of the Q²Forge API.
    """
    args = setup_cli()
    read_configuration(args=args)
    yaml_data = get_configuration()

    return json.dumps(yaml_data, indent=4)


@app.post("/api/q2forge/config/create")
def create_config_endpoint(config_request: CreateConfig):
    """
    This endpoint is used to create a new configuration Yaml file.

    Args:
        config_request (CreateConfig): The request object containing the necessary information to create a configuration.

    Returns:
        dict:
            A dictionary containing the created configuration.
    """
    try:

        logger.debug(f"Received configuration request: {config_request}")

        config_path = (
            Path(__file__).resolve().parent.parent.parent
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

        # Save the query examples locally to be used in the embedding
        save_query_examples_to_file(
            kg_short_name=config_request.kg_short_name,
            query_examples=config_request.queryExamples,
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


@app.post("/api/q2forge/config/activate")
def activate_config_endpoint(config_request: ActivateConfig):
    """
    This endpoint is used to create a new configuration Yaml file.

    Args:
        config_request (ActivateConfig): The request object containing the necessary information to activate a configuration.

    Returns:
        dict:
            A dictionary containing the created configuration.
    """
    try:

        logger.debug(f"Configuration to activate: {config_request}")

        config_to_activate_path = (
            Path(__file__).resolve().parent.parent.parent
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
            Path(__file__).resolve().parent.parent.parent / "config" / "params.yml"
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


@app.post("/api/q2forge/config/kg_descriptions")
def generate_kg_descriptions_endpoint(
    config_request: ActivateConfig,
):
    """
    This endpoint is used to generate KG description of a given Knowledge Graph.

    Args:
        config_request (ActivateConfig): The request object containing the necessary information to generate KG descriptions.

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


@app.post("/api/q2forge/config/kg_embeddings")
def generate_kg_embeddings_endpoint(
    config_request: ActivateConfig,
):
    """
    This endpoint is used to generate KG embeddings of a given Knowledge Graph.

    Args:
        config_request (ActivateConfig): The request object containing the necessary information to generate KG embeddings.

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
