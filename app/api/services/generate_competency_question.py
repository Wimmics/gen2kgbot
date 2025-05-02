from langchain.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from app.api.services.prompts.generate_competency_question import (
    generate_competency_question_prompt,
    enforce_structured_output_prompt,
)
import json
from app.api.services.utils import serialize_aimessagechunk
import app.utils.config_manager as config


async def generate_competency_questions(
    model_config_id: str,
    kg_schema: str,
    kg_description: str,
    additional_context: str,
    number_of_questions: int,
    enforce_structured_output: bool,
):
    """
    Generate a fixed number of questions from a given KG schema, description and additional context using a language model.

    Args:
        model_config_id (str): The ID of the selected model configuration available in the config file
        kg_schema (str): The schema of the knowledge graph e.g. a list of used ontologies or a list of classes and properties to be used in the questions
        kg_description (str): The description of the knowledge graph
        additional_context (str): Some additional context to be used in the question generation, e.g. the abstract of the paper presenting the KG
        number_of_questions (int): The number of questions to generate
        enforce_structured_output (bool): Whether to enforce structured output by adding a prefix to the prompt

    Yields:
        str: A stream of the generated questions in JSON format containing the keys "event" and "data"
    """

    query_test_prompt_template = ChatPromptTemplate.from_template(
        generate_competency_question_prompt
    )

    llm: BaseChatModel = config.get_seq2seq_model_by_config_id(model_config_id)

    chain_for_json_mode = query_test_prompt_template | llm

    async for event in chain_for_json_mode.astream_events(
        {
            "kg_schema": kg_schema,
            "kg_description": kg_description,
            "additional_context": additional_context,
            "number_of_questions": number_of_questions,
            "enforce_structured_output_prompt": (
                enforce_structured_output_prompt
                if enforce_structured_output
                else ""
            ),
        },
        version="v2",
    ):

        if event["event"] == "on_chat_model_stream":
            chunk_content = serialize_aimessagechunk(event["data"]["chunk"])
            response_part = {
                "event": "on_chat_model_stream",
                "data": chunk_content,
            }
            # print(response_part)
            yield json.dumps(response_part)

        elif event["event"] == "on_chat_model_end":
            response_part = {
                "event": "on_chat_model_end",
            }
            yield json.dumps(response_part)

        elif event["event"] == "on_chat_model_start":

            response_part = {
                "event": "on_chat_model_start",
            }
            yield json.dumps(response_part)
