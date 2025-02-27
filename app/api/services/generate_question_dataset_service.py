import os
from fastapi import HTTPException
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langchain.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama
from app.api.services.prompts.question_generation_prompt import (
    question_generation_prompt,
    enforce_structured_output_prompt,
)
import json
from langchain_core.messages import AIMessageChunk


def serialize_aimessagechunk(chunk):
    if isinstance(chunk, AIMessageChunk):
        return chunk.content
    else:
        raise TypeError(
            f"Object of type {type(chunk).__name__} is not correctly formatted for serialization"
        )


async def generate_questions(
    model_provider: str,
    model_name: str,
    base_uri: str,
    kg_schema: str,
    kg_description: str,
    additional_context: str,
    number_of_questions: int,
    enforce_structured_output: bool,
):
    """
    Generate a fixed number of questions from a given KG schema, description and additional context using a language model.

    Args:
        model_provider (str): The provider of the language model
        model_name (str): The name of the language model
        base_uri (str): The base URI of the language model
        kg_schema (str): The schema of the knowledge graph e.g. a list of used ontologies or a list of classes and properties to be used in the questions
        kg_description (str): The description of the knowledge graph
        additional_context (str): Some additional context to be used in the question generation, e.g. the abstract of the paper presenting the KG
        number_of_questions (int): The number of questions to generate
        enforce_structured_output (bool): Whether to enforce structured output by adding a prefix to the prompt

    Returns:
        str: The generated questions

    Raises:
        HTTPException: If an error occurs during the question generation
    """

    query_test_prompt_template = ChatPromptTemplate.from_template(
        question_generation_prompt
    )

    llm: BaseChatModel

    # try:
    if model_provider == "Ollama-local":
        llm = ChatOllama(base_url="http://localhost:11434", model=model_name)
    elif model_provider == "DeepSeek":
        llm = ChatDeepSeek(model=model_name, api_key=os.getenv("DEEPSEEK_API_KEY"))
    elif model_provider == "Ovh":
        llm = ChatOpenAI(
            model=model_name,
            base_url=base_uri,
            api_key=os.getenv("OVHCLOUD_API_KEY"),
        )
    elif model_provider == "OpenAI":
        llm = ChatOpenAI(
            model=model_name,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    else:
        raise ValueError("Unsupported LLM provider")

    chain_for_json_mode = query_test_prompt_template | llm

    # result_from_json_mode = await chain_for_json_mode.ainvoke(
    #     {
    #         "kg_schema": kg_schema,
    #         "kg_description": kg_description,
    #         "additional_context": additional_context,
    #         "number_of_questions": number_of_questions,
    #         "enforce_structured_output_prompt": (
    #             enforce_structured_output_prompt
    #             if enforce_structured_output
    #             else ""
    #         ),
    #     }
    # )

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

    # except Exception as e:
    #     raise HTTPException(detail=str(e), status_code=500)


async def generate_questions1(
    model_provider: str,
    model_name: str,
    base_uri: str,
    kg_schema: str,
    kg_description: str,
    additional_context: str,
    number_of_questions: int,
    enforce_structured_output: bool,
):
    """
    Generate a fixed number of questions from a given KG schema, description and additional context using a language model.

    Args:
        model_provider (str): The provider of the language model
        model_name (str): The name of the language model
        base_uri (str): The base URI of the language model
        kg_schema (str): The schema of the knowledge graph e.g. a list of used ontologies or a list of classes and properties to be used in the questions
        kg_description (str): The description of the knowledge graph
        additional_context (str): Some additional context to be used in the question generation, e.g. the abstract of the paper presenting the KG
        number_of_questions (int): The number of questions to generate
        enforce_structured_output (bool): Whether to enforce structured output by adding a prefix to the prompt

    Returns:
        str: The generated questions

    Raises:
        HTTPException: If an error occurs during the question generation
    """

    query_test_prompt_template = ChatPromptTemplate.from_template(
        question_generation_prompt
    )

    llm: BaseChatModel

    try:
        if model_provider == "Ollama-local":
            llm = ChatOllama(base_url="http://localhost:11434", model=model_name)
        elif model_provider == "DeepSeek":
            llm = ChatDeepSeek(model=model_name, api_key=os.getenv("DEEPSEEK_API_KEY"))
        elif model_provider == "Ovh":
            llm = ChatOpenAI(
                model=model_name,
                base_url=base_uri,
                api_key=os.getenv("OVHCLOUD_API_KEY"),
            )
        elif model_provider == "OpenAI":
            llm = ChatOpenAI(
                model=model_name,
                api_key=os.getenv("OPENAI_API_KEY"),
            )
        else:
            raise ValueError("Unsupported LLM provider")

        chain_for_json_mode = query_test_prompt_template | llm

        result_from_json_mode = await chain_for_json_mode.ainvoke(
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
            }
        )
        return result_from_json_mode.content
    except Exception as e:
        raise HTTPException(detail=str(e), status_code=500)
