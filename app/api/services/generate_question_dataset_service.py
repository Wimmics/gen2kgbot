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
