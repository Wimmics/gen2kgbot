import os
from fastapi import HTTPException
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langchain.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama

from app.api.services.prompts.query_test_prompt import query_test_prompt


async def get_query_test_answer(
    model_provider: str,
    model_name: str,
    base_uri: str,
    question: str,
    sparql_query: str,
    sparql_query_context: str,
):

    query_test_prompt_template = ChatPromptTemplate.from_template(query_test_prompt)

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
                "question": question,
                "sparql": sparql_query,
                "qname_info": sparql_query_context,
            }
        )
        return result_from_json_mode.content
    except Exception as e:
        raise HTTPException(detail=str(e), status_code=500)
