import os
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langchain.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama
from app.api.services.prompts.refine_query import refine_query_prompt
import json
from langchain_core.messages import AIMessageChunk


def serialize_aimessagechunk(chunk):
    if isinstance(chunk, AIMessageChunk):
        return chunk.content
    else:
        raise TypeError(
            f"Object of type {type(chunk).__name__} is not correctly formatted for serialization"
        )


async def refine_query(
    model_provider: str,
    model_name: str,
    base_uri: str,
    question: str,
    sparql_query: str,
    sparql_query_context: str,
):
    """
    Judge the sparql query correctness given a question and the sparql query context using a language model.

    Args:
        model_provider (str): The provider of the language model
        model_name (str): The name of the language model
        base_uri (str): The base URI of the language model
        question (str): The asked question to be judged in natural language
        sparql_query (str): The sparql query to be judged
        sparql_query_context (str): The list of QNames and Full QNames used in the sparql query with some additional context e.g. rdfs:label, rdfs:comment

    Returns:
        str: The judged answer

    Raises:
        HTTPException: If an error occurs during the question answering
    """
    query_test_prompt_template = ChatPromptTemplate.from_template(refine_query_prompt)

    llm: BaseChatModel

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

    async for event in chain_for_json_mode.astream_events(
        {
            "question": question,
            "sparql": sparql_query,
            "qname_info": sparql_query_context,
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
