from langchain.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from app.api.services.prompts.refine_query import refine_query_prompt
import json
from app.api.services.utils import serialize_aimessagechunk
import app.utils.config_manager as config


async def refine_query(
    model_config_id: str,
    question: str,
    sparql_query: str,
    sparql_query_context: str,
):
    """
    Judge the sparql query correctness given a question and the sparql query context using a language model.

    Args:
        model_config_id (str): The ID of the selected model configuration
        question (str): The asked question to be judged in natural language
        sparql_query (str): The sparql query to be judged
        sparql_query_context (str): The list of QNames and Full QNames used in the sparql query with some additional context e.g. rdfs:label, rdfs:comment

    Yields:
        str: A stream of the generated questions in JSON format containing the keys "event" and "data"
    """
    query_test_prompt_template = ChatPromptTemplate.from_template(refine_query_prompt)

    llm: BaseChatModel = config.get_seq2seq_model_by_config_id(model_config_id)

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
