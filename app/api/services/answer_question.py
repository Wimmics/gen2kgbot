import json
from app.api.services.utils import serialize_aimessagechunk
from app.utils.config_manager import get_scenario_module
from langgraph.graph.state import CompiledStateGraph
from app.utils.graph_state import EnumEncoder


async def answer_question(scenario_id: int, question: str):
    """
    Asynchronously answers a question based on the provided scenario ID and a question.
    Args:
        scenario_id (int): The ID of the scenario to use for answering the question.
        question (str): The question to be answered.
    Yields:
        str: A JSON string containing the event type, node name, and relevant data.
    """

    graph: CompiledStateGraph = get_scenario_module(scenario_id).graph

    state_nodes_filter = [
        "init",
        "preprocess_question",
        "select_similar_query_examples",
        "select_similar_classes",
        "get_context_class_from_cache",
        "get_context_class_from_kg",
        "create_prompt",
        "create_retry_prompt",
        "verify_query",
        "validate_sparql_syntax",
        "extract_query_qnames",
        "find_qnames_info",
        "judge_regeneration_prompt",
        "run_query",
    ]

    chat_nodes_filter = [
        "validate_question",
        "generate_query",
        "interpret_results",
        "ask_question",
        "judge_query",
        "judge_regenerate_query",
    ]

    # with open("data/custom_inputs/input_judging_subgraph.json", "r") as f:
    #     input_judging_subgraph = json.load(f)

    # async for event in graph.astream_events(input_judging_subgraph, version="v2"):
    async for event in graph.astream_events(
        {"initial_question": question}, version="v2"
    ):
        if event["event"] == "on_chat_model_stream" and "metadata" in event:
            if event["metadata"]["langgraph_node"] in chat_nodes_filter and "chunk" in event["data"]:
                chunk_content = serialize_aimessagechunk(event["data"]["chunk"])
                response_part = {
                    "event": "on_chat_model_stream",
                    "node": event["metadata"]["langgraph_node"],
                    "data": chunk_content,
                }
                # print(response_part)
                yield json.dumps(response_part)

        elif event["event"] == "on_chat_model_end" and "metadata" in event:
            response_part = {
                "event": "on_chat_model_end",
                "node": event["metadata"]["langgraph_node"],
            }
            yield json.dumps(response_part)

        elif event["event"] == "on_chain_end" and "metadata" in event:
            if "langgraph_node" in event["metadata"]:
                # print(event)
                if (
                    event["metadata"]["langgraph_node"] in state_nodes_filter
                    and event["name"] in state_nodes_filter
                    and "output" in event["data"]
                ):
                    data = event["data"]["output"]
                    if "messages" in data:
                        del data["messages"]
                    # print(data)
                    response_part = {
                        "event": "on_chain_end",
                        "node": event["metadata"]["langgraph_node"],
                        "data": event["data"]["output"],
                    }
                    yield json.dumps(response_part, cls=EnumEncoder)

        elif event["event"] == "on_chat_model_start" and "metadata" in event:

            response_part = {
                "event": "on_chat_model_start",
                "node": event["metadata"]["langgraph_node"],
            }
            yield json.dumps(response_part)
