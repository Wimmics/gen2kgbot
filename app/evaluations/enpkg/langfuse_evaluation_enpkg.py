import asyncio
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from sentence_transformers import SentenceTransformer, util
from app.api.services.answer_question import (
    get_scenario_instance,
    get_scenario_module_str,
)
from app.utils.config_manager import ConfigManager
from app.utils.logger_manager import setup_logger
from Levenshtein import distance
from langgraph.graph.state import CompiledStateGraph

logger = setup_logger(__package__, __file__)

evaluation_config = {
    "dataset": "enpkg",
    "scenario_id": 7,
    "experiment_name": "enpkg_7_llama-3_1-70",
    "cosine_model": "all-MiniLM-L6-v2",
    "scenario_config": {
        "generate_query": "llama-3_1-70B@ovh",
        "interpret_results": "llama-3_1-70B@ovh",
        "judge_query": "llama-3_1-70B@ovh",
        "judge_regenerate_query": "llama-3_1-70B@ovh",
        "judging_grade_threshold_retry": 8,
        "judging_grade_threshold_run": 5,
        "text_embedding_model": "nomic-embed-text_faiss@local",
        "validate_question": "llama-3_1-70B@ovh",
    },
}


async def run_experiment(experiment_name, scenario_id: int):

    dataset = langfuse.get_dataset(evaluation_config["dataset"])

    langfuse_handler = CallbackHandler()

    for item in dataset.items:

        with item.run(
            run_name=experiment_name,
            run_metadata={
                # "model": "TODO",
                "scenario": f"{scenario_id}",
            },
        ) as root_span:

            output = await run_my_langchain_llm_app(
                item.input["question"], scenario_id, langfuse_handler
            )

            root_span.score_trace(
                name="Cosine Similarity",
                value=cosine_evaluation(output, item.expected_output),
            )

            root_span.score_trace(
                name="Levenshtein Distance",
                value=levenshtein_evaluation(output, item.expected_output),
            )

    print(
        f"\nFinished processing dataset 'capital_cities' for run '{experiment_name}'."
    )


async def run_my_langchain_llm_app(question: str, scenario_id: int, callback_handler):

    with langfuse.start_as_current_span(name="gen2kgbot") as root_span:

        config = ConfigManager()
        config.read_configuration()
        scenario = get_scenario_instance(
            f"Scenario{scenario_id}",
            get_scenario_module_str(scenario_id=scenario_id),
            config,
        )
        graph: CompiledStateGraph = scenario.construct_graph()
        result = await graph.ainvoke(
            {"initial_question": question},
            config={"callbacks": [callback_handler]},
        )

        root_span.update_trace(input=input, output=result["last_generated_query"])

    return result["last_generated_query"]


def levenshtein_evaluation(output, expected_output):
    score = distance(output, expected_output) / max(len(output), len(expected_output))
    logger.info(f"Levenshtein Score: {score}")
    return score


def cosine_evaluation(output, expected_output):
    model = SentenceTransformer(evaluation_config["cosine_model"])
    emb1 = model.encode(output, convert_to_tensor=True)
    emb2 = model.encode(expected_output, convert_to_tensor=True)

    similarity = float(util.cos_sim(emb1, emb2)[0][0])
    normalized_score = (similarity + 1) / 2
    logger.info(f"Cosine Score: {normalized_score}")

    return normalized_score


if __name__ == "__main__":
    langfuse = get_client()

    if langfuse.auth_check():
        print("Langfuse client is authenticated and ready!")
        asyncio.run(
            run_experiment(
                experiment_name=evaluation_config["experiment_name"],
                scenario_id=evaluation_config["scenario_id"],
            )
        )
    else:
        print("Authentication failed. Please check your credentials and host.")
