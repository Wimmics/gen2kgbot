import asyncio
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from app.api.services.answer_question import (
    get_scenario_instance,
    get_scenario_module_str,
)
from app.evaluations.evaluation_metrics import (
    bleu_score,
    cosine_evaluation,
    levenshtein_evaluation,
    rouge_scores,
)
from app.utils.config_manager import ConfigManager
from app.utils.logger_manager import setup_logger
from langgraph.graph.state import CompiledStateGraph

logger = setup_logger(__package__, __file__)

evaluation_config = {
    "dataset": "enpkg",
    "scenario_id": 7,
    "experiment_name": "enpkg_7_llama-3_1-70",
    "cosine_model": "all-MiniLM-L6-v2",
    "scenario_config": {
        "generate_query": "gemma-3_4b@local",
        "interpret_results": "gemma-3_4b@local",
        "judge_query": "gemma-3_4b@local",
        "judge_regenerate_query": "gemma-3_4b@local",
        "text_embedding_model": "nomic-embed-text_faiss@local",
        "validate_question": "gemma-3_4b@local",
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
                value=cosine_evaluation(
                    output, item.expected_output, evaluation_config["cosine_model"]
                ),
            )

            root_span.score_trace(
                name="Levenshtein Distance",
                value=levenshtein_evaluation(output, item.expected_output),
            )

            root_span.score_trace(
                name="Blue Score",
                value=bleu_score(output, item.expected_output),
            )

            root_span.score_trace(
                name="Rouge 1 Score",
                value=rouge_scores(output, item.expected_output, "rouge-1"),
            )

            root_span.score_trace(
                name="Rouge 2 Score",
                value=rouge_scores(output, item.expected_output, "rouge-2"),
            )

            root_span.score_trace(
                name="Rouge l Score",
                value=rouge_scores(output, item.expected_output, "rouge-l"),
            )

    logger.info(
        f"\nFinished processing dataset 'capital_cities' for run '{experiment_name}'."
    )


async def run_my_langchain_llm_app(question: str, scenario_id: int, callback_handler):

    with langfuse.start_as_current_span(name="gen2kgbot") as root_span:
        eval_conf = evaluation_config["scenario_config"]
        config = ConfigManager()
        config.read_configuration()
        config.set_custom_scenario_configuration(
            scenario_id=evaluation_config["scenario_id"],
            validate_question_model=eval_conf["validate_question"],
            ask_question_model=eval_conf["ask_question"],
            generate_query_model=eval_conf["generate_query"],
            judge_query_model=eval_conf["judge_query"],
            judge_regenerate_query_model=eval_conf["judge_regenerate_query"],
            interpret_results_model=eval_conf["interpret_results"],
            text_embedding_model=eval_conf["text_embedding_model"],
        )
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
