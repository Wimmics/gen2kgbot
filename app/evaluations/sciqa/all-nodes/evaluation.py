import asyncio
from datetime import datetime
from pathlib import Path
from langfuse import get_client
from langfuse.langchain import CallbackHandler
import yaml
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
now = datetime.now()
evaluation_config = {
    "dataset": "sciqa",
    "scenario_id": 7,
    "config_file": "params_sciqa.yml",
    "experiment_name": f"sciqa_7_deepseek-chat {now.strftime('%Y-%m-%d_%H-%M-%S')}",
    "cosine_model": "all-MiniLM-L6-v2",
    "scenario_config": {
        "ask_question": "deepseek-chat@deepseek",
        "generate_query": "deepseek-chat@deepseek",
        "interpret_results": "deepseek-chat@deepseek",
        "judge_query": "deepseek-chat@deepseek",
        "judge_regenerate_query": "deepseek-chat@deepseek",
        "text_embedding_model": "nomic-embed-text_faiss@local",
        "validate_question": "deepseek-chat@deepseek",
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
                    output,
                    item.expected_output["query"],
                    evaluation_config["cosine_model"],
                ),
            )

            root_span.score_trace(
                name="Levenshtein Distance",
                value=levenshtein_evaluation(output, item.expected_output["query"]),
            )

            root_span.score_trace(
                name="Blue Score",
                value=bleu_score(output, item.expected_output["query"]),
            )

            root_span.score_trace(
                name="Rouge 1 Score",
                value=rouge_scores(output, item.expected_output["query"], "rouge-1"),
            )

            root_span.score_trace(
                name="Rouge 2 Score",
                value=rouge_scores(output, item.expected_output["query"], "rouge-2"),
            )

            root_span.score_trace(
                name="Rouge l Score",
                value=rouge_scores(output, item.expected_output["query"], "rouge-l"),
            )

    logger.info(
        f"\nFinished processing dataset {evaluation_config["dataset"]} for run '{experiment_name}'."
    )


def get_output_from_result(result: dict) -> str:
    output = "No output generated"

    if "last_generated_query" in result:
        output = result["last_generated_query"]
    elif "query_judgements" in result and len(result["query_judgements"]) > 0:
        if "query" in result["query_judgements"][-1]:
            output = result["query_judgements"][-1]["query"]
        else:
            output = result["query_judgements"][-1]["generated_answer"]
    else:
        messages = result["messages"].reverse()
        for message in messages:
            if message.content.find("```sparql") != -1:
                output = message["content"]
                break
    return output


async def run_my_langchain_llm_app(question: str, scenario_id: int, callback_handler):

    config_path = (
        Path(__file__).resolve().parent.parent.parent.parent.parent
        / "config"
        / evaluation_config["config_file"]
    )
    with open(config_path, "rt", encoding="utf8") as f:
        default_config = yaml.safe_load(f.read())
        f.close()

    with langfuse.start_as_current_span(name="gen2kgbot") as root_span:
        eval_conf = evaluation_config["scenario_config"]
        config = ConfigManager()
        config.set_configuration(default_config)
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

        output = get_output_from_result(result)
        root_span.update_trace(input=question, output=output)

    return output


if __name__ == "__main__":
    langfuse = get_client()

    try:
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
    except Exception as e:
        print(f"An error occurred: {e}")
