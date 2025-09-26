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
from app.evaluations.evaluation_metrics import exact_match
from app.utils.config_manager import ConfigManager
from app.utils.graph_routers import GraphRouters
from app.utils.logger_manager import setup_logger
from langgraph.graph.state import CompiledStateGraph

logger = setup_logger(__package__, __file__)
now = datetime.now()
evaluation_config = {
    "dataset": "sciqa-validation-node",
    "scenario_id": 1,
    "config_file": "params_sciqa.yml",
    "experiment_name": f"sciqa_7_deepseek-chat {now.strftime('%Y-%m-%d_%H-%M-%S')}",
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

orkg_descriptions = [
    "The Open Research Knowledge Graph (ORKG) provides a structured representation of research contributions with a strong emphasis on benchmarking and evaluation. It captures information about datasets, models, evaluation metrics, results, and their associated scholarly publications. By interlinking these elements, the ORKG enables researchers to query for evaluation settings, discover which models were tested on specific datasets, analyze performance metrics, and compare results across publications.",
    "The ORKG is designed to make research contributions in benchmarking and evaluation accessible in a structured, machine-readable format. It represents the key entities that define evaluation settings, including datasets, models, metrics, benchmark scores, and related papers. This interconnected structure allows for systematic comparison of models, exploration of evaluation methods, and retrieval of benchmark results reported across scientific literature.",
    "The Open Research Knowledge Graph structures knowledge about scientific benchmarking to support systematic comparison and synthesis of research findings. It models datasets, algorithms and models, evaluation metrics, benchmark results, and publications as interlinked entities in a knowledge graph. With this representation, researchers can ask precise questions about which datasets were used, what metrics were applied, and which models achieved specific performance outcomes.",
    "The ORKG captures benchmarking-related contributions in scientific research by organizing them into a structured knowledge graph. Core elements include datasets, models, evaluation metrics, performance results, and the scholarly works in which these results are published. This representation enables the retrieval of benchmark data, the comparison of evaluation metrics, and the exploration of model performance across multiple research efforts.",
    "The Open Research Knowledge Graph is a digital infrastructure for representing and interlinking benchmarking knowledge in science. It structures research contributions in terms of datasets, models, evaluation methods, metrics, benchmark scores, and papers. Researchers can use the ORKG to trace evaluation practices, compare performance outcomes across different models, and navigate the scholarly publications reporting benchmark results.",
    "The ORKG makes benchmarking knowledge explicit by transforming textual research contributions into structured, interconnected entities. It captures the essential components of evaluation—datasets, models, metrics, results, and publications—and links them in a graph-based format. This design allows researchers to explore how models are tested, what results are achieved, and which publications provide benchmark evidence.",
    "The Open Research Knowledge Graph is focused on the structured representation of research benchmarking. It organizes datasets, models, evaluation metrics, results, and associated publications into a unified knowledge graph. This supports querying across contributions to identify evaluation settings, compare reported outcomes, and synthesize insights about benchmarking practices across domains.",
    "The ORKG provides a machine-readable map of benchmarking knowledge in science. It records datasets, models, evaluation metrics, benchmark scores, and their related publications as interconnected entities. This structured approach enables systematic exploration of model evaluations, allowing users to compare results, track performance, and understand the evaluation practices used in scholarly research.",
    "The Open Research Knowledge Graph is designed to facilitate structured access to benchmarking knowledge. It represents datasets, models, evaluation metrics, results, and publications in a graph-based format. This enables researchers to query the evaluation landscape, ask detailed questions about benchmarks, and compare results across different studies and models.",
    "The ORKG focuses on capturing the benchmarking dimension of scientific research. By structuring datasets, models, metrics, results, and associated publications, it creates a knowledge graph that supports evaluation-focused discovery. Users can investigate what benchmarks have been applied, which models were tested, what performance values were obtained, and where the results were published.",
    "The Open Research Knowledge Graph systematically represents benchmarking knowledge in a structured way. It interlinks datasets, models, metrics, results, and publications, making them queryable as part of a knowledge graph. This allows for targeted exploration of evaluation methods, performance comparisons across models, and navigation of publications reporting benchmark outcomes.",
    "The ORKG provides a structured representation of research evaluation knowledge, focusing on benchmarks. It models datasets, models, evaluation metrics, results, and related publications as entities that can be interlinked and compared. This structure makes it possible to analyze evaluation settings across studies, identify best-performing models, and trace benchmarking practices in the literature.",
    "The Open Research Knowledge Graph is dedicated to capturing benchmarking contributions from scientific literature. It organizes datasets, models, metrics, results, and scholarly papers into a connected knowledge graph. With this representation, users can search for evaluation details, compare benchmark outcomes, and explore how different models perform under specific evaluation metrics.",
    "The ORKG structures knowledge about benchmarking in research. It represents datasets, models, metrics, results, and related publications, building a knowledge graph that enables queries about evaluation practices. This structured framework helps researchers compare models, analyze reported performance, and understand benchmarking evidence in a systematic way.",
    "The Open Research Knowledge Graph turns research contributions related to benchmarking into structured knowledge. It captures datasets, models, evaluation metrics, benchmark results, and their linked publications. This enables researchers to perform targeted queries, such as which models achieved top results on a dataset or which papers report benchmark scores.",
    "The ORKG enables structured exploration of benchmarking knowledge by capturing datasets, models, evaluation metrics, results, and associated publications. Its graph-based design supports queries about evaluation setups, reported scores, and benchmark outcomes. Researchers can use the ORKG to compare contributions across the literature and synthesize benchmarking knowledge in a structured way.",
    "The Open Research Knowledge Graph is a structured knowledge base for benchmarking contributions. It interlinks datasets, models, metrics, results, and publications, making benchmarking practices accessible and comparable. This representation supports systematic discovery of model evaluations, performance results, and the scientific works in which they are reported.",
    "The ORKG structures benchmarking information from research papers into a connected knowledge graph. It captures datasets, models, metrics, results, and publications, allowing benchmarking data to be explored in a consistent way. Researchers can use this graph to identify evaluation metrics, compare results, and track benchmark practices across studies.",
    "The Open Research Knowledge Graph is an infrastructure for structured benchmarking knowledge. It captures datasets, models, evaluation metrics, results, and scholarly publications in a graph representation. By doing so, it enables precise querying of benchmark data, model evaluations, and reported outcomes across the scientific literature.",
    "The ORKG builds a structured representation of benchmarking knowledge across research domains. It records datasets, models, evaluation metrics, results, and publications in a graph structure. This enables researchers to explore evaluation setups, track model performance, and compare benchmark findings across scientific contributions in a systematic way.",
]


async def run_experiment(experiment_name, scenario_id: int, kg_description: str | None = None):

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
                item.input["question"], scenario_id, langfuse_handler, kg_description
            )

            root_span.score_trace(
                name="Exact Match",
                value=exact_match(output, item.expected_output["output"]),
            )

    logger.info(
        f"\nFinished processing dataset {evaluation_config["dataset"]} for run '{experiment_name}'."
    )


async def run_my_langchain_llm_app(
    question: str, scenario_id: int, callback_handler, kg_description: str | None = None
):

    config_path = (
        Path(__file__).resolve().parent.parent.parent.parent.parent
        / "config"
        / evaluation_config["config_file"]
    )
    with open(config_path, "rt", encoding="utf8") as f:
        default_config = yaml.safe_load(f.read())
        f.close()

    if kg_description is not None:
        default_config["kg_description"] = kg_description

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

        result = await graph.nodes["validate_question"].ainvoke(
            {"initial_question": question, "scenario_id": f"scenario_{scenario_id}"},
            config={"callbacks": [callback_handler]},
        )

        graphRouters = GraphRouters(config, None)
        output = graphRouters.validate_question_router(result)

        root_span.update_trace(input=question, output=output)

    return output


if __name__ == "__main__":
    langfuse = get_client()

    try:
        if langfuse.auth_check():
            print("Langfuse client is authenticated and ready!")

            # for index, desc in enumerate(orkg_descriptions):
            asyncio.run(
                run_experiment(
                    experiment_name=f"{evaluation_config['experiment_name']}",
                    scenario_id=evaluation_config["scenario_id"],
                )
            )
        else:
            print("Authentication failed. Please check your credentials and host.")
    except Exception as e:
        print(f"An error occurred: {e}")
