from langsmith import Client
from pandas import DataFrame
import importlib


def evaluate_provide_query(outputs: dict, reference_outputs: dict) -> dict:
    """Evaluate if the model did provide a valid SPARQL query or didn't."""
    score = 1 if "last_generated_query" in outputs else 0
    return {
        "key": "query_provided",
        "score": score,
    }


def evaluate_provide_answer(outputs: dict, reference_outputs: dict) -> dict:
    """Evaluate if the model did provide an answer or didn't answer."""
    response: str = outputs["messages"][-1].content
    score = 1 if response.lower().find("i don't know") == -1 else 0
    return {
        "key": "answer_provided",
        "score": score,
    }


def run_example(input):
    scenario_module = importlib.import_module(
        f"app.scenarios.scenario_{scenario}.scenario_{scenario}"
    )
    return scenario_module.run_scenario(input["question"])


def start_evaluation(client: Client, dataset_name: str):
    experiment_results = client.evaluate(
        run_example,
        data=dataset_name,
        evaluators=[evaluate_provide_query],
        experiment_prefix=f"genkgbot-scenario_{scenario}-deepseek-reasoner",
        num_repetitions=1,
        max_concurrency=4,
        metadata={"scenario": f"{scenario}", "llm_model": "deepseek-reasoner"},
    )

    pandas_results: DataFrame = experiment_results.to_pandas()
    print(pandas_results.to_string())


scenario = -1


def main():
    for scenario in range(6, 7):
        globals()["scenario"] = scenario
        client = Client()
        dataset_name = "GenKGBot Evaluation v1_1"
        start_evaluation(client=client, dataset_name=dataset_name)


if __name__ == "__main__":
    main()
