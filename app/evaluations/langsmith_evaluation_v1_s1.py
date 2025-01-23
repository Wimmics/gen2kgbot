from ast import main
from langsmith import Client
from pandas import DataFrame
from app.core.scenarios.scenario_1.scenario_1 import run_scenario


def evaluate_provide_answer(outputs: dict, reference_outputs: dict) -> dict:
    """Evaluate if the model did provide an answer or didn't answer."""
    response: str = outputs["messages"][-1].content
    score = 1 if response.lower().find("i don't know") == -1 else 0
    return {
        "key": "answer_provided",
        "score": score,
    }


def run_example(input):
    return run_scenario(input["question"])


def start_evaluation(client: Client, dataset_name: str):
    experiment_results = client.evaluate(
        run_example,
        data=dataset_name,
        evaluators=[evaluate_provide_answer],
        experiment_prefix="genkgbot-scenario_1-deepseek-r1:1.5b",
        num_repetitions=1,
        max_concurrency=4,
        metadata= {"scenario": f"1", "llm_model":"deepseek-r1:1.5b"}

    )
    # pandas_results: DataFrame = experiment_results.to_pandas()
    # print(pandas_results.to_string())


def main():
    client = Client()
    dataset_name = "GenKGBot Evaluation v1"
    start_evaluation(client=client, dataset_name=dataset_name)


if __name__ == "__main__":
    main()
