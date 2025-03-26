import asyncio
from langsmith import Client
from pandas import DataFrame
from langgraph.graph.state import CompiledStateGraph
from app.core.utils.config_manager import get_scenario_module
from app.core.utils.graph_state import InputState


def evaluate_last_judging_score(outputs: dict, reference_outputs: dict) -> dict:
    """Evaluate if the model generated a query judged runnable."""
    if "query_judgements" in outputs and len(outputs["query_judgements"]) > 0:
        score = outputs["query_judgements"][-1]["judging_grade"]
    elif (
        outputs["question_validation_results"].find("true") != -1
        and outputs["question_validation_results"].find("false") == -1
    ):
        score = 0
    else:
        score = -1

    return {
        "key": "last_judging_score",
        "score": score,
    }


async def run_example(input):

    graph: CompiledStateGraph = get_scenario_module(scenario).graph

    return await graph.ainvoke(
        input=InputState({"initial_question": input["initial_question"]})
    )


async def start_evaluation(client: Client, dataset_name: str):
    model_name = "llama3.1:70b"
    experiment_results = await client.aevaluate(
        run_example,
        data=dataset_name,
        evaluators=[evaluate_last_judging_score],
        experiment_prefix=f"gen2kgbot-scenario_{scenario}-{model_name}",
        num_repetitions=1,
        max_concurrency=4,
        metadata={"scenario": scenario, "llm_model": model_name},
    )

    pandas_results: DataFrame = experiment_results.to_pandas()
    print(pandas_results.to_string())


scenario = -1


async def main():
    for scenario in range(7, 8):
        globals()["scenario"] = scenario
        client = Client()
        dataset_name = "Gen2KGBot Gold PubChem dataset v1 - test"
        await start_evaluation(client=client, dataset_name=dataset_name)


if __name__ == "__main__":
    asyncio.run(main())
