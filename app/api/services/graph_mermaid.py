import os
from pathlib import Path
from app.api.responses.scenario_schema import ScenarioSchema
from app.utils.config_manager import get_scenario_module


def get_scenarios_schema() -> list[ScenarioSchema]:
    """
    Get the schema for all scenarios.

    Returns:
        list: A list of dictionaries each containing the scenario ID (scenario_id) and its schema
            (schema) in a mermaid code block.
    """

    scenarios_path = Path(__file__).resolve().parent.parent.parent / "scenarios"
    scenario_ids = [int(f.split("_")[1]) for f in os.listdir(scenarios_path) if os.path.isdir(os.path.join(scenarios_path, f))]

    scenarios_schema = []
    for scenario_id in scenario_ids:
        scenarios_schema.append(
            ScenarioSchema(
                scenario_id=str(scenario_id),
                graph_schema="```mermaid\n" + get_graph_schema(scenario_id) + "\n```",
            )
        )
    return scenarios_schema


def get_graph_schema(scenario_id: int) -> str:
    """
    Get the graph schema for a given scenario.
    Args:
        scenario_id (int): The ID of the scenario.
    Returns:
        str: The graph schema in mermaid format.
    """

    scenario_module = get_scenario_module(scenario_id)
    return scenario_module.graph.get_graph(xray=1).draw_mermaid()
