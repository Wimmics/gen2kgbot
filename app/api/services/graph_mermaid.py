from app.api.responses.scenario_schema import ScenarioSchema
from app.utils.config_manager import ConfigManager


def get_scenarios_schema(config: ConfigManager) -> list[ScenarioSchema]:
    """
    Get the schema for all scenarios.

    Returns:
        list: A list of dictionaries each containing the scenario ID (scenario_id) and its schema
            (schema) in a mermaid code block.
    """

    scenario_ids = [
        int(id.split("_")[1]) for id in config.get_configuration().keys() if id.startswith("scenario_")
    ]
    scenarios_schema = []
    for scenario_id in scenario_ids:
        scenarios_schema.append(
            ScenarioSchema(
                scenario_id=str(scenario_id),
                graph_schema="```mermaid\n" + get_graph_schema(config, scenario_id) + "\n```",
            )
        )
    return scenarios_schema


def get_graph_schema(config: ConfigManager, scenario_id: int) -> str:
    """
    Get the graph schema for a given scenario.
    Args:
        scenario_id (int): The ID of the scenario.
    Returns:
        str: The graph schema in mermaid format.
    """

    scenario_module = config.get_scenario_module(scenario_id)
    return scenario_module.graph.get_graph(xray=1).draw_mermaid()
