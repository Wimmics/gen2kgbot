from app.utils.config_manager import get_configuration, get_scenario_module


def get_scenarios_schema():
    """
    Get the schema for all scenarios.

    Returns:
        list: A list of dictionaries each containing the scenario ID (scenario_id) and its schema
            (schema) in a mermaid code block.
    """

    config = get_configuration()

    scenario_ids = [
        int(id.split("_")[1]) for id in config.keys() if id.startswith("scenario_")
    ]
    scenarios_schema = []
    for scenario_id in scenario_ids:
        scenarios_schema.append(
            {
                "scenario_id": scenario_id,
                "schema": "```mermaid\n" + get_graph_schema(scenario_id) + "\n```",
            }
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
