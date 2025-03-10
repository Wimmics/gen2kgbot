from app.core.utils.config_manager import get_configuration, get_scenario_module


def get_scenarios_schema():
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
    scenario_module = get_scenario_module(scenario_id)
    return scenario_module.graph.get_graph().draw_mermaid()
