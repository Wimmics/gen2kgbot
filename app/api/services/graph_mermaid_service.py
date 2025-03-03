from app.core.utils.config_manager import get_scenario_module


def get_graph_schema(scenario_id: int):
    scenario_module = get_scenario_module(scenario_id)
    return scenario_module.graph.get_graph().draw_mermaid()
