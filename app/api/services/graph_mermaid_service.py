import importlib


def get_graph_schema(scenario_id: str):
    scenario_module = importlib.import_module(
        f"app.core.scenarios.scenario_{scenario_id}.scenario_{scenario_id}"
    )
    return scenario_module.graph.get_graph().draw_mermaid()
