import asyncio
from langgraph.graph import StateGraph, START, END
from app.scenarios.scenario_3.prompt import system_prompt_template
from app.utils import config_manager
from app.utils.config_manager import ConfigManager
from app.utils.construct_util import ConstructUtil
from app.utils.graph_nodes import GraphNodes
from app.utils.graph_routers import GraphRouters
from app.utils.graph_state import InputState, OverallState
from app.utils.logger_manager import setup_logger

logger = setup_logger(__package__, __file__)


class Scenario3:

    def __init__(self, config_manager: ConfigManager = None):
        self.SCENARIO = "scenario_3"
        self.config = ConfigManager() if config_manager is None else config_manager
        self.constructUtil = ConstructUtil(self.config)
        self.graphNodes = GraphNodes(self.config, self.constructUtil)
        self.graphRouters = GraphRouters(self.config, self.constructUtil)

    def init(self, state: OverallState) -> OverallState:
        logger.info(f"Running scenario: {self.SCENARIO}")
        return OverallState({"scenario_id": self.SCENARIO})

    def create_prompt(self, state: OverallState) -> OverallState:
        return self.graphNodes.create_query_generation_prompt(
            system_prompt_template, state
        )

    def construct_graph(self):
        """
        Construct the state graph for the scenario.
        """
        logger.info("Constructing state graph for Scenario 3")

        builder = StateGraph(
            state_schema=OverallState, input=InputState, output=OverallState
        )

        builder.add_node("init", self.init)
        builder.add_node("validate_question", self.graphNodes.validate_question)
        builder.add_node("preprocess_question", self.graphNodes.preprocess_question)
        builder.add_node(
            "select_similar_classes", self.graphNodes.select_similar_classes
        )
        builder.add_node("create_prompt", self.create_prompt)
        builder.add_node("generate_query", self.graphNodes.generate_query)
        builder.add_node("run_query", self.graphNodes.run_query)
        builder.add_node("interpret_results", self.graphNodes.interpret_results)

        builder.add_edge(START, "init")
        builder.add_edge("init", "validate_question")
        builder.add_conditional_edges(
            "validate_question", self.graphRouters.validate_question_router
        )
        builder.add_edge("preprocess_question", "select_similar_classes")
        builder.add_edge("select_similar_classes", "create_prompt")
        builder.add_edge("create_prompt", "generate_query")
        builder.add_conditional_edges(
            "generate_query", self.graphRouters.generate_query_router
        )
        builder.add_conditional_edges("run_query", self.graphRouters.run_query_router)
        builder.add_edge("interpret_results", END)

        return builder.compile()


scenario = Scenario3()
graph = scenario.construct_graph()
config_manager.setup_langgraph_studio(scenario.config)


if __name__ == "__main__":
    asyncio.run(config_manager.main(scenario.config, graph))
