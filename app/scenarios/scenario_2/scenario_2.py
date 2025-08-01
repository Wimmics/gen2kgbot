import asyncio
from typing import Literal
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from app.scenarios.scenario_2.prompt import system_prompt_template
from app.utils import config_manager
from app.utils.config_manager import ConfigManager
from app.utils.construct_util import ConstructUtil
from app.utils.graph_nodes import GraphNodes
from app.utils.graph_routers import GraphRouters
from app.utils.graph_state import InputState, OverallState
from app.utils.logger_manager import setup_logger

logger = setup_logger(__package__, __file__)


class Scenario2:

    def __init__(self, config_manager: ConfigManager = None):
        self.SCENARIO = "scenario_2"
        self.config = ConfigManager() if config_manager is None else config_manager
        self.constructUtil = ConstructUtil(self.config)
        self.graphNodes = GraphNodes(self.config, self.constructUtil)
        self.graphRouters = GraphRouters(self.config, self.constructUtil)

    def validate_question_router(
        self,
        state: OverallState,
    ) -> Literal["generate_query", "__end__"]:
        """
        Check the question validation results and decide whether to continue the process or stop.

        Args:
            state (OverallState): current state of the conversation

        Returns:
            Literal["generate_query", END]: next step in the conversation
        """
        results = state["question_validation_results"]
        if results.find("true") != -1 and results.find("false") == -1:
            logger.info("Question validation passed.")
            return "generate_query"
        else:
            logger.warning("Question validation failed.")
            return END

    def init(self, state: OverallState) -> OverallState:
        logger.info(f"Running scenario: {self.SCENARIO}")
        return OverallState({"scenario_id": self.SCENARIO})

    async def generate_query(self, state: OverallState) -> OverallState:
        logger.info(f"Question: {state["initial_question"]}")

        template = system_prompt_template

        if "kg_full_name" in template.input_variables:
            template = template.partial(kg_full_name=self.config.get_kg_full_name())

        if "kg_description" in template.input_variables:
            template = template.partial(kg_description=self.config.get_kg_description())

        if "initial_question" in state.keys():
            template = template.partial(initial_question=state["initial_question"])

        prompt = template.format()
        logger.debug(f"Prompt created:\n{prompt}")

        result = await self.config.get_seq2seq_model(
            scenario_id=state["scenario_id"], node_name="generate_query"
        ).ainvoke(template.format())
        return OverallState(
            {"messages": [HumanMessage(state["initial_question"]), result]}
        )

    def construct_graph(self):
        """
        Construct the state graph for the scenario.
        """
        logger.info("Constructing state graph for Scenario 2")

        builder = StateGraph(
            state_schema=OverallState, input=InputState, output=OverallState
        )

        builder.add_node("init", self.init)
        builder.add_node("validate_question", self.graphNodes.validate_question)
        builder.add_node("generate_query", self.generate_query)
        builder.add_node("run_query", self.graphNodes.run_query)
        builder.add_node("interpret_results", self.graphNodes.interpret_results)

        builder.add_edge(START, "init")
        builder.add_edge("init", "validate_question")
        builder.add_conditional_edges(
            "validate_question", self.validate_question_router
        )
        builder.add_conditional_edges(
            "generate_query", self.graphRouters.generate_query_router
        )
        builder.add_conditional_edges("run_query", self.graphRouters.run_query_router)
        builder.add_edge("interpret_results", END)

        return builder.compile()


scenario = Scenario2()
graph = scenario.construct_graph()
config_manager.setup_langgraph_studio(scenario.config)


if __name__ == "__main__":
    asyncio.run(config_manager.main(scenario.config, graph))
