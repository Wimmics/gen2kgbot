import asyncio
from typing import Literal
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from app.utils import config_manager
from app.utils.config_manager import ConfigManager
from app.utils.construct_util import ConstructUtil
from app.utils.graph_nodes import GraphNodes
from app.utils.graph_state import InputState, OverallState
from app.utils.logger_manager import setup_logger
from app.scenarios.scenario_1.prompt import system_prompt_template

logger = setup_logger(__package__, __file__)


class Scenario1:

    def __init__(self, config_manager: ConfigManager = None):
        self.SCENARIO = "scenario_1"
        self.config = ConfigManager() if config_manager is None else config_manager
        self.constructUtil = ConstructUtil(self.config)
        self.graphNodes = GraphNodes(self.config, self.constructUtil)

    def init(self, state: OverallState) -> OverallState:
        logger.info(f"Running scenario: {self.SCENARIO}")
        return OverallState({"scenario_id": self.SCENARIO})

    async def ask_question(self, state: OverallState):
        logger.info(f"Question: {state["initial_question"]}")
        result = await self.config.get_seq2seq_model(
            scenario_id=state["scenario_id"], node_name="ask_question"
        ).ainvoke(
            system_prompt_template.format(initial_question=state["initial_question"])
        )
        logger.info(f"Model's response:\n{result.content}")
        return {"messages": [HumanMessage(state["initial_question"]), result]}

    def validate_question_router(
        self, state: OverallState
    ) -> Literal["ask_question", "__end__"]:
        """
        Check the question validation results and decide whether to continue the process or stop.

        Args:
            state (OverallState): current state of the conversation

        Returns:
            Literal["ask_question", END]: next step in the conversation
        """
        results = state["question_validation_results"]
        if results.find("true") != -1 and results.find("false") == -1:
            logger.info("Question validation passed.")
            return "ask_question"
        else:
            logger.warning("Question validation failed.")
            return END

    def construct_graph(self):
        """
        Construct the state graph for the scenario.
        """
        logger.info("Constructing state graph for Scenario 1")

        builder = StateGraph(
            state_schema=OverallState, input=InputState, output=OverallState
        )

        builder.add_node("init", self.init)
        builder.add_node("validate_question", self.graphNodes.validate_question)
        builder.add_node("ask_question", self.ask_question)

        builder.add_edge(START, "init")
        builder.add_edge("init", "validate_question")
        builder.add_conditional_edges(
            "validate_question", self.validate_question_router
        )
        builder.add_edge("ask_question", END)

        return builder.compile()


scenario = Scenario1()
graph = scenario.construct_graph()


if __name__ == "__main__":
    asyncio.run(config_manager.main(scenario.config, graph))
