import asyncio
from langgraph.graph import StateGraph, START, END
from app.scenarios.scenario_6.prompt import (
    system_prompt_template,
    retry_system_prompt_template,
)
from app.utils import config_manager
from app.utils.config_manager import ConfigManager
from app.utils.graph_nodes import (
    preprocess_question,
    select_similar_classes,
    get_class_context_from_cache,
    get_class_context_from_kg,
    select_similar_query_examples,
    create_query_generation_prompt,
    generate_query,
    validate_question,
    verify_query,
    run_query,
    interpret_results,
)
from app.utils.graph_routers import (
    get_class_context_router,
    preprocessing_subgraph_router,
    validate_question_router,
    verify_query_router,
    run_query_router,
)
from app.utils.graph_state import InputState, OverallState
from app.utils.logger_manager import setup_logger

logger = setup_logger(__package__, __file__)


class Scenario6:

    def __init__(self):
        self.config = None
        self.SCENARIO = "scenario_6"

    def init(self, state: OverallState) -> OverallState:
        self.config = self.get_config()
        logger.info(f"Running scenario: {self.SCENARIO}")
        return OverallState({"scenario_id": self.SCENARIO})

    def create_prompt(self, state: OverallState) -> OverallState:
        return create_query_generation_prompt(system_prompt_template, state)

    def create_retry_prompt(self, state: OverallState) -> OverallState:
        return create_query_generation_prompt(retry_system_prompt_template, state)

    def construct_graph(self):
        """
        Construct the state graph for the scenario.
        """
        logger.info("Constructing state graph for Scenario 6")

        # Subgraph for preprocessing the question: generate context with classes and examples queries
        prepro_builder = StateGraph(
            state_schema=OverallState, input=OverallState, output=OverallState
        )

        prepro_builder.add_node("init", self.init)
        prepro_builder.add_node("validate_question", validate_question)
        prepro_builder.add_node("preprocess_question", preprocess_question)
        prepro_builder.add_node("select_similar_classes", select_similar_classes)
        prepro_builder.add_node(
            "get_context_class_from_cache", get_class_context_from_cache
        )
        prepro_builder.add_node("get_context_class_from_kg", get_class_context_from_kg)
        prepro_builder.add_node(
            "select_similar_query_examples", select_similar_query_examples
        )

        prepro_builder.add_edge(START, "init")
        prepro_builder.add_edge("init", "validate_question")
        prepro_builder.add_conditional_edges("validate_question", validate_question_router)
        prepro_builder.add_edge("preprocess_question", "select_similar_query_examples")
        prepro_builder.add_edge("preprocess_question", "select_similar_classes")
        prepro_builder.add_edge("select_similar_query_examples", END)
        prepro_builder.add_conditional_edges(
            "select_similar_classes", get_class_context_router
        )
        prepro_builder.add_edge("get_context_class_from_cache", END)
        prepro_builder.add_edge("get_context_class_from_kg", END)

        # Main graph for generating and executing the query
        builder = StateGraph(
            state_schema=OverallState, input=InputState, output=OverallState
        )

        builder.add_node("preprocessing_subgraph", prepro_builder.compile())
        builder.add_node("create_prompt", self.create_prompt)
        builder.add_node("generate_query", generate_query)
        builder.add_node("verify_query", verify_query)
        builder.add_node("run_query", run_query)
        builder.add_node("create_retry_prompt", self.create_retry_prompt)

        builder.add_node("interpret_results", interpret_results)

        builder.add_edge(START, "preprocessing_subgraph")
        builder.add_conditional_edges(
            "preprocessing_subgraph", preprocessing_subgraph_router
        )
        builder.add_edge("create_prompt", "generate_query")
        builder.add_edge("generate_query", "verify_query")
        builder.add_conditional_edges("verify_query", verify_query_router)
        builder.add_edge("create_retry_prompt", "generate_query")
        builder.add_conditional_edges("run_query", run_query_router)
        builder.add_edge("interpret_results", END)

        return builder.compile()

    def get_config(self) -> ConfigManager:
        if not self.config:
            self.config = ConfigManager()

        return self.config


scenario = Scenario6()
graph = scenario.construct_graph()


if __name__ == "__main__":
    asyncio.run(config_manager.main(scenario.get_config(), graph))
