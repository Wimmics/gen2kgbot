from app.core.utils.graph_state import OverallState
from app.core.utils.preprocessing import extract_relevant_entities_spacy
from app.core.utils.utils import (
    get_class_vector_db_from_config,
    get_current_llm,
    setup_logger,
)
from langchain_core.messages import AIMessage
from app.core.utils.prompts import interpret_csv_query_results_prompt

logger = setup_logger(__package__, __file__)


def preprocess_question(input: OverallState) -> OverallState:
    result = AIMessage(
        f"{",".join(extract_relevant_entities_spacy(input["initial_question"]))}"
    )
    logger.info("Preprocessing the question was done succesfully")
    return {
        "messages": result,
        "initial_question": input["initial_question"],
        "number_of_tries": 0,
    }


def select_similar_classes(state: OverallState) -> OverallState:

    db = get_class_vector_db_from_config()

    query = state["initial_question"]
    logger.debug(f"query: {query}")

    # Retrieve the most similar text
    retrieved_documents = db.similarity_search(query, k=10)

    result = "These are some relevant classes for the query generation:"
    for doc in retrieved_documents:
        result = f"{result}\n{doc.page_content}"
    result = f"{result}\n\n"

    logger.debug("Done with selecting similar classes to help query generation")

    return {"messages": AIMessage(result), "selected_classes": retrieved_documents}


async def interpret_csv_query_results(state: OverallState) -> OverallState:
    csv_results_message = state["messages"][-1]
    llm = get_current_llm()
    result = await llm.ainvoke(
        interpret_csv_query_results_prompt.format(
            question=state["initial_question"], results=csv_results_message.content
        )
    )
    return OverallState({"messages": result, "results_interpretation": result})
