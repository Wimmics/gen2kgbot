from pathlib import Path
from langfuse import get_client
import json
from app.utils.logger_manager import setup_logger

logger = setup_logger(__package__, __file__)


def create_dataset(langfuse):
    """
    Create a dataset in Langfuse with the name 'sciqa'.
    """
    langfuse.create_dataset(name="sciqa")

    logger.info("Dataset 'sciqa' created successfully.")

    path_to_file = Path(__file__).resolve().parent / "eval_dataset.json"

    with open(path_to_file, "r", encoding="utf-8") as f:
        data = json.loads(f.read())

    local_items = []

    for item in data:
        i = {
            "input": {
                "id": item["id"],
                "question": item["question"],
            },
            "expected_output": {
                "query": item["query"],
                "query_result": item["query_result"],
            },
        }
        local_items.append(i)

    # Upload to Langfuse
    for item in local_items:
        langfuse.create_dataset_item(
            dataset_name="sciqa",
            # any python object or value
            input=item["input"],
            # any python object or value, optional
            expected_output=item["expected_output"],
        )

    logger.info("Items uploaded to dataset successfully.")


if __name__ == "__main__":

    langfuse = get_client()

    # Verify connection
    if langfuse.auth_check():
        logger.info("Langfuse client is authenticated and ready!")
        create_dataset(langfuse)
    else:
        logger.error("Authentication failed. Please check your credentials and host.")
