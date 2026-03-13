from pathlib import Path
from langfuse import get_client
import json
from app.utils.logger_manager import setup_logger

logger = setup_logger(__package__, __file__)


def create_dataset(langfuse):
    """
    Create a dataset in Langfuse with the name 'sciqa-validation-node'.
    """
    langfuse.create_dataset(name="sciqa-validation-node")

    logger.info("Dataset 'sciqa-validation-node' created successfully.")

    path_to_file = Path(__file__).resolve().parent / "validation-node-dataset.json"

    with open(path_to_file, "r", encoding="utf-8") as f:
        data = json.loads(f.read())

    local_items = []

    for item in data:
        i = {
            "input": {
                "question": item["question"],
            },
            "expected_output": {
                "output": item["output"],
            },
        }
        local_items.append(i)

    # Upload to Langfuse
    for item in local_items:
        langfuse.create_dataset_item(
            dataset_name="sciqa-validation-node",
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
