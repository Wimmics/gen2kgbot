import os
from pathlib import Path
import yaml
from app.api.requests.create_config import QueryExample
from app.utils.logger_manager import setup_logger

logger = setup_logger(__package__, __file__)


def add_missing_config_params(config: dict):
    """
    Adds missing configuration parameters to the provided config dictionary.
    This function loads additional parameters from a configuration file and merges them into the config dictionary.
    Args:
        config (dict): The original configuration dictionary.
    Returns:
        dict: The updated configuration dictionary with additional parameters.
    """

    params_to_add_path = (
        Path(__file__).resolve().parent.parent.parent.parent
        / "config"
        / "params_to_add.yml"
    )

    with open(params_to_add_path, "r") as to_add_file:
        config_to_add = yaml.safe_load(to_add_file) or {}

    config.update(config_to_add)

    return config


def save_query_examples_to_file(kg_short_name: str, query_examples: list[QueryExample]):
    """
    Saves the provided query examples to the embedding folder.

    Args:
        query_examples (dict): The query examples to save.
    """

    query_examples_folder = (
        Path(__file__).resolve().parent.parent.parent.parent
        / "data"
        / kg_short_name
        / "example_queries"
    )

    if not query_examples_folder.exists():
        query_examples_folder.mkdir(parents=True, exist_ok=True)

    for index, query_example in enumerate(query_examples, start=1):
        with open(f"{query_examples_folder}/{index}.rq", "w") as file:
            content = f"# {query_example.question}\n\n" f"{query_example.query}\n"
            file.write(content)


def get_available_configurations():
    """
    Retrieves the available configurations from the configuration directory.
    Returns:
        list: A list of available configuration short names.
    """

    config_path = Path(__file__).resolve().parent.parent.parent.parent / "config"

    config_short_names = set()

    for filename in os.listdir(config_path):
        with open(file=os.path.join(config_path, filename), mode="r") as f:
            config = yaml.safe_load(f) or {}
            kg_short_name = config.get("kg_short_name")

            if kg_short_name is not None:
                config_short_names.add(kg_short_name)

    logger.info(f"{len(config_short_names)} config found.")

    return config_short_names
