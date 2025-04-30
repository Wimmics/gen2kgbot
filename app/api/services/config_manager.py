from pathlib import Path
import yaml

from app.api.requests.create_config import QueryExample


def add_missing_config_params(config: dict):
    """
    Adds missing configuration parameters to the provided config dictionary.
    This function loads additional parameters from a YAML file and merges them into the config dictionary.
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
            content = (
                f"# {query_example.question}\n\n"
                f"{query_example.query}\n"
            )
            file.write(content)
