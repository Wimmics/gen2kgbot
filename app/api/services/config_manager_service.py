from pathlib import Path
import yaml


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
        Path(__file__).resolve().parent.parent.parent / "config" / "params_to_add.yml"
    )

    with open(params_to_add_path, "r") as to_add_file:
        config_to_add = yaml.safe_load(to_add_file) or {}

    # Merge the dictionaries =
    config.update(config_to_add)

    return config
