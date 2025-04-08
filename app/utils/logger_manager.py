import logging.config
import os
from pathlib import Path
import yaml


def setup_logger(
    package: str = __package__ or "", file: str = __file__
) -> logging.Logger:
    """
    Set up logging configuration.

    Args:
      package (str): specify the package name of the module for which the logger is being set up.
    If no package name is provided, it defaults to `__package__` which is the package name of the current module.
      file (str): specify the file path of the module where the logger is being set up.
    It is used to determine the name of the logger based on the module's file path.

    Returns:
      returns a configured `logging.Logger` object with module name.
    """

    # Normalize the file path: in case of Langgraph Studio, it is always '/'
    file = file.replace("/", os.path.sep)

    if package == "":
        package = "[no_mod]"
    _mod_name = package + "." + file.split(os.path.sep)[-1]
    if _mod_name.endswith(".py"):
        _mod_name = _mod_name[: -len(".py")]

    # Resolve the path to the configuration file
    parent_dir = Path(__file__).resolve().parent.parent.parent
    config_path = parent_dir / "config" / "logging.yml"

    # Configure logging
    with open(config_path, "rt", encoding="utf8") as f:
        log_config = yaml.safe_load(f.read())
        # Create log folder if it not exists
        log_file_handler = (
            Path(log_config["handlers"]["file_handler"]["filename"]).resolve().parent
        )
        if not os.path.exists(log_file_handler):
            os.makedirs(log_file_handler)

    logging.config.dictConfig(log_config)

    logger = logging.getLogger(_mod_name)
    # logger.info(f"Setup Logger Done for {package} - {file}")

    # Get and return the logger
    return logger
