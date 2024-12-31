import json
import logging
import logging.config
from pathlib import Path
import tiktoken
import tempfile
from uuid import uuid4


def setup_logger(name):
    """
    Set up logging configuration.

    Parameters:
    - name (str): Typically __name__ from the calling module.

    Returns:
    - logger (logging.Logger): Configured logger object.
    """
    # Resolve the path to the configuration file
    parent_dir = Path(__file__).resolve().parent.parent.parent
    config_path = parent_dir / "config" / "logging.ini"

    # Configure logging
    logging.config.fileConfig(config_path, disable_existing_loggers=False)

    # Get and return the logger
    return logging.getLogger(name)


logger = setup_logger(__name__)


def create_user_session(session_id=None, user_session_dir=False, input_dir=False):
    """
    If no session_id is provided, creates a new session_id and all the temporary directory for the kgbot to save generated and input files.
    If session_id is provided, returns the path to the temporary directory for the kgbot depending on the provided argument.

    Args:
        session_id (str, optional): The session_id to use. Defaults to None.
        user_session_dir (bool, optional): If True, returns the path to the temporary directory for the kgbot. Defaults to False.
        input_dir (bool, optional): If True, returns the path to the input files directory for the kgbot. Defaults to False.

    Returns:
        str: The session_id if no session_id is provided.
        Path: The path to the temporary directory for the kgbot if user_session_dir is True.
        Path: The path to the input files directory for the kgbot if input_dir is True.
    """

    if session_id is None:
        session_id = str(uuid4().hex)

        # Create a temporary directory for the kgbot
        kgbot_temp_dir = Path(tempfile.gettempdir()) / "kgbot"
        kgbot_temp_dir.mkdir(parents=True, exist_ok=True)

        user_session_dir = kgbot_temp_dir / session_id
        user_session_dir.mkdir(parents=True, exist_ok=True)

        input_dir = user_session_dir / "input_files"
        input_dir.mkdir(parents=True, exist_ok=True)

        return session_id

    else:
        if user_session_dir:
            user_session_dir = Path(tempfile.gettempdir()) / "kgbot" / session_id
            return user_session_dir

        if input_dir:
            input_dir = (
                Path(tempfile.gettempdir()) / "kgbot" / session_id / "input_files"
            )
            return input_dir


def load_config():
    config_path = Path(__file__).resolve().parent.parent / "config" / "langgraph.json"
    with open(config_path, "r") as file:
        return json.load(file)


def get_module_prefix(name):
    """
    Extracts the module prefix based on the current file's __name__,
    excluding the last part to get the parent module path.

    Example:
    __name__ = "app.core.agents.enpkg.agent"
    get_module_prefix(__name__) -> "app.core.agents.enpkg"
    """
    current_module = name
    module_parts = current_module.split(".")
    return ".".join(module_parts[:-1])


def token_counter(text: str) -> int:
    tokenizer = tiktoken.encoding_for_model(model_name="gpt-4o")
    # TODO [Franck]: the model name should be a config param
    tokens = tokenizer.encode(text)
    return len(tokens)
