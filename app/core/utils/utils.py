import argparse
from langchain_core.messages import HumanMessage
import json
import logging
import logging.config
import os
from pathlib import Path
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
import yaml
from langgraph.graph.state import CompiledStateGraph

from app.core.utils.printing import new_log


def setup_logger(package: str = __package__, file: str = __file__) -> logging.Logger:
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
    with open(config_path, "rt") as f:
        log_config = yaml.safe_load(f.read())
    logging.config.dictConfig(log_config)

    # Get and return the logger
    return logging.getLogger(_mod_name)


logger = setup_logger(__package__, __file__)


def get_yml_config():

    # # Resolve the path to the configuration file
    parent_dir = Path(__file__).resolve().parent.parent.parent
    config_path = parent_dir / "config" / "params.yml"

    # # Configure logging
    with open(config_path, "rt") as f:
        return yaml.safe_load(f.read())


config = get_yml_config()


def get_openai_key():
    return os.getenv("OPENAI_API_KEY")

def get_ovh_key():
    return os.getenv("OVHCLOUD_API_KEY")


def get_llm_from_config(scenario: str) -> BaseChatModel:

    model_type = config[scenario]["seq2seq_llm"]["type"]
    model_id = config[scenario]["seq2seq_llm"]["id"]
    base_url = config[scenario]["seq2seq_llm"]["base_url"]
    temperature = config[scenario]["seq2seq_llm"]["temperature"]
    max_retries = config[scenario]["seq2seq_llm"]["max_retries"]
    model_kwargs = config[scenario]["seq2seq_llm"]["model_kwargs"]

    if model_type == "openai":
        llm = ChatOpenAI(
            temperature=temperature,
            model=model_id,
            max_retries=max_retries,
            verbose=True,
            openai_api_key=get_openai_key(),
            model_kwargs=model_kwargs,
        )
        logger.info(f"LLM initialized: OpenAI")

    elif model_type == "ollama":
        llm = ChatOllama(
            temperature=temperature,
            model=model_id,
            max_retries=max_retries,
            verbose=True,
            model_kwargs=model_kwargs,
        )
    
    elif model_type == "ovh":
        llm = ChatOpenAI(
            temperature=temperature,
            model=model_id,
            max_retries=max_retries,
            verbose=True,
            base_url=base_url,  
            api_key=get_ovh_key(),
            model_kwargs=model_kwargs,
        )
        logger.info(f"LLM initialized: OVH")

    return llm


def get_embedding_type_from_config(scenario: str) -> Embeddings:

    embedding_type = config[scenario]["text_embedding_llm"]["type"]
    model_id = config[scenario]["text_embedding_llm"]["id"]

    if embedding_type == "ollama-embeddings":
        embeddings = OllamaEmbeddings(model=model_id)
        logger.info(f"Embedding initialized: OllamaEmbeddings")

    elif embedding_type == "openai-embeddings":
        embeddings = OpenAIEmbeddings(model=model_id)

        logger.info(f"Embedding initialized: OpenAiEmbeddings")

    return embeddings


def get_class_vector_db_from_config(scenario: str) -> VectorStore:

    embeddings = get_embedding_type_from_config(scenario=scenario)
    model_id = config[scenario]["text_embedding_llm"]["id"]

    vector_db = config[scenario]["text_embedding_llm"]["vector_db"]

    embedding_map = {
        "nomic-embed-text": "nomic",
        "mxbai-embed-large": "mxbai",
        "all-minilm": "minilm"
    }
    embedding_id = embedding_map.get(model_id)

    embedding_directory = f"data/{vector_db}_embeddings/idsm/v3_4_full_{embedding_id}_{vector_db}_index"

    if vector_db == "faiss":
        db = FAISS.load_local(
            embedding_directory,
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info(f"Class Vector DB initialized: {embedding_directory}")

    elif vector_db == "chroma":
        db = Chroma(persist_directory=embedding_directory, embedding_function=embeddings)
        logger.info(f"Class Vector DB initialized: {embedding_directory}")


    return db

def get_query_vector_db_from_config(scenario: str) -> VectorStore:

    embeddings = get_embedding_type_from_config(scenario=scenario)
    model_id = config[scenario]["text_embedding_llm"]["id"]

    vector_db = config[scenario]["text_embedding_llm"]["vector_db"]

    embedding_map = {
        "nomic-embed-text": "nomic",
        "mxbai-embed-large": "mxbai",
        "all-minilm": "minilm"
    }
    embedding_id = embedding_map.get(model_id)

    embedding_directory = f"data/{vector_db}_embeddings/idsm/query_v1_{embedding_id}_{vector_db}_index"

    if vector_db == "faiss":
        db = FAISS.load_local(
            embedding_directory,
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info(f"Query Vector DB initialized: {embedding_directory}")

    elif vector_db == "chroma":
        db = Chroma(persist_directory=embedding_directory, embedding_function=embeddings)
        logger.info(f"Query Vector DB initialized: {embedding_directory}")


    return db


def main(graph: CompiledStateGraph):
    """
    Process a predefined or custom question, invokes a graph with the question, and logs the messages returned by the graph.
    """

    parser = argparse.ArgumentParser(
        description="Process the scenario with the predifined or custom question."
    )
    parser.add_argument("-c", "--custom", type=str, help="Provide a custom question.")
    args = parser.parse_args()

    if args.custom:
        question = args.custom
    else:
        question = "What protein targets does donepezil (CHEBI_53289) inhibit with an IC50 less than 20 µM?"

    state = graph.invoke({"messages": HumanMessage(question)})

    new_log()
    for m in state["messages"]:
        m.pretty_print()
    new_log()