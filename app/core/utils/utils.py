import argparse
from multiprocessing.connection import Client
import logging
import logging.config
import os
from pathlib import Path
import re
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
import yaml
from langgraph.graph.state import CompiledStateGraph
from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.utils.printing import new_log


def setup_cli():
    parser = argparse.ArgumentParser(
        description="Process the scenario with the predifined or custom question and configuration."
    )
    parser.add_argument("-c", "--custom", type=str, help="Provide a custom question.")
    parser.add_argument(
        "-p", "--params", type=str, help="Provide a custom configuration path."
    )
    globals()["args"] = parser.parse_args()


args = None


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
        # Create log folder if it not exists
        log_file_handler = (
            Path(log_config["handlers"]["file_handler"]["filename"]).resolve().parent
        )
        if os.path.exists(log_file_handler) == False:
            os.makedirs(log_file_handler)

    logging.config.dictConfig(log_config)

    logger = logging.getLogger(_mod_name)
    # logger.info(f"Setup Logger Done for {package} - {file}")

    # Get and return the logger
    return logger


logger = setup_logger(__package__, __file__)


def get_yml_config():
    # if args.params:
    #     config_path = args.params
    # else:
    # # Resolve the path to the configuration file
    config_path = (
        Path(__file__).resolve().parent.parent.parent / "config" / "params.yml"
    )

    # # Configure logging
    with open(config_path, "rt") as f:
        return yaml.safe_load(f.read())


config = get_yml_config()

current_llm = None
current_scenario = None


def get_current_llm() -> BaseChatModel:
    if current_llm:
        return current_llm
    else:
        logger.error("No LLM is currently initialised")


def get_current_scenario() -> str:
    if current_scenario:
        return current_scenario
    else:
        logger.error("No Current scenario is currently running!")


def get_openai_key():
    return os.getenv("OPENAI_API_KEY")


def get_ovh_key():
    return os.getenv("OVHCLOUD_API_KEY")


def get_huggingface_key():
    return os.getenv("HF_TOKEN")

def get_google_key():
    return os.getenv("GOOGLE_API_KEY")

def get_deepseek_key():
    return os.getenv("DEEPSEEK_API_KEY")

def get_llm_from_config(scenario: str) -> BaseChatModel:

    model_type = config[scenario]["seq2seq_llm"]["type"]
    model_id = config[scenario]["seq2seq_llm"]["id"]
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

    elif model_type == "ollama":
        llm = ChatOllama(
            temperature=temperature,
            model=model_id,
            max_retries=max_retries,
            verbose=True,
            model_kwargs=model_kwargs,
        )
    
    elif model_type == "ollama-server":
        base_url = config[scenario]["seq2seq_llm"]["base_url"]

        # TODO Hundle Ollama Servers with Auth
        llm = ChatOllama(
            temperature=temperature,
            model=model_id,
            max_retries=max_retries,
            verbose=True,
            model_kwargs=model_kwargs,
            auth=("username","password")
        )

    elif model_type == "ovh":
        base_url = config[scenario]["seq2seq_llm"]["base_url"]

        llm = ChatOpenAI(
            temperature=temperature,
            model=model_id,
            max_retries=max_retries,
            verbose=True,
            base_url=base_url,
            api_key=get_ovh_key(),
            model_kwargs=model_kwargs,
        )

    elif model_type == "hugface":
        hfe = HuggingFaceEndpoint(
            repo_id=model_id,
            task="text-generation",
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.03,
        )

        llm = ChatHuggingFace(llm=hfe, verbose=True)

    elif model_type == "google":
        llm = ChatGoogleGenerativeAI(
            model=model_id,
            temperature=temperature,
            max_retries=max_retries,
            api_key=get_google_key(),
            verbose=True,
            model_kwargs=model_kwargs,
        )
    
    elif model_type == "deepseek":
        base_url = config[scenario]["seq2seq_llm"]["base_url"]
        llm = ChatOpenAI(
            temperature=temperature,
            model=model_id,
            max_retries=max_retries,
            verbose=True,
            openai_api_base=base_url,
            openai_api_key=get_deepseek_key(),
            model_kwargs=model_kwargs,
        )

    logger.info(f"LLM initialized : {model_type} - {model_id} ")
    globals()["current_llm"] = llm
    globals()["current_scenario"] = scenario
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


def get_class_vector_db_from_config() -> VectorStore:

    scenario = get_current_scenario()
    embeddings = get_embedding_type_from_config(scenario=scenario)
    model_id = config[scenario]["text_embedding_llm"]["id"]

    vector_db = config[scenario]["text_embedding_llm"]["vector_db"]

    embedding_map = {
        "nomic-embed-text": "nomic",
        "mxbai-embed-large": "mxbai",
        "all-minilm": "minilm",
    }
    embedding_id = embedding_map.get(model_id)

    embedding_directory = (
        f"data/{vector_db}_embeddings/idsm/v3_4_full_{embedding_id}_{vector_db}_index"
    )

    if vector_db == "faiss":
        db = FAISS.load_local(
            embedding_directory,
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info(f"Class Vector DB initialized: {embedding_directory}")

    elif vector_db == "chroma":
        db = Chroma(
            persist_directory=embedding_directory, embedding_function=embeddings
        )
        logger.info(f"Class Vector DB initialized: {embedding_directory}")

    return db


def get_query_vector_db_from_config(scenario: str) -> VectorStore:

    embeddings = get_embedding_type_from_config(scenario=scenario)
    model_id = config[scenario]["text_embedding_llm"]["id"]

    vector_db = config[scenario]["text_embedding_llm"]["vector_db"]

    embedding_map = {
        "nomic-embed-text": "nomic",
        "mxbai-embed-large": "mxbai",
        "all-minilm": "minilm",
    }
    embedding_id = embedding_map.get(model_id)

    embedding_directory = (
        f"data/{vector_db}_embeddings/idsm/query_v1_{embedding_id}_{vector_db}_index"
    )

    if vector_db == "faiss":
        db = FAISS.load_local(
            embedding_directory,
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info(f"Query Vector DB initialized: {embedding_directory}")

    elif vector_db == "chroma":
        db = Chroma(
            persist_directory=embedding_directory, embedding_function=embeddings
        )
        logger.info(f"Query Vector DB initialized: {embedding_directory}")

    return db


def main(graph: CompiledStateGraph):
    """
    Process a predefined or custom question, invokes a graph with the question, and logs the messages returned by the graph.
    """

    if args != None and args.custom:

        question = args.custom
    else:
        question = "What protein targets does donepezil (CHEBI_53289) inhibit with an IC50 less than 5 µM?"

    state = graph.invoke({"initial_question": question})

    new_log()
    for m in state["messages"]:
        m.pretty_print()
    new_log()

    if "last_generated_query" in state:
        new_log()
        print(state["last_generated_query"])
        new_log()


def langsmith_setup():
    # Setting up the LangSmith
    # For now, all runs will be stored in the "KGBot Testing - GPT4"
    # If you want to separate the traces to have a better control of specific traces.
    # Metadata as llm version and temperature can be obtained from traces.

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = (
        f"Gen KGBot Refactoring"  # Please update the name here if you want to create a new project for separating the traces.
    )
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

    client = Client()

    # #Check if the client was initialized
    print(f"Langchain client was initialized: {client}")


def find_sparql_queries(message: str):
    return re.findall("```sparql(.*)```", message, re.DOTALL)
