import os
from pathlib import Path
import yaml
from argparse import ArgumentParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langgraph.graph.state import CompiledStateGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from app.core.utils.envkey_manager import (
    get_deepseek_key,
    get_google_key,
    get_openai_key,
    get_ovh_key,
)
from app.core.utils.graph_state import InputState
from app.core.utils.logger_manager import setup_logger

logger = setup_logger(__package__, __file__)

# Selected seq2seq LLM
current_llm = None

# Running scenario id
current_scenario = None

# Vector db that contains the documents describing the classes in the form: "(uri, label, description)"
classes_vector_db = None

# Vector db that contains the example SPARQL queries and associated questions
queries_vector_db = None


def setup_cli():
    parser = ArgumentParser(
        description="Process the scenario with the predifined or custom question and configuration."
    )
    parser.add_argument(
        "-q",
        "--question",
        type=str,
        help='User\'s question. Defaults to "What protein targets does donepezil (CHEBI_53289) inhibit with an IC50 less than 5 µM?"',
        default="What protein targets does donepezil (CHEBI_53289) inhibit with an IC50 less than 5 µM?",
    )
    parser.add_argument("-p", "--params", type=str, help="Custom configuration file")
    parser.add_argument(
        "dev",
        nargs="*",
        default="",
        type=str,
        help="Ignored. Fake argument to allow Langgraph Studio start with option dev",
    )
    globals()["args"] = parser.parse_args()


# Initialize the CLI
args = None
setup_cli()


def get_configuration() -> dict:
    """
    Load the configuration file
    """
    if args.params:
        config_path = args.params
    else:
        # Resolve the path to the configuration file
        config_path = (
            Path(__file__).resolve().parent.parent.parent / "config" / "params.yml"
        )

    logger.info(f"Using configuration file: {config_path}")

    # Configure logging
    with open(config_path, "rt") as f:
        return yaml.safe_load(f.read())


# Load the configuration
config = get_configuration()


def get_current_llm() -> BaseChatModel:
    if current_llm:
        return current_llm
    else:
        logger.error("No LLM is currently initialised")
        raise Exception("No LLM is currently initialised")


def get_current_scenario() -> str:
    if current_scenario:
        return current_scenario
    else:
        logger.error("No scenario id has been set up")
        raise Exception("No scenario id has been set up")


def get_kg_full_name() -> str:
    return config["kg_full_name"]


def get_kg_short_name() -> str:
    return config["kg_short_name"]


def get_vector_db_name() -> str:
    return config[get_current_scenario()]["text_embedding_llm"]["vector_db"]


def get_embeddings_model_id() -> str:
    return config[get_current_scenario()]["text_embedding_llm"]["id"]


def get_kg_sparql_endpoint_url() -> str:
    return config["kg_sparql_endpoint_url"]


def get_known_prefixes() -> dict:
    """
    Get the prefixes and associated namespaces from configuration file
    """
    return config["prefixes"]


def get_class_context_cache_directory() -> Path:
    """
    Generate the path for the cache of class context files, and
    create the directory structure if it does not exist.

    The path includes the KG short name (e.g. "idsm") and "classes_context" sub-directories.
    E.g. "./data/idsm/classes_context"
    """
    str_path = (
        config["class_context_cache_directory"]
        + f"/{get_kg_short_name().lower()}/classes_context"
    )
    if os.path.isabs(str_path):
        path = Path(str_path)
    else:
        path = Path(__file__).resolve().parent.parent.parent.parent / str_path

    if not os.path.exists(path):
        os.makedirs(path)
    return path


def get_embeddings_directory() -> Path:
    """
    Generate the path of the pre-computed embedding files, and
    create the directory structure if it does not exist.

    The path includes the KG short name (e.g. "idsm"), vector db name (e.g. "faiss") sub-directories.
    E.g. "./data/idsm/faiss_embeddings"
    """
    str_path = (
        config["embeddings_directory"]
        + f"/{get_kg_short_name().lower()}/{get_vector_db_name()}_embeddings"
    )
    if os.path.isabs(str_path):
        path = Path(str_path)
    else:
        path = Path(__file__).resolve().parent.parent.parent.parent / str_path

    if not os.path.exists(path):
        os.makedirs(path)
    return path


def get_temp_directory() -> Path:
    str_path = config["temp_directory"]
    if os.path.isabs(str_path):
        path = Path(str_path)
    else:
        path = Path(__file__).resolve().parent.parent.parent.parent / str_path

    if not os.path.exists(path):
        os.makedirs(path)
    return path


def get_llm(scenario: str) -> BaseChatModel:
    """
    Create a seq2seq LLM based on the scenario configuration
    """

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
            auth=("username", "password"),
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

    logger.info(f"Seq2seq LLM initialized: {model_type} - {model_id} ")
    globals()["current_llm"] = llm
    globals()["current_scenario"] = scenario
    return llm


def get_embedding_type() -> Embeddings:
    """
    Instantiate a text embedding model based on the current scenario configuration

    Returns:
        Embeddings: text embedding model
    """

    scenario = get_current_scenario()
    embedding_type = config[scenario]["text_embedding_llm"]["type"]
    model_id = config[scenario]["text_embedding_llm"]["id"]

    if embedding_type == "ollama-embeddings":
        embeddings = OllamaEmbeddings(model=model_id)
        logger.info("Text embedding model initialized: OllamaEmbeddings")

    elif embedding_type == "openai-embeddings":
        embeddings = OpenAIEmbeddings(model=model_id)
        logger.info("Text embedding model initialized: OpenAiEmbeddings")

    return embeddings


def get_class_vector_db() -> VectorStore:
    """
    Instantiate a vector db based on the configuration, to store the
    pre-computed embeddings of the RDFS/OWL classes

    Returns:
        VectorStore: vector db
    """

    # Already initialized?
    if globals()["classes_vector_db"] != None:
        return globals()["classes_vector_db"]

    embeddings = get_embedding_type()
    model_id = get_embeddings_model_id()
    vector_db = get_vector_db_name()

    embeddings_directory = f"{get_embeddings_directory()}/{config["class_context_embeddings_prefix"]}{model_id}_{vector_db}_index"
    logger.debug(f"Classes context embeddings directory: {embeddings_directory}")

    if vector_db == "faiss":
        db = FAISS.load_local(
            embeddings_directory,
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info(f"Classes context vector DB initialized: {embeddings_directory}")

    elif vector_db == "chroma":
        db = Chroma(
            persist_directory=embeddings_directory, embedding_function=embeddings
        )
        logger.info(f"Classes context vector DB initialized: {embeddings_directory}")

    else:
        logger.error(f"Unsupported type of vector db: {vector_db}")
        raise Exception(f"Unsupported type of vector db: {vector_db}")

    globals()["classes_vector_db"] = db
    return db


def get_query_vector_db() -> VectorStore:
    """
    Instantiate a vector db based on the configuration, to store the
    pre-computed embeddings of the SPARQL queries

    Returns:
        VectorStore: vector db
    """

    # Already initialized?
    if globals()["queries_vector_db"] != None:
        return globals()["queries_vector_db"]

    embeddings = get_embedding_type()
    model_id = get_embeddings_model_id()
    vector_db = get_vector_db_name()

    embeddings_directory = f"{get_embeddings_directory()}/{config["queries_embeddings_prefix"]}{model_id}_{vector_db}_index"
    logger.debug(f"SPARQL queries embeddings directory: {embeddings_directory}")

    if vector_db == "faiss":
        db = FAISS.load_local(
            embeddings_directory,
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info(f"SPARQL queries vector DB initialized: {embeddings_directory}")

    elif vector_db == "chroma":
        db = Chroma(
            persist_directory=embeddings_directory, embedding_function=embeddings
        )
        logger.info(f"SPARQL queries vector DB initialized: {embeddings_directory}")

    else:
        logger.error(f"Unsupported type of vector db: {vector_db}")
        raise Exception(f"Unsupported type of vector db: {vector_db}")

    globals()["queries_vector_db"] = db
    return db


async def main(graph: CompiledStateGraph):
    """
    Entry point when invoked from the CLI

    Args:
        graph (CompiledStateGraph): Langraph cmopiled state graph
    """

    question = args.question
    state = await graph.ainvoke(input=InputState({"initial_question": question}))

    logger.info("==============================================================")
    for m in state["messages"]:
        logger.info(m.pretty_repr())

    if "last_generated_query" in state:
        logger.info("==============================================================")
        logger.info("last_generated_query: " + state["last_generated_query"])
    logger.info("==============================================================")
