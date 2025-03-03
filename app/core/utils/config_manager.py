import os
from typing import Literal
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

# Selected seq2seq LLM. Dictionary with the scenario id as key
current_llm = {}

# Vector db that contains the documents describing the classes in the form: "(uri, label, description)".
# Dictionary with the scenario id as key
classes_vector_db = {}

# Vector db that contains the example SPARQL queries and associated questions.
# Dictionary with the scenario id as key
queries_vector_db = {}


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


def get_kg_full_name() -> str:
    return config["kg_full_name"]


def get_kg_short_name() -> str:
    return config["kg_short_name"]


def get_kg_description() -> str:
    return config["kg_description"]


def get_kg_sparql_endpoint_url() -> str:
    return config["kg_sparql_endpoint_url"]


def get_known_prefixes() -> dict:
    """
    Get the prefixes and associated namespaces from configuration file
    """
    return config["prefixes"]


def get_class_context_format() -> Literal["turtle", "tuple"]:
    format = config["class_context_format"]
    if format != "turtle" and format != "tuple":
        raise ValueError(f"Invalid parameter class_context_format: {format}")
    return format


def get_class_context_cache_directory() -> Path:
    """
    Generate the path for the cache of class context files, and
    create the directory structure if it does not exist.

    The path includes sub-dir: KG short name (e.g. "idsm"), "classes_context", the format (e.g. "turtle" or "tuple")
    E.g. "./data/idsm/classes_context/turtle" or "./data/idsm/classes_context/tuple"
    """
    str_path = (
        config["data_directory"]
        + f"/{get_kg_short_name().lower()}/classes_context/{get_class_context_format()}"
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


def get_vector_db_name(scenario_id: str) -> str:
    embed_id = config[scenario_id]["text_embedding_model"]
    embed_config = config["text_embedding_models"][embed_id]
    return embed_config["vector_db"]


def get_embeddings_model_id(scenario_id: str) -> str:
    embed_id = config[scenario_id]["text_embedding_model"]
    embed_config = config["text_embedding_models"][embed_id]
    return embed_config["id"]


def get_embeddings_directory(vector_db_name: str) -> Path:
    """
    Generate the path of the pre-computed embedding files, and
    create the directory structure if it does not exist.

    The path includes the KG short name (e.g. "idsm"), vector db name (e.g. "faiss") sub-directories.
    E.g. "./data/idsm/faiss_embeddings"
    """
    str_path = (
        config["data_directory"]
        + f"/{get_kg_short_name().lower()}/{vector_db_name}_embeddings"
    )
    if os.path.isabs(str_path):
        path = Path(str_path)
    else:
        path = Path(__file__).resolve().parent.parent.parent.parent / str_path

    if not os.path.exists(path):
        os.makedirs(path)
    return path


def get_seq2seq_model(scenario_id: str) -> BaseChatModel:
    """
    Create a seq2seq LLM based on the scenario configuration
    """

    if scenario_id in current_llm.keys() and current_llm[scenario_id] is not None:
        return current_llm[scenario_id]

    llm_id = config[scenario_id]["seq2seq_model"]
    llm_config = config["seq2seq_models"][llm_id]

    server_type = llm_config["server_type"]
    model_id = llm_config["id"]

    if "temperature" in llm_config.keys():
        temperature = llm_config["temperature"]
    else:
        temperature = None

    if "max_retries" in llm_config.keys():
        max_retries = llm_config["max_retries"]
    else:
        max_retries = None

    if "model_kwargs" in llm_config.keys():
        model_kwargs = llm_config["model_kwargs"]
    else:
        model_kwargs = {}

    if server_type == "openai":
        llm_config = ChatOpenAI(
            temperature=temperature,
            model=model_id,
            max_retries=max_retries,
            verbose=True,
            openai_api_key=get_openai_key(),
            model_kwargs=model_kwargs,
        )

    elif server_type == "ollama":
        llm_config = ChatOllama(
            temperature=temperature,
            model=model_id,
            max_retries=max_retries,
            verbose=True,
            model_kwargs=model_kwargs,
        )

    elif server_type == "ollama-server":
        base_url = llm_config["base_url"]

        # TODO Hundle Ollama Servers with Auth
        llm_config = ChatOllama(
            temperature=temperature,
            model=model_id,
            max_retries=max_retries,
            verbose=True,
            model_kwargs=model_kwargs,
            auth=("username", "password"),
        )

    elif server_type == "ovh":
        base_url = llm_config["base_url"]

        llm_config = ChatOpenAI(
            temperature=temperature,
            model=model_id,
            max_retries=max_retries,
            verbose=True,
            base_url=base_url,
            api_key=get_ovh_key(),
            model_kwargs=model_kwargs,
        )

    elif server_type == "hugface":
        hfe = HuggingFaceEndpoint(
            repo_id=model_id,
            task="text-generation",
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.03,
        )

        llm_config = ChatHuggingFace(llm=hfe, verbose=True)

    elif server_type == "google":
        llm_config = ChatGoogleGenerativeAI(
            model=model_id,
            temperature=temperature,
            max_retries=max_retries,
            api_key=get_google_key(),
            verbose=True,
            model_kwargs=model_kwargs,
        )

    elif server_type == "deepseek":
        base_url = llm_config["base_url"]
        llm_config = ChatOpenAI(
            temperature=temperature,
            model=model_id,
            max_retries=max_retries,
            verbose=True,
            openai_api_base=base_url,
            openai_api_key=get_deepseek_key(),
            model_kwargs=model_kwargs,
        )

    else:
        logger.error(f"Unsupported type of seq2seq model: {server_type}")
        raise Exception(f"Unsupported type of seq2seq model: {server_type}")

    logger.info(f"Seq2seq model initialized: {server_type} - {model_id} ")
    globals()["current_llm"][scenario_id] = llm_config
    return llm_config


def get_embedding_model(scenario_id: str) -> Embeddings:
    """
    Instantiate a text embedding model based on the scenario configuration

    Args:
        scenario_id (str): scenario ID

    Returns:
        Embeddings: text embedding model
    """

    embed_id = config[scenario_id]["text_embedding_model"]
    embed_config = config["text_embedding_models"][embed_id]

    server_type = embed_config["server_type"]
    model_id = embed_config["id"]

    if server_type == "ollama-embeddings":
        embeddings = OllamaEmbeddings(model=model_id)

    elif server_type == "openai-embeddings":
        embeddings = OpenAIEmbeddings(model=model_id)

    else:
        logger.error(f"Unsupported type of embedding model: {server_type}")
        raise Exception(f"Unsupported type of embedding model: {server_type}")

    logger.info(f"Text embedding model initialized: {server_type} - {model_id} ")
    return embeddings


def create_vector_db(
    scenario_id: str, vector_db_name: str, embeddings_directory: str
) -> VectorStore:
    """
    Create a vector db based on the configuration,
    and load the pre-computed embeddings from the given directory

    Args:
        scenario_id (str): scenario ID
        vector_db_name (str): type of vector db (currently supported: "faiss", "chroma")
        embeddings_directory (str): directory containing the pre-computed embeddings

    Returns:
        VectorStore: vector db
    """

    embeddings = get_embedding_model(scenario_id)

    if vector_db_name == "faiss":
        db = FAISS.load_local(
            embeddings_directory,
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )
    elif vector_db_name == "chroma":
        db = Chroma(
            persist_directory=embeddings_directory, embedding_function=embeddings
        )
    else:
        logger.error(f"Unsupported type of vector DB: {vector_db_name}")
        raise Exception(f"Unsupported type of vector DB: {vector_db_name}")

    return db


def get_class_context_vector_db(scenario_id: str) -> VectorStore:
    """
    Create a vector db based on the scenario configuration,
    and load the pre-computed embeddings of the RDFS/OWL classes

    Args:
        scenario_id (str): scenario ID

    Returns:
        VectorStore: vector db
    """

    # Already initialized?
    if (
        scenario_id in classes_vector_db.keys()
        and classes_vector_db[scenario_id] is not None
    ):
        return classes_vector_db[scenario_id]

    model_id = get_embeddings_model_id(scenario_id)
    vector_db_name = get_vector_db_name(scenario_id)
    embeddings_directory = f"{get_embeddings_directory(vector_db_name)}/{config["class_context_embeddings_prefix"]}{model_id}_{vector_db_name}_index"
    logger.debug(f"Classes context embeddings directory: {embeddings_directory}")

    db = create_vector_db(scenario_id, vector_db_name, embeddings_directory)
    logger.info("Classes context vector DB initialized.")
    globals()["classes_vector_db"][scenario_id] = db
    return db


def get_query_vector_db(scenario_id: str) -> VectorStore:
    """
    Create a vector db based on the scenario configuration,
    and load the pre-computed embeddings of the SPARQL queries

    Args:
        scenario_id (str): scenario ID

    Returns:
        VectorStore: vector db
    """

    # Already initialized?
    if (
        scenario_id in queries_vector_db.keys()
        and queries_vector_db[scenario_id] is not None
    ):
        return queries_vector_db[scenario_id]

    model_id = get_embeddings_model_id(scenario_id)
    vector_db_name = get_vector_db_name(scenario_id)
    embeddings_directory = f"{get_embeddings_directory(vector_db_name)}/{config["queries_embeddings_prefix"]}{model_id}_{vector_db_name}_index"
    logger.debug(f"SPARQL queries embeddings directory: {embeddings_directory}")

    db = create_vector_db(scenario_id, vector_db_name, embeddings_directory)
    logger.info("SPARQL queries vector DB initialized.")
    globals()["queries_vector_db"][scenario_id] = db
    return db


async def main(graph: CompiledStateGraph):
    """
    Entry point when invoked from the CLI

    Args:
        graph (CompiledStateGraph): Langraph compiled state graph
    """

    question = args.question
    logger.info(f"Users' question: {question}")
    state = await graph.ainvoke(input=InputState({"initial_question": question}))

    logger.info("==============================================================")
    for m in state["messages"]:
        logger.info(m.pretty_repr())

    if "last_generated_query" in state:
        logger.info("==============================================================")
        logger.info("last_generated_query: " + state["last_generated_query"])
    logger.info("==============================================================")
