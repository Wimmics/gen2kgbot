import importlib
import os
from typing import Literal
from pathlib import Path
import yaml
from argparse import ArgumentParser, Namespace
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

# Global config. Shall be initialized by read_configuration()
config = None

# Selected seq2seq LLM. Dictionary with the scenario id as key
current_llm = {}

# Vector db that contains the documents describing the classes in the form: "(uri, label, description)".
# Dictionary with the scenario id as key
classes_vector_db = {}

# Vector db that contains the example SPARQL queries and associated questions.
# Dictionary with the scenario id as key
queries_vector_db = {}


def setup_cli() -> Namespace:
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
        "--prod",
        action="store_true",
        help="Load the production configuration file",
        default=False,
    )
    return parser.parse_args()


def get_configuration() -> dict:
    return config


def read_configuration(args: Namespace = None):
    """
    Load the configuration file and set it in global variable 'config'

    Args:
        args (Namespace): command line arguments. Optional. If not provided,
            the default configuration file is used.
    """
    if args is None:
        # Set the default configuration file: used when starting from Langgraph Studio
        config_path = (
            Path(__file__).resolve().parent.parent.parent / "config" / "params.yml"
        )
        logger.info(f"Loading default configuration file: {config_path}")
    elif args.params:
        config_path = args.params
        logger.info(f"Loading custom configuration file: {config_path}")
    elif args.__contains__("prod") and args.prod:
        config_path = (
            Path(__file__).resolve().parent.parent.parent / "config" / "params_prod.yml"
        )
        logger.info(f"Loading configuration file: {config_path}")
    else:
        # Set the default configuration file
        config_path = (
            Path(__file__).resolve().parent.parent.parent / "config" / "params.yml"
        )
        logger.info(f"Loading default configuration file: {config_path}")

    with open(config_path, "rt", encoding="utf8") as f:
        config = yaml.safe_load(f.read())
        f.close()

    # Make a deep copy of the configuration and assign it to the global variable
    cfg = {}
    for key in config.keys():
        cfg[key] = config[key]
    globals()["config"] = cfg


# Load the default config file.
# Necessary when using Langragraph Studio as it loads the scenarios without CLI arguments.
# If calling from CLI, the default config will be overridden.
read_configuration()


def get_kg_full_name() -> str:
    return config["kg_full_name"]


def get_kg_short_name() -> str:
    return config["kg_short_name"]


def get_kg_description() -> str:
    return config["kg_description"]


def get_kg_sparql_endpoint_url() -> str:
    return config["kg_sparql_endpoint_url"]


def get_ontologies_sparql_endpoint_url() -> str:
    if "ontologies_sparql_endpoint_url" in config.keys():
        return config["ontologies_sparql_endpoint_url"]
    else:
        return config["kg_sparql_endpoint_url"]


def get_properties_qnames_info() -> str:
    return config["properties_qnames_info"]


def get_judging_grade_threshold_retry() -> int:
    return config["judging_grade_threshold_retry"]


def get_judging_grade_threshold_run() -> int:
    return config["judging_grade_threshold_run"]


def get_known_prefixes() -> dict:
    """
    Get the prefixes and associated namespaces from configuration file
    """
    return config["prefixes"]


def get_prefixes_as_sparql() -> str:
    """
    Get the prefixes and associated namespaces as SPARQL prefix declarations
    """
    prefixes = ""
    for prefix, ns in config["prefixes"].items():
        prefixes += f"PREFIX {prefix}: <{ns}>\n"
    return prefixes + "\n"


def get_max_similar_classes() -> int:
    if "max_similar_classes" in config.keys():
        return config["max_similar_classes"]
    else:
        return 10


def expand_similar_classes() -> bool:
    if "expand_similar_classes" in config.keys():
        return config["expand_similar_classes"]
    else:
        return False


def get_class_context_format() -> Literal["turtle", "tuple", "nl"]:
    if "class_context_format" in config.keys():
        format = config["class_context_format"]
        if format not in ["turtle", "tuple", "nl"]:
            raise ValueError(f"Invalid parameter class_context_format: {format}")
    else:
        format = "turtle"
    return format


def get_excluded_classes_namespaces() -> list[str]:
    if "excluded_classes_namespaces" in config.keys():
        return config["excluded_classes_namespaces"]
    else:
        return []


def get_class_context_cache_directory() -> Path:
    """
    Generate the path for the cache of class context files, and
    create the directory structure if it does not exist.

    The path includes sub-dir: KG short name (e.g. "idsm"), "classes_context", the format (e.g. "turtle" or "tuple")
    E.g. "./data/idsm/classes_context/turtle" or "./data/idsm/classes_context/tuple"
    """

    # Format "nl" applies to serialization in prompts but is still stored as "tuple"
    format = get_class_context_format()
    if format == "nl":
        format = "tuple"

    str_path = (
        config["data_directory"]
        + f"/{get_kg_short_name().lower()}/classes_context/{format}"
    )
    if os.path.isabs(str_path):
        path = Path(str_path)
    else:
        path = Path(__file__).resolve().parent.parent.parent.parent / str_path

    if not os.path.exists(path):
        os.makedirs(path)
    return path


def get_preprocessing_directory() -> Path:
    """
    Generate the path for the directory where to store the class and property textual descriptions.
    Create the directory structure if it does not exist.
    """
    str_path = (
        config["data_directory"] + f"/{get_kg_short_name().lower()}/preprocessing"
    )
    if os.path.isabs(str_path):
        path = Path(str_path)
    else:
        path = Path(__file__).resolve().parent.parent.parent.parent / str_path

    if not os.path.exists(path):
        os.makedirs(path)
    return path


def get_class_embeddings_subdir() -> str:
    return config["class_embeddings_subdir"]


def get_property_embeddings_subdir() -> str:
    return config["property_embeddings_subdir"]


def get_temp_directory() -> Path:
    str_path = config["temp_directory"]
    if os.path.isabs(str_path):
        path = Path(str_path)
    else:
        path = Path(__file__).resolve().parent.parent.parent.parent / str_path

    if not os.path.exists(path):
        os.makedirs(path)
    return path


def get_embedding_model_config_by_name(embed_name: str) -> dict:
    """
    Return the configuration of a text embedding model given by its name
    """
    if embed_name not in config["text_embedding_models"]:
        raise ValueError(f"Unknown text embedding model name: {embed_name}")
    else:
        return config["text_embedding_models"][embed_name]


def get_embedding_model_config_by_scenario(scenario_id: str) -> dict:
    """
    Return the configuration of the text embedding model of a given scenario
    """
    embed_name = config[scenario_id]["text_embedding_model"]
    return get_embedding_model_config_by_name(embed_name)


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


def get_seq2seq_model(scenario_id: str, node_name: str) -> BaseChatModel:
    """
    Create a seq2seq LLM based on the scenario configuration
    """
    logger.debug(
        f"Getting seq2seq model for scenario: {scenario_id} and Node: {node_name}"
    )

    if (
        scenario_id in current_llm.keys()
        and current_llm[scenario_id] is not None
        and node_name in dict(current_llm[scenario_id]).keys()
        and current_llm[scenario_id][node_name] is not None
    ):
        logger.debug(f"Using cached current_llm for scenario: {current_llm}")
        return current_llm[scenario_id][node_name]

    logger.debug(f"Current LLM config used: {current_llm}")

    llm_id = config[scenario_id][node_name]
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

    if "top_p" in llm_config.keys():
        top_p = llm_config["top_p"]
    else:
        top_p = 0.95

    if server_type == "openai":
        if model_id.startswith("o"):  # o1/o3 do not support parameter top_p
            llm_config = ChatOpenAI(
                temperature=temperature,
                model=model_id,
                max_retries=max_retries,
                verbose=True,
                openai_api_key=get_openai_key(),
                model_kwargs=model_kwargs,
            )
        else:
            llm_config = ChatOpenAI(
                temperature=temperature,
                model=model_id,
                max_retries=max_retries,
                verbose=True,
                openai_api_key=get_openai_key(),
                model_kwargs=model_kwargs,
                top_p=top_p,
            )

    elif server_type == "ollama":
        llm_config = ChatOllama(
            temperature=temperature,
            model=model_id,
            max_retries=max_retries,
            verbose=True,
            top_p=top_p,
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
            top_p=top_p,
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
            top_p=top_p,
            model_kwargs=model_kwargs,
        )

    elif server_type == "hugface":
        base_url = llm_config["base_url"]

        hfe = HuggingFaceEndpoint(
            model=model_id,
            task="text-generation",
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.03,
            top_p=top_p,
            inference_server_url=base_url,
        )

        llm_config = ChatHuggingFace(llm=hfe, verbose=True)

    elif server_type == "google":
        llm_config = ChatGoogleGenerativeAI(
            model=model_id,
            temperature=temperature,
            max_retries=max_retries,
            api_key=get_google_key(),
            verbose=True,
            top_p=top_p,
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
            top_p=top_p,
            model_kwargs=model_kwargs,
        )

    else:
        logger.error(f"Unsupported type of seq2seq model: {server_type}")
        raise Exception(f"Unsupported type of seq2seq model: {server_type}")

    logger.info(
        f"Seq2Seq model initialized for {scenario_id} and Node: {node_name} with: {server_type} - {model_id} "
    )

    globals()["current_llm"].setdefault(scenario_id, {}).update({node_name: llm_config})

    logger.debug(f"Current LLM config used: {current_llm}")

    return llm_config


def get_embedding_model_by_embed_name(embed_name: str) -> Embeddings:
    """
    Instantiate a text embedding model based on the model name in the configuration

    Args:
        embed_name (str): name of the config in section text_embedding_models of the configuration file

    Returns:
        Embeddings: text embedding model
    """

    embed_config = get_embedding_model_config_by_name(embed_name)
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


def get_embedding_model_by_scenario(scenario_id: str) -> Embeddings:
    """
    Instantiate a text embedding model based on the scenario configuration

    Args:
        scenario_id (str): scenario ID

    Returns:
        Embeddings: text embedding model
    """
    embed_name = config[scenario_id]["text_embedding_model"]
    return get_embedding_model_by_embed_name(embed_name)


def create_vector_db(
    embeddings: Embeddings, vector_db_name: str, embeddings_directory: str
) -> VectorStore:
    """
    Create a vector db based on the configuration,
    and load the pre-computed embeddings from the given directory

    Args:
        embeddings (Embeddings): instantiated embeddings model
        vector_db_name (str): type of vector db (currently supported: "faiss", "chroma")
        embeddings_directory (str): directory containing the pre-computed embeddings

    Returns:
        VectorStore: vector db
    """

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


def create_vector_db_by_scenario(scenario_id: str, embeddings_dir_name) -> VectorStore:
    """
    Create a vector db based on the configuration,
    and load the pre-computed embeddings from the given directory

    Args:
        scenario_id (str): scenario ID
        vector_db_name (str): type of vector db (currently supported: "faiss", "chroma")
        embeddings_directory_name (str): name of the subdirectory that contains the pre-computed embeddings

    Returns:
        VectorStore: vector db
    """

    vector_db_name = get_embedding_model_config_by_scenario(scenario_id)["vector_db"]

    embeddings_directory = (
        f"{get_embeddings_directory(vector_db_name)}/{embeddings_dir_name}"
    )
    logger.debug(f"Embeddings directory: {embeddings_directory}")

    embeddings = get_embedding_model_by_scenario(scenario_id)

    return create_vector_db(embeddings, vector_db_name, embeddings_directory)


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

    db = create_vector_db_by_scenario(scenario_id, config["class_embeddings_subdir"])
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

    db = create_vector_db_by_scenario(scenario_id, config["queries_embeddings_subdir"])
    logger.info("SPARQL queries vector DB initialized.")
    globals()["queries_vector_db"][scenario_id] = db
    return db


def get_scenario_module(scenario_id: int):
    scenario_module = importlib.import_module(
        f"app.core.scenarios.scenario_{scenario_id}.scenario_{scenario_id}"
    )
    return scenario_module


def set_custom_scenario_configuration(
    scenario_id: int,
    validate_question_model: str,
    ask_question_model: str,
    generate_query_model: str,
    interpret_csv_query_results_model: str,
    text_embedding_model: str,
):
    """
    Set a custom configuration to use in a scenario
    """

    config[f"scenario_{scenario_id}"]["validate_question"] = validate_question_model

    if ask_question_model is not None:
        config[f"scenario_{scenario_id}"]["ask_question"] = ask_question_model

    if generate_query_model is not None:
        config[f"scenario_{scenario_id}"]["generate_query"] = generate_query_model

    if interpret_csv_query_results_model is not None:
        config[f"scenario_{scenario_id}"][
            "interpret_csv_query_results"
        ] = interpret_csv_query_results_model

    if f"scenario_{scenario_id}" in globals()["current_llm"].keys():
        del globals()["current_llm"][f"scenario_{scenario_id}"]

    if text_embedding_model is not None:
        config[f"scenario_{scenario_id}"]["text_embedding_model"] = text_embedding_model

        if f"scenario_{scenario_id}" in globals()["classes_vector_db"].keys():
            del globals()["classes_vector_db"][f"scenario_{scenario_id}"]

        if f"scenario_{scenario_id}" in globals()["queries_vector_db"].keys():
            globals()["queries_vector_db"][f"scenario_{scenario_id}"]

    logger.info(f"Custom configuration set for scenario_{scenario_id}")
    logger.debug(
        f"The custom configuration for scenario_{scenario_id} is : {config[f"scenario_{scenario_id}"]}"
    )


async def main(graph: CompiledStateGraph):
    """
    Entry point when invoked from the CLI

    Args:
        graph (CompiledStateGraph): Langraph compiled state graph
    """

    # Parse the command line arguments
    args = setup_cli()

    # Load the configuration file and assign to global variable 'config'
    read_configuration(args)

    question = args.question
    logger.info(f"Users' question: {question}")
    state = await graph.ainvoke(input=InputState({"initial_question": question}))

    # logger.info("==============================================================")
    # for m in state["messages"]:
    #     logger.info(m.pretty_repr())
    # if "last_generated_query" in state:
    #     logger.info("==============================================================")
    #     logger.info("last_generated_query: " + state["last_generated_query"])
    # logger.info("==============================================================")
