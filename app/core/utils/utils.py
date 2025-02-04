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
from app.core.utils.config_manager import get_yml_config
from app.core.utils.envkey_manager import (
    get_deepseek_key,
    get_google_key,
    get_openai_key,
    get_ovh_key,
)
from app.core.utils.logger_manager import setup_logger
from app.core.utils.printing import new_log


args = None
logger = setup_logger(__package__, __file__)
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

    logger.info(f"LLM initialized : {model_type} - {model_id} ")
    globals()["current_llm"] = llm
    globals()["current_scenario"] = scenario
    return llm


def get_embedding_type_from_config(scenario: str) -> Embeddings:

    embedding_type = config[scenario]["text_embedding_llm"]["type"]
    model_id = config[scenario]["text_embedding_llm"]["id"]

    if embedding_type == "ollama-embeddings":
        embeddings = OllamaEmbeddings(model=model_id)
        logger.info("Embedding initialized: OllamaEmbeddings")

    elif embedding_type == "openai-embeddings":
        embeddings = OpenAIEmbeddings(model=model_id)

        logger.info("Embedding initialized: OpenAiEmbeddings")

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


async def main(graph: CompiledStateGraph):
    """
    Process a predefined or custom question, invokes a graph with the question, and logs the messages returned by the graph.
    """

    if args is None and args.custom:
        question = args.custom
    else:
        question = "What protein targets does donepezil (CHEBI_53289) inhibit with an IC50 less than 5 µM?"

    state = await graph.ainvoke({"initial_question": question})

    new_log()
    for m in state["messages"]:
        m.pretty_print()
    new_log()

    if "last_generated_query" in state:
        new_log()
        print(state["last_generated_query"])
        new_log()
