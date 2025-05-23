from typing import List
from rdflib import Graph
from rdflib.exceptions import ParserError
from rdflib.plugins.stores import sparqlstore
import re
from app.utils.logger_manager import setup_logger
import app.utils.config_manager as config


logger = setup_logger(__package__, __file__)


def run_sparql_query(query: str, endpoint_url: str = None) -> str:
    """
    Submit a SPARQL query to the endpoint and return the result in CSV SPARQL Results format.

    Args:
        query (str): SPARQL query to be executed

    Returns:
        str: CSV SPARQL Results (https://www.w3.org/2009/sparql/docs/csv-tsv-results/results-csv-tsv.html)

    Raises:
        ValueError: non parsable SPARQL query or any other error
    """

    if endpoint_url is None:
        endpoint_url = config.get_kg_sparql_endpoint_url()
    try:
        # logger.debug(f"Submiting SPARQL query:\n{query}")
        logger.debug(f"Submiting to SPARQL endpoint: {endpoint_url}")

        _store = sparqlstore.SPARQLStore()
        _store.open(endpoint_url)
        graph = Graph(store=_store)

        res = graph.query(query_object=query, initNs={}, initBindings={})

    except ParserError as e:
        raise ValueError("SPARQL query is invalid\n" f"{e}")

    except Exception as e:
        raise ValueError(f"An error occurred while executing the SPARQL query: {e}")

    csv_str = res.serialize(format="csv").decode("utf-8")
    return csv_str


def find_sparql_queries(message: str) -> List[str]:
    """
    Extract, from the LLM's response, SPARQL queries embedded in a sparql markdown block.
    """
    return re.findall("```sparql(.*)```", message, re.DOTALL)


def find_json(message: str) -> List[str]:
    """
    Extract, from the LLM's response, JSON embedded in a json markdown block.
    """
    return re.findall("```json(.*)```", message, re.DOTALL)
