import csv
from typing import List
from rdflib import Graph
from rdflib.exceptions import ParserError
from rdflib.plugins.stores import sparqlstore
import re
from app.core.utils.logger_manager import setup_logger
from app.core.utils.utils import get_kg_sparql_endpoint_url


logger = setup_logger(__package__, __file__)


def run_sparql_query(query: str) -> List[csv.DictReader]:
    """
    queries a graph using a SPARQL statement and returns the results as a list of
    dictionaries.

    Args:
        query (str): a string that represents a SPARQL query to be executed on the graph data.

    Returns:
        List[csv.DictReader]: a list of dictionaries containing the results of the query.
    """

    try:
        endpoint_url = get_kg_sparql_endpoint_url()

        logger.debug(f"trying to running the query: {query} in the endpoint: {endpoint_url}")

        _store = sparqlstore.SPARQLStore()
        _store.open(endpoint_url)
        graph = Graph(store=_store)

        res = graph.query(query_object=query, initNs={}, initBindings={})

    except ParserError as e:
        raise ValueError("Generated SPARQL statement is invalid\n" f"{e}")

    except Exception as e:
        raise ValueError(f"An error occurred while querying the graph: {e}")

    csv_str = res.serialize(format="csv").decode("utf-8")

    logger.debug(f"running the query yelded the following CSV:\n {csv_str}")

    return csv_str


def find_sparql_queries(message: str):
    return re.findall("```sparql(.*)```", message, re.DOTALL)
