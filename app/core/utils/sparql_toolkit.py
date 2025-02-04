import csv
from typing import List
from rdflib import Graph
from rdflib.exceptions import ParserError
from rdflib.plugins.stores import sparqlstore
import re

endpoint_url_idsm = "https://idsm.elixir-czech.cz/sparql/endpoint/idsm"


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
        _store = sparqlstore.SPARQLStore()
        _store.open(endpoint_url_idsm)
        graph = Graph(store=_store)

        res = graph.query(query_object=query, initNs={}, initBindings={})

    except ParserError as e:
        raise ValueError("Generated SPARQL statement is invalid\n" f"{e}")

    except Exception as e:
        raise ValueError(f"An error occurred while querying the graph: {e}")

    csv_str = res.serialize(format="csv").decode("utf-8")
    return csv_str


def find_sparql_queries(message: str):
    return re.findall("```sparql(.*)```", message, re.DOTALL)
