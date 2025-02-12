import os
from pathlib import Path
from typing import List, Tuple
from SPARQLWrapper import JSON, TURTLE, SPARQLWrapper
from rdflib import Graph, URIRef, BNode, RDFS, term
from app.core.utils.config_manager import (
    get_class_context_directory,
    get_kg_sparql_endpoint_url,
    setup_logger,
    get_known_prefixes,
)


logger = setup_logger(__package__, __file__)

query_cls_rel = """
SELECT ?property (SAMPLE(COALESCE(?type, STR(DATATYPE(?value)), "Untyped")) AS ?valueType) WHERE {
    {
        SELECT ?instance WHERE {
            ?instance a <{class_uri}> .
        } LIMIT 100
    }
    { ?instance ?property ?value . }
    OPTIONAL { ?value a ?type . }
}
GROUP BY ?property ?type
LIMIT 300
"""

tmp_directory = Path(__file__).resolve().parent.parent.parent.parent / "tmp"


def get_class_context(class_label_comment: tuple) -> str:
    """
    Retrieve a class context from the knowledge graph and save it to the cache

    Args:
        class_label_comment (tuple): (class URI, label, description)

    Returns:
        str: Turtle serialization of the class context
    """

    class_uri = class_label_comment[0]
    class_ref = URIRef(class_uri)
    class_label = class_label_comment[1]
    class_comment = class_label_comment[2]

    graph = get_empty_graph_with_prefixes()
    endpoint_url = get_kg_sparql_endpoint_url()

    if class_label:
        graph.add((class_ref, RDFS.label, term.Literal(class_label)))
    if class_comment:
        graph.add((class_ref, RDFS.comment, term.Literal(class_comment)))

    properties_and_values = get_prop_and_val_types(class_uri, endpoint_url=endpoint_url)
    for property_uri, prop_type in properties_and_values:
        value_ref = (
            BNode()
            if (prop_type == "Untyped" or prop_type is None)
            else URIRef(prop_type)
        )
        graph.add((class_ref, URIRef(property_uri), value_ref))

    # Save the graph to the cache
    graph.serialize(
        format="turtle", destination=generate_class_context_filename(class_uri)
    )

    return graph.serialize(format="turtle")


def get_prop_and_val_types(cls: str, endpoint_url: str) -> List[Tuple[str, str]]:
    """
    Retrieve what properties are used with instances of a given class, and the types of their values.

    Args:
        cls (str): class URI
        endpoint_url (str): SPARQL endpoint URL

    Returns:
        List[Tuple[str, str]]: property URIs with associated value types.
            Types may be a URI, datatype, or "Untyped"
    """

    query = query_cls_rel.replace("{class_uri}", cls)
    logger.debug(f"SPARQL query to retrieve class properties and types:\n{query}")

    values = [
        (
            nested_value(x, ["property", "value"]),
            (nested_value(x, ["valueType", "value"])),
        )
        for x in run_sparql(query, endpoint_url)
    ]
    logger.debug(f"Retrieved {len(values)} (property,type) couples for class {cls}")

    return [] if values == [(None, None)] else values


def run_sparql(query, endpoint_url):
    sparql = SPARQLWrapper(endpoint_url)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    sparql.setTimeout(600)
    results = sparql.query().convert()
    results = nested_value(results, ["results", "bindings"])
    return results


def run_sparql_construct(query, filename, endpoint_url):
    sparql = SPARQLWrapper(endpoint_url)
    sparql.setQuery(query)
    sparql.setReturnFormat(TURTLE)
    sparql.setTimeout(600)
    results = sparql.queryAndConvert()
    graph = Graph()
    graph.parse(data=results, format="turtle")
    graph.serialize(destination=filename, format="turtle")
    return results


def generate_class_context_filename(class_uri: str) -> str:
    """
    Generate a file name for the description of a class
    """

    class_name = class_uri.split("/")[-1]

    context_directory = Path(get_class_context_directory())
    if not os.path.exists(context_directory):
        os.makedirs(context_directory)

    return f"{context_directory}/{class_name}.ttl"


def add_known_prefixes_to_query(query: str) -> str:

    prefixes = get_known_prefixes()
    final_query = ""
    for prefix, namespace in prefixes.items():
        final_query += f"prefix {prefix}: <{namespace}>\n"

    final_query += query

    return final_query


def get_empty_graph_with_prefixes() -> Graph:
    """
    Creates an empty RDF graph with predefined prefixes.
    """
    g = Graph()

    prefix_map = get_known_prefixes()
    for prefix, namespace in prefix_map.items():
        g.bind(prefix, namespace, override=True)
    return g


def nested_value(data: dict, path: list):
    current = data
    for key in path:
        try:
            current = current[key]
        except Exception:
            return None
    return current
