from typing import Literal
from pathlib import Path
from typing import List, Tuple
from SPARQLWrapper import JSON, TURTLE, SPARQLWrapper, POST
from rdflib import Graph, URIRef, BNode, RDFS, term
import app.core.utils.config_manager as config
from app.core.utils.logger_manager import setup_logger


logger = setup_logger(__package__, __file__)

query_cls_rel = """
SELECT DISTINCT ?property (SAMPLE(COALESCE(?type, STR(DATATYPE(?value)), "Untyped")) AS ?valueType) WHERE {
    {
        SELECT ?instance WHERE {
            ?instance a <{class_uri}> . 
        } LIMIT 100
    }
    {
        ?instance ?property ?value . 
        OPTIONAL { ?value a ?type . }
    }    
}
GROUP BY ?property ?type
LIMIT 300
"""

query_cls_props = """
SELECT DISTINCT ?property (COALESCE(?lbl, "None") as ?label) (COALESCE(?comment, "None") as ?description) WHERE {
    {
        SELECT ?instance WHERE {
            ?instance a <{class_uri}> . 
        } LIMIT 1000
    }
    {
        ?instance ?property ?value . 
        OPTIONAL { ?property rdfs:label ?lbl . }
        OPTIONAL { ?property rdfs:comment ?comment . }
    }    
}
LIMIT 100
"""


def get_class_context(class_label_description: tuple) -> str:
    """
    Retrieve a class context from the KG, format it according to parameter class_context_format, and save it to the cache.
    The context contains the properties used by instances of the class and their value types:
    In Turtle: `<class URI> <property> <property type>`, or as tuples: `('class URI', 'property', 'property type')`,
    where `property type` maybe be a class URI, a datatype URI, or a blank node if untyped.

    Args:
        class_label_description (tuple): (class URI, label, description)

    Returns:
        str: serialization of the class context formatted according to parameter class_context_format
    """

    class_uri = class_label_description[0]
    endpoint_url = config.get_kg_sparql_endpoint_url()
    properties_and_types = get_class_properties_and_val_types(class_uri, endpoint_url)

    dest_file = generate_context_filename(class_uri)
    format = config.get_class_context_format()

    if format == "turtle":
        graph = get_empty_graph_with_prefixes()
        class_ref = URIRef(class_uri)
        # class_label = class_label_description[1]
        # class_description = class_label_description[2]
        # if class_label:
        #     graph.add((class_ref, RDFS.label, term.Literal(class_label)))
        # if class_description:
        #     graph.add((class_ref, RDFS.comment, term.Literal(class_description)))
        for property_uri, property_type in properties_and_types:
            value_ref = (
                BNode()
                if (property_type == "Untyped" or property_type is None)
                else URIRef(property_type)
            )
            graph.add((class_ref, URIRef(property_uri), value_ref))

        # Save the graph to the cache
        graph.serialize(format="turtle", destination=dest_file, encoding="utf-8")
        logger.debug(f"Class context stored in: {dest_file}.")
        return graph.serialize(format="turtle")

    elif format == "tuple":
        result = ""
        for property_uri, property_type in properties_and_types:
            result += f"('{class_uri}', '{property_uri}', '{property_type}')\n"
        with open(dest_file, "w", encoding="utf-8") as f:
            f.write(result)
        logger.debug(f"Class context stored in: {dest_file}.")
        return result

    else:
        raise ValueError(f"Invalid requested format for class context: {format}")


def get_class_properties_context(class_label_description: tuple) -> str:
    """
    Retrieve from the KG the properties used by instances of the class with label and description,
    and save them as tuples to a file in cache.

    Args:
        class_label_description (tuple): (class URI, label, description)

    Returns:
        str: serialization of the properties as tuples formatted as ('prop URI', 'label', 'description')
    """

    class_uri = class_label_description[0]
    endpoint_url = config.get_kg_sparql_endpoint_url()
    properties_tuples = get_class_properties_description(class_uri, endpoint_url)

    result = ""
    for pro_uri, label, description in properties_tuples:
        result += f"('{pro_uri}', '{label}', '{description}')\n"

    dest_file = generate_context_filename(class_uri) + "_properties"
    with open(dest_file, "w", encoding="utf-8") as f:
        f.write(result)
    logger.debug(f"Class properties stored in: {dest_file}.")
    return result


def get_class_properties_and_val_types(
    cls: str, endpoint_url: str
) -> List[Tuple[str, str]]:
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
    # logger.debug(f"SPARQL query to retrieve class properties and types:\n{query}")

    values = []
    for result in run_sparql_query(query, endpoint_url):
        if "property" in result.keys() and "valueType" in result.keys():
            values.append((result["property"]["value"], result["valueType"]["value"]))
        else:
            logger.warning(f"Unexpected SPARQL result format: {result}")

    logger.debug(f"Retrieved {len(values)} (property,type) couples for class {cls}")
    return values


def get_class_properties_description(
    cls: str, endpoint_url: str
) -> List[Tuple[str, str, str]]:
    """
    Retrieve the label and description of properties used with instances of a given class

    Args:
        cls (str): class URI
        endpoint_url (str): SPARQL endpoint URL

    Returns:
        List[Tuple[str, str,str ]]: property URI, label, description
    """

    query = query_cls_props.replace("{class_uri}", cls)
    # logger.debug(f"SPARQL query to retrieve class properties and types:\n{query}")

    values = []
    for result in run_sparql_query(query, endpoint_url):
        if (
            "property" in result.keys()
            and "label" in result.keys()
            and "description" in result.keys()
        ):
            values.append(
                (
                    result["property"]["value"],
                    result["label"]["value"],
                    result["description"]["value"],
                )
            )
        else:
            logger.warning(f"Unexpected SPARQL result format: {result}")

    logger.debug(f"Retrieved {len(values)} property descriptions for class {cls}")
    return values


def run_sparql_query(query, endpoint_url) -> list:
    """
    Execute a SPARQL query and return the results the list of bindings from
    the SPARQL Results JSON Format (https://www.w3.org/TR/sparql11-results-json/).

    Invocation uses the HTTP POST method.

    In case of failure, the function logger an warning and returns [].
    """
    sparql = SPARQLWrapper(endpoint_url)
    sparql.setMethod(POST)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    sparql.setTimeout(600)
    try:
        results = sparql.queryAndConvert()
        return results["results"]["bindings"]
    except Exception as e:
        logger.warning(f"Error while executing SPARQL query: {e}")
        return []


def run_sparql_construct(query, filename, endpoint_url):
    sparql = SPARQLWrapper(endpoint_url)
    sparql.setMethod(POST)
    sparql.setQuery(query)
    sparql.setReturnFormat(TURTLE)
    sparql.setTimeout(600)
    results = sparql.queryAndConvert()
    graph = Graph()
    graph.parse(data=results, format="turtle")
    graph.serialize(destination=filename, format="turtle")
    return results


def generate_context_filename(uri: str) -> str:
    """
    Generate a file name for a resource given by its uri.
    Example:
        `http://example.org/Person -> ./data/idsm/classes_context/tuple/ex_Person`

    Args:
        uri (str): resource uri
        
    Return:
        str: file name using based on the uri replaced by its prefixed name 
            and the ":" replaced with a "_"
    """
    class_name = fulliri_to_prefixed(uri).replace(":", "_")
    context_directory = Path(config.get_class_context_cache_directory())
    return f"{context_directory}/{class_name}"


def add_known_prefixes_to_query(query: str) -> str:
    """
    Insert the prefix definitions (from the config file) before the SPARQL query
    """
    prefixes = config.get_known_prefixes()
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

    prefix_map = config.get_known_prefixes()
    for prefix, namespace in prefix_map.items():
        g.bind(prefix, namespace, override=True)
    return g


def fulliri_to_prefixed(uri: str) -> str:
    """
    Transform a full IRI into its equivalent prefixed name
    """
    for prefix, namespace in config.get_known_prefixes().items():
        uri = uri.replace(namespace, f"{prefix}:")
    return uri
