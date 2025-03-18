from pathlib import Path
import os
import re
from typing import List, Tuple
from SPARQLWrapper import JSON, TURTLE, SPARQLWrapper, POST
from rdflib import Graph, URIRef, BNode, RDF, RDFS, OWL, term
import app.core.utils.config_manager as config
from app.core.utils.logger_manager import setup_logger


logger = setup_logger(__package__, __file__)

# SPARQL query to retrieve the properties used by instances of a given class, with their value types.
# The value type can be a class URI, a datatype URI, or "None" if the value is unknown/unspecified
class_properties_valuetypes_query = (
    config.get_prefixes_as_sparql()
    + """
SELECT DISTINCT 
    ?property
    (COALESCE(?lbl, "None") as ?label)
    (SAMPLE(COALESCE(STR(?type), STR(DATATYPE(?value)), "None")) AS ?valueType)
WHERE {
    {
        SELECT ?instance WHERE {
            ?instance a {class_uri} .
        } LIMIT 100
    }
    {
        ?instance ?property ?value .
        OPTIONAL { ?property rdfs:label ?lbl . }
        OPTIONAL { ?value a ?type . }
    }
}
GROUP BY ?property ?lbl
LIMIT 100
"""
)

# SPARQL query to retrieve the label/description of classes "connected" to a given class.
# Connected meaning: for class A, retrieve all classes B such that:
#   ?a rdf:type A. ?b rdf:type B.
# and either
#    ?a ?p ?b.
# or
#    ?b ?p ?a.
connected_classes_query = (
    config.get_prefixes_as_sparql()
    + """
SELECT DISTINCT ?class (COALESCE(?lbl, "None") as ?label) (COALESCE(?comment, "None") as ?description) WHERE {
  { 
  	SELECT DISTINCT ?class WHERE {
      ?seed a {class_uri}.
      { ?seed ?p ?other. } UNION { ?other ?p ?seed. }
	  ?other a ?class.
      FILTER (?class != owl:Class && ?class != rdfs:Class)
    } LIMIT 1000
  }
    
  OPTIONAL { ?class rdfs:label ?lbl. }
  OPTIONAL { ?class rdfs:comment ?comment. }
}
LIMIT 20
"""
)


def get_class_context(class_label_description: tuple) -> str:
    """
    Retrieve a class context from the KG, format it according to parameter class_context_format, and save it to the cache.
    The context contains the properties used by instances of the class and their value types:
    In Turtle:
        `[] a <class URI>; <property> []. <property>  rdfs:label "property label".` # if value type is None
        `[] a <class URI>; <property> [ a <value type> ]. <property>  rdfs:label "property label".`
    or as tuples:
        `('class URI', 'property', 'property label', 'value type')`,
    where `value type` maybe be a class URI or a datatype URI.

    Args:
        class_label_description (tuple): (class URI, label, description)

    Returns:
        str: serialization of the class context formatted according to parameter class_context_format
    """

    class_uri = class_label_description[0]
    endpoint_url = config.get_kg_sparql_endpoint_url()
    properties_results = get_class_properties_and_val_types(class_uri, endpoint_url)

    dest_file = generate_context_filename(class_uri)
    format = config.get_class_context_format()

    if format == "turtle":
        # Create a graph containing the class context
        graph = get_empty_graph_with_prefixes()

        subj = BNode()
        # URIRef takes a full IRI, so we need to convert a possible prefixed IRI to a full IRI
        class_ref = URIRef(prefixed_to_fulliri(class_uri))
        # First triple: [] rdf:type [ a <CLS> ]
        graph.add((subj, RDF.type, class_ref))

        for property_uri, property_label, property_type in properties_results:
            # do not add the triple: "[] rdf:type [ a owl:Class ]"
            if (
                URIRef(property_uri) != RDF.type
                and property_type != str(OWL.Class)
            ):
                obj = BNode()
                graph.add((subj, URIRef(property_uri), obj))
                if property_type != None:
                    graph.add((obj, RDF.type, URIRef(property_type)))

                # Add a separate triple for the property label
                if property_label != None:
                    graph.add(
                        (URIRef(property_uri), RDFS.label, term.Literal(property_label))
                    )

        # Save the graph to the cache
        graph.serialize(format="turtle", destination=dest_file, encoding="utf-8")
        logger.debug(f"Class context stored in: {dest_file}.")
        return graph.serialize(format="turtle")

    elif format == "tuple" or format == "nl":
        # Format "nl" applies to serialization in prompts, but it is still stored as "tuple"
        result = ""
        for property_uri, property_label, property_type in properties_results:
            result += f"{(class_uri, property_uri, property_label, property_type)}\n"
        with open(dest_file, "w", encoding="utf-8") as f:
            f.write(result)
        logger.debug(f"Class context stored in: {dest_file}.")
        return result

    else:
        raise ValueError(f"Invalid requested format for class context: {format}")


def get_class_properties_and_val_types(
    class_uri: str, endpoint_url: str
) -> List[Tuple[str, str, str]]:
    """
    Retrieve what properties are used with instances of a given class, and the types of their values.

    Args:
        class_uri (str): class URI
        endpoint_url (str): SPARQL endpoint URL

    Returns:
        List[Tuple[str, str, str]]: property URIs with associated label and value type.
            Value type may be a URI, a datatype, or None. Label may be None.
    """

    if isPrefixed(class_uri):
        query = class_properties_valuetypes_query.replace("{class_uri}", class_uri)
    else:
        query = class_properties_valuetypes_query.replace(
            "{class_uri}", f"<{class_uri}>"
        )
    # logger.debug(f"SPARQL query to retrieve class properties and types:\n{query}")

    results = []
    _sparql_results = run_sparql_query(query, endpoint_url)
    if _sparql_results is None:
        logger.warning(
            f"Failed to retrieve the properties and value types for class {class_uri}"
        )
    else:
        for result in _sparql_results:
            if (
                "property" in result.keys()
                and "label" in result.keys()
                and "valueType" in result.keys()
            ):
                label = result["label"]["value"]
                valueType = result["valueType"]["value"]
                results.append(
                    (
                        result["property"]["value"],
                        (None if label == "None" else label),
                        (None if valueType == "None" else valueType),
                    )
                )
            else:
                logger.warning(
                    f"Unexpected SPARQL result format for properties/value_types of class {class_uri}: {result}"
                )

        logger.debug(
            f"Retrieved {len(results)} (property,label,type) tuples for class {class_uri}"
        )
    return results


def isPrefixed(uri: str) -> bool:
    """
    Check if a URI is prefixed or full.
    """
    return not uri.startswith("http:") and not uri.startswith("https:")


def get_connected_classes(class_uris: list[str]) -> list[tuple]:
    """
    Retrieve, from the KG, the classes connected to a list of "seed" classes, with their labels and descriptions,

    The seed classes are those initially found because they are similar to the user's question.
    The connected classes are those whose instances are connected to instances of the seed classes by at least one predicate.

    This is used to expand the list of classes that can be relevant for generating the SPARQL query.

    Args:
        class_uris (list[str]): list of seed class URIs

    Returns:
        list[tuple]: list of tuples (class URI, label, description) gathering all the connected classes
            for all the seed classes, after removing duplicates.
            If label or description is "None" it is replaced by None.
    """

    endpoint_url = config.get_kg_sparql_endpoint_url()
    results = []

    for class_uri in class_uris:

        logger.debug(f"Retrieving classes connected to class {class_uri}")
        dest_file = generate_context_filename(class_uri) + "_conntected_classes"

        if os.path.exists(dest_file):
            logger.debug(f"Connected classes found in cache: {dest_file}.")
            f = open(dest_file, "r", encoding="utf8")
            for line in f.readlines():
                results.append(eval(line))
            f.close()

        else:
            logger.debug(f"Connected classes not found in cache for class {class_uri}.")
            results_one_class = []
            if isPrefixed(class_uri):
                query = connected_classes_query.replace("{class_uri}", class_uri)
            else:
                query = connected_classes_query.replace("{class_uri}", f"<{class_uri}>")

            _sparql_results = run_sparql_query(query, endpoint_url)
            if _sparql_results is None:
                logger.warning(
                    f"Failed to retrieve connected classes for class {class_uri}"
                )
                continue

            for result in _sparql_results:
                if (
                    "class" in result.keys()
                    and "label" in result.keys()
                    and "description" in result.keys()
                ):
                    label = result["label"]["value"]
                    descr = result["description"]["value"]
                    results_one_class.append(
                        (
                            result["class"]["value"],
                            (None if label == "None" else label),
                            (None if descr == "None" else descr),
                        )
                    )
                else:
                    logger.warning(
                        f"Unexpected SPARQL result format for classes connected to class {class_uri}:\n{result}"
                    )

            # Save the connected classes to cache
            with open(dest_file, "w", encoding="utf-8") as f:
                for cls, label, description in results_one_class:
                    f.write(f"{(cls, label, description)}\n")
                f.close()
                logger.debug(f"Saved connected classes into cache: {dest_file}.")

            # Add the results for that class to the results for all classes
            results += results_one_class

    # Remove duplicates
    results = list(set(results))
    logger.info(f"Retrieved the descriptions of {len(results)} connected classes")

    return results


def run_sparql_query(query, endpoint_url, timeout=600) -> list:
    """
    Execute a SPARQL query and return the list of bindings from
    the SPARQL Results JSON Format (https://www.w3.org/TR/sparql11-results-json/).

    Invocation uses the HTTP POST method.

    In case of failure, the function logs an error and returns None.

    Args:
        query (str): SPARQL query
        endpoint_url (str): SPARQL endpoint URL
        timeout (int): timeout in seconds, default is 600
    """
    sparql = SPARQLWrapper(endpoint_url)
    sparql.setMethod(POST)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    sparql.setTimeout(timeout)
    try:
        results = sparql.queryAndConvert()
        return results["results"]["bindings"]
    except Exception as e:
        logger.error(f"Error while executing SPARQL query: {e}")
        return None


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

    class_name = re.sub(r"[:/\\#]", "_", fulliri_to_prefixed(uri))
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
    Transform a string containing full IRIs into their equivalent prefixed names
    """
    for prefix, namespace in config.get_known_prefixes().items():
        uri = uri.replace(namespace, f"{prefix}:")
    return uri


def prefixed_to_fulliri(uri: str) -> str:
    """
    Transform a string containing prefixed IRIs into their equivalent full IRIs
    """
    for prefix, namespace in config.get_known_prefixes().items():
        uri = uri.replace(f"{prefix}:", namespace)
    return uri
