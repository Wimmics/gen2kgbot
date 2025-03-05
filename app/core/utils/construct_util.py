from pathlib import Path
import os
import re
from typing import List, Tuple
from SPARQLWrapper import JSON, TURTLE, SPARQLWrapper, POST
from rdflib import Graph, URIRef, BNode, RDFS, term
import app.core.utils.config_manager as config
from app.core.utils.logger_manager import setup_logger


logger = setup_logger(__package__, __file__)

# SPARQL query to retrieve the properties used by instances of a given class, with their value types.
# The value type can be a class URI, a datatype URI, or "Untyped" if the value is unknown/unspecified
class_properties_valuetypes_query = """
SELECT DISTINCT ?property (SAMPLE(COALESCE(STR(?type), STR(DATATYPE(?value)), "Untyped")) AS ?valueType) WHERE {
    {
        SELECT ?instance WHERE {
            ?instance a <{class_uri}> .
        } LIMIT 1000 # the limit assumes that this sample is representative of the class instances
    }
    {
        ?instance ?property ?value .
        OPTIONAL { ?value a ?type . }
    }
}
GROUP BY ?property
LIMIT 100
"""

# SPARQL query to retrieve the label/description of properties used by the instances of a given class
class_properties_query = """
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

# SPARQL query to retrieve the label/description of classes "connected" to a given class.
# Connected meaning: for class A, retrieve all classes B such that:
#   ?a rdf:type A. ?b rdf:type B.
# and either
#    ?a ?p ?b.
# or
#    ?b ?p ?a.
connected_classes_query = """
SELECT DISTINCT ?class ?label (COALESCE(?comment, "None") as ?description) WHERE {
  { 
  	SELECT DISTINCT ?class WHERE {
      ?seed a <{class_uri}>.
      { ?seed ?p ?other. } UNION { ?other ?p ?seed. }
	  ?other a ?class.
      FILTER (?class != owl:Class && ?class != rdfs:Class)
    }
  }
  
  ?class rdfs:label ?label.
  OPTIONAL { ?class rdfs:comment ?comment. }
}
LIMIT 50
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
        # -- commented out: label and description are already in the classes retreived from the vector db
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


def get_connected_classes(class_uris: list[str]) -> list[tuple]:
    """
    Retrieve the classes connected to a list of "seed" classes, with their labels and descriptions.
    The seed classes are those initially found because they are similar to the user's question.
    The connected classes are those whose instances are connected to instances of the seed classes by at least one predicate.

    This is used to expand the list of classes that can be relevant for generating the SPARQL query.

    Args:
        class_uris (list[str]): list of seed class URIs

    Returns:
        list[tuple]: list of tuples (class URI, label, description) gathering all the connected classes
            for all the seed classes, after removing duplicates.
    """

    endpoint_url = config.get_kg_sparql_endpoint_url()
    results = []

    for class_uri in class_uris:
        logger.debug(f"Retrieving classes connected to class {class_uri}")

        dest_file = generate_context_filename(class_uri) + "_conntected_classes"
        if os.path.exists(dest_file):
            logger.debug(f"Connected classes found in cache: {dest_file}.")
            f = open(dest_file, "r")
            for line in f.readlines():
                results.append(eval(line))
            f.close()

        else:
            logger.debug(f"Connected classes not found in cache for class {class_uri}.")
            results_one_class = []
            query = connected_classes_query.replace("{class_uri}", class_uri)
            for result in run_sparql_query(query, endpoint_url):
                if (
                    "class" in result.keys()
                    and "label" in result.keys()
                    and "description" in result.keys()
                ):
                    descr = result["description"]["value"]
                    results_one_class.append(
                        (
                            result["class"]["value"],
                            result["label"]["value"],
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
                    descr = None if description == "None" else description
                    f.write(f"('{cls}', '{label}', {descr})\n")
                f.close()
                logger.debug(f"Saved connected classes into cache: {dest_file}.")

            # Add the results for that class to the results for all classes
            results += results_one_class

    # Remove duplicates
    results = list(set(results))
    logger.info(f"Retrieved the descriptions of {len(results)} connected classes")

    return results


def get_class_properties(class_label_description: tuple) -> str:
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

    query = class_properties_valuetypes_query.replace("{class_uri}", cls)
    # logger.debug(f"SPARQL query to retrieve class properties and types:\n{query}")

    values = []
    for result in run_sparql_query(query, endpoint_url):
        if "property" in result.keys() and "valueType" in result.keys():
            values.append((result["property"]["value"], result["valueType"]["value"]))
        else:
            logger.warning(
                f"Unexpected SPARQL result format for properties/value_types of class {cls}: {result}"
            )

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

    query = class_properties_query.replace("{class_uri}", cls)
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
            logger.warning(
                f"Unexpected SPARQL result format for description of properties of class {cls}:\n{result}"
            )

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
    Transform a full IRI into its equivalent prefixed name
    """
    for prefix, namespace in config.get_known_prefixes().items():
        uri = uri.replace(namespace, f"{prefix}:")
    return uri
