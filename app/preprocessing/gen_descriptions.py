"""
This module generates a textual description of the classes from the ontologies in the form (class, label, description),
and of the properties .
These descriptions will be used later on to compute per-class text embeddings.

The ontology classes are retrieved from the KG SPARQL endpoint (param: kg_sparql_endpoint_url),
or from a dedicated SPARQL endpoint if defined (param: ontologies_sparql_endpoint_url).

Several files are generated in directory `{data_directory}/{KG short name}/preprocessing`:
- `classes_description.txt`: description of all the classes found in the ontologies
- `classes_with_instances.txt`: list of classes that have at least one instance in the KG
- `classes_with_instances_description.txt`: description of the classes that have at least one instance in the KG

If files `classes_description.txt` or `classes_with_instances.txt` already exist, they are simply reloaded.

Either `classes_description.txt` or `classes_with_instances_description.txt` shall be used to compute the embeddings of the classes.
"""

import os
import app.core.utils.config_manager as config
from app.core.utils.construct_util import run_sparql_query, fulliri_to_prefixed


logger = config.setup_logger(__package__, __file__)


get_classes_query = (
    config.get_prefixes_as_sparql()
    + """
SELECT DISTINCT
    ?class
    (group_concat(distinct ?lbl_str, "--") as ?label)
    (group_concat(distinct ?comment_str, "--") as ?description)
    
WHERE {
    ?class a owl:Class .
    FILTER(isIRI(?class))   # Ignore anonymous classes that are mostly owl constructs

    OPTIONAL {
    	{ SELECT DISTINCT ?class ?lbl WHERE {
          { ?class rdfs:label ?lbl }
          UNION 
          { ?class skos:prefLabel ?lbl }
          UNION 
          { ?class skos:altLabel ?lbl }
          UNION 
          { ?class schema:name ?lbl }
          UNION 
          { ?class schema:alternateName ?lbl }
          UNION 
          { ?class obo:IAO_0000118 ?lbl }       # alt label
          UNION 
          { ?class obo:OBI_9991118 ?lbl }       # IEDB alternative term
          UNION 
          { ?class obo:OBI_0001847 ?lbl }       # ISA alternative term
		}}
	}
	BIND(COALESCE(str(?lbl), "None") as ?lbl_str)

    OPTIONAL {
    	{ SELECT DISTINCT ?class ?comment WHERE {
          { ?class rdfs:comment ?comment }
          UNION 
          { ?class skos:definition ?comment }
          UNION 
          { ?class dc:description ?comment }
          UNION 
          { ?class dcterms:description ?comment }
          UNION 
          { ?class schema:description ?comment }
          UNION 
          { ?class obo:IAO_0000115 ?comment }   # definition
		}}
	}
 	BIND(COALESCE(str(?comment), "None") as ?comment_str)

    #FILTER (?comment_str != "None" && ?lbl_str != "None")   # removes lots of deprecated classes, but is it a good idea?

} GROUP BY ?class
"""
)

get_classes_with_instances_query = """
SELECT distinct ?class WHERE { ?s a ?class. } LIMIT 100
"""

get_properties_query = (
    config.get_prefixes_as_sparql()
    + """
SELECT DISTINCT
    ?prop
	?domain ?domain_lbl ?range ?range_lbl
	(group_concat(distinct ?lbl_str) as ?label)
    (group_concat(distinct ?comment_str) as ?description)
WHERE {
    {
        { ?prop a owl:ObjectProperty. }
        UNION
        { ?prop a owl:DataTypeProperty. }
        UNION
        { ?prop rdfs:subPropertyOf []. }
	}
    
  	OPTIONAL { 
        ?prop rdfs:domain ?domain. FILTER(isIRI(?domain))
        OPTIONAL { ?domain rdfs:label ?domain_lbl }
    }
    OPTIONAL { 
        ?prop rdfs:range ?range. FILTER(isIRI(?range)) 
        OPTIONAL { ?range rdfs:label ?range_lbl }
    }

    OPTIONAL {
    	{ SELECT DISTINCT ?prop ?lbl WHERE {
          { ?prop rdfs:label ?lbl }
          UNION 
          { ?prop skos:prefLabel ?lbl }
          UNION 
          { ?prop skos:altLabel ?lbl }
          UNION 
          { ?prop schema:name ?lbl }
          UNION 
          { ?prop schema:alternateName ?lbl }
          UNION 
          { ?prop obo:IAO_0000118 ?lbl }       # alt label
          UNION 
          { ?prop obo:OBI_9991118 ?lbl }       # IEDB alternative term
          UNION 
          { ?prop obo:OBI_0001847 ?lbl }       # ISA alternative term
		}}
	}
	BIND(COALESCE(str(?lbl), "None") as ?lbl_str)

    OPTIONAL {
    	{ SELECT DISTINCT ?prop ?comment WHERE {
          { ?prop rdfs:comment ?comment }
          UNION 
          { ?prop skos:definition ?comment }
          UNION 
          { ?prop dc:description ?comment }
          UNION 
          { ?prop dcterms:description ?comment }
          UNION 
          { ?prop schema:description ?comment }
          UNION 
          { ?prop obo:IAO_0000115 ?comment }   # definition
		}}
	}
 	BIND(COALESCE(str(?comment), "None") as ?comment_str)

    #FILTER (?comment_str != "None" && ?lbl_str != "None")   # removes lots of deprecated classes, but is it a good idea?
    
} GROUP BY ?prop
"""
)


def save_to_txt(filename: str, data: list):
    """
    Utilitary function to save a list to a text file
    """
    with open(filename, "w", encoding="utf-8") as f:
        for result in data:
            f.write(f"{result}\n")
        f.close()


def make_classes_description() -> list[tuple]:
    """
    Get a description of all the classes from the ontologies as tuples (class, label, description).
    The URIs are prefixed based on the prefixes defined in the config file.

    Data is either read from an existing file or from the SPARQL endpoint and saved in a file.

    The query is done against the SPARQL endpoint that contains the ontologies
    (ontologies_sparql_endpoint_url), which may be the same as the one that hosts
    the KG itself (kg_sparql_endpoint_url), or not.

    Returns:
        list[tuple]: list of tuples (class, label, description)
    """
    results = []
    _sparql_results = run_sparql_query(
        get_classes_query, config.get_ontologies_sparql_endpoint_url(), timeout=3600
    )
    if _sparql_results is not None:
        for result in _sparql_results:
            if (
                "class" in result.keys()
                and "label" in result.keys()
                and "description" in result.keys()
            ):
                label = (
                    None
                    if result["label"]["value"] == "None"
                    else result["label"]["value"]
                )
                descr = (
                    None
                    if result["description"]["value"] == "None"
                    else result["description"]["value"]
                )
                results.append(
                    (
                        fulliri_to_prefixed(result["class"]["value"]),
                        label,
                        descr,
                    )
                )
            else:
                logger.warning(f"Unexpected SPARQL result format: {result}")
    return results


def make_properties_description() -> list[tuple]:
    """
    Get a description of all the properties from the ontologies as tuples (prop, label, description).
    The URIs are prefixed based on the prefixes defined in the config file.

    Data is either read from an existing file or from the SPARQL endpoint and saved in a file.

    The query is done against the SPARQL endpoint that contains the ontologies
    (ontologies_sparql_endpoint_url), which may be the same as the one that hosts
    the KG itself (kg_sparql_endpoint_url), or not.

    Returns:
        list[tuple]: list of tuples (prop, label, description)
    """
    results = []
    _sparql_results = run_sparql_query(
        get_properties_query, config.get_ontologies_sparql_endpoint_url(), timeout=3600
    )
    if _sparql_results is not None:
        for result in _sparql_results:
            if (
                "prop" in result.keys()
                and "label" in result.keys()
                and "description" in result.keys()
            ):

                domain_range = ""

                domain = ""
                if "domain_lbl" in result.keys():
                    domain += result["domain_lbl"]["value"]
                    if "domain" in result.keys():
                        domain += f" ({result["domain"]["value"]})"
                else:
                    if "domain" in result.keys():
                        domain += result["domain"]["value"]
                if domain.strip() != "":
                    domain_range += f"The subject of this property is a {fulliri_to_prefixed(domain.strip())}. "

                range = ""
                if "range_lbl" in result.keys():
                    range += result["range_lbl"]["value"]
                    if "range" in result.keys():
                        range += f" ({result["range"]["value"]})"
                else:
                    if "range" in result.keys():
                        range += result["range"]["value"]
                if range.strip() != "":
                    domain_range += f"The object of this property is a {fulliri_to_prefixed(range.strip())}."

                label = ""
                if result["label"]["value"] != "None":
                    label = result["label"]["value"] + ". "
                if domain_range != "":
                    label += domain_range
                if label.strip() == "":
                    label = None

                descr = (
                    None
                    if result["description"]["value"] == "None"
                    else result["description"]["value"]
                )

                results.append(
                    (
                        fulliri_to_prefixed(result["prop"]["value"]),
                        label,
                        descr,
                    )
                )
            else:
                logger.warning(f"Unexpected SPARQL result format: {result}")
    return results


def get_classes_with_instances() -> list[str]:
    """
    Retrieve the list of classes that have at least one instance in the KG.
    The class URIs are prefixed based on the prefixes defined in the config file.

    Data is either read from an existing file or from the SPARQL endpoint and saved in a file.

    Returns:
        list[str]: list of the class URIS (with prefixed)
    """
    results = []
    _sparql_results = run_sparql_query(
        get_classes_with_instances_query,
        config.get_kg_sparql_endpoint_url(),
        timeout=3600,
    )
    if _sparql_results is not None:
        for result in _sparql_results:
            if "class" in result.keys():
                results.append(fulliri_to_prefixed(result["class"]["value"]))
            else:
                logger.warning(f"Unexpected SPARQL result format: {result}")
    return results


if __name__ == "__main__":

    # Collect the description of the properties and save them
    descr_txt_file = os.path.join(
        config.get_preprocessing_directory(), "properties_description.txt"
    )
    descriptions = []
    if os.path.exists(descr_txt_file):
        logger.info(f"Reading property descriptions from {descr_txt_file}")
        f = open(descr_txt_file, "r", encoding="utf8")
        descriptions = [eval(line) for line in f.readlines()]
        f.close()
    else:
        logger.info(f"Retrieving property descriptions from the SPARQL endpoint")
        descriptions = make_properties_description()
        save_to_txt(descr_txt_file, descriptions)

    logger.info(f"Retrieved {len(descriptions)} (property,label,description) tuples.")

    # Collect the description of the classes and save them
    descr_txt_file = os.path.join(
        config.get_preprocessing_directory(), "classes_description.txt"
    )
    descriptions = []
    if os.path.exists(descr_txt_file):
        logger.info(f"Reading class descriptions from {descr_txt_file}")
        f = open(descr_txt_file, "r", encoding="utf8")
        descriptions = [eval(line) for line in f.readlines()]
        f.close()
    else:
        logger.info(f"Retrieving class descriptions from the SPARQL endpoint")
        descriptions = make_classes_description()
        save_to_txt(descr_txt_file, descriptions)

    logger.info(f"Retrieved {len(descriptions)} (class,label,description) tuples.")

    # Retrieve the classes with at least 1 instance in the KG
    classes_with_instances_file = os.path.join(
        config.get_preprocessing_directory(), "classes_with_instances.txt"
    )
    classes_with_instances = []
    if os.path.exists(classes_with_instances_file):
        logger.info(
            f"Reading classes with instances from {classes_with_instances_file}."
        )
        f = open(classes_with_instances_file, "r", encoding="utf8")
        classes_with_instances = [line.strip() for line in f.readlines()]
        f.close()
    else:
        logger.info(f"Retrieving classes with instances from the SPARQL endpoint")
        classes_with_instances = get_classes_with_instances()
        save_to_txt(classes_with_instances_file, classes_with_instances)

        with open(classes_with_instances_file, "w", encoding="utf-8") as f:
            for result in classes_with_instances:
                f.write(f"{result}\n")
            f.close()
        logger.info(f"Saved classes with instances to {classes_with_instances_file}.")
    logger.info(f"Retrieved {len(classes_with_instances)} classes with instances.")

    # Filter classes_description to keep only the classes with instances
    classes_description_filtered = []
    for c in descriptions:
        if c[0] in classes_with_instances:
            classes_description_filtered.append(c)
        else:
            logger.debug(f"Ignoring empty class {c[0]}")

    logger.info(
        f"Keeping {len(classes_description_filtered)} classes after removing those with no instance."
    )
    classes_with_instances_description_file = os.path.join(
        config.get_preprocessing_directory(),
        "classes_with_instances_description.txt",
    )
    save_to_txt(classes_with_instances_description_file, classes_description_filtered)
