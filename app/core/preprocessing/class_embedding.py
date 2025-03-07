import os
import pickle
import app.core.utils.config_manager as config
from app.core.utils.construct_util import run_sparql_query, fulliri_to_prefixed


logger = config.setup_logger(__package__, __file__)


get_classes_query = """
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
          { ?class rdfs:commentX ?comment }
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
    
} GROUP BY ?class
"""


def collect_classes() -> None:
    """
    Collect all classes from the ontologies as tuples (class, label, description),
    and save them in a pickle file.

    Args:
        query (str): SPARQL query that must return 3 variables: class, label, description
        filename (str): pickle file where to save the class tuples
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
                logger.warning(
                    f"Unexpected SPARQL result format for properties/value_types of class: {result}"
                )
    return results


if __name__ == "__main__":

    # Collect the description of the classes of interest and save them
    results = collect_classes()
    logger.info(f"Retrieved {len(results)} (class,label,description) tuples.")
    logger.debug(f"Here are the first results:")
    for result in results[:10]:
        logger.debug(f"{result}")

    # Save the classes in a pickle for later use
    class_descr_pkl = os.path.join(
        config.get_classes_preprocessing_directory(), "classes.pkl"
    )
    with open(f"{class_descr_pkl}", "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

    # Save the classes in a text file
    class_descr_txt = os.path.join(
        config.get_classes_preprocessing_directory(), "classes.txt"
    )
    with open(f"{class_descr_txt}", "w", encoding="utf-8") as f:
        for result in results:
            f.write(f"{result}\n")
        f.close()

    logger.info(f"Tuples saved to {class_descr_txt} and {class_descr_pkl}.")
