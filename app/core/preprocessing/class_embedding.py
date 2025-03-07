import os
import pickle
import app.core.utils.config_manager as config
from app.core.utils.construct_util import run_sparql_query, fulliri_to_prefixed


logger = config.setup_logger(__package__, __file__)


get_classes_query = """
SELECT DISTINCT
    ?class 
    (COALESCE(?lbl, "None") as ?label)
    (COALESCE(?comment, "None") as ?description)
WHERE {
    ?class a owl:Class .
    OPTIONAL { ?class rdfs:label ?lbl }
    OPTIONAL { ?class rdfs:comment ?comment }
}
"""


get_classes_with_members_query = """
SELECT DISTINCT
    ?class 
    (COALESCE(?lbl, "None") as ?label)
    (COALESCE(?comment, "None") as ?description)
WHERE {
    ?class a owl:Class .
    OPTIONAL { ?class rdfs:label ?lbl }
    OPTIONAL { ?class rdfs:comment ?comment }

    ?s a ?class .           # Keep only the classes with at least one individual
}
"""

get_classes_with_members_fed_query = """
SELECT DISTINCT
    ?class 
    (COALESCE(?lbl, "None") as ?label)
    (COALESCE(?comment, "None") as ?description)
WHERE {
    ?class a owl:Class .
    OPTIONAL { ?class rdfs:label ?lbl }
    OPTIONAL { ?class rdfs:comment ?comment }
    
    SERVICE <{endpoint_url}> { ?s a ?class . }  # Keep only the classes with at least one individual
}
"""


def collect_classes(query: str, filename: str) -> None:
    """
    Collect all classes from the ontologies as tuples (class, label, description),
    and save them in a pickle file.

    Args:
        query (str): SPARQL query that must return 3 variables: class, label, description
        filename (str): pickle file where to save the class tuples
    """
    query = query.replace("{endpoint_url}", config.get_kg_sparql_endpoint_url())
    results = []
    _sparql_results = run_sparql_query(
        query, config.get_ontologies_sparql_endpoint_url(), timeout=3600
    )
    if _sparql_results is not None:
        for result in _sparql_results:
            if (
                "class" in result.keys()
                and "label" in result.keys()
                and "description" in result.keys()
            ):
                results.append(
                    (
                        fulliri_to_prefixed(result["class"]["value"]),
                        result["label"]["value"],
                        result["description"]["value"],
                    )
                )
            else:
                logger.warning(
                    f"Unexpected SPARQL result format for properties/value_types of class: {result}"
                )

    logger.info(f"Retrieved {len(results)} (class,label,description) tuples.")

    # Save the classes in a pickle for later use
    with open(f"{filename}", "wb") as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"Tuples saved to {filename}.")


if __name__ == "__main__":

    # Collect the description of the classes of interest and save them
    class_descr_pkl = os.path.join(config.get_classes_preprocessing_directory(), "classes.pkl")
    collect_classes(get_classes_query, class_descr_pkl)

