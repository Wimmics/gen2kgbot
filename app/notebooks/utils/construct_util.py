import logging
import os
from pathlib import Path
from typing import List, Tuple
from SPARQLWrapper import JSON, TURTLE, SPARQLWrapper
from rdflib import Graph, Namespace, URIRef, BNode, RDFS, term

def setup_logger(name):
    """
    Set up logging configuration.

    Parameters:
    - name (str): Typically __name__ from the calling module.

    Returns:
    - logger (logging.Logger): Configured logger object.
    """
    # Resolve the path to the configuration file
    parent_dir = Path(__file__).resolve().parent.parent.parent
    config_path = parent_dir / "config" / "logging.ini"

    # Configure logging
    logging.config.fileConfig(config_path, disable_existing_loggers=False)

    # Get and return the logger
    return logging.getLogger(name)


logger = setup_logger(__name__)

query_cls_rel = """
SELECT ?property (SAMPLE(COALESCE(?type, STR(DATATYPE(?value)), "Untyped")) AS ?valueType) WHERE {{
        {{
        SELECT ?instance WHERE {{
            ?instance a <{class_uri}> .
        }} LIMIT 100
        }}
        {
          {?instance ?property ?value .}
        }
        OPTIONAL {{
        ?value a ?type .
        }}
    }}
    GROUP BY ?property ?type
    LIMIT 300
"""

classes_directory = Path(__file__).resolve().parent.parent.parent / "data" / "classes_context" / "idsm"
tmp_directory = Path(__file__).resolve().parent.parent.parent / "tmp" 
endpoint_url_corese = 'http://localhost:8080/sparql'
endpoint_url_idsm = 'https://idsm.elixir-czech.cz/sparql/endpoint/idsm'



def get_context_class(cl:str) ->Graph:
    graph = get_graph_with_prefixes()

    class_ref = URIRef(cl[0])
    properties_and_values = get_prop_and_val_types(cl[0],)
    # print(properties_and_values)
    if (cl[1]): graph.add((class_ref, RDFS.label, term.Literal(cl[1])))
    if (cl[2]): graph.add((class_ref, RDFS.comment, term.Literal(cl[2])))
    for property_uri, prop_type in properties_and_values:
        value_ref = (
            BNode() if (prop_type == "Untyped" or prop_type == None) else URIRef(prop_type)
        )
        graph.add((class_ref, URIRef(property_uri), value_ref))
    
    # save the graph
    class_file_path = format_class_graph_file(cl[0])

    return graph.serialize(destination=class_file_path)


def get_prop_and_val_types(cls: str,endpoint_url:str=endpoint_url_idsm) -> List[Tuple[str, str]]:
    query = query_cls_rel.replace("{class_uri}",cls)

    values = [(nested_value(x, ['property','value']),(nested_value(x, ['valueType','value']) )) for x in run_sparql(query,endpoint_url)]

    return [] if values == [(None,None)] else values


def run_sparql(query, url=endpoint_url_idsm):
    sparql = SPARQLWrapper(url)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    sparql.setTimeout(600)
    results = sparql.query().convert()
    results = nested_value(results, ['results', 'bindings'])
    return results

def run_sparql_construct(query, filename, url=endpoint_url_idsm):
    sparql = SPARQLWrapper(url)
    sparql.setQuery(query)
    sparql.setReturnFormat(TURTLE)
    sparql.setTimeout(600)
    results = sparql.queryAndConvert()
    graph = Graph()
    graph.parse(data=results, format='turtle')
    graph.serialize(destination=filename, format='turtle')
    return results
    
def format_class_graph_file(class_uri:str) -> str: 
    class_name = class_uri.split('/')[-1]
    return f"{classes_directory}/{class_name}.ttl"


def get_known_prefixes() -> dict:
    return {'http://schema.org/':'schema',
                'https://enpkg.commons-lab.org/module/':'enpkg_module',
                'http://purl.org/pav/':'pav',
                'http://example.org/':'example',
                'https://enpkg.commons-lab.org/kg/':'enpkg',
                'http://purl.obolibrary.org/obo/':'obo',
                'http://purl.org/spar/cito/':'cito',
                'http://rdf.ncbi.nlm.nih.gov/pubchem/vocabulary#':'pubchem',
                'http://semanticscience.org/resource/':'sio',
                'http://www.bioassayontology.org/bao#':'bao',
                'http://purl.obolibrary.org/obo/CHEBI_':'chebi',
                'http://semanticscience.org/resource/CHEMINF_':'cheminf',
                'http://rdf.ebi.ac.uk/terms/chembl#':'chembl'
                }

def add_known_prefixes_to_query(query:str) -> str :
    prefixes = get_known_prefixes()

    final_query = ""
    for k,v in prefixes.items():
        final_query += f"prefix {v}: <{k}>\n" 

    final_query += query

    return final_query

def get_graph_with_prefixes() -> Graph:
    prefix_map = get_known_prefixes()

    g = Graph()

    # Update prefix definitions
    for namespace, prefix in prefix_map.items():
        g.bind(prefix, namespace,override=True)
    
    return g


def get_class_context_found(cls,cls_path)->Graph:
    if os.path.exists(cls_path):
        logger.info(f"Classe context file path at {cls_path} found.")
        g = Graph()
        return g.parse(cls_path, format='turtle')
    else:
        logger.info(f"Classe context file path at {cls_path} not found.")
        return get_context_class(cls)

def get_class_context_not_found(cls,cls_path)->Graph:
    if os.path.exists(cls_path):
        logger.info(f"Classe context file path at {cls_path} found.")
        g = Graph()
        return g.parse(cls_path, format='turtle')
    else:
        logger.info(f"Classe context file path at {cls_path} not found.")
        return get_context_class(cls)


def get_context_if_not_found(cls,cls_path)->Graph:
    if os.path.exists(cls_path):
        logger.info(f"Classe context file path at {cls_path} found.")
        g = Graph()
        return g.parse(cls_path, format='turtle')
    else:
        logger.info(f"Classe context file path at {cls_path} not found.")
        return get_context_class(cls)
    


def nested_value(data: dict, path: list):
    current = data
    for key in path:
        try:
            current = current[key]
        except:
            return None
    return current