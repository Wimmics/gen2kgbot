import json
from pathlib import Path
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
from fastapi_mcp import FastApiMCP
import yaml
from app.api.requests.activate_config import ActivateConfig
from app.api.requests.answer_question import AnswerQuestion
from app.api.requests.create_config import CreateConfig
from app.api.requests.generate_competency_question import GenerateCompetencyQuestion
from app.api.requests.refine_query import RefineQuery
from app.api.responses.kg_config import KGConfig
from app.api.responses.scenario_schema import ScenarioSchema
from app.api.services.answer_question import answer_question
from app.api.services.config_manager import (
    add_missing_config_params,
    save_query_examples_to_file,
    get_available_configurations,
)
from app.api.services.generate_competency_question import generate_competency_questions
from fastapi.middleware.cors import CORSMiddleware
from app.api.services.graph_mermaid import get_scenarios_schema
from app.api.services.refine_query import refine_query
from app.utils.config_manager import (
    get_configuration,
    read_configuration,
    set_custom_scenario_configuration,
    setup_cli,
)
from app.utils.logger_manager import setup_logger
from app.preprocessing.compute_embeddings import start_compute_embeddings
from app.preprocessing.gen_descriptions import generate_descriptions
import app.utils.config_manager as config
import uvicorn


# setup logger
logger = setup_logger(__package__, __file__)

# setup FastAPI
app = FastAPI(
    title="Q²Forge API",
    docs_url=None,
    openapi_url="/api/q2forge/openapi.json",
    redoc_url="/api/q2forge/docs",
    description=(
        "**Q²Forge** is a resource that addresses the challenge of generating new competency questions for a KG and corresponding SPARQL queries that reflect those questions. It iteratively validates those queries with human "
        + "feedback and LLM as a judge. Q²Forge is open source, extensible and modular, meaning that different parts of the system (CQ generation, query generation and query refinement) can be used as a whole or as parts depending on the context, "
        + "or replaced by alternative services. The end result is a complete pipeline from competency question formulation to query evaluation, supporting the creation of reference query sets for any target KG."
    ),
    version="1.0",
)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    path="/api/q2forge/scenarios_graph_schema",
    summary="Get the scenarios graph schemas",
    operation_id="get_scenarios_graph_schema",
    description=(
        "This endpoint returns the different scenarios graph schemas. "
        + "The schemas are represented in a mermaid format, which can be used to visualize the flow of the scenarios."
    ),
    responses={
        200: {
            "description": "A list containing the different scenarios schemas",
            "content": {
                "application/json": {
                    "example": [
                        {
                            "scenario_id": 1,
                            "schema": "```mermaid\n%%{init: {'flowchart': {'curve': 'linear'}}}%%\ngraph TD;\n\t__start__([<p>__start__</p>]):::first\n\tinit(init)\n\tvalidate_question(validate_question)\n\task_question(ask_question)\n\t__end__([<p>__end__</p>]):::last\n\t__start__ --> init;\n\task_question --> __end__;\n\tinit --> validate_question;\n\tvalidate_question -.-> ask_question;\n\tvalidate_question -.-> __end__;\n\tclassDef default fill:#f2f0ff,line-height:1.2\n\tclassDef first fill-opacity:0\n\tclassDef last fill:#bfb6fc\n\n```",
                        },
                        {
                            "scenario_id": 2,
                            "schema": "```mermaid\n ...```",
                        },
                    ]
                }
            },
        }
    },
)
def get_scenario_schema_endpoint() -> list[ScenarioSchema]:
    return get_scenarios_schema()


@app.get(
    path="/api/q2forge/config/available",
    summary="Get the Available Configurations",
    operation_id="get_available_configurations",
    description=(
        "This endpoint returns the short name of each available configuration that can be used by the Q²Forge resource."
    ),
    responses={
        200: {
            "description": "A list of the configuration short names.",
            "content": {"application/json": {"example": ["idsm", "d2kab"]}},
        }
    },
)
def get_available_configurations_endpoint() -> list[str]:
    return get_available_configurations()


@app.get(
    path="/api/q2forge/config/active",
    operation_id="get_active_config",
    summary="Get the currently active configuration",
    description=(
        "This endpoint returns the currently active configuration of the Q²Forge API. "
        + "The configuration contains information about the knowledge graph, "
        + "the SPARQL endpoint, the properties to be used, the prefixes, and the models to be used for each scenario."
    ),
    responses={
        200: {
            "description": "The currently active configuration",
            "content": {
                "application/json": {
                    "example": {
                        "kg_full_name": "PubChem knowledge graph",
                        "kg_short_name": "idsm",
                        "kg_description": "The IDSM SPARQL endpoint provides fast similarity and structural search functionality in knowledge graph such as ChEMBL, ChEBI or PubChem.",
                        "kg_sparql_endpoint_url": "https://idsm.elixir-czech.cz/sparql/endpoint/idsm",
                        "ontologies_sparql_endpoint_url": "http://gen2kgbot.i3s.unice.fr/corese",
                        "properties_qnames_info": [
                            "rdfs:label",
                            "skos:prefLabel",
                            "skos:altLabel",
                            "schema:name",
                            "schema:alternateName",
                            "obo:IAO_0000118",
                            "obo:OBI_9991118",
                            "obo:OBI_0001847",
                            "rdfs:comment",
                            "skos:definition",
                            "dc:description",
                            "dcterms:description",
                            "schema:description",
                            "obo:IAO_0000115",
                        ],
                        "prefixes": {
                            "bao": "http://www.bioassayontology.org/bao#",
                            "biopax": "http://www.biopax.org/release/biopax-level3.owl#",
                            "cito": "http://purl.org/spar/cito/",
                            "chembl": "http://rdf.ebi.ac.uk/terms/chembl#",
                            "dc": "http://purl.org/dc/elements/1.1/",
                            "dcterms": "http://purl.org/dc/terms/",
                            "enpkg": "https://enpkg.commons-lab.org/kg/",
                            "enpkg_module": "https://enpkg.commons-lab.org/module/",
                            "fabio": "http://purl.org/spar/fabio/",
                            "foaf": "http://xmlns.com/foaf/0.1/",
                            "frbr": "http://purl.org/vocab/frbr/core#",
                            "ndfrt": "http://purl.bioontology.org/ontology/NDFRT/",
                            "obo": "http://purl.obolibrary.org/obo/",
                            "owl": "http://www.w3.org/2002/07/owl#",
                            "patent": "http://data.epo.org/linked-data/def/patent/",
                            "pav": "http://purl.org/pav/",
                            "pubchem": "http://rdf.ncbi.nlm.nih.gov/pubchem/vocabulary#",
                            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
                            "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
                            "schema": "http://schema.org/",
                            "skos": "http://www.w3.org/2004/02/skos/core#",
                            "sio": "http://semanticscience.org/resource/",
                            "snomedct": "http://purl.bioontology.org/ontology/SNOMEDCT/",
                            "xsd": "http://www.w3.org/2001/XMLSchema#",
                            "up": "http://purl.uniprot.org/core/",
                        },
                        "excluded_classes_namespaces": [
                            "http://data.epo.org/linked-data/def/patent/Publication",
                            "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#",
                        ],
                        "seq2seq_models": {
                            "llama3_2-1b@local": {
                                "server_type": "ollama",
                                "id": "llama3.2:1b",
                                "temperature": 0,
                                "max_retries": 3,
                                "top_p": 0.95,
                            },
                            "llama3_2:3b@local": {
                                "server_type": "ollama",
                                "id": "llama3.2:latest",
                                "temperature": 0,
                                "max_retries": 3,
                                "top_p": 0.95,
                            },
                            "gemma-3_1b@local": {
                                "server_type": "ollama",
                                "id": "gemma3:1b",
                                "temperature": 0,
                                "max_retries": 3,
                                "top_p": 0.95,
                            },
                            "gemma-3_4b@local": {
                                "server_type": "ollama",
                                "id": "gemma3:4b",
                                "temperature": 0,
                                "max_retries": 3,
                                "top_p": 0.95,
                            },
                            "gemma-3_12b@local": {
                                "server_type": "ollama",
                                "id": "gemma3:12b",
                                "temperature": 0,
                                "max_retries": 3,
                                "top_p": 0.95,
                            },
                            "gemma-3_27b@local": {
                                "server_type": "ollama",
                                "id": "gemma3:27b",
                                "temperature": 0,
                                "max_retries": 3,
                                "top_p": 0.95,
                            },
                            "deepseek-r1_1_5b@local": {
                                "server_type": "ollama",
                                "id": "deepseek-r1:1.5b",
                                "temperature": 0,
                                "max_retries": 3,
                                "top_p": 0.95,
                            },
                            "llama-3_1-70B@ovh": {
                                "server_type": "ovh",
                                "id": "Meta-Llama-3_1-70B-Instruct",
                                "base_url": "https://llama-3-1-70b-instruct.endpoints.kepler.ai.cloud.ovh.net/api/openai_compat/v1",
                                "temperature": 0,
                                "max_retries": 3,
                                "top_p": 0.95,
                            },
                            "llama-3_3-70B@ovh": {
                                "server_type": "ovh",
                                "id": "Meta-Llama-3_3-70B-Instruct",
                                "base_url": "https://llama-3-3-70b-instruct.endpoints.kepler.ai.cloud.ovh.net/api/openai_compat/v1",
                                "temperature": 0,
                                "max_retries": 3,
                                "top_p": 0.95,
                            },
                            "llama-3_1-8B@ovh": {
                                "server_type": "ovh",
                                "id": "Llama-3.1-8B-Instruct",
                                "base_url": "https://llama-3-1-8b-instruct.endpoints.kepler.ai.cloud.ovh.net/api/openai_compat/v1/",
                                "temperature": 0,
                                "max_retries": 3,
                                "top_p": 0.95,
                            },
                            "deepseek-r1-70B@ovh": {
                                "server_type": "ovh",
                                "id": "DeepSeek-R1-Distill-Llama-70B",
                                "base_url": "https://deepseek-r1-distill-llama-70b.endpoints.kepler.ai.cloud.ovh.net/api/openai_compat/v1",
                                "temperature": 0,
                                "max_retries": 3,
                                "top_p": 0.95,
                            },
                            "gpt-4o@openai": {
                                "server_type": "openai",
                                "id": "gpt-4o",
                                "temperature": 0,
                                "max_retries": 3,
                                "top_p": 0.95,
                            },
                            "o3-mini@openai": {
                                "server_type": "openai",
                                "id": "o3-mini",
                            },
                            "o1@openai": {"server_type": "openai", "id": "o1"},
                            "deepseek-chat@deepseek": {
                                "server_type": "deepseek",
                                "id": "deepseek-chat",
                                "base_url": "https://api.deepseek.com",
                                "temperature": 0,
                                "max_retries": 3,
                                "top_p": 0.95,
                            },
                            "deepseek-reasoner@deepseek": {
                                "server_type": "deepseek",
                                "id": "deepseek-reasoner",
                                "base_url": "https://api.deepseek.com",
                                "temperature": 0,
                                "max_retries": 3,
                                "top_p": 0.95,
                            },
                            "deepseek-reasoner@hf": {
                                "server_type": "hugface",
                                "id": "deepseek-ai/DeepSeek-R1",
                                "base_url": "https://huggingface.co/api/inference-proxy/together",
                                "top_p": 0.95,
                            },
                        },
                        "text_embedding_models": {
                            "nomic-embed-text_faiss@local": {
                                "server_type": "ollama-embeddings",
                                "id": "nomic-embed-text",
                                "vector_db": "faiss",
                            },
                            "mxbai-embed-large_faiss@local": {
                                "server_type": "ollama-embeddings",
                                "id": "mxbai-embed-large",
                                "vector_db": "faiss",
                            },
                            "nomic-embed-text_chroma@local": {
                                "server_type": "ollama-embeddings",
                                "id": "nomic-embed-text",
                                "vector_db": "chroma",
                            },
                        },
                        "scenario_1": {
                            "validate_question": "deepseek-reasoner@hf",
                            "ask_question": "llama3_2-1b@local",
                        },
                        "scenario_2": {
                            "validate_question": "llama3_2-1b@local",
                            "generate_query": "llama-3_1-70B@ovh",
                            "interpret_results": "llama3_2-1b@local",
                        },
                        "scenario_3": {
                            "validate_question": "llama3_2-1b@local",
                            "generate_query": "llama-3_1-70B@ovh",
                            "interpret_results": "llama3_2-1b@local",
                            "text_embedding_model": "nomic-embed-text_faiss@local",
                        },
                        "scenario_4": {
                            "validate_question": "llama-3_1-70B@ovh",
                            "generate_query": "llama3_2-1b@local",
                            "interpret_results": "llama3_2-1b@local",
                            "text_embedding_model": "nomic-embed-text_faiss@local",
                        },
                        "scenario_5": {
                            "validate_question": "llama3_2-1b@local",
                            "generate_query": "llama3_2-1b@local",
                            "interpret_results": "llama-3_1-70B@ovh",
                            "text_embedding_model": "nomic-embed-text_faiss@local",
                        },
                        "scenario_6": {
                            "validate_question": "llama-3_1-70B@ovh",
                            "generate_query": "deepseek-r1_1_5b@local",
                            "interpret_results": "llama3_2-1b@local",
                            "text_embedding_model": "nomic-embed-text_faiss@local",
                        },
                        "scenario_7": {
                            "validate_question": "llama3_2-1b@local",
                            "generate_query": "llama-3_1-70B@ovh",
                            "judge_query": "llama-3_1-70B@ovh",
                            "judge_regenerate_query": "llama-3_1-70B@ovh",
                            "interpret_results": "llama-3_1-70B@ovh",
                            "text_embedding_model": "nomic-embed-text_faiss@local",
                            "judging_grade_threshold_retry": 8,
                            "judging_grade_threshold_run": 5,
                        },
                    }
                }
            },
        }
    },
)
def get_active_config_endpoint() -> KGConfig:

    args = setup_cli()
    read_configuration(args=args)
    yaml_data = get_configuration()
    return yaml_data


@app.post(
    path="/api/q2forge/config/create",
    operation_id="create_config",
    summary="Create a new configuration",
    description=(
        "This endpoint creates a new configuration file to be used by the Q²Forge resource. "
        + "The configuration file contains information about a knowledge graph, "
    ),
    responses={
        200: {
            "description": "The created configuration",
            "content": {
                "application/json": {
                    "example": {
                        "kg_full_name": "WheatGenomic Scienctific Literature Knowledge Graph",
                        "kg_short_name": "d2kab",
                        "kg_description": "The Wheat Genomics Scientific Literature Knowledge Graph (WheatGenomicsSLKG) is a FAIR knowledge graph that exploits the Semantic Web technologies to describe PubMed scientific articles on wheat genetics and genomics. It represents Named Entities (NE) about genes, phenotypes, taxa and varieties, mentioned in the title and the abstract of the articles, and the relationships between wheat mentions of varieties and phenotypes.",
                        "kg_sparql_endpoint_url": "http://d2kab.i3s.unice.fr/sparql",
                        "ontologies_sparql_endpoint_url": "http://d2kab.i3s.unice.fr/sparql",
                        "properties_qnames_info": [],
                        "prefixes": {
                            "bibo": "http://purl.org/ontology/bibo/",
                            "d2kab": "http://ns.inria.fr/d2kab/",
                            "dc": "http://purl.org/dc/elements/1.1/",
                            "dcterms": "http://purl.org/dc/terms/",
                            "fabio": "http://purl.org/spar/fabio/",
                            "foaf": "http://xmlns.com/foaf/0.1/",
                            "frbr": "http://purl.org/vocab/frbr/core#",
                            "obo": "http://purl.obolibrary.org/obo/",
                            "oio": "http://www.geneontology.org/formats/oboInOwl#",
                            "oa": "http://www.w3.org/ns/oa#",
                            "owl": "http://www.w3.org/2002/07/owl#",
                            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
                            "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
                            "schema": "http://schema.org/",
                            "skos": "http://www.w3.org/2004/02/skos/core#",
                            "xsd": "http://www.w3.org/2001/XMLSchema#",
                            "wto": "http://opendata.inrae.fr/wto/",
                        },
                        "ontology_named_graphs": [
                            "http://ns.inria.fr/d2kab/graph/wheatgenomicsslkg",
                            "http://ns.inria.fr/d2kab/ontology/wto/v3",
                            "http://purl.org/dc/elements/1.1/",
                            "http://purl.org/dc/terms/",
                            "http://purl.org/obo/owl/",
                            "http://purl.org/ontology/bibo/",
                            "http://purl.org/spar/fabio",
                            "http://purl.org/vocab/frbr/core#",
                            "http://www.w3.org/2002/07/owl#",
                            "http://www.w3.org/2004/02/skos/core",
                            "http://www.w3.org/ns/oa#",
                        ],
                        "excluded_classes_namespaces": [],
                        "queryExamples": [
                            {
                                "question": "Retrieve genes that are mentioned proximal to the a given phenotype (resistance to leaf rust in this example).",
                                "query": 'prefix rdf:     <http://www.w3.org/1999/02/22-rdf-syntax-ns#> \nprefix rdfs:    <http://www.w3.org/2000/01/rdf-schema#> \nprefix xsd:     <http://www.w3.org/2001/XMLSchema#> \nprefix schema:  <http://schema.org/> \nprefix owl:     <http://www.w3.org/2002/07/owl#> \nprefix skos:    <http://www.w3.org/2004/02/skos/core#> \nprefix oa:      <http://www.w3.org/ns/oa#> \nprefix ncbi:    <http://identifiers.org/taxonomy/> \nprefix dct:     <http://purl.org/dc/terms/> \nprefix frbr:    <http://purl.org/vocab/frbr/core#> \nprefix fabio:   <http://purl.org/spar/fabio/> \nprefix obo:     <http://purl.obolibrary.org/obo/> \nprefix bibo:    <http://purl.org/ontology/bibo/> \nprefix d2kab:   <http://ns.inria.fr/d2kab/> \nprefix dc:      <http://purl.org/dc/terms/> \nprefix d2kab_bsv:   <http://ontology.inrae.fr/bsv/ontology/>\nprefix dul: <http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#>\nprefix dct:     <http://purl.org/dc/terms/> \nprefix taxref: <http://taxref.mnhn.fr/lod/property/>\n\nSELECT ?GeneName (count(distinct ?paper) as ?NbOcc)\nFROM NAMED <http://ns.inria.fr/d2kab/graph/wheatgenomicsslkg>\nFROM NAMED <http://ns.inria.fr/d2kab/ontology/wto/v3>\nWHERE {\n  GRAPH <http://ns.inria.fr/d2kab/graph/wheatgenomicsslkg> { \n     ?a1 a oa:Annotation; \n      oa:hasTarget [ oa:hasSource ?source1 ] ;  \n      oa:hasBody ?WTOtraitURI .\n\n   ?source1 frbr:partOf+ ?paper .\n    \n   ?a a oa:Annotation ; \n      oa:hasTarget [ oa:hasSource ?source ] ;\n      oa:hasBody [ a d2kab:Gene; skos:prefLabel ?GeneName ].\n\n   ?source frbr:partOf+ ?paper.\n\n   ?paper a fabio:ResearchPaper.\n}\n   GRAPH <http://ns.inria.fr/d2kab/ontology/wto/v3> {\n       ?WTOtraitURI skos:prefLabel "resistance to Leaf rust" .\n}\n}\nGROUP BY ?GeneName \nHAVING (count(distinct ?paper) > 1)\nORDER BY DESC(?NbOcc)',
                            },
                            {
                                "question": "Retrieve publications in which genes are mentioned proximal to wheat varieties and traits from a specific class, e.g., all wheat traits related to resistance to fungal pathogens.",
                                "query": "prefix rdf:     <http://www.w3.org/1999/02/22-rdf-syntax-ns#> \nprefix rdfs:    <http://www.w3.org/2000/01/rdf-schema#> \nprefix xsd:     <http://www.w3.org/2001/XMLSchema#> \nprefix schema:  <http://schema.org/> \nprefix owl:     <http://www.w3.org/2002/07/owl#> \nprefix skos:    <http://www.w3.org/2004/02/skos/core#> \nprefix oa:      <http://www.w3.org/ns/oa#> \nprefix ncbi:    <http://identifiers.org/taxonomy/> \nprefix dct:     <http://purl.org/dc/terms/> \nprefix frbr:    <http://purl.org/vocab/frbr/core#> \nprefix fabio:   <http://purl.org/spar/fabio/> \nprefix obo:     <http://purl.obolibrary.org/obo/> \nprefix bibo:    <http://purl.org/ontology/bibo/> \nprefix d2kab:   <http://ns.inria.fr/d2kab/> \nprefix dc:      <http://purl.org/dc/terms/> \nprefix d2kab_bsv:   <http://ontology.inrae.fr/bsv/ontology/>\nprefix dul: <http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#>\nprefix dct:     <http://purl.org/dc/terms/> \nprefix taxref: <http://taxref.mnhn.fr/lod/property/>\n\nSELECT *\nFROM NAMED <http://ns.inria.fr/d2kab/ontology/wto/v3>\nWHERE {\n  GRAPH <http://ns.inria.fr/d2kab/ontology/wto/v3> {\n    { ?body a ?class ; skos:prefLabel ?WTOtrait.\n      ?class rdfs:subClassOf* <http://opendata.inrae.fr/wto/0000340>.\n    }\n    UNION\n    { ?body rdfs:label ?WTOtrait ;\n        rdfs:subClassOf* <http://opendata.inrae.fr/wto/0000340>.\n    }\n    UNION\n    { ?body skos:prefLabel ?WTOtrait ; skos:broader* ?concept .\n      ?concept a ?class.\n      ?class rdfs:subClassOf* <http://opendata.inrae.fr/wto/0000340>.\n    }\n  }\n}",
                            },
                            {
                                "question": 'Retrieve genetic markers mentioned proximal to genes which are in turn mentioned proximal to a wheat phenotype ("resistance to Stripe rust" in this example) considering the same scientific publication.',
                                "query": 'prefix rdf:     <http://www.w3.org/1999/02/22-rdf-syntax-ns#> \nprefix rdfs:    <http://www.w3.org/2000/01/rdf-schema#> \nprefix xsd:     <http://www.w3.org/2001/XMLSchema#> \nprefix schema:  <http://schema.org/> \nprefix owl:     <http://www.w3.org/2002/07/owl#> \nprefix skos:    <http://www.w3.org/2004/02/skos/core#> \nprefix oa:      <http://www.w3.org/ns/oa#> \nprefix ncbi:    <http://identifiers.org/taxonomy/> \nprefix dct:     <http://purl.org/dc/terms/> \nprefix frbr:    <http://purl.org/vocab/frbr/core#> \nprefix fabio:   <http://purl.org/spar/fabio/> \nprefix obo:     <http://purl.obolibrary.org/obo/> \nprefix bibo:    <http://purl.org/ontology/bibo/> \nprefix d2kab:   <http://ns.inria.fr/d2kab/> \nprefix dc:      <http://purl.org/dc/terms/> \nprefix d2kab_bsv:   <http://ontology.inrae.fr/bsv/ontology/>\nprefix dul: <http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#>\nprefix dct:     <http://purl.org/dc/terms/> \nprefix taxref: <http://taxref.mnhn.fr/lod/property/>\n\nSELECT (GROUP_CONCAT(distinct ?GeneName; SEPARATOR="-") as ?genes) \n(GROUP_CONCAT(distinct ?marker; SEPARATOR="-") as ?markers) \n?paper ?year ?WTOtrait\nFROM NAMED <http://ns.inria.fr/d2kab/graph/wheatgenomicsslkg>\nFROM NAMED <http://ns.inria.fr/d2kab/ontology/wto/v3>\nWHERE {\nVALUES ?WTOtrait { "resistance to Stripe rust" }\nGRAPH <http://ns.inria.fr/d2kab/graph/wheatgenomicsslkg> { \n?a1 a oa:Annotation ;\n    oa:hasTarget [ oa:hasSource ?source1 ];\n    oa:hasBody [ a d2kab:Gene ; skos:prefLabel ?GeneName].\n\n?source1 frbr:partOf+ ?paper .\n\n?a2 a oa:Annotation ;\n    oa:hasTarget [ oa:hasSource ?source2 ] ;\n    oa:hasBody [ a d2kab:Marker ; skos:prefLabel ?marker ]. \n\n?source2 frbr:partOf+ ?paper .\n\n?a3 a oa:Annotation; \n    oa:hasTarget [ oa:hasSource ?source3 ];\n    oa:hasBody ?WTOtraitURI.\n\n?source3 frbr:partOf+ ?paper . \n\n?paper a fabio:ResearchPaper; dct:title ?source3; dct:issued ?year .\nFILTER (?year >= "2010"^^xsd:gYear)\n}\nGRAPH <http://ns.inria.fr/d2kab/ontology/wto/v3> {\n       ?WTOtraitURI skos:prefLabel ?WTOtrait.\n}\n}\nGROUP BY ?paper ?year ?WTOtrait',
                            },
                            {
                                "question": "Retrieves couples of scientific publications such as a first publication mentions a given phenotype and a gene and the second one mentions the same gene name with a genetic marker.",
                                "query": 'prefix rdf:     <http://www.w3.org/1999/02/22-rdf-syntax-ns#> \nprefix rdfs:    <http://www.w3.org/2000/01/rdf-schema#> \nprefix xsd:     <http://www.w3.org/2001/XMLSchema#> \nprefix schema:  <http://schema.org/> \nprefix owl:     <http://www.w3.org/2002/07/owl#> \nprefix skos:    <http://www.w3.org/2004/02/skos/core#> \nprefix oa:      <http://www.w3.org/ns/oa#> \nprefix ncbi:    <http://identifiers.org/taxonomy/> \nprefix dct:     <http://purl.org/dc/terms/> \nprefix frbr:    <http://purl.org/vocab/frbr/core#> \nprefix fabio:   <http://purl.org/spar/fabio/> \nprefix obo:     <http://purl.obolibrary.org/obo/> \nprefix bibo:    <http://purl.org/ontology/bibo/> \nprefix d2kab:   <http://ns.inria.fr/d2kab/> \nprefix dc:      <http://purl.org/dc/terms/> \nprefix d2kab_bsv:   <http://ontology.inrae.fr/bsv/ontology/>\nprefix dul: <http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#>\nprefix dct:     <http://purl.org/dc/terms/> \nprefix taxref: <http://taxref.mnhn.fr/lod/property/>\n\nSELECT distinct ?paper1 ?WTOtrait ?Title1 ?geneName ?paper2 ?Title2 (GROUP_CONCAT(distinct ?marker; SEPARATOR="-") as ?markers) \nFROM <http://ns.inria.fr/d2kab/graph/wheatgenomicsslkg>\nFROM <http://ns.inria.fr/d2kab/ontology/wto/v3>\nWHERE {\n{\n\nSELECT distinct ?geneName ?gene ?paper1 ?Title1 ?WTOtrait WHERE \n{\n    VALUES ?WTOtrait { "resistance to Stripe rust" }\n    ?a1 a oa:Annotation ; \n        oa:hasTarget [ oa:hasSource ?source1 ] ;\n        oa:hasBody ?body .\n\n    GRAPH <http://ns.inria.fr/d2kab/ontology/wto/v3> {\n        ?body skos:prefLabel ?WTOtrait.\n    }\n\n    ?a2 a oa:Annotation ;\n        oa:hasTarget [ oa:hasSource ?source2 ] ;\n        oa:hasBody ?gene .\n        ?gene a d2kab:Gene ; skos:prefLabel ?geneName . \n        ?source1 frbr:partOf+ ?paper1 .\n        ?source2 frbr:partOf+ ?paper1 .\n        ?paper1 a fabio:ResearchPaper ; dct:title ?source1 .\n        ?source1 rdf:value ?Title1.\n}\nLIMIT 20\n}\n?a3 a oa:Annotation ;\n    oa:hasTarget [ oa:hasSource ?source3 ] ;\n    oa:hasBody [a d2kab:Marker ; skos:prefLabel ?marker ] .\n \n?a4 a oa:Annotation ;\n    oa:hasTarget [ oa:hasSource ?source4 ] ;\n    oa:hasBody ?gene .\n \n?source3 frbr:partOf+ ?paper2 .\n?source4 frbr:partOf+ ?paper2 .\n?paper2 a fabio:ResearchPaper ; dct:title ?titleURI .\n?titleURI rdf:value ?Title2.\nFILTER (URI(?paper1) != URI(?paper2))\n}\nGROUP BY ?WTOtrait ?geneName ?paper1 ?Title1 ?paper2 ?Title2\nLIMIT 50',
                            },
                        ],
                    }
                }
            },
        },
        400: {
            "description": "The configuration file already exists",
            "content": {
                "application/json": {
                    "example": {"error": "Configuration file already exists: d2kab"}
                }
            },
        },
        500: {
            "description": "An error occurred while creating the configuration",
            "content": {
                "application/json": {
                    "example": {
                        "error": "Error creating configuration file: <error_message>"
                    }
                }
            },
        },
    },
)
def create_config_endpoint(config_request: CreateConfig) -> CreateConfig:
    try:

        logger.debug(f"Received configuration request: {config_request}")

        config_path = (
            Path(__file__).resolve().parent.parent.parent
            / "config"
            / f"params_{config_request.kg_short_name}.yml"
        )

        # Check if the file already exists
        if config_path.exists():
            logger.error(f"Configuration file already exists at {config_path}")
            return Response(
                status_code=400,
                content=json.dumps(
                    {
                        "error": f"Configuration file already exists: {config_request.kg_short_name}"
                    }
                ),
                media_type="application/json",
            )

        # Save the query examples locally to be used in the embedding
        save_query_examples_to_file(
            kg_short_name=config_request.kg_short_name,
            query_examples=config_request.queryExamples,
        )

        # Create the configuration dictionary from the request
        with open(config_path, "w", encoding="utf-8") as file:
            # Convert the request to a dictionary
            config_dict = config_request.model_dump()

            # Remove the query examples from the request to avoid adding them to the config file
            del config_dict["queryExamples"]

            # Add missing parameters to the configuration
            updated_config = add_missing_config_params(config_dict)

            # Write the configuration to the file
            yaml.safe_dump(updated_config, file)

            logger.info(f"Configuration file created at {config_path}")

            return Response(
                status_code=200,
                content=config_request.model_dump_json(),
                media_type="application/json",
            )
    except Exception as e:
        logger.error(f"Error creating configuration file: {str(e)}")
        return Response(
            status_code=500,
            content={"error": f"Error creating configuration file: {str(e)}"},
            media_type="application/json",
        )


@app.post(
    path="/api/q2forge/config/activate",
    operation_id="activate_config",
    summary="Activate a configuration",
    description=(
        "This endpoint activates a configuration file to be used by the Q²Forge resource. "
    ),
    responses={
        200: {
            "description": "The short name of the activated configuration",
            "content": {"application/json": {"example": {"kg_short_name": "d2kab"}}},
        },
        400: {
            "description": "The configuration file does not exist",
            "content": {
                "application/json": {
                    "example": {"error": "Configuration file does not exist: d2kab"}
                }
            },
        },
        500: {
            "description": "An error occurred while activating the configuration",
            "content": {
                "application/json": {
                    "example": {
                        "error": "Error activating configuration file: <error_message>"
                    }
                }
            },
        },
    },
)
def activate_config_endpoint(config_request: ActivateConfig):
    try:

        logger.debug(f"Configuration to activate: {config_request}")

        config_to_activate_path = (
            Path(__file__).resolve().parent.parent.parent
            / "config"
            / f"params_{config_request.kg_short_name}.yml"
        )

        # Check if the file already exists
        if not config_to_activate_path.exists():
            logger.error(
                f"Configuration file does not exist at {config_to_activate_path}"
            )
            return Response(
                status_code=400,
                content=json.dumps(
                    {
                        "error": f"Configuration file does not exist {config_request.kg_short_name}"
                    }
                ),
                media_type="application/json",
            )

        active_config_path = (
            Path(__file__).resolve().parent.parent.parent / "config" / "params.yml"
        )

        with open(config_to_activate_path, "r", encoding="utf-8") as file_to_activate:
            config_data = yaml.safe_load(file_to_activate)

            # Activate the configuration
            with open(active_config_path, "w", encoding="utf-8") as active_file:
                yaml.safe_dump(config_data, active_file)
                logger.info(f"Configuration file activated at {active_file}")
                return Response(
                    status_code=200,
                    content=config_request.model_dump_json(),
                    media_type="application/json",
                )
    except Exception as e:
        logger.error(f"Error activating configuration file: {str(e)}")
        return Response(
            status_code=500,
            content={"error": f"Error activating configuration file: {str(e)}"},
            media_type="application/json",
        )


@app.post(
    path="/api/q2forge/config/kg_descriptions",
    operation_id="generate_kg_descriptions",
    summary="Generate KG descriptions",
    description=(
        "This endpoint generates KG descriptions of a given Knowledge Graph. "
        + "The KG descriptions are used to create KG embeddings, which are then used to provide the language model with more context through a RAG technique."
    ),
    responses={
        200: {
            "description": "The generated KG descriptions",
            "content": {
                "application/json": {
                    "example": {
                        "kg_description": {
                            "classes_description.txt": "('obo:CHEBI_100', '(-)-medicarpin', 'The (-)-enantiomer of medicarpin.')\n...",
                            "classes_with_instances_description.txt": "('obo:CHEBI_10', '(+)-Atherospermoline', None)\n...",
                            "properties_description.txt": "('http://prismstandard.org/namespaces/basic/2.0/alternateTitle', 'alternate title. ', 'An alternative title for a resource.')\n...",
                        }
                    }
                }
            },
        },
        500: {
            "description": "An error occurred while generating the KG descriptions",
            "content": {
                "application/json": {
                    "example": {
                        "error": "Error generating KG descriptions: <error_message>"
                    }
                }
            },
        },
    },
)
def generate_kg_descriptions_endpoint(config_request: ActivateConfig):
    try:
        generate_descriptions()

        directory = config.get_preprocessing_directory()
        generated_files = {}
        for file in directory.iterdir():
            if file.is_file():
                with file.open("r", encoding="utf-8") as f:
                    content = f.read()
                    generated_files[file.name] = content

        return Response(
            status_code=200,
            content=json.dumps(generated_files),
            media_type="application/json",
        )
    except Exception as e:
        logger.error(f"Error generating KG descriptions: {str(e)}")
        return Response(
            status_code=500,
            content={"error": f"Error generating KG descriptions: {str(e)}"},
            media_type="application/json",
        )


@app.post(
    path="/api/q2forge/config/kg_embeddings",
    operation_id="generate_kg_embeddings",
    summary="Generate KG embeddings",
    description=(
        "This endpoint generates the embedding of the textual description of the ontology classes used in the KG. "
        + "The KG embeddings are used in the different scenarios to generate "
        + "SPARQL queries from questions in natural language."
    ),
    responses={
        200: {
            "description": "The generated KG embeddings",
            "content": {
                "application/json": {
                    "example": {"message": "KG embeddings generated successfully"}
                }
            },
        },
        500: {
            "description": "An error occurred while generating the KG embeddings",
            "content": {
                "application/json": {
                    "example": {
                        "error": "Error generating KG embeddings: <error_message>"
                    }
                }
            },
        },
    },
)
def generate_kg_embeddings_endpoint(config_request: ActivateConfig):
    try:

        start_compute_embeddings(is_api_call=True)

        return Response(
            status_code=200,
            content=json.dumps({"message": "KG embeddings generated successfully"}),
            media_type="application/json",
        )
    except Exception as e:
        logger.error(f"Error generating KG embeddings: {str(e)}")
        return Response(
            status_code=500,
            content={"error": f"Error generating KG embeddings: {str(e)}"},
            media_type="application/json",
        )


@app.post(
    path="/api/q2forge/generate_questions",
    operation_id="generate_questions",
    summary="Generate competency questions",
    description=(
        "This endpoint generates competency questions about a given Knowledge Graph using a given LLM. "
    ),
    responses={
        200: {
            "description": "The stream of generated competency questions",
            "content": {
                "application/json": {
                    "example": [
                        {"event": "on_chat_model_start"},
                        {
                            "event": "on_chat_model_stream",
                            "data": "{'question': 'What protein targets does donepezil (CHEBI_53289) inhibit with an IC50 less than 10 µM?'}",
                        },
                        {"event": "on_chat_model_end"},
                    ]
                }
            },
        },
        500: {
            "description": "An error occurred while generating competency questions",
            "content": {
                "application/json": {
                    "example": {
                        "error": "Error generating competency questions: <error_message>"
                    }
                }
            },
        },
    },
)
async def generate_question_endpoint(
    generate_competency_question_request: GenerateCompetencyQuestion,
) -> StreamingResponse:

    return StreamingResponse(
        generate_competency_questions(
            model_config_id=generate_competency_question_request.model_config_id,
            number_of_questions=generate_competency_question_request.number_of_questions,
            additional_context=generate_competency_question_request.additional_context,
            kg_description=generate_competency_question_request.kg_description,
            kg_schema=generate_competency_question_request.kg_schema,
            enforce_structured_output=generate_competency_question_request.enforce_structured_output,
        ),
        media_type="application/json",
    )


@app.post(
    path="/api/q2forge/answer_question",
    operation_id="answer_question",
    summary="Generate and Execute a SPARQL query",
    description=(
        "This endpoint answers a question about a given Knowledge Graph using a given LLMs configuration."
    ),
    responses={
        200: {
            "description": "The stream of the answer to the question",
            "content": {
                "application/json": {
                    "example": [
                        {
                            "event": "on_chain_end",
                            "node": "init",
                            "data": {"scenario_id": "6"},
                        },
                        {
                            "event": "on_chain_end",
                            "node": "validate_question",
                            "data": {"question_validation_results": "true"},
                        },
                    ]
                }
            },
        },
        500: {
            "description": "An error occurred while answering the question",
            "content": {
                "application/json": {
                    "example": {"error": "Error answering question: <error_message>"}
                }
            },
        },
    },
)
def answer_question_endpoint(answer_question_request: AnswerQuestion):

    set_custom_scenario_configuration(
        scenario_id=answer_question_request.scenario_id,
        validate_question_model=answer_question_request.validate_question_model,
        ask_question_model=answer_question_request.ask_question_model,
        generate_query_model=answer_question_request.generate_query_model,
        judge_query_model=answer_question_request.judge_query_model,
        judge_regenerate_query_model=answer_question_request.judge_regenerate_query_model,
        interpret_results_model=answer_question_request.interpret_results_model,
        text_embedding_model=answer_question_request.text_embedding_model,
    )
    return StreamingResponse(
        answer_question(
            scenario_id=answer_question_request.scenario_id,
            question=answer_question_request.question,
        ),
        # media_type="text/event-stream",
        media_type="application/json",
    )


@app.post(
    path="/api/q2forge/judge_query",
    operation_id="judge_query",
    summary="Judge a SPARQL query",
    description=(
        "This endpoint judges a SPARQL query given a natural language question using a given LLM."
    ),
    responses={
        200: {
            "description": "The stream of the model judgement",
            "content": {
                "application/json": {
                    "example": [
                        {"event": "on_chat_model_start"},
                        {
                            "event": "on_chat_model_stream",
                            "data": "Grade: 8/10\nJustification: The student has attempted to retrieve publications in which genes are mentioned proximal to wheat varieties and traits from a specific class, e.g., all wheat traits related to resistance to fungal pathogens. The query is well-structured and uses the necessary prefixes for RDF data modeling.",
                        },
                        {"event": "on_chat_model_end"},
                    ]
                }
            },
        },
        500: {
            "description": "An error occurred while judging the query",
            "content": {
                "application/json": {
                    "example": {"error": "Error judging query: <error_message>"}
                }
            },
        },
    },
)
async def judge_query_endpoint(refine_query_request: RefineQuery):
    return StreamingResponse(
        refine_query(
            model_config_id=refine_query_request.model_config_id,
            question=refine_query_request.question,
            sparql_query=refine_query_request.sparql_query,
            sparql_query_context=refine_query_request.sparql_query_context,
        ),
        media_type="application/json",
    )


# Add MCP server to FastAPI app
mcp = FastApiMCP(fastapi=app, name="Q2Forge")

# Mount the MCP server directly to your FastAPI app
mcp.mount()

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
