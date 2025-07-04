from app.utils.database_manager import close_db, db
from bson import ObjectId
import shutil

collections = [
    {
        "name": "users",
        "initial_data": [
            {
                "_id": ObjectId("686649779160ed2c4733657e"),
                "username": "admin",
                "disabled": False,
                "hashed_password": "$2b$12$yZMqaDhT.nlaAtYe4m0qFeQF1bM5fRMJ5A8viek5Z7Z6maXFH/r7i",
                "active_config_id": ObjectId("686668b8b3daabccac9fce03"),
            }
        ],
    },
    {
        "name": "configurations",
        "initial_data": [
            {
                "_id": ObjectId("686668b8b3daabccac9fce03"),
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
                "data_directory": "./data",
                "class_embeddings_subdir": "classes_with_instance_nomic",
                "property_embeddings_subdir": "properties_nomic",
                "queries_embeddings_subdir": "sparql_queries_nomic",
                "temp_directory": "./tmp",
                "max_similar_classes": 10,
                "expand_similar_classes": False,
                "class_context_format": "turtle",
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
                    "o3-mini@openai": {"server_type": "openai", "id": "o3-mini"},
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
        ],
    },
]


def initialize_database():

    for col in collections:
        name = col["name"]
        initial_data = col["initial_data"]

        if name not in db.list_collection_names():
            db.create_collection(name)
            print(f"📁 Created collection: {name}")
        else:
            print(f"ℹ️ Collection already exists: {name}")

        if db[name].count_documents({}) == 0:
            db[name].insert_many(initial_data)
            print(f"✅ Inserted initial data into: {name}")
        else:
            print(f"⚠️ Collection '{name}' already has data. Skipping initial insert.")

    close_db()
    print("🔌 MongoDB connection closed")


def copy_default_embeddings():
    source_folder = 'setup/data'
    destination_folder = 'data'

    # This copies everything from source to destination
    shutil.copytree(source_folder, destination_folder, dirs_exist_ok=True)


if __name__ == "__main__":
    copy_default_embeddings()
    initialize_database()
