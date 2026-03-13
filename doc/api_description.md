# Web API description

Gen²KGBot exposes a Web API to allow applications to use its services remotely, [Q²Forge]([Q²Forge](https://github.com/Wimmics/q2forge)) Application is an example.

## Set up for multiple users

Multiple users can access the Web API. To enable this, you must first set up a MongoDB database. Follow these steps:

1. Install [MongoDB](https://www.mongodb.com/try/download/community).
2. Configure the environment variables `Q2FORGE_SECRET_KEY` (used when generating users' passwords) and `MONGODB_CONNECTION_STRING`.
3. Run the following command to initialize the database with a default user and the [d2kab](https://d2kab.mystrikingly.com/) knowledge graph configuration.


```bash
python -m setup_db.init_db
```

The default values are:
* `Q2FORGE_SECRET_KEY` = "d7655ea104320f61fc49ac859501609b9595b994472d473ad736dc0ec657d512"
* `MONGODB_CONNECTION_STRING` = "mongodb://admin:admin@localhost:27017"
* admin/admin for username/password


## Start the API

To run the API use the following command: `python -m app.api.q2forge_api`


## API documentation

All the available endpoints are documented in [redoc](https://redocly.com/docs/redoc) and accessible at the following URI: <http://localhost:8000/q2forge/docs>. They are listed below with their respective descriptions:


- **Get the scenarios graph schemas:** This endpoint returns the different scenarios graph schemas. The schemas are represented in a mermaid format, which can be used to visualize the flow of the scenarios.

- **Get the currently active configuration:** This endpoint returns the currently active configuration of the Q²Forge API. The configuration contains information about the knowledge graph, the SPARQL endpoint, the properties to be used, the prefixes, and the models to be used for each scenario.

- **Create a new configuration:** This endpoint creates a new configuration file to be used by the Q²Forge resource. The configuration file contains information about a knowledge graph,

- **Activate a configuration:** This endpoint activates a configuration file to be used by the Q²Forge resource.

- **Generate KG descriptions:** This endpoint generates KG descriptions of a given Knowledge Graph. The KG descriptions are used to create the KG embeddings.

- **Generate KG embeddings:** This endpoint generates KG embeddings of a given Knowledge Graph. The KG embeddings are used in the different scenarios to generate SPARQL queries from questions in natural language.

- **Generate competency questions:** This endpoint generates competency questions about a given Knowledge Graph using a given LLM.

- **Generate and Execute a SPARQL query:** This endpoint answers a question about a given Knowledge Graph using a given LLMs configuration.

- **Judge a SPARQL query:** This endpoint judges a SPARQL query given a natural language question using a given LLM.
