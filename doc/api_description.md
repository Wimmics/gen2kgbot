# Web API description

Gen²KGBot exposes a Web API to allow applications to use its services remotely, [Q²Forge]([Q²Forge](https://github.com/Wimmics/q2forge)) Application is an example.

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
