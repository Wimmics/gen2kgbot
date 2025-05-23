# Gen²KGBot - Generic Generative Knowledge Graph Robot

Gen²KGBot intends to allow users to "speak to a knowledge graph", that is, **use natural language to query knowledge graphs** in a generic manner, with the help of generative large language models (LLM).

It provides a generic framework to translate a natural-language (NL) question into its counterpart SPARQL query, execute the query and interpret the SPARQL results.

Several steps are involved, depending on the scenario selected:
- explore the schema of the KG, including the used ontolgies;
- generate a textual description of the ontology classes and turn them into text embeddings
- generate a description of how the ontology classes are used in the KG. This description can follow 3 formats: Turtle, tuples, natural language.
- ask an LLM to translate a NL question into a SPARQL query using a context including the textual description of the ontology classes related to the question, and how these classes are used in this KG.
- if the SPARQL query is invalid, as the LLM to fix it;
- execute the SPARQL query against the KG and ask an LLM to interpret the results.

Gen²KGBot can be used from the **command-line interface**, from **Langgraph Studio**, or remotely through a **Web API**.

## Documentation

- [Envionment setup](#environment-setup)
- [Startup instructions](#startup-instructions)
- [Web API description](doc/api_description.md)
- [Development Guidelines](doc/dev_guidelines.md)

## License

AGPLv3: see the [LICENSE file](LICENSE).

## Cite this work

Yousouf TAGHZOUTI, Franck MICHEL, Tao JIANG, Louis-Félix NOTHIAS, Fabien GANDON (2025). **Gen²KGBot - Generic Generative Knowledge Graph Robot**. <https://github.com/Wimmics/gen2kgbot> 


<details>
<summary>See BibTex</summary>
@software{taghzouti_gen2kgbot:2025,
    author = {TAGHZOUTI, Yousouf and MICHEL, Franck and JIANG, Tao and NOTHIAS, Louis-Félix , and GANDON Fabien},
    title = {{Gen²KGBot - Generic Generative Knowledge Graph Robot}},
    url = {https://github.com/Wimmics/gen2kgbot},
    version = {1.0},
    year = {2025}
}
</details>

## Scenarios

Gen²KGBot implements multiple scenarios of increasing complexity to translate NL questions into SPARQL, and refine the generated query.

### Scenario 1
Simply ask the user's question to the language model. This naive scenario is used to figure out what the language model "knows" about the topic. The KG is not involved.

### Scenario 2
Ask the language model to directly **translate the user's question into a SPARQL query without any other information**.

This scenario is used to figure out what the language model may "know" about the target KG.
It can be used as a baseline for the construction of a SPARQL query.

### Scenario 3
Ask the language model to translate the user's question into a SPARQL query based on a context containing a **list of classes related to the question**.
These classes are selected using a **similarity search between the question and the class descriptions**.

This involves a preprocessing step where a **textual description of the classes** used in the KG is generated, and **text embeddings** of the descriptions are computed.

### Scenario 4
Extends the context in Scenario 3 with a **description of the properties and value types used with the instances of selected classes**.

This additional context can be provided in multiples syntaxes: as Turtle, as tuples _(class, property, property label, value type)_, 
or in natural language like _"Instances of class 'class' have property 'prop' (label) with value type 'value_type'"_.

### Scenario 5
Extends the Scenario 4 with a **retry mechanism if the generated SPARQL query is not syntactically correct**.

In this case the language model is provided with the previously generated query, the parsing error, and asked to reformulate it.

### Scenario 6
Extends the context in Scenario 5 with some **example SPARQL queries** related to the question.
These queries are selected using a **similarity search with the question**.

This involves a preprocessing step where existing SPARQL queries are provided, and **text embeddings** thereof are computed.

### Scenario 7
Extends the Scenario 6 with a **query judge component that can evaluates the quality of the generated SPARQL** and may start a **query improvement cycle**.



## Environment setup

Conda (or one of its distributions) is required for setting up the environment.

1) Install [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) or an equivalent distribution e.g. [mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html).
2) File `environment.yml` shall be used to install the dependencies. 
⚠️ Some packages are hardware-dependent (notably faiss-cpu vs. faiss-gpu). 
**Update `environment.yml` accordingly** before runnung the command below:
```sh
conda env create -f ./environment.yml
```

3) Install the [Ollama](https://github.com/ollama/ollama) application for your platform and, as a startup config, install models `ollama3.2:1b`, `nomic-embed-text`:
```sh
ollama pull ollama3.2:1b
ollama pull nomic-embed-text
```

4) Gen²KGBot relies on LangChain. Set up environment variable `LANGCHAIN_API_KEY` with your own key.

5) Set up the environment variables providing your own keys for using the LLMs and services of your choice.
Currently, Gen²KGBot supports the following ones: `OPENAI_API_KEY`, `OVHCLOUD_API_KEY`, `HF_TOKEN`, `GOOGLE_API_KEY`, `DEEPSEEK_API_KEY`.



## Startup instructions

### Adapt the configuration for your KG

To create a new configuration, follow these steps:

1) Copy [this existing config](./config/params.yml) as a starting point.

2) Modify these fields:
- `kg_full_name`: The full name of the knowledge graph.
- `kg_short_name`: The short name of the knowledge graph.
- `kg_description`: A description of the knowledge graph.
- `kg_sparql_endpoint_url`: The URL of the SPARQL endpoint for the knowledge graph.
- `ontologies_sparql_endpoint_url`: The URL of the SPARQL endpoint for the ontologies.
- `properties_qnames_info`: A list of property QNames information.
- `prefixes`: A dictionary of prefixes and their corresponding URIs.
- `ontology_named_graphs`: A list of ontology named graphs.
- `excluded_classes_namespaces`: A list of excluded class namespaces.

3) Optionally, if you want to use **scenarios 6+**, you will need a list of (question, SPARQL query) examples. Put each of the pair in a file under `{data_directory}/{kg_short_name}/example_queries`. It should have the following format: 

```
# {question}

{query}
```

### KG preprocessing

The KG preprocessing consists of two steps:
1) **Generate the KG descriptions**:

In the terminal, run the command: `python -m app.preprocessing.gen_descriptions -p="path/to/your/config"`

2) **Compute the classes and examples embeddings**:

In the terminal, run the command: `python -m app.preprocessing.compute_embeddings {+parameters}`

The parameters are:

- **--params**: Custom configuration file. Default: `config/params.yaml`
- **--model**: Embedding model description in the configuration file. Default: `nomic-embed-text_faiss@local`
- **--classes**  File with the description of the classes. Must be located in `{data_directory}/{KG short name}/preprocessing`. For example: `classes_with_instances_description.txt`
- **--properties** : File with the description of the properties. Must be located in `{data_directory}/{KG short name}/preprocessing`. For example: `properties_description.txt`
- **--sparql**: Sub-directory containing the example SPARQL queries. Must be located in `{data_directory}/{KG short name}`. For example:                        `example_queries`

### Running scenarios using CLI

Each scenario can be run in the terminal. 

Option `-q|--question` sets a custom NL question. Otherwise a default NL question.

Option `-p|--params` sets a custom configuration file. Otherwise file `config/params.yml` is used.

Use python's option `-m` to run one of the scenarios. For instance:

````bash
python -m app.scenarios.scenario_1.scenario_1 -c "What is the name of proteine X"
````

Or with additional parameters:

````bash
python -m app.scenarios.scenario_1.scenario_1 \
    --params config/params_d2kab.yml \ 
    --question "What articles mention taxon 'wheat' (Triticum aestivum) and trait 'resistance to Leaf rust'?"
````

### Running scenarios using Langgraph Studio

You may use the [LangGraph Studio](https://studio.langchain.com/) inteface to interact with the application. Simply run the following command in the root directory.

````bash
langgraph dev
````
This will initialize LangGraph studio based on local file `langgraph.json` and the default configuration file `config/params.yml`.

Then select the scenario and fill in a question in natural language.
