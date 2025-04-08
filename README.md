# Gen²KGBot - Generic Generative Knowledge Graph Robot

Gen²KGBot addresses the problem of implementing GraphRAG applied to RDF knowledge graphs in a generic manner.
It provides two components:
- An application to generate a validation dataset, that is, a set of natural language questions and equivalent SPARQL queries.
- A framework to query an RDF knowledge graph (KG) using natural language questions: this involves the generation of a SPARQL query, its execution and the interpretation of the SPARQL results.


## Documentation

- [Envionment setup](#environment-setup)
- [Startup instructions](#startup-instructions)
- [Development Guidelines](doc/dev_guidelines.md)

## License

See the [LICENSE file](LICENSE).


## KGQueryForge: (NL question, SPARQL query) generator


## NL-to-SPARQL translation and execution

Gen²KGBot implements multiple scenarios of increasing complexity to translate natural language questions into SPARQL, and refining the generated query.

### Scenario 1
Simply ask the user's question to the language model. This naive scenario is used to figure out what the language model "knows" about the topic. The KG is not involved.

### Scenario 2
Ask the language model to directly **translate the user's question into a SPARQL query without any other information**.

This scenario is used to figure out what the language model may "know" about the target KG.
It can be used as a baseline for the construction of a SPARQL query.

### Scenario 3
Ask the language model to generate a SPARQL query based on the **user's question** and a context containing a **list of classes related to the question**.
These classes are selected using a **similarity search between the question and the class descriptions**.

This involves a preprocessing step where a **textual description of the classes** used in the KG is generated, and **text embeddings** of the descriptions are computed.

### Scenario 4
Extends the context in Scenario 3 with a **description of the properties and value types used with the instances of selected classes**.

This additional context can be provided in multiples syntaxes: as Turtle, as tuples _(class, property, property label, value type)_, 
or in natural language like _"Instances of class 'class' have property 'prop' (label) with value type 'value_type'"_.

### Scenario 5
Extends the Scenario 4 with a **retry mechanism if the generated SPARQL query is not syntactically correct**.

In this case the language model is provided with the previously generated query, and asked to reformulate it.

### Scenario 6
Extends the context in Scenario 5 with some **example SPARQL queries** related to the question.

These queries are selected using a **similarity search with the question**.

This involves a preprocessing step where existing SPARQL queries are provided, and **text embeddings** thereof are computed.

### Scenario 7
Extends the Scenario 6 with a **query judge** component that can evaluates the quality of the generated SPARQL and may start a **query improvement cycle**.



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
Also, set up the environment variables providing your own keys for using the LLMs and services of your choice.
Currently, Gen²KGBot supports the following ones: `OPENAI_API_KEY`, `OVHCLOUD_API_KEY`, `HF_TOKEN`, `GOOGLE_API_KEY`, `DEEPSEEK_API_KEY`.



## Startup instructions

### KG-dependent Preprocessing

[TBC]

### Running scenarios using CLI

Each scenario can be run in the terminal. 

Option `-q|--question` allows to set a custom NL question. Otherwise a default NL question.

Option `-p|--params` allows to set a custom configuration faile. Otherwise file `config/params.yml` is used.

Use python's option `-m` to run one of the scenarios. For instance:

````bash
python -m app.scenarios.scenario_1.scenario_1 -c "What is the name of proteine X"
````

Or with additional parameters:

````bash
python -m app.scenarios.scenario_1.scenario_1 \
    --params app/config/params_d2kab.yml \ 
    --question "What articles mention taxon 'wheat' (Triticum aestivum) and trait 'resistance to Leaf rust'?"
````

### Running scenarios using Langgraph Studio

You may use the [LangGraph Studio](https://studio.langchain.com/) inteface to interact with the application. Simply run the following command in the root directory.

````bash
langgraph dev
````
This will initialize LangGraph studio based on local file `langgraph.json` and the defauult configuration file `config/params.yml`.

Then select the scenario and fill in a NL question.
