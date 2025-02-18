# Gen²KGBot - Generic Generative Knowledge Graph Robot

## Scenarios

- Scenario 1: simply ask the question to the LLM, to figure out what it "knows" about the topic. No query to the KG.
- Scenario 2: ask the model to create a SPARQL query without any information other than the user's question.
- Scenario 3: ask the model to create a SPARQL query based on the user's question, a list of classes related to the question, in the form (class name, label, description)
- Scenario 4: Scenario 3 + add in the context a description (in Turtle) of the prioperties used with the instances of selected classes. Triples of the form: `<class> <prop> <datatype or class>.`
- Scenario 5: Scenario 4 + retry mechanism if the generated SPQRQ query is not syntactically correct.
- Scenario 6: Scenario 6 + add in the context some example SPARQL queries related to the question.


## General information

We use the ```dev``` branch for pushing contributions. Please create your own branch like (either user-centric or feature-centric) and do a pull request to the ```dev``` branch when ready for reviewing.


## Environment setup

Conda is required for setting up the environment. For installation instructions, see: https://docs.conda.io/projects/conda/en/latest/user-guide/install/
1) Install conda
2) To install the environment from the `environment.yml` file, use the following command:
```sh
conda env create -f ./environment_{os}.yml
```

⚠️ Some packages are os and hardware dependent. If you have a problem creating the environment you can delete it and retry without the dependencies causing the problem.

☢️ To delete the environment use:
```sh
conda env remove --name kgbot-rag-backend
```

3) At the root of the repository, create the follwing directories:
    - `logs`: log files produces while running the application (see configuration in `app/config/logging.yml`)
    - `data`: descriptions extracted from the knowledge graph, as well as the embeddings thereof.
    - `tmp`: files containing the descripton of the knowledge graph

4) Install the [Ollama](https://github.com/ollama/ollama) application for your platform and, as a startup config, insttall the models `ollama3.2:1b`, `nomic-embed-text`
```sh
ollama pull ollama3.2:1b
ollama pull nomic-embed-text
```

⚠️ To run Ollama on Windows, you may need to install Microsoft Visual C++ 14+: install and run [Build Tools](https://visualstudio.microsoft.com/fr/visual-cpp-build-tools/) then install Desktop C++ Development with default packages.

5) Set up the following environment variables:
    - `LANGCHAIN_API_KEY`
    - `OPENAI_API_KEY` if you use OpenAI models


## Application Startup Instructions

The application has been structured as a module and adheres to the dot notation convention for Python imports. To import a module within the Python script, you can either use an absolute path (e.g., app.core.module1.module2) or a relative import (e.g., ..core.module1.module2).

### Launching the Application

It is recomanded to use the [LangGraph Studio](https://studio.langchain.com/) inteface to interact with the application. To achieve this, you simply run the following command in the root directory.

````bash
langgraph dev
````
This will initialize LangGraph studio based on local file langgraph.json.

Then you can select the scenario and input the question.

However, it is also possible to run each of the scenarios in the terminal. You should use the `-m` option from the Python command line interface, for example, to run scenario_6 with the default question use: 
````bash
python -m app.core.scenarios.scenario_1.scenario_1
````

Custom questions are allowed and can be asked with the following command:

````bash
python -m app.core.scenarios.scenario_1.scenario_1 -c "What is the name of protene X"
````

## Project Structure
- **Each scenario** have its own [subfolder](./app/core/scenarios) in `app.core.scenarios`
- **The notebooks** for creating the embeddings are in [the folder](./app/notebooks) `app.notebooks`


## Development Guidelines

To ensure that all contributors are aligned and to facilitate smoother integration of our work, we kindly ask that you adhere to the following guidelines:

**Documentation Standards**
- **Google Docstring Format**: All documentation for classes and functions should be written following the Google Docstring format. This format is both natural language and supports automatic documentation generation tools. The documentation is also parsed by the LLM to know about class/function signature, so natural language is more indicated.

- **Mintlify Doc Writer for VSCode**: To simplify the process of writing docstrings, we recommend using the Mintlify Doc Writer extension available in Visual Studio Code. This tool automates the creation of docstrings. To use this extension effectively:
    Install Mintlify Doc Writer from the VSCode extensions marketplace.
    In the extension's settings, set the docstring format to Google.
    To generate a docstring for a class or function, simply right-click on the code element and select the Generate Documentation option.
    Review and adjust the generated docstrings as necessary to accurately reflect the code’s purpose and behavior.

**Code Formatting**

To maintain a unified code style across our project, we adhere to the PEP8 convention. This style guide helps in keeping our code readable and maintainable. Here's how to ensure your code meets these standards:

- **Black Formatter** in VSCode: The easiest way to format your code according to PEP8 is by using the Black Formatter extension in Visual Studio Code. Here’s how to use it:
    Install Black Formatter from the VSCode extensions marketplace.
    Right-click inside any Python file and select Format Document to automatically format your code.


## Logging guidelines

These guidelines will help us efficiently track application behavior, debug issues, and understand application flow.

**Configuration**

Our logging configuration is centralized in an INI file located at app/config/logging.yml. This setup allows us to manage logging behavior across all scripts from a single location.


**Integrating Logging into Your Scripts**

To leverage logging setup, please incorporate the following code at the beginning of each Python script:

```python
from app.core.utils.logger_manager import setup_logger

logger = setup_logger(__package__, __file__)
```

**Usage Recommendations**

**Prefer Logging Over Print**: For any output meant for debugging or information tracking, use the logger object instead of the print function. 

**Logging Levels**: Please use the appropriate level when emitting log messages:
- logger.DEBUG: Detailed information, typically of interest only when diagnosing problems.
- logger.INFO: Confirmation that things are working as expected.
- logger.WARNING: Indicates a deviation from the norm but doesn't prevent the program from working
- logger.ERROR: Issues that prevent certain functionalities from operating correctly but do not necessarily affect the overall application's ability to run.
- logger.CRITICAL: These are used for errors that require immediate attention, such as a complete system failure or a critical resource unavailability.

**Logs Outputs**

Our configuration supports outputting log messages to two destinations:

- Console: Log messages at the INFO level and above will be outputted to the console. This setup is intended for general monitoring and quick diagnostics.
- File: A more detailed log, including messages at the DEBUG level and above, is written to a file. 

The log files are located within the /logs directory.
