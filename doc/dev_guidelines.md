# Development Guidelines

To ensure that all contributors are aligned and to facilitate smooth integration of contributions, we kindly ask that developers adhere to the following guidelines:

## Branches

We use the `dev` branch for pushing contributions. Please create your own branch (either user-centric or feature-centric) and do a pull request to the ```dev``` branch when ready for reviewing.
Branch `master` shall be updated when necessary.

## Code Formatting 

To maintain a unified code style across our project, we adhere to the PEP8 convention. This style guide helps in keeping code readable and maintainable. Here's how to ensure your code meets these standards:

- **Black Formatter**: in VSCode, install Black Formatter from the VSCode extensions marketplace.
    Right-click inside any Python file and select Format Document to automatically format your code.

## Code Documentation

- **Google Docstring Format**: All documentation for classes and functions should be written following the Google Docstring format. This format is both natural language and supports automatic documentation generation tools. The documentation is also parsed by the LLM to know about class/function signature, so natural language is more indicated.

- **Mintlify Doc Writer for VSCode**: Mintlify Doc Writer automates the creation of docstrings. To use this extension effectively:
    Install Mintlify Doc Writer from the VSCode extensions marketplace.
    In the extension's settings, set the docstring format to Google.
    To generate a docstring for a class or function, simply right-click on the code element and select the Generate Documentation option.
    **Important**: generated documentation may be very simple or useless. Carefully review and adjust the generated docstrings as necessary to provide relevant information abd accurately reflect the code’s purpose and behavior.

## Logging

Logging configuration is described in file `app/config/logging.yml`. This setup centralizes logging behavior across all scripts.


#### Integrating Logging into Your Scripts

To leverage logging setup, please incorporate the following code at the beginning of each Python script:

```python
from app.core.utils.logger_manager import setup_logger

logger = setup_logger(__package__, __file__)
```

#### Usage Recommendations

**Use Logging, not Print**: For any output meant for debugging or information tracking, use the logger object, **do NOT use the print function**.

**Logging Levels**: use the appropriate level when emitting log messages:
- logger.DEBUG: Detailed information, typically of interest only when diagnosing problems.
- logger.INFO: Confirmation that things are working as expected.
- logger.WARNING: Indicates a deviation from the norm but doesn't prevent the program from working
- logger.ERROR: Issues that prevent certain functionalities from operating correctly but do not necessarily affect the overall application's ability to run.
- logger.CRITICAL: These are used for errors that require immediate attention, such as a complete system failure or a critical resource unavailability.

**Logs Outputs**

The configuration supports writing log messages to two destinations:
- Console: this setup is intended for development.
- File: includes timestamped traces

By default, log files are located within the `./logs` directory. This can be changed in the file handler in `app/config/logging.yml`.

**Run time**

Change the log level to DEBUG, INFO, WARNING etc. in the handlers' configuration in `app/config/logging.yml`.
