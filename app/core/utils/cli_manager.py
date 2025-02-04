import argparse


def setup_cli():
    parser = argparse.ArgumentParser(
        description="Process the scenario with the predifined or custom question and configuration."
    )
    parser.add_argument("-c", "--custom", type=str, help="Provide a custom question.")
    parser.add_argument(
        "-p", "--params", type=str, help="Provide a custom configuration path."
    )
    globals()["args"] = parser.parse_args()
