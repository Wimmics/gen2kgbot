from pathlib import Path
import yaml


def get_yml_config():
    # if args.params:
    #     config_path = args.params
    # else:
    # # Resolve the path to the configuration file
    config_path = (
        Path(__file__).resolve().parent.parent.parent / "config" / "params.yml"
    )

    # # Configure logging
    with open(config_path, "rt") as f:
        return yaml.safe_load(f.read())
