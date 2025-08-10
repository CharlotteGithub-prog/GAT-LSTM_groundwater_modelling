# Imports
import os
import re
import yaml
import logging

# Set up file logger
logger = logging.getLogger(__name__)

# Define project config file loader function
def load_project_config(config_path="config/eden_project_config.yaml"):
    """
    Loads the project configuration from a YAML file.

    Args:
        config_path (str): The path to the configuration file, relative to the project's
                           working directory (project root: Dissertation_Code).

    Returns:
        dict: The loaded configuration as a dictionary.
    """
    # Join the cwd with the relative config_path.
    full_config_path = os.path.abspath(os.path.join(os.getcwd(), config_path))

    # Log error if config file not found
    if not os.path.exists(full_config_path):
        logger.error(f"Configuration file not found at: {full_config_path}")
        logger.error(f"Current working directory: {os.getcwd()}")
        raise FileNotFoundError(f"Configuration file not found at: {full_config_path}. "
                                f"Please ensure the working directory is the project root.")

    # Log when config file successfully loaded
    logger.info(f"Loading configuration from: {full_config_path}")
    with open(full_config_path, 'r') as file:
        return yaml.safe_load(file)

# Build a regex that matches exactly the keys passed, e.g.. {results_root} or {raw_data_root}
def deep_format(obj, **replacements):
    if replacements:
        pattern = re.compile(r'\{(' + '|'.join(map(re.escape, replacements.keys())) + r')\}')
    else:
        pattern = None

    def _fmt(x):
        if isinstance(x, str) and pattern:
            return pattern.sub(lambda m: str(replacements[m.group(1)]), x)
        if isinstance(x, dict):
            return {k: _fmt(v) for k, v in x.items()}
        if isinstance(x, list):
            return [_fmt(v) for v in x]
        return x

    return _fmt(obj)

def expanduser_tree(obj):
    if isinstance(obj, str):
        return os.path.expanduser(obj)
    if isinstance(obj, dict):
        return {k: expanduser_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [expanduser_tree(v) for v in obj]
    return obj
