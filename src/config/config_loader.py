# src/config/config_loader.py
import yaml
import os

def load_config(config_path=None):
    """
    Loads the YAML configuration file.
    :param config_path: Optional override path to config file.
    :return: Parsed config dictionary.
    """
    if config_path is None:
        # Default relative path from project root
        config_path = os.path.join(os.path.dirname(__file__), '../../config/config.yaml')
        config_path = os.path.abspath(config_path)

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found at {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML config: {e}")
