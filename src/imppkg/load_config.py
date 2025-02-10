import yaml
from typing import Dict, Any


def load_config(config_path: str, active_tag: str) -> dict:
    """
    Load the YAML configuration file and return the config for the active tag.

    Args:
        config_path (str): Path to the YAML file.
        active_tag (str): The current tag to load (e.g., 'kopparberg').

    Returns:
        dict: Combined configuration for the active tag.
    """
    with open(config_path, "r") as file:
        config: Dict[str, Any] = yaml.safe_load(file)  # YAML file contents as a dictionary.

    # Get base settings
    base_config: Dict[str, Any] = config["base"]

    # Get tag-specific settings
    tag_config: Dict[str, Any] = config["tags"].get(active_tag)  # Might be None if missing.
    if not tag_config:
        raise ValueError(f"Tag '{active_tag}' not found in configuration.")

    # Combine base and tag-specific settings
    combined_config: Dict[str, Any] = {**base_config, **tag_config}
    return combined_config


# Test the loader
if __name__ == "__main__":
    # Path to the YAML file
    config_path: str = "config.yaml"

    # Set the active tag (e.g., 'kopparberg')
    active_tag: str = "kopparberg"

    # Load the configuration
    config: Dict[str, Any] = load_config(config_path, active_tag)

    # Access combined configuration values
    database_path: str = config["database_path"]
    collection_name: str = config["collection_name"]
    document_column: str = config["document_column"]
    metadata_columns: list = config["metadata_columns"]  # Assuming it's a list.
    persist_directory: str = config["persist_directory"]
    type_filter: str = config["type"]
    value_filter: str = config["value"]

    # Print the configuration for testing
    print("Database Path:", database_path)
    print("Collection Name:", collection_name)
    print("Document Column:", document_column)
    print("Metadata Columns:", metadata_columns)
    print("Persist Directory:", persist_directory)
    print(f"SQL Query: SELECT * FROM rosenberg WHERE {type_filter} = '{value_filter}'")
