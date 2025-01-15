import argparse
import yaml

def load_yaml_into_namespace(
    yaml_file: str, namespace: argparse.Namespace
) -> argparse.Namespace:
    """
    Load a YAML file and merge its content into the given argparse Namespace.

    Args:
        yaml_file (str): Path to the YAML file.
        namespace (argparse.Namespace): The current Namespace object.

    Returns:
        argparse.Namespace: Updated Namespace with the values from the YAML file.
    """
    # Load the YAML file
    with open(yaml_file, "r") as file:
        yaml_data = yaml.safe_load(file)  # Parse the YAML content as a dictionary

    # Merge the YAML data into the Namespace
    namespace_dict = vars(namespace)  # Convert Namespace to dictionary
    namespace_dict.update(yaml_data)  # Update with YAML data
    print(f"Loaded YAML file: {yaml_file} as config")
    return argparse.Namespace(**namespace_dict)  # Convert back to Namespace