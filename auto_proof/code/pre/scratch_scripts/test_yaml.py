import yaml

def parse_yaml_to_dict(file_path):
    """Parses a YAML file into a Python dictionary.

    Args:
        file_path (str): The path to the YAML file.

    Returns:
        dict: A Python dictionary representing the YAML content, or None if an error occurs.
    """
    try:
        with open(file_path, 'r') as f:
            yaml_data = yaml.safe_load(f)  # Use safe_load for security
            return yaml_data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML: {e}")
        return None

# Example usage:
if __name__ == "__main__":
    yaml_file = 'config.yaml'  # Replace with your YAML file name

    # Create a sample YAML file for demonstration
    sample_yaml_content = """
    default_settings: &default
      timeout: 30
      retries: 3
      logging:
        level: "INFO"

    server1:
      <<: *default
      host: "server1.example.com"
      port: &port 8080
      timeout: 

    server2:
      <<: *default
      host: "server2.example.com"
      port: 8081
    """
    with open(yaml_file, 'w') as f:
        f.write(sample_yaml_content)

    config_dict = parse_yaml_to_dict(yaml_file)

    if config_dict:
        print("Parsed YAML Dictionary:")
        import pprint
        pprint.pprint(config_dict)

        # Accessing values from the dictionary
        print(f"\nServer 1 Host: {config_dict['server1']['host']}")
        print(f"Server 1 Timeout: {config_dict['server1']['timeout']}") # Note the overridden value
        print(f"Server 1 Retries (inherited): {config_dict['server1']['retries']}")
        print(f"Default Logging Level: {config_dict['default_settings']['logging']['level']}")

