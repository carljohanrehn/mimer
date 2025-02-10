import yaml


# Helper function to create a single tag dictionary
def create_tag(
        county: dict[str, str],
        base_directory: str
) -> dict[str, str]:
    """
    Transform a single county dictionary into its YAML representation.

    Args:
        county (dict[str, str]): A dictionary containing 'name' and 'id' keys.
        base_directory (str): The base directory for persisting data.

    Returns:
        dict[str, str]: A dictionary representing the YAML structure for the county.
    """
    tag: dict[str, str] = {
        'type': 'Län',
        'value': county['name'],
        'collection_name': f'rosenberg_{county["id"]}',
        'persist_directory': f'{base_directory}/{county["id"]}',
        'sql_query': f"SELECT * FROM rosenberg WHERE Län = '{county['name']}'"
    }
    return tag


# Functional-style YAML generation
def generate_yaml_from_counties(
        counties: list[dict[str, str]],
        base_directory: str = 'chroma/rosenberg/län'
) -> str:
    """
    Generate a YAML structure for a list of counties as dicts, in a functional style.

    Args:
        counties (list[dict[str, str]]): A list of county dictionaries, each with 'name' and 'id' keys.
        base_directory (str): The base directory for persisting data.

    Returns:
        str: YAML string representation of the generated structure.
    """
    # Use a dictionary comprehension to create all tags in one expression
    tags: dict[str, dict[str, str]] = {
        county['id']: create_tag(county, base_directory) for county in counties
    }

    # Convert the tags structure into a YAML-compatible dictionary
    yaml_structure: dict[str, dict[str, dict[str, str]]] = {'tags': tags}

    # Dump the dictionary to a YAML string
    return yaml.dump(yaml_structure, allow_unicode=True, sort_keys=False)


# Example Usage
if __name__ == '__main__':
    # Input: List of counties as dicts (name = display name, id = identifier)
    counties: list[dict[str, str]] = [
        {"name": "Blekinge län", "id": "blekinge"},
        {"name": "Gotlands län", "id": "gotland"},
        {"name": "Gävleborgs län", "id": "gavleborg"},
        {"name": "Göteborgs och Bohus län", "id": "goteborg_bohus"},
        {"name": "Hallands län", "id": "halland"},
        {"name": "Jämtlands län", "id": "jamtland"},
        {"name": "Jönköpings län", "id": "jonkoping"},
        {"name": "Kalmar län", "id": "kalmar"},
        {"name": "Kopparbergs län", "id": "kopparberg"},
        {"name": "Kristianstads län", "id": "kristianstad"},
        {"name": "Kronobergs län", "id": "kronoberg"},
        {"name": "Malmöhus län", "id": "malmohus"},
        {"name": "Norrbottens län", "id": "norrbotten"},
        {"name": "Skaraborgs län", "id": "skaraborg"},
        {"name": "Stockholms län", "id": "stockholm"},
        {"name": "Södermanlands län", "id": "sodermanland"},
        {"name": "Uppsala län", "id": "uppsala"},
        {"name": "Värmlands län", "id": "varmland"},
        {"name": "Västerbottens län", "id": "vasterbotten"},
        {"name": "Västernorrlands län", "id": "vasternorrland"},
        {"name": "Västmanlands län", "id": "vastmanland"},
        {"name": "Älvsborgs län", "id": "alvsborg"},
        {"name": "Örebro län", "id": "orebro"},
        {"name": "Östergötlands län", "id": "ostergotland"}
    ]

    # SELECT DISTINCT "Län" FROM rosenberg ORDER BY "Län" ASC;
    #
    # Blekinge län
    # Gotlands län
    # Gävleborgs län
    # Göteborgs och Bohus län
    # Hallands län
    # Jämtlands län
    # Jönköpings län
    # Kalmar län
    # Kopparbergs län
    # Kristianstads län
    # Kronobergs län
    # Malmöhus län
    # Norrbottens län
    # Skaraborgs län
    # Stockholms län
    # Södermanlands län
    # Uppsala län
    # Värmlands län
    # Västerbottens län
    # Västernorrlands län
    # Västmanlands län
    # Älvsborgs län
    # Örebro län
    # Östergötlands län

    # Generate YAML
    generated_yaml: str = generate_yaml_from_counties(counties)

    # Print or save the generated YAML structure
    print(generated_yaml)

    # Optionally, write it to a file
    with open('../config/generated_tags.yaml', 'w') as file:
        file.write(generated_yaml)
