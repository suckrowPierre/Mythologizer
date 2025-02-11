import json
from typing import Tuple, List, Optional, Dict
from mythologizer.sandbox import Sandbox

def get_sandbox_from_config_json(json_str: str) -> Tuple[int, List[Dict[str, str]], Optional[int], Optional[int]]:

    ### TODO!!!!!! add file_path
    """
        Parse the simulation configuration JSON and return a tuple.

        :param json_str: JSON string containing the configuration.
        :return: Tuple with population, culture list, epochs_to_simulate (optional), and seed (optional).
        :raises ValueError: If required fields are missing or invalid.
        """
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")

    if "population" not in data or not isinstance(data["population"], int):
        raise ValueError("Missing or invalid 'population' field (must be an integer).")

    if "cultures" not in data or not isinstance(data["culture"], list):
        raise ValueError("Missing or invalid 'culture' field (must be a list of objects).")

    for item in data["culture"]:
        if not isinstance(item, dict) or "name" not in item or "description" not in item:
            raise ValueError("Each item in 'culture' must be a dictionary with 'name' and 'description' fields.")

    # TODO add validation for example populAtion cant be equal under 0 ? or add validation in the classes. This seems smarter

    population = int(data["population"])
    cultures = data["cultures"]
    epochs_to_simulate = int(data.get("epochs_to_simulate"))
    seed = data.get("seed")

    sandbox = Sandbox(seed=seed)
    sandbox.culture_registry.add(cultures)
    sandbox.randomly_populate(population)

    # TODO: DO WE WANT TO RUN THIS BEFORE RETURNING ????
    sandbox.simulate(epochs_to_simulate)
    return sandbox


def load_sandbox_from_config_file(path: string):
    # load json from file and parse



def load_sandbox_checkpoint(path: string):

    # read file
    # parse files
    parsed_data = None
    sandbox = Sandbox(...parsed_data)
    return sandbox