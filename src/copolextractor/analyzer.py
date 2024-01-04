import yaml
from pathlib import Path
from typing import Union

def load_yaml(file_path: Union[str, Path]) -> dict:
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)
    return data 

def get_number_of_reactions(data: dict) -> int:
    return len(data["reaction"])

def get_total_number_of_combinations(data: dict) -> int:
    combinations_count = 0
    for reaction in data["reaction"]:
        combinations_count += len(reaction["combinations"])
    return combinations_count


def compare_number_of_reactions(test_file: Union[str, Path], model_file: Union[str, Path]) -> dict:
    test_data = load_yaml(test_file)
    model_data = load_yaml(model_file)
    test_number_of_reactions = get_number_of_reactions(test_data)
    model_number_of_reactions = get_number_of_reactions(model_data)
    return {"test": test_number_of_reactions, "model": model_number_of_reactions, "equal": test_number_of_reactions == model_number_of_reactions, "mae": abs(test_number_of_reactions - model_number_of_reactions)}