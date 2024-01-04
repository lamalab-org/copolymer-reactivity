import yaml
from pathlib import Path
from typing import Union
from typing import List
from copolextractor.utils import name_to_smiles


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


def _extract_reactions(data: dict) -> list:
    reactions = []
    for reaction in data["reaction"]:
        reactions.append(reaction)
    return reactions


def _compare_monomers(test_monomers: List[str], model_monomers: List[str]) -> bool:
    if set(test_monomers) == set(model_monomers):
        return True
    else:
        test_monomer_smiles = [name_to_smiles(monomer) for monomer in test_monomers]
        model_monomer_smiles = [name_to_smiles(monomer) for monomer in model_monomers]
        return set(test_monomer_smiles) == set(model_monomer_smiles)


def find_matching_reaction(data: dict, monomers: List[str]) -> int:
    """Find matching reaction in data

    Args:
        data (dict): Extracted data
        monomers (List[str]): Expected monomers

    Raises:
        ValueError: Multiple matching reactions found

    Returns:
        int: Index of matching reaction
    """
    matching_rxn_ids = []
    rxns = _extract_reactions(data)
    for i, rxn in enumerate(rxns):
        if _compare_monomers(monomers, rxn["monomers"]):
            matching_rxn_ids.append(i)

    if len(matching_rxn_ids) == 0:
        return None
    elif len(matching_rxn_ids) > 1:
        raise ValueError("Multiple matching reactions found")
    else:
        return matching_rxn_ids[0]

def find_matching_combination(combination, polymerization_type, solvent, temperature, method):
    ...


def compare_number_of_reactions(test_file: Union[str, Path], model_file: Union[str, Path]) -> dict:
    test_data = load_yaml(test_file)
    model_data = load_yaml(model_file)
    test_number_of_reactions = get_number_of_reactions(test_data)
    model_number_of_reactions = get_number_of_reactions(model_data)
    output = {
        "test": test_number_of_reactions,
        "model": model_number_of_reactions,
        "equal": test_number_of_reactions == model_number_of_reactions,
        "mae": abs(test_number_of_reactions - model_number_of_reactions),
    }
    return output
