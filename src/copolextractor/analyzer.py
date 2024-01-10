import yaml
from pathlib import Path
from typing import Union, List, Tuple
from src.copolextractor.utils import name_to_smiles
from thefuzz import fuzz


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


def find_matching_reaction(model_data: dict, test_data: dict) -> int:
    """Find matching reaction in data

    Args:
        model_data (dict): Extracted data
        test_data (dict): test data

    Raises:
        ValueError: Multiple matching reactions found

    Returns:
        int: Index of matching reaction
    """
    matching_rxn_ids = []
    rxns_model = _extract_reactions(model_data)
    rxns_test = _extract_reactions(test_data)
    for i, (rxn_model, rxn_test) in enumerate(zip(rxns_model, rxns_test)):
        if _compare_monomers(rxn_model['monomers'], rxn_model["monomers"]):
            matching_rxn_ids.append(i)

    if len(matching_rxn_ids) == 0:
        return 1
    elif len(matching_rxn_ids) > 1:
        raise ValueError("Multiple matching reactions found")
    else:
        return 0


def find_matching_combination(combination: List[dict], polymerization_type: str, solvent: str, temperature: Union[str, float, int], method: str) -> Tuple[int, float]:
    # We need to do fuzzy matching here and take the best match but also return the confidence
    # of the match
    # first we check if we are lucky and find an exact match, then confidence would
    solvent_smiles = name_to_smiles(solvent)
    matching_idxs = []
    for i, comb in enumerate(combination):
        if (
            comb["polymerization_type"] == polymerization_type
            and name_to_smiles(comb["solvent"]) == solvent_smiles
            and comb["temperature"] == temperature
            and comb["method"] == method
        ):
            matching_idxs.append(i)
    
    if len(matching_idxs) == 1:
        return matching_idxs[0], 1
    elif len(matching_idxs) > 1:
        raise ValueError("Multiple matching combinations found")

    # if we are not lucky we need to do fuzzy matching
    combination_string = f"{polymerization_type} {solvent} {temperature} {method}"
    combination_strings = [
        f"{comb['polymerization_type']} {comb['solvent']} {comb['temperature']} {comb['method']}"
        for comb in combination
    ]
    scores = [fuzz.ratio(combination_string, comb_string)/100 for comb_string in combination_strings]
    best_score = max(scores)
    best_score_index = scores.index(best_score)
    return best_score_index, best_score


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
