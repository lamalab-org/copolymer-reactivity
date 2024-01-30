import yaml
from pathlib import Path
from typing import Union, List, Tuple
from copolextractor.utils import name_to_smiles
from thefuzz import fuzz
from pint import UnitRegistry

ureg = UnitRegistry()


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


def extract_combinations(data: list) -> list:
    combinations = []
    for combination in data:
            combinations.append(combination)
    return combinations


def _extract_reactions(data: dict) -> list:
    reactions = []
    for reaction in data["reaction"]:
        reactions.append(reaction)
    return reactions


def _extract_monomers(data: dict) -> list:
    monomers = []
    for reaction in data['reaction']:
            monomers.append(reaction['monomers'])
    return monomers


def get_temp(data: dict):
    temp = data['temperature']
    temp_unit = data['temperature_unit']
    return temp, temp_unit


def convert_unit(temp1: int, temp2: int, unit1: str, unit2: str):
    temp1_unit = ureg.Quantity(temp1, ureg.parse_units(unit1))
    temp2_unit = ureg.Quantity(temp2, ureg.parse_units(unit2))
    if ureg.parse_units(unit1) is not ureg.degC:
            temp1_unit.ito(ureg.degC)
    if ureg.parse_units(unit2) is not ureg.degC:
            temp2_unit.ito(ureg.degC)
    return temp1_unit.magnitude, temp2_unit.magnitude


def get_metadata_polymerization(data: dict):
    temp = data['temperature']
    temp_unit = data['temperature_unit']
    method = data['method']
    type = data['polymerization_type']
    solvent = data['solvent']
    reaction_constants = data['reaction_constants']
    reaction_constant_confidence = data['reaction_constant_conf']
    determination_method = data['determination_method']
    return temp, temp_unit, method, type, solvent, reaction_constants, reaction_constant_confidence, determination_method


def _get_sequence_of_monomers(test_monomers, model_monomers):
    if set(test_monomers) == set(model_monomers):
        if test_monomers[0] == model_monomers[0]:
            sequence_change = 0
        else:
            sequence_change = 1
    else:
        test_monomer_smiles = [name_to_smiles(monomer) for monomer in test_monomers]
        model_monomer_smiles = [name_to_smiles(monomer) for monomer in model_monomers]
        if test_monomer_smiles[0] == model_monomer_smiles[0]:
            sequence_change = 0
        else:
            sequence_change = 1
    return sequence_change


def _compare_monomers(test_monomers: List[str], model_monomers: dict[str]) -> bool:
    print(test_monomers)
    print(model_monomers)
    if set(test_monomers) == set(model_monomers):
        return True
    else:
        test_monomer_smiles = [name_to_smiles(monomer) for monomer in test_monomers]
        model_monomer_smiles = [name_to_smiles(monomer) for monomer in model_monomers]
        return set(test_monomer_smiles) == set(model_monomer_smiles)


def find_matching_reaction(data1: dict, data2: dict):
    """Find matching reaction in data

    Args:
        data1 (dict): model data
        data2 (dict): test monomers

    Raises:
        ValueError: Multiple matching reactions found

    Returns:
        int: Index of matching test reaction
    """
    matching_rxn_ids = []
    monomers1 = _extract_monomers(data1)
    for i, rxn1 in enumerate(monomers1):
        if _compare_monomers(rxn1, data2):
                print('Matching monomers:', data2, rxn1)
                matching_rxn_ids.append(i)
                sequence = _get_sequence_of_monomers(rxn1, data2)

    if len(matching_rxn_ids) == 0:
        return None, None
    elif len(matching_rxn_ids) > 1:
        raise ValueError("Multiple matching reactions found")
    else:
        return matching_rxn_ids[0], sequence


def find_matching_combination(combination: List[dict], polymerization_type: str, solvent: str, method: str, determination_method: str) -> Tuple[int, float]:
    # We need to do fuzzy matching here and take the best match but also return the confidence
    # of the match
    # first we check if we are lucky and find an exact match, then confidence would
    solvent_smiles = name_to_smiles(solvent)
    matching_idxs = []
    for i, comb in enumerate(combination):
        if (
                comb["polymerization_type"] == polymerization_type
                and name_to_smiles(comb["solvent"]) == solvent_smiles
                and comb["method"] == method
                and comb['determination_method'] == determination_method
        ):
            print(i)
            print(solvent)
            print(comb["solvent"])
            matching_idxs.append(i)
    print(matching_idxs)
    if len(matching_idxs) == 1:
        return matching_idxs[0], 1
    elif len(matching_idxs) > 1:
        raise ValueError("Multiple matching combinations found")

    # if we are not lucky we need to do fuzzy matching
    combination_string = f"{polymerization_type} {solvent} {method} {determination_method}"
    combination_strings = [
        f"{comb['polymerization_type']} {comb['solvent']} {comb['method']} {comb['determination_method']}"
        for comb in combination
    ]
    scores = [fuzz.ratio(combination_string, comb_string) / 100 for comb_string in combination_strings]
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


def get_reaction_constant(reaction_constant: dict, reaction_const_conf: dict):
    reaction_constants = []
    reaction_constants_conf = []
    for constant in reaction_constant:
        reaction_constants.append(reaction_constant[constant])
    for conf in reaction_const_conf:
        reaction_constants_conf.append(reaction_const_conf[conf])
    return reaction_constants, reaction_constants_conf


def change_sequence(constant1: list, constant2: list):
    constant1[0], constant1[1] = constant1[1], constant1[0]
    constant2[0], constant2[1] = constant2[1], constant2[0]
    return constant1, constant2
