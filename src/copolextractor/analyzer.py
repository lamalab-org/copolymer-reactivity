from pathlib import Path
from typing import Union, List, Tuple
from copolextractor.utils import name_to_smiles, load_yaml
from thefuzz import fuzz
from pint import UnitRegistry


def get_number_of_reactions(data: dict) -> int:
    return len(data["reactions"])


def get_total_number_of_reaction_conditions(data: dict) -> int:
    reaction_conditions_count = 0
    if isinstance(data["reactions"], list):
        for reaction in data["reactions"]:
            if "reaction_conditions" in reaction:
                reaction_conditions_count += len(reaction["reaction_conditions"])
    elif "reaction_conditions" in data["reactions"]:
        reaction_conditions_count = len(data["reactions"]["reaction_conditions"])
    return reaction_conditions_count


def extract_reaction_conditions(data: list) -> list:
    reaction_conditions = []
    for reaction_condition in data:
        reaction_conditions.append(reaction_condition)
    return reaction_conditions


def _extract_reactions(data: dict) -> list:
    reactions = []
    for reaction in data["reactions"]:
        reactions.append(reaction)
    return reactions


def _extract_monomers(data: dict) -> list:
    monomers = []
    if isinstance(data["reactions"], list):
        for reaction in data["reactions"]:
            if "monomers" in reaction:
                monomers.extend(reaction["monomers"])
    elif "monomers" in data["reactions"]:
        monomers.extend(data["reactions"]["monomers"])
    return monomers


def get_temp(data: list, index: int) -> tuple:
    specific_cond = data[index]
    temp = specific_cond["temperature"]
    temp_unit = specific_cond["temperature_unit"]
    return temp, temp_unit


def get_solvent(data: list, index: int) -> tuple:
    specific_comb = data[index]
    solvent = specific_comb["solvent"]
    solvent_smiles = name_to_smiles(solvent)
    return solvent, solvent_smiles


ureg = UnitRegistry()


def convert_unit(temp1, unit1):
    """
    Converts a temperature to Celsius if the unit is not already Celsius.
    Handles 'na' entries by returning None.
    """
    # Check for 'na' entries
    if temp1 == 'na' or unit1 == 'na':
        return None

    try:
        # Parse and convert the temperature
        temp1_unit = ureg.Quantity(temp1, ureg.parse_units(unit1))
        if ureg.parse_units(unit1) is not ureg.degC:
            temp1_unit.ito(ureg.degC)
        return temp1_unit.magnitude
    except (AttributeError, ValueError, TypeError, KeyError) as e:
        # Log the error details
        print("Error converting units:", e)
        return None



def get_metadata_polymerization(data: dict):
    temp = data["temperature"]
    temp_unit = data["temperature_unit"]
    method = data["method"]
    polymer_type = data["polymerization_type"]
    solvent = data["solvent"]
    reaction_constants = data["reaction_constants"]
    reaction_constant_confidence = data["reaction_constant_conf"]
    determination_method = data["determination_method"]
    return (
        temp,
        temp_unit,
        method,
        polymer_type,
        solvent,
        reaction_constants,
        reaction_constant_confidence,
        determination_method,
    )


def get_sequence_of_monomers(test_monomers, model_monomers):
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


def _compare_monomers(model_monomers: List[str], test_monomers: List[str]) -> bool:
    print(f"test monomers: {test_monomers} vs model monomers: {model_monomers}")
    if set(test_monomers) == set(model_monomers):
        return True
    else:
        test_monomer_smiles = [name_to_smiles(monomer) for monomer in test_monomers]
        model_monomer_smiles = [name_to_smiles(monomer) for monomer in model_monomers]
        return set(test_monomer_smiles) == set(model_monomer_smiles)


def find_matching_reaction(data1: dict, data2: list):
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
    if isinstance(data1.get("reactions"), list):
        matching_rxn_ids = []
        for i, rxn in enumerate(data1["reactions"]):
            monomers1 = rxn["monomers"]
            if _compare_monomers(monomers1, data2):
                matching_rxn_ids.append(i)

        if len(matching_rxn_ids) == 0:
            return None
        elif len(matching_rxn_ids) > 1:
            raise ValueError("Multiple matching reactions found")
        else:
            return matching_rxn_ids[0]
    else:
        monomers1 = data1["reactions"]["monomers"]
        if _compare_monomers(monomers1, data2):
            return 0
        else:
            return None


def find_matching_reaction_conditions(
    reaction_conditions: List[dict],
    solvent: str,
    temperature: int,
    temp_unit: str,
    polymerization_type: str,
    method: str,
    determination_method: str,
) -> Tuple[int, float]:
    # We need to do fuzzy matching here and take the best match but also return the confidence
    # of the match
    # first we check if we are lucky and find an exact match, then confidence would
    matching_idxs = []
    for i, comb in enumerate(reaction_conditions):
        if comb["temperature"] != "NA" and comb["temperature_unit"] != "NA":
            temp = convert_unit(temperature, temp_unit)
            temperature_model = convert_unit(
                comb["temperature"], comb["temperature_unit"])
        else:
            temperature_model = comb["temperature"]
            temp = temperature
        if (
            name_to_smiles(comb["solvent"]) == name_to_smiles(solvent)
            and temperature_model == temp
            and comb["polymerization_type"] == polymerization_type
            and comb["method"] == method
            and comb["determination_method"] == determination_method
        ):
            matching_idxs.append(i)
    if len(matching_idxs) == 1:
        return matching_idxs[0], 1
    elif len(matching_idxs) > 1:
        raise ValueError("Multiple matching reaction_conditions found")

    # if we are not lucky we need to do fuzzy matching
    reaction_conditions_string = (
        f"{solvent} {temperature} {polymerization_type} {method} {determination_method}"
    )
    reaction_conditions_strings = [
        f"{comb['solvent']} {comb['temperature']} {comb['polymerization_type']} {comb['method']} {comb['determination_method']}"
        for comb in reaction_conditions
    ]
    scores = [
        fuzz.ratio(reaction_conditions_string, comb_string) / 100
        for comb_string in reaction_conditions_strings
    ]
    best_score = max(scores)
    best_score_index = scores.index(best_score)
    return best_score_index, best_score


def compare_smiles(smiles1: str, smiles2: str):
    return int(smiles1 != smiles2)


def compare_number_of_reactions(
    test_file: Union[str, Path], model_file: Union[str, Path]
) -> dict:
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


def get_reaction_constant(data: list, index: int) -> tuple:
    specific_comb = data[index]
    reaction_const = specific_comb["reaction_constants"]
    reaction_const_conf = specific_comb["reaction_constant_conf"]
    reaction_const, reaction_const_conf = get_reaction_const_list(
        reaction_const, reaction_const_conf
    )
    return reaction_const, reaction_const_conf


def get_reaction_const_list(reaction_const: list, reaction_const_conf: list):
    reaction_constants = []
    reaction_constants_conf = []
    if reaction_const is None:
        reaction_constants = [None, None]
    else:
        reaction_constants = list(reaction_const.values())
    if reaction_const_conf is None:
        reaction_constants_conf = [None, None]
    else:
        reaction_constants_conf = [
            None if value == "None" else value for value in reaction_const_conf.values()
        ]

    return reaction_constants, reaction_constants_conf


def change_sequence(constant1: list, constant2: list):
    print(constant1, constant2)
    if constant1 and constant2:
        constant1[0], constant1[1] = constant1[1], constant1[0]
        constant2[0], constant2[1] = constant2[1], constant2[0]
    return constant1, constant2


def average(const):
    if len(const) != 0:
        average_value = sum(const) / len(const)
        return average_value


def count_na_values(data, null_value="na"):
    """
    Count occurrences of a specific null value (e.g., 'na') in nested dictionaries or lists.

    Args:
        data: The data structure (dict, list, or scalar) to search.
        null_value (str): The value to count as 'null', defaults to "na".

    Returns:
        int: The count of occurrences of the null value.
    """
    null_count = 0

    # If data is a dictionary, iterate through its values
    if isinstance(data, dict):
        for key, value in data.items():
            null_count += count_na_values(value, null_value)

    # If data is a list, iterate through its items
    elif isinstance(data, list):
        for item in data:
            null_count += count_na_values(item, null_value)

    # If data matches the null_value directly, increment the count
    elif isinstance(data, str):
        if data == null_value:
            null_count += 1

    return null_count


def count_total_entries(data):
    """
    Count the total number of scalar entries in nested dictionaries or lists.

    Args:
        data: The data structure (dict, list, or scalar) to search.

    Returns:
        int: The total number of scalar entries.
    """
    count = 0

    # If data is a dictionary, iterate through its values
    if isinstance(data, dict):
        for value in data.values():
            count += count_total_entries(value)

    # If data is a list, iterate through its items
    elif isinstance(data, list):
        for item in data:
            count += count_total_entries(item)

    # Count scalar entries (non-dict, non-list)
    else:
        count += 1

    return count


def calculate_rate(na_count, total_count):
    """
    Calculate the rate of 'na' values in the total count.

    Args:
        na_count (int): Number of 'na' occurrences.
        total_count (int): Total number of entries.

    Returns:
        float: Rate of 'na' values.
    """
    if total_count != 0:
        return na_count / total_count
    else:
        return 0.0  # Return 0 if total_count is zero to avoid division by zero.

