from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from pint import UnitRegistry
from thefuzz import fuzz

from copolextractor.utils import load_yaml, name_to_smiles


def get_number_of_reactions(data: dict) -> int:
    """Return the number of reactions stored under the "reactions" key of an extraction record.

    Args:
        data: Extraction record containing a "reactions" key (list or dict).

    Returns:
        The number of reactions.
    """
    return len(data["reactions"])


def get_total_number_of_reaction_conditions(data: dict) -> int:
    """Count the total number of reaction-condition entries across all reactions.

    Args:
        data: Extraction record whose "reactions" value is either a list of reaction
            dicts or a single reaction dict, each optionally containing a
            "reaction_conditions" list.

    Returns:
        The summed number of reaction-condition entries.
    """
    reaction_conditions_count = 0
    if isinstance(data["reactions"], list):
        for reaction in data["reactions"]:
            if "reaction_conditions" in reaction:
                reaction_conditions_count += len(reaction["reaction_conditions"])
    elif "reaction_conditions" in data["reactions"]:
        reaction_conditions_count = len(data["reactions"]["reaction_conditions"])
    return reaction_conditions_count


def extract_reaction_conditions(data: List[dict]) -> List[dict]:
    """Return a shallow copy of a list of reaction-condition entries.

    Args:
        data: List of reaction-condition dicts.

    Returns:
        A new list containing the same reaction-condition entries.
    """
    reaction_conditions = []
    for reaction_condition in data:
        reaction_conditions.append(reaction_condition)
    return reaction_conditions


def _extract_reactions(data: dict) -> List[dict]:
    """Return the list of reaction dicts stored under the "reactions" key.

    Args:
        data: Extraction record with a "reactions" key holding an iterable of reaction dicts.

    Returns:
        A new list containing the reaction dicts.
    """
    reactions = []
    for reaction in data["reactions"]:
        reactions.append(reaction)
    return reactions


def _extract_monomers(data: dict) -> List[str]:
    """Collect all monomer names referenced across the reactions of an extraction record.

    Args:
        data: Extraction record whose "reactions" value is either a list of reaction
            dicts or a single reaction dict, each optionally containing a "monomers" list.

    Returns:
        A flat list of monomer names (duplicates possible if reused across reactions).
    """
    monomers = []
    if isinstance(data["reactions"], list):
        for reaction in data["reactions"]:
            if "monomers" in reaction:
                monomers.extend(reaction["monomers"])
    elif "monomers" in data["reactions"]:
        monomers.extend(data["reactions"]["monomers"])
    return monomers


def get_temp(data: List[dict], index: int) -> Tuple[Any, Any]:
    """Return the temperature and its unit for one reaction-condition entry.

    Args:
        data: List of reaction-condition dicts, each with "temperature" and
            "temperature_unit" keys.
        index: Position of the entry to read within `data`.

    Returns:
        A (temperature, temperature_unit) tuple, values as stored (may be "na").
    """
    specific_cond = data[index]
    temp = specific_cond["temperature"]
    temp_unit = specific_cond["temperature_unit"]
    return temp, temp_unit


def get_solvent(data: List[dict], index: int) -> Tuple[str, Optional[str]]:
    """Return the solvent name and its SMILES representation for one reaction-condition entry.

    Args:
        data: List of reaction-condition dicts, each with a "solvent" key.
        index: Position of the entry to read within `data`.

    Returns:
        A (solvent_name, solvent_smiles) tuple. `solvent_smiles` is None if the name
        could not be resolved (see `name_to_smiles`).
    """
    specific_comb = data[index]
    solvent = specific_comb["solvent"]
    solvent_smiles = name_to_smiles(solvent)
    return solvent, solvent_smiles


ureg = UnitRegistry()


def convert_unit(temp1: Union[float, str], unit1: str) -> Optional[float]:
    """Convert a temperature to Celsius if the unit is not already Celsius.

    Args:
        temp1: Temperature value, or the literal string "na" for a missing value.
        unit1: Unit of `temp1` as understood by `pint` (e.g. "degF", "kelvin"), or "na".

    Returns:
        The temperature in degrees Celsius, or None if the input is "na" or cannot
        be parsed/converted.
    """
    # Check for 'na' entries
    if temp1 == "na" or unit1 == "na":
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


def get_metadata_polymerization(
    data: dict,
) -> Tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
    """Read the polymerization metadata fields of a single reaction-condition entry.

    Args:
        data: Reaction-condition dict with the keys "temperature", "temperature_unit",
            "method", "polymerization_type", "solvent", "reaction_constants",
            "reaction_constant_conf" and "determination_method".

    Returns:
        A tuple of (temperature, temperature_unit, method, polymerization_type,
        solvent, reaction_constants, reaction_constant_conf, determination_method),
        values as stored in `data`.
    """
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


def get_sequence_of_monomers(test_monomers: List[str], model_monomers: List[str]) -> int:
    """Check whether the monomer order in `model_monomers` matches `test_monomers`.

    Monomers are compared by name first; if the name sets differ, SMILES strings are
    used instead (so equivalent monomers with different names are still recognized).

    Args:
        test_monomers: Reference (ground-truth) monomer names, in their original order.
        model_monomers: Monomer names to compare, in their order.

    Returns:
        0 if the first monomer of both lists matches (same order), 1 if the order is
        swapped.
    """
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
    """Check whether two monomer lists refer to the same set of monomers.

    Falls back to comparing SMILES strings when the raw names don't match, so that
    monomers referred to by different synonyms are still recognized as equal.

    Args:
        model_monomers: Monomer names produced by the model.
        test_monomers: Reference (ground-truth) monomer names.

    Returns:
        True if both lists represent the same set of monomers, False otherwise.
    """
    print(f"test monomers: {test_monomers} vs model monomers: {model_monomers}")
    if set(test_monomers) == set(model_monomers):
        return True
    else:
        test_monomer_smiles = [name_to_smiles(monomer) for monomer in test_monomers]
        model_monomer_smiles = [name_to_smiles(monomer) for monomer in model_monomers]
        return set(test_monomer_smiles) == set(model_monomer_smiles)


def find_matching_reaction(data1: dict, data2: List[str]) -> Optional[int]:
    """Find the reaction in `data1` whose monomers match `data2`.

    Args:
        data1: Model data, either {"reactions": [reaction, ...]} or
            {"reactions": reaction} for a single-reaction record.
        data2: Monomer names to match against each reaction's "monomers" list.

    Raises:
        ValueError: If more than one reaction in `data1` matches `data2`.

    Returns:
        Index of the matching reaction (0 for the single-reaction case), or None if
        no reaction matches.
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
    """Find the reaction-condition entry that best matches the given attributes.

    An exact match is preferred (temperature is unit-converted before comparison).
    If no exact match is found, falls back to fuzzy string matching over all
    attributes combined, so a best guess with a confidence score is always returned.

    Args:
        reaction_conditions: Candidate reaction-condition dicts to search.
        solvent: Solvent name to match.
        temperature: Temperature value to match.
        temp_unit: Unit of `temperature`.
        polymerization_type: Polymerization type to match.
        method: Method to match.
        determination_method: Determination method to match.

    Raises:
        ValueError: If more than one entry is an exact match.

    Returns:
        A (index, confidence) tuple. `confidence` is 1.0 for an exact match, or a
        fuzzy-match ratio in [0, 1] otherwise.
    """
    # We need to do fuzzy matching here and take the best match but also return the confidence
    # of the match
    # first we check if we are lucky and find an exact match, then confidence would
    matching_idxs = []
    for i, comb in enumerate(reaction_conditions):
        if comb["temperature"] != "NA" and comb["temperature_unit"] != "NA":
            temp = convert_unit(temperature, temp_unit)
            temperature_model = convert_unit(comb["temperature"], comb["temperature_unit"])
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


def compare_smiles(smiles1: str, smiles2: str) -> int:
    """Return whether two SMILES strings differ.

    Args:
        smiles1: First SMILES string.
        smiles2: Second SMILES string.

    Returns:
        1 if the strings differ, 0 if they are identical.
    """
    return int(smiles1 != smiles2)


def compare_number_of_reactions(test_file: Union[str, Path], model_file: Union[str, Path]) -> dict:
    """Compare the number of reactions found in a reference and a model YAML file.

    Args:
        test_file: Path to the reference (ground-truth) YAML file.
        model_file: Path to the model-generated YAML file.

    Returns:
        A dict with keys "test", "model" (reaction counts), "equal" (whether they
        match) and "mae" (absolute difference).
    """
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


def get_reaction_constant(
    data: List[dict], index: int
) -> Tuple[List[Optional[float]], List[Optional[float]]]:
    """Return the reaction constants and their confidence intervals for one entry.

    Args:
        data: List of reaction-condition dicts, each with "reaction_constants" and
            "reaction_constant_conf" keys.
        index: Position of the entry to read within `data`.

    Returns:
        A (reaction_constants, reaction_constants_conf) tuple, see
        `get_reaction_const_list` for details.
    """
    specific_comb = data[index]
    reaction_const = specific_comb["reaction_constants"]
    reaction_const_conf = specific_comb["reaction_constant_conf"]
    reaction_const, reaction_const_conf = get_reaction_const_list(
        reaction_const, reaction_const_conf
    )
    return reaction_const, reaction_const_conf


def get_reaction_const_list(
    reaction_const: Optional[Dict[str, Any]], reaction_const_conf: Optional[Dict[str, Any]]
) -> Tuple[List[Optional[float]], List[Optional[float]]]:
    """
    Extract reaction constants and their confidence intervals.

    Parameters:
        reaction_const: A dictionary containing reaction constants or None.
        reaction_const_conf: A dictionary containing reaction confidence intervals or None.

    Returns:
        A tuple containing two lists:
        - reaction_constants: A list of reaction constants, or [None, None] if input is None.
        - reaction_constants_conf: A list of confidence intervals, or [None, None] if input is None.
    """
    # Extract reaction constants
    if reaction_const is None:
        reaction_constants = [None, None]
    else:
        reaction_constants = list(reaction_const.values())

    # Extract reaction confidence intervals
    if reaction_const_conf is None:
        reaction_constants_conf = [None, None]
    else:
        reaction_constants_conf = [
            None if value == "None" else value for value in reaction_const_conf.values()
        ]

    return reaction_constants, reaction_constants_conf


def change_sequence(
    constant1: List[Optional[float]], constant2: List[Optional[float]]
) -> Tuple[List[Optional[float]], List[Optional[float]]]:
    """Swap the two elements of each of the given two-element lists in place.

    Used to realign reaction constants when the monomer order between a reference
    and a model reaction has been flipped (see `get_sequence_of_monomers`).

    Args:
        constant1: Two-element list to swap in place (e.g. [r1, r2]). Left unchanged
            if empty/falsy.
        constant2: Second two-element list to swap in place.

    Returns:
        The same (constant1, constant2) objects, with elements swapped.
    """
    print(constant1, constant2)
    if constant1 and constant2:
        constant1[0], constant1[1] = constant1[1], constant1[0]
        constant2[0], constant2[1] = constant2[1], constant2[0]
    return constant1, constant2


def average(const: List[float]) -> Optional[float]:
    """Return the arithmetic mean of a list of numbers.

    Args:
        const: List of numeric values.

    Returns:
        The mean, or None if `const` is empty.
    """
    if len(const) != 0:
        average_value = sum(const) / len(const)
        return average_value


def count_na_values(data: Any, null_value: str = "na") -> int:
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


def count_total_entries(data: Any) -> int:
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


def calculate_rate(na_count: int, total_count: int) -> float:
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
