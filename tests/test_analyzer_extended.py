import pytest

from copolextractor.analyzer import (
    _compare_monomers,
    _extract_monomers,
    average,
    count_na_values,
    extract_reaction_conditions,
)


def test_extract_monomers_from_list():
    """Test extracting monomers from a data structure with reactions as a list"""
    data = {
        "reactions": [
            {"monomers": ["ethene", "propene"]},
            {"monomers": ["styrene", "butadiene"]},
        ]
    }
    monomers = _extract_monomers(data)
    assert len(monomers) == 4
    assert "ethene" in monomers
    assert "propene" in monomers
    assert "styrene" in monomers
    assert "butadiene" in monomers


def test_extract_monomers_from_dict():
    """Test extracting monomers from a data structure with reactions as a dict"""
    data = {"reactions": {"monomers": ["vinyl chloride", "vinyl acetate"]}}
    monomers = _extract_monomers(data)
    assert len(monomers) == 2
    assert "vinyl chloride" in monomers
    assert "vinyl acetate" in monomers


def test_extract_monomers_empty():
    """Test extracting monomers when none are present"""
    data = {"reactions": [{"temperature": 60}]}
    monomers = _extract_monomers(data)
    assert len(monomers) == 0


@pytest.mark.parametrize(
    "model_monomers, test_monomers, expected_match",
    [
        (["ethene", "propene"], ["ethene", "propene"], True),
        (["ethene", "propene"], ["propene", "ethene"], True),  # Order doesn't matter
        (["ethene", "propene"], ["ethene", "butene"], False),
        (["water", "ethanol"], ["oxidane", "ethanol"], True),  # Synonyms
        (["styrol", "benzene"], ["styren", "benzene"], True),  # Synonyms
    ],
)
def test_compare_monomers(model_monomers, test_monomers, expected_match):
    """Test comparing monomer lists"""
    result = _compare_monomers(model_monomers, test_monomers)
    assert result == expected_match


@pytest.mark.parametrize(
    "values, expected_avg",
    [
        ([1, 2, 3, 4, 5], 3.0),
        ([10, 20, 30], 20.0),
        ([1.5, 2.5, 3.5], 2.5),
        ([100], 100.0),
        ([-1, 0, 1], 0.0),
        ([0, 0, 0], 0.0),
    ],
)
def test_average(values, expected_avg):
    """Test average calculation"""
    result = average(values)
    assert result == expected_avg


def test_average_empty_list():
    """Test average with empty list"""
    result = average([])
    assert result is None


@pytest.mark.parametrize(
    "data, expected_count",
    [
        ({"temp": "na", "pressure": 10}, 1),
        ({"temp": "na", "pressure": "na"}, 2),
        ({"temp": 60, "pressure": 1}, 0),
        ({"nested": {"value1": "na", "value2": "na"}}, 2),
        ([{"temp": "na"}, {"temp": "na"}], 2),
        ({"list": ["na", "na", "na"]}, 3),
        (
            {
                "reactions": [
                    {"temp": "na", "solvent": "water"},
                    {"temp": 60, "solvent": "na"},
                ]
            },
            2,
        ),
    ],
)
def test_count_na_values(data, expected_count):
    """Test counting 'na' values in nested structures"""
    result = count_na_values(data, null_value="na")
    assert result == expected_count


def test_count_na_values_custom_null():
    """Test counting with custom null value"""
    data = {"temp": "N/A", "pressure": "N/A", "method": "A"}
    result = count_na_values(data, null_value="N/A")
    assert result == 2


def test_extract_reaction_conditions():
    """Test extracting reaction conditions from a list"""
    conditions = [
        {"temp": 60, "solvent": "water"},
        {"temp": 80, "solvent": "ethanol"},
        {"temp": 100, "solvent": "methanol"},
    ]
    result = extract_reaction_conditions(conditions)
    assert len(result) == 3
    assert result[0]["temp"] == 60
    assert result[1]["temp"] == 80
    assert result[2]["temp"] == 100


def test_extract_reaction_conditions_empty():
    """Test extracting reaction conditions from empty list"""
    result = extract_reaction_conditions([])
    assert len(result) == 0
