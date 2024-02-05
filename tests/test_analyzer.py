from copolextractor.analyzer import (
    get_total_number_of_combinations,
    find_matching_reaction,
    find_matching_combination,
    convert_unit,
    _get_sequence_of_monomers
)
import pytest


def test_get_total_number_of_combinations():
    assert (
        get_total_number_of_combinations(
            {
                "reaction": [
                    {"combinations": [{"monomers": ["a", "b"]}, {"monomers": ["c", "d"]}]},
                    {"combinations": [{"monomers": ["e", "f"]}, {"monomers": ["g", "h"]}]},
                ]
            }
        )
        == 4
    )
    assert (
        get_total_number_of_combinations(
            {
                "reaction": [
                    {"combinations": [{"monomers": ["a", "b"]}, {"monomers": ["c", "d"]}]},
                    {
                        "combinations": [
                            {"monomers": ["e", "f"]},
                            {"monomers": ["g", "h"]},
                            {"monomers": ["i", "j"]},
                        ]
                    },
                ]
            }
        )
        == 5
    )


@pytest.mark.parametrize(
    "data,monomers,expected",
    [
        (
            {
                "reaction": [
                    {"monomers": ["water", "benzene"]},
                    {"monomers": ["water", "ethanol"]}
                ]
            },
            ["water", "benzene"],
            0,
        ),
        (
            {
                "reaction": [
                    {"monomers": ["water", "benzene"]},
                    {"monomers": ["oxidane", "ethanol"]},
                ]
            },
            ["water", "ethanol"],
            1,
        ),
(
            {
                "reaction": [
                    {"monomers": ["styrol", "benzene"]},
                    {"monomers": ["oxidane", "ethanol"]},
                ]
            },
            ["styren", "benzene"],
            0,
        ),
    ],
)
def test_find_matching_reaction(data, monomers, expected):
    assert find_matching_reaction(data, monomers) == expected


@pytest.mark.parametrize(
    "data,ground_truth,expected_index ,expected_score",
    [
        (
            [
                {
                    "polymerization_type": "radical",
                    "solvent": "water",
                    "method": "A",
                    "determination_method": "Tudor"
                },
                {
                    "polymerization_type": "radical",
                    "solvent": "ethanol",
                    "method": "A",
                    "determination_method": "Tudor"
                },
            ],
            ("radical", "water", "A", "Tudor"),
            0,
            1,
        ),
        (
            [
                {
                    "polymerization_type": "radical (A)",
                    "solvent": "water",
                    "method": "A",
                    "determination_method": "Tudor"
                },
                {
                    "polymerization_type": "radical",
                    "solvent": "ethanol",
                    "method": "A",
                    "determination_method": "B"
                },
            ],
            ("radical", "water", "A", "Tudor"),
            0,
            0.9,
        ),
        (
            [
                {
                    "polymerization_type": "radical (A)",
                    "solvent": "oxidane",
                    "method": "A",
                    "determination_method": "B"
                },
                {
                    "polymerization_type": "radical",
                    "solvent": "ethanol",
                    "method": "A",
                    "determination_method": "Tudor"
                },
            ],
            ("radical", "water", "A", "B"),
            0,
            0.9,
        )
    ],
)
def test_find_matching_combination(data, ground_truth, expected_index, expected_score):
    idx, score = find_matching_combination(data, *ground_truth)
    assert idx == expected_index
    if expected_score != 1:
        assert score <= 1
    else:
        assert score == expected_score


@pytest.mark.parametrize(
    "temp1, temp2, unit1, unit2, expected_temp1, expected_temp2",
        [
            (30, 35, "DegC", "DegC", 30, 35),
            (30, 35, "°C", "DegC", 30, 35),
            (32, 50, "DegF", "K", 0, -223.15),
            (200, 200, "K", "°C", -73.15, 200)
        ]
    )
def test_covert_unit(temp1, temp2, unit1, unit2, expected_temp1, expected_temp2):
    temp_conv1, temp_conv2 = convert_unit(temp1, temp2, unit1, unit2)
    assert abs(expected_temp1 - temp_conv1) < 0.01
    assert abs(expected_temp2 - temp_conv2) < 0.01


@pytest.mark.parametrize(
    "monomers1, monomers2, exp_sequence_change",
    [
        (["e","f"], ["e", "f"], 0),
        (["a", "b"], ["b", "a"], 1)
    ]
)
def test__get_sequence_of_monomers(monomers1, monomers2, exp_sequence_change):
    sequence_change = _get_sequence_of_monomers(monomers1, monomers2)
    assert sequence_change == exp_sequence_change
