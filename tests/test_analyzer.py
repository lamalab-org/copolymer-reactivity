from copolextractor.analyzer import (
    get_total_number_of_reaction_conditions,
    find_matching_reaction,
    find_matching_reaction_conditions,
    convert_unit,
    get_sequence_of_monomers,
    get_number_of_reactions,
    change_sequence
)
import pytest


def test_get_total_number_of_reaction_conditions():
    assert (
        get_total_number_of_reaction_conditions(
            {
                "reactions": [
                    {"reaction_conditions": [{"monomers": ["a", "b"]}, {"monomers": ["c", "d"]}]},
                    {"reaction_conditions": [{"monomers": ["e", "f"]}, {"monomers": ["g", "h"]}]},
                ]
            }
        )
        == 4
    )
    assert (
        get_total_number_of_reaction_conditions(
            {
                "reactions": [
                    {"reaction_conditions": [{"monomers": ["a", "b"]}, {"monomers": ["c", "d"]}]},
                    {
                        "reaction_conditions": [
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
                "reactions": [
                    {"monomers": ["water", "benzene"]},
                    {"monomers": ["water", "ethanol"]}
                ]
            },
            ["water", "benzene"],
            0,
        ),
        (
            {
                "reactions": [
                    {"monomers": ["water", "benzene"]},
                    {"monomers": ["oxidane", "ethanol"]},
                ]
            },
            ["water", "ethanol"],
            1,
        ),
(
            {
                "reactions": [
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
                    "solvent": "H2O",
                    "temperature": 10,
                    "temperature_unit": "°C",
                    "polymerization_type": "radical",
                    "method": "A",
                    "determination_method": "Tudor"
                },
                {
                    "solvent": "ethanol",
                    "temperature": 10,
                    "temperature_unit": "°C",
                    "polymerization_type": "radical",
                    "method": "A",
                    "determination_method": "B"
                },
            ],
            ("water", 10, "°C", "radical", "A", "Tudor"),
            0,
            1,
        ),
        (
            [
                {
                    "solvent": "H2O",
                    "temperature": 10,
                    "temperature_unit": "°C",
                    "polymerization_type": "radical (A)",
                    "method": "A",
                    "determination_method": "Tudor"
                },
                {
                    "solvent": "ethanol",
                    "temperature": 10,
                    "temperature_unit": "°C",
                    "polymerization_type": "radical",
                    "method": "A",
                    "determination_method": "B"
                },
            ],
            ("water", 10, "°C", "radical", "A", "Tudor"),
            0,
            0.9,
        ),
        (
            [
                {
                    "solvent": "water",
                    "temperature": 10,
                    "temperature_unit": "°C",
                    "polymerization_type": "radical (A)",
                    "method": "A",
                    "determination_method": "B"
                },
                {
                    "solvent": "ethanol",
                    "temperature": 20,
                    "temperature_unit": "K",
                    "polymerization_type": "radical",
                    "method": "A",
                    "determination_method": "Tudor"
                },
            ],
            ("ethanol", 20, "K" "radical", "A", "Tudor"),
            1,
            0.9,
        )
    ],
)
def test_find_matching_reaction_conditions(data, ground_truth, expected_index, expected_score):
    idx, score = find_matching_reaction_conditions(data, *ground_truth)
    assert idx == expected_index
    if expected_score != 1:
        assert score <= 1
    else:
        assert score == expected_score


@pytest.mark.parametrize(
    "temp1, temp2, unit1, unit2, expected_temp1, expected_temp2",
        [
            (30, 35, "°C", "°C", 30, 35),
            (32, 50, "°F", "K", 0, -223.15),
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
def test_get_sequence_of_monomers(monomers1, monomers2, exp_sequence_change):
    sequence_change = get_sequence_of_monomers(monomers1, monomers2)
    assert sequence_change == exp_sequence_change


def test_get_number_of_reactions():
    assert (
            get_number_of_reactions(
                {
                    "reactions": [
                        {"reaction_conditions": [{"monomers": ["a", "b"]}, {"monomers": ["c", "d"]}]},
                        {"reaction_conditions": [{"monomers": ["e", "f"]}, {"monomers": ["g", "h"]}]},
                    ]
                }
            )
            == 2
    )
    assert (
            get_number_of_reactions(
                {
                    "reactions": [
                        {"reaction_conditions": [{"monomers": ["a", "b"]}, {"monomers": ["c", "d"]}]},
                        {
                            "reaction_conditions": [
                                {"monomers": ["e", "f"]},
                                {"monomers": ["g", "h"]},
                                {"monomers": ["i", "j"]},
                            ]
                        },
                    ]
                }
            )
            == 2
    )
    assert (
            get_number_of_reactions(
                {
                    "reactions": [
                        {"reaction_conditions": [{"monomers": ["a", "b"]}, {"monomers": ["c", "d"]}]},
                    ]
                }
            )
            == 1
    )


@pytest.mark.parametrize(
    "const1, const2, exp_output1, exp_output2",
    [
        (["e","f"], ["a", "b"], ["f","e"], ["b", "a"]),
        (["a", "b"], ["b", "a"], ["b", "a"], ["a", "b"])
    ]
)
def test_change_sequence(const1, const2, exp_output1, exp_output2):
    assert change_sequence(const1, const2) == (exp_output1, exp_output2)
