from copolextractor.analyzer import (
    get_total_number_of_combinations,
    find_matching_reaction,
    find_matching_combination,
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
            {"reaction": [{"monomers": ["water", "benzene"]}, {"monomers": ["water", "ethanol"]}]},
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
    ],
)
def test_find_matching_reaction(data, monomers, expected):
    assert find_matching_reaction(data, monomers) == expected


@pytest.mark.parametrize(
    "data,ground_truth,expected,expected_score",
    [
        (
            [
                {
                    "polymerization_type": "radical",
                    "solvent": "water",
                    "temperature": 10,
                    "method": "A",
                },
                {
                    "polymerization_type": "radical",
                    "solvent": "ethanol",
                    "temperature": 20,
                    "method": "A",
                },
            ],
            ("radical", "water", 10, "A"),
            0,
            1,
        ),
        (
            [
                {
                    "polymerization_type": "radical (A)",
                    "solvent": "water",
                    "temperature": 10,
                    "method": "A",
                },
                {
                    "polymerization_type": "radical",
                    "solvent": "ethanol",
                    "temperature": 20,
                    "method": "A",
                },
            ],
            ("radical", "water", 10, "A"),
            0,
            0.9,
        ),
        (
            [
                {
                    "polymerization_type": "radical (A)",
                    "solvent": "oxidane",
                    "temperature": 10,
                    "method": "A",
                },
                {
                    "polymerization_type": "radical",
                    "solvent": "ethanol",
                    "temperature": 20,
                    "method": "A",
                },
            ],
            ("radical", "water", 10, "A"),
            0,
            0.9,
        )
    ],
)
def test_find_matching_combination(data, ground_truth, expected, expected_score):
    idx, score =  find_matching_combination(data, *ground_truth)
    assert idx == expected
    if expected_score != 1:
        assert score <= 1
    else:
        assert score == expected_score