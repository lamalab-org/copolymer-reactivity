from copolextractor.analyzer import get_total_number_of_combinations, find_matching_reaction
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
