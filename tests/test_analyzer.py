from copolextractor.analyzer import get_total_number_of_combinations


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
