import os
import tempfile
from pathlib import Path

import pytest

from copolextractor.utils import (
    calculate_logP,
    canonicalize_smiles,
    is_within_deviation,
    load_json,
    sanitize_filename,
    save_json,
)


@pytest.mark.parametrize(
    "filename, expected",
    [
        ("normal_file.txt", "normal_file.txt"),
        ("file<with>special:chars.pdf", "file_with_special_chars.pdf"),
        ('file"with"quotes.txt', "file_with_quotes.txt"),
        ("file/with\\slashes.doc", "file_with_slashes.doc"),
        ("file|with*question?.xml", "file_with_question_.xml"),
        ("", ""),
    ],
)
def test_sanitize_filename(filename, expected):
    result = sanitize_filename(filename)
    assert result == expected


@pytest.mark.parametrize(
    "smiles, expected_logp_range",
    [
        ("C", (-1, 2)),  # Methane - very low logP
        ("CCCCCC", (2, 5)),  # Hexane - higher logP
        ("CCO", (-1, 1)),  # Ethanol - negative logP
        ("c1ccccc1", (1, 3)),  # Benzene - moderate logP
        ("CC(=O)O", (-1, 1)),  # Acetic acid - low logP
    ],
)
def test_calculate_logP(smiles, expected_logp_range):
    result = calculate_logP(smiles)
    assert result is not None
    assert expected_logp_range[0] <= result <= expected_logp_range[1]


def test_calculate_logP_invalid_smiles():
    """Test with invalid SMILES string"""
    result = calculate_logP("INVALID_SMILES_123")
    assert result is None


@pytest.mark.parametrize(
    "smiles, expected_canonical",
    [
        ("C(C)C", "CCC"),  # Propane
        ("C(C)(C)C", "CC(C)C"),  # Isobutane
        ("C1CCCCC1", "C1CCCCC1"),  # Cyclohexane
        ("c1ccccc1", "c1ccccc1"),  # Benzene
        ("CC(O)C", "CC(C)O"),  # 2-propanol
    ],
)
def test_canonicalize_smiles(smiles, expected_canonical):
    result = canonicalize_smiles(smiles)
    assert result == expected_canonical


@pytest.mark.parametrize(
    "actual, expected, deviation, should_match",
    [
        (10, 10, 0.1, True),  # Exact match
        (10, 11, 0.1, True),  # Within 10% deviation
        (10, 9, 0.1, False),  # abs(10-9)/abs(9) = 0.111 > 0.1
        (10, 12, 0.1, False),  # Outside 10% deviation
        (10, 8, 0.1, False),  # Outside 10% deviation
        (0, 0, 0.1, True),  # Both zero
        (1, 0, 0.1, False),  # Expected zero, actual non-zero
        (0, 1, 0.1, False),  # Actual zero, expected non-zero
        (100, 105, 0.1, True),  # Within 10% for larger numbers
        (100, 110, 0.1, True),  # abs(100-110)/abs(110) = 0.0909 < 0.1
    ],
)
def test_is_within_deviation(actual, expected, deviation, should_match):
    result = is_within_deviation(actual, expected, deviation)
    assert result == should_match


def test_save_and_load_json():
    """Test saving and loading JSON files"""
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.json"

        # Test data
        test_data = {
            "key1": "value1",
            "key2": 123,
            "key3": [1, 2, 3],
            "key4": {"nested": "data"},
        }

        # Save JSON
        save_json(test_data, test_file)

        # Check file exists
        assert test_file.exists()

        # Load JSON
        loaded_data = load_json(test_file)

        # Verify data matches
        assert loaded_data == test_data


def test_save_json_requires_existing_directory():
    """Test that save_json requires parent directories to exist"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create the directory first
        nested_dir = Path(tmpdir) / "subdir1" / "subdir2"
        nested_dir.mkdir(parents=True, exist_ok=True)

        nested_path = nested_dir / "test.json"
        test_data = {"test": "data"}

        # Save should work with existing directory
        save_json(test_data, nested_path)

        # Verify file was created
        assert nested_path.exists()

        # Verify data is correct
        loaded_data = load_json(nested_path)
        assert loaded_data == test_data
