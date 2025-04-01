import backoff
import requests
import pubchempy as pcp
import diskcache as dc
from pathlib import Path
from typing import Union
import yaml
import json
from rdkit import Chem
from rdkit.Chem import Descriptors
import re


def load_yaml(file_path: Union[str, Path]) -> dict:
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)
    return data


def load_json(file_path: Union[str, Path]) -> dict:
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def save_json(data, file_path):
    """
    Save JSON data to a file.
    """
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)


def sanitize_filename(filename):
    """Replace invalid characters in filename with underscores."""
    return re.sub(r'[<>:"/\\|?*]', "_", filename)


def calculate_logP(smiles):
    """
    Calculate the logP value for a given SMILES string.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            logP = Descriptors.MolLogP(mol)
            return logP
        else:
            print(f"Conversion failed for SMILES: {smiles}")
            return None
    except Exception as e:
        print(f"Error processing SMILES: {smiles} with error {e}")
        return None


CACTUS = "https://cactus.nci.nih.gov/chemical/structure/{0}/{1}"


def canonicalize_smiles(smiles: str) -> str:
    """Canonicalize smiles using RDKit"""
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol)


@backoff.on_exception(backoff.expo, requests.exceptions.RequestException, max_time=10)
def cactus_request_w_backoff(inp, rep="SMILES"):
    url = CACTUS.format(inp, rep)
    response = requests.get(url, allow_redirects=True, timeout=5)
    response.raise_for_status()
    resp = response.text
    if "html" in resp:
        return None
    return resp


cache = dc.Cache("cache")


def name_to_smiles(name: str, force_retry: bool = True) -> str:
    """Use the chemical name resolver https://cactus.nci.nih.gov/chemical/structure.
    If this does not work, use pubchem.

    Args:
        name: Chemical name to convert
        force_retry: If True, attempts online lookup for None values.
                    If False, uses only cached values.
    """
    cache_key = f"name_to_smiles_{name}"

    # Get from cache
    cached_value = cache.get(cache_key)

    # If we have a cached value and we're not force retrying, return it
    # (even if it's None)
    if not force_retry:
        return cached_value

    # Only proceed with conversion if we're force retrying and the cached value is None
    if cached_value is None:
        try:
            smiles = cactus_request_w_backoff(name, rep="SMILES")
            if smiles is None:
                raise Exception
            result = canonicalize_smiles(smiles)
        except Exception:
            try:
                compound = pcp.get_compounds(name, "name")
                result = canonicalize_smiles(compound[0].canonical_smiles)
            except Exception:
                result = None

        # Cache the result
        cache.set(cache_key, result)
        return result

    return cached_value


@cache.memoize()
def smiles_to_name(smiles: str) -> str:
    """Convert SMILES to a chemical name using CACTUS and PubChem as fallback."""
    canonical_smiles = canonicalize_smiles(smiles)
    try:
        # First try with CACTUS
        name = cactus_request_w_backoff_name(canonical_smiles, rep="name")
        if name is not None:
            return name.strip()
    except Exception:
        pass

    # If CACTUS fails, try with PubChem
    try:
        compound = pcp.get_compounds(canonical_smiles, "smiles")
        if compound:
            return compound[0].iupac_name
    except Exception:
        pass

    return None


def cactus_request_w_backoff_name(smiles, rep="name"):
    url = CACTUS.format(smiles, rep)
    response = requests.get(url, allow_redirects=True, timeout=10)
    response.raise_for_status()
    resp = response.text
    if "html" in resp:
        return None
    return resp

