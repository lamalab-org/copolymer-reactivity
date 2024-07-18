import backoff
import requests
import pubchempy as pcp
import diskcache as dc
from rdkit import Chem
from pathlib import Path
from typing import Union
import yaml
import json


def load_yaml(file_path: Union[str, Path]) -> dict:
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)
    return data


def load_json(file_path: Union[str, Path]) -> dict:
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


CACTUS = "https://cactus.nci.nih.gov/chemical/structure/{0}/{1}"


def canonicalize_smiles(smiles: str) -> str:
    """Canonicalize smiles using RDKit"""
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol)


@backoff.on_exception(backoff.expo, requests.exceptions.RequestException, max_time=10)
def cactus_request_w_backoff(inp, rep="SMILES"):
    url = CACTUS.format(inp, rep)
    response = requests.get(url, allow_redirects=True, timeout=10)
    response.raise_for_status()
    resp = response.text
    if "html" in resp:
        return None
    return resp


cache = dc.Cache("cache")


@cache.memoize()
def name_to_smiles(name: str) -> str:
    """Use the chemical name resolver https://cactus.nci.nih.gov/chemical/structure.
    If this does not work, use pubchem.
    """
    try:
        smiles = cactus_request_w_backoff(name, rep="SMILES")
        if smiles is None:
            raise Exception
        return canonicalize_smiles(smiles)
    except Exception:
        try:
            compound = pcp.get_compounds(name, "name")
            return canonicalize_smiles(compound[0].canonical_smiles)
        except Exception:
            return None


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
        compound = pcp.get_compounds(canonical_smiles, 'smiles')
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