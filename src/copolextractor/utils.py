import backoff
import requests
import pubchempy as pcp
import diskcache as dc
from pathlib import Path
from typing import Union
import yaml
from rdkit import Chem
from rdkit.Chem import Descriptors
import re
import os
import json
import numpy as np
import pandas as pd
import hashlib
from openai import OpenAI


# Global embedding cache
embedding_cache = {}


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


# Initialize OpenAI client
try:
    client = OpenAI()
except Exception as e:
    print(f"Warning: Could not initialize OpenAI client: {e}")
    print("Will use hash-based embeddings instead.")
    client = None


def load_embeddings(file_path="embeddings.json"):
    """Load embeddings from a JSON file if it exists."""
    global embedding_cache

    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            embeddings = json.load(file)
            print(f"Loaded {len(embeddings)} embeddings from {file_path}.")
            embedding_cache = {item["name"]: item["embedding"] for item in embeddings}
            return embedding_cache
    return {}


def save_embeddings(embeddings_dict, file_path="output/embeddings.json"):
    """Save embeddings to a JSON file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    embeddings_list = [{"name": name, "embedding": embedding} for name, embedding in embeddings_dict.items()]
    with open(file_path, "w") as file:
        json.dump(embeddings_list, file, indent=4)
    print(f"Saved {len(embeddings_list)} embeddings to {file_path}.")


def get_or_create_embedding(text, model="text-embedding-3-small"):
    """
    Retrieve embeddings for a given text.
    Uses cache if available, otherwise generates new embeddings.
    """
    global embedding_cache
    global client

    # Handle None or NaN text inputs
    if text is None or pd.isna(text):
        print("Warning: Found None or NaN in text, skipping embedding.")
        return None

    # Clean the text by removing newlines for consistency
    text_cleaned = text.replace("\n", " ")

    # Check if the embedding already exists in the cache
    if text_cleaned in embedding_cache:
        return embedding_cache[text_cleaned]

    # If OpenAI client is not available, use a hash-based embedding
    if client is None:
        print(f"Using hash-based embedding for: {text_cleaned}")
        hash_val = int(hashlib.md5(text_cleaned.encode()).hexdigest(), 16)
        np.random.seed(hash_val % 2 ** 32)
        hash_embedding = np.random.normal(0, 1, 1536).tolist()
        embedding_cache[text_cleaned] = hash_embedding
        return hash_embedding

    try:
        # Generate the embedding using the OpenAI API
        response = client.embeddings.create(input=[text_cleaned], model=model)
        embedding = response.data[0].embedding

        # Store the new embedding in the cache
        embedding_cache[text_cleaned] = embedding
        return embedding
    except Exception as e:
        print(f"Error getting embedding for '{text_cleaned}': {e}")
        return None


def is_within_deviation(actual_product, expected_product, deviation=0.10):
    """Check if product is within acceptable deviation"""
    if expected_product == 0:
        return actual_product == 0
    return abs(actual_product - expected_product) / abs(expected_product) <= deviation


# Load embeddings when the module is imported
load_embeddings()


def create_grouped_kfold_splits(df, n_splits=5, random_state=42, id_column='reaction_id'):
    """
    Create K-Fold splits that keep reactions with the same ID together
    in either train or test set.
    """
    # Check if the ID column exists
    if id_column not in df.columns:
        print(f"Warning: {id_column} not found in DataFrame")
        df['temp_reaction_id'] = df.index
        id_column = 'temp_reaction_id'

    # Get unique reaction IDs
    unique_reaction_ids = df[id_column].unique()

    # Shuffle the unique IDs
    np.random.seed(random_state)
    np.random.shuffle(unique_reaction_ids)

    # Create folds based on unique reaction IDs
    folds = np.array_split(unique_reaction_ids, n_splits)

    # Create train/test splits
    splits = []
    for i in range(n_splits):
        test_ids = folds[i]
        test_mask = df[id_column].isin(test_ids)
        test_indices = np.where(test_mask)[0]
        train_indices = np.where(~test_mask)[0]
        splits.append((train_indices, test_indices))

    return splits


# List of radical polymerization types
RADICAL_TYPES = [
    'free radical', 'Free radical', 'Free Radical',
    'atom transfer radical polymerization',
    'atom-transfer radical polymerization',
    'nickel-mediated radical', 'bulk',
    'Radical', 'radical',
    'controlled radical',
    'controlled/living radical',
    'conventional radical polymerization',
    'reversible deactivation radical polymerization',
    'reversible addition-fragmentation chain transfer polymerization',
    'reversible addition-fragmentation chain transfer',
    'Homogeneous Radical',
    'Radiation-induced', 'radiation-induced',
    'Radiation-Initiated',
    'photo-induced polymerization',
    'photopolymerization',
    'thermal polymerization',
    'thermal',
    'group transfer polymerization',
    'Emulsion',
    'Homogeneous Radical',
    'semicontinuous emulsion',
    'emulsion'
]