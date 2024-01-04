import backoff
import requests
import pubchempy as pcp
import diskcache as dc

CACTUS = "https://cactus.nci.nih.gov/chemical/structure/{0}/{1}"


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
        return smiles
    except Exception:
        try:
            compound = pcp.get_compounds(smiles, "name")
            return compound[0].canonical_smiles
        except Exception:
            return None