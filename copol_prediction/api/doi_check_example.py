"""
Example script to check if a DOI exists in the dataset using the API.
"""

import requests

# API URL (change this to your ngrok URL or localhost)
API_URL = "http://localhost:8000"


def check_doi(doi: str) -> dict:
    """
    Check if a DOI exists in the dataset.
    
    Args:
        doi: The DOI to check (can be full URL or just the DOI number)
        
    Returns:
        Dictionary with check results
    """
    response = requests.post(
        f"{API_URL}/check_doi",
        json={"doi": doi}
    )
    response.raise_for_status()
    return response.json()


# Beispiel 1: DOI mit vollständiger URL
print("=" * 50)
print("Beispiel 1: DOI mit vollständiger URL")
print("=" * 50)
result = check_doi("https://doi.org/10.1002/macp.1973.021650110")
print(f"DOI: {result['doi']}")
print(f"Existiert: {'JA ✓' if result['exists'] else 'NEIN ✗'}")
print(f"Normalisierte DOI: {result['normalized_doi']}")
print(f"Zeitstempel: {result['timestamp']}")
print()

# Beispiel 2: Nur die DOI-Nummer
print("=" * 50)
print("Beispiel 2: Nur DOI-Nummer")
print("=" * 50)
result = check_doi("10.1002/pol.1973.170110204")
print(f"DOI: {result['doi']}")
print(f"Existiert: {'JA ✓' if result['exists'] else 'NEIN ✗'}")
print()

# Beispiel 3: Eine DOI, die wahrscheinlich nicht existiert
print("=" * 50)
print("Beispiel 3: Nicht existierende DOI")
print("=" * 50)
result = check_doi("10.1234/nicht-existierend")
print(f"DOI: {result['doi']}")
print(f"Existiert: {'JA ✓' if result['exists'] else 'NEIN ✗'}")
print()

# Beispiel 4: Batch-Check für mehrere DOIs
print("=" * 50)
print("Beispiel 4: Batch-Check für mehrere DOIs")
print("=" * 50)
dois_to_check = [
    "10.1016/0014-3057(84)90010-7",
    "10.1234/beispiel-doi-1",
    "10.1234/beispiel-doi-2"
]

for doi in dois_to_check:
    try:
        result = check_doi(doi)
        status = "✓ IM DATENSATZ" if result['exists'] else "✗ NICHT IM DATENSATZ"
        print(f"{doi}: {status}")
    except Exception as e:
        print(f"{doi}: ERROR - {e}")


