import json
import re
from crossref.restful import Works

# JSON-Dateien laden
with open('enhanced_doi_list.json', 'r') as file:
    data_languages = json.load(file)

# JSON-Datei mit DOI-URLs laden
with open('../../doi_extraction/doi_output.json', 'r') as file:
    data_dois = json.load(file)

# Liste der DOIs extrahieren
doi_list = data_dois['doi_list']


def extract_year_from_crossref(doi):
    works = Works()
    try:
        meta = works.doi(doi)
        return meta['published']['date-parts'][0][0]
    except Exception:
        return None

# Dictionary zur Speicherung der Jahreszahlen für jedes Paper
doi_to_year = {}

# Jahre aus den DOIs extrahieren
for doi in doi_list:
    year = extract_year_from_crossref(doi)
    if year:
        doi_to_year[doi] = year

# Aktualisierung der enhanced_doi_list.json mit den Jahreszahlen
for entry in data_languages:
    doi = entry.get('paper', '')
    year = doi_to_year.get(doi, None)
    if year:
        entry['year'] = year

# Aktualisierte JSON-Daten speichern
with open('data_languages_updated.json', 'w') as file:
    json.dump(data_languages, file, indent=4)

print("Die Jahreszahlen wurden erfolgreich extrahiert und hinzugefügt.")
