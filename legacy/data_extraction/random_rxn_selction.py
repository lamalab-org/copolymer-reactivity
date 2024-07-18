import random
import json


input_file = 'enhanced_doi_list1.json'

with open(input_file, 'r', encoding='utf-8') as file:
    data = json.load(file)


max_rxn_number = 15


print("Rohdaten:", data[:10])

selected_entries = []
rxn_groups = {i: [] for i in range(max_rxn_number + 1)}

for entry in data:
    if entry.get('downloaded', False) and entry.get('processed', False) and 'rxn_number' in entry:
        rxn_number = int(entry['rxn_number'])
        if rxn_number <= max_rxn_number:
            rxn_groups[rxn_number].append({'out': entry['out'], 'rxn_number': rxn_number})
    else:
        if not ('rxn_number' in entry):
            print("Fehlender 'rxn_number' bei Eintrag:", entry)  # Gibt den gesamten Eintrag aus

# Auswahl und Speicherung der Daten
for rxn_number, entries in rxn_groups.items():
    if len(entries) >= 2:
        selected_entries.extend(random.sample(entries, 2))
    elif len(entries) == 1:
        selected_entries.append(entries[0])

with open('collected_data/selected_entries.json', 'w', encoding='utf-8') as file:
    json.dump(selected_entries, file, ensure_ascii=False, indent=4)

print("Die ausgewählten Einträge wurden in 'selected_entries.json' gespeichert.")