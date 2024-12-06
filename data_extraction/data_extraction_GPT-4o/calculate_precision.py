import json


with open('output/copol_database/RF_training_paper.json', 'r') as file:
    data = json.load(file)


for entry in data:
    number_of_extracted_rxn = entry.get("number_of_extracted_rxn", 0)
    correct_rxn_number = entry.get("correct_rxn_number", 0)
    if number_of_extracted_rxn > 0:
        precision = correct_rxn_number / number_of_extracted_rxn
    elif number_of_extracted_rxn == 0 and correct_rxn_number:
        precision = 0
    else:
        precision = None
    entry["precision"] = precision


with open('output/copol_database/RF_training_paper.json', 'w') as file:
    json.dump(data, file, indent=4)

print(json.dumps(data, indent=4))
