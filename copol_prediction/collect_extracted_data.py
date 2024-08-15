import os
from rdkit import Chem
import json
import copolextractor.analyzer as az
import copolextractor.utils as utils


folder_path = '../data_extraction/data_extraction_GPT-4o/model_output_extraction'

results = []

file_count = 0
reaction_count = 0
input_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".json")])

for filename in input_files:
    if filename.endswith(".json"):
        file_count += 1
        file_path = os.path.join(folder_path, filename)
        print(file_path)
        try:
            with open(file_path, 'r') as file:
                file_content = json.load(file)
                data = json.loads(file_content)
        except json.JSONDecodeError:
            continue
        try:
            if isinstance(data, dict):
                reaction_count += az.get_total_number_of_reaction_conditions(data)
                reactions = data.get('reactions', [])
                source = data.get('source')
                for reaction in reactions:
                    monomers = reaction.get('monomers', [])
                    for monomer_pair in monomers:
                        monomer1 = monomer_pair.get('monomer1')
                        monomer2 = monomer_pair.get('monomer2')

                        monomer1_smiles = utils.name_to_smiles(monomer1)
                        monomer2_smiles = utils.name_to_smiles(monomer2)

                        reaction_conditions = reaction.get('reaction_conditions', [])
                        for condition in reaction_conditions:
                            temp = condition.get('temperature')
                            unit = condition.get('temperature_unit')
                            temperature = az.convert_unit(temp, unit)
                            solvent_smiles = utils.name_to_smiles(condition.get('solvent'))
                            r_values = condition.get('reaction_constants', {})
                            conf_intervals = condition.get('reaction_constant_conf', {})
                            method = condition.get('method')
                            product = condition.get('r_product')

                            result = {
                                'file': filename,
                                'monomer1_s': monomer1_smiles,
                                'monomer2_s': monomer2_smiles,
                                'monomer1': monomer1,
                                'monomer2': monomer2,
                                'r_values': r_values,
                                'conf_intervals': conf_intervals,
                                'temperature': temperature,
                                'temperature_unit': '°C',
                                'solvent': solvent_smiles,
                                'method': method,
                                'r-product': product,
                                'source': source
                            }
                            results.append(result)
                            print(result)
            else:
                print(f"Unexpected file format of {filename}")
        except (AttributeError, json.JSONDecodeError):
            continue

print(f"Number of files: {file_count}")
print(f"Number of reactions: {reaction_count}")
for result in results:
    print(result)

with open('../data_extraction/data_extraction_GPT-4o/collected_data/extracted_data_collected_without_fp.json', 'w') as file:
    json.dump(results, file, indent=4)

