import json
import pubchempy as pcp


def get_smiles_from_name(monomer):
    try:
        compound = pcp.get_compounds(monomer, 'name')
        if compound:
            return compound[0].canonical_smiles
        else:
            return "SMILES not found for this chemical name"
    except pcp.PubChemHTTPError as e:
        return f"PubChem Error: {str(e)}"


with open('test_data.json', 'r') as file:
    test_data = json.load(file)

with open('extracted_data.json', 'r') as file:
    extracted_data = json.load(file)

monomer_error = 0
reaction_const_error = 0
sum_error = 0
not_found_error = 0

# Iteration über beide Datensätze gleichzeitig
for test_entry, extracted_entry in zip(test_data['reaction'], extracted_data['reaction']):
    for test_combination, extracted_combination in zip(test_entry['combinations'], extracted_entry['combinations']):

        test_monomer = test_combination['monomers']
        extracted_monomer = extracted_combination['monomers']
        smiles_test_monomer = get_smiles_from_name(test_monomer)
        smiles_extracted_monomer = get_smiles_from_name(extracted_monomer)

        if set(smiles_test_monomer) == set(smiles_extracted_monomer):
            print("Monomers match:", test_monomer)
        else:
            monomer_error += 1
            print("Monomers do not match:", test_monomer, "vs", extracted_monomer)

        test_constants = test_combination['reaction_constants']
        extracted_constants = extracted_combination['reaction_constants']

        for key, value in test_constants.items():
            if key in extracted_constants:
                if abs(value - extracted_constants[key]) == 0:
                    print(f"Reaction constant {key} matches:", value)
                else:
                    reaction_const_error += 1
                    print(f"Reaction constant {key} does not match:", value, "vs", extracted_constants[key])
            else:
                not_found_error += 1
                print(f"Reaction constant {key} not found in extracted data")

sum_error = monomer_error + reaction_const_error + not_found_error
print("monomer names do not match:" + str(monomer_error))
print("reaction constant do not match:" + str(reaction_const_error))
print("data was not found in the document:" + str(not_found_error))
print("sum of errors:" + str(sum_error))
